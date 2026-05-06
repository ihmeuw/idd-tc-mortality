"""
Poisson component: count-outcome rate model.

Fit: Poisson GLM (log link) with log(exposed) offset, coefficient fixed at 1.
y is raw death counts (integers >= 0); exposure enters only as offset, not as a
free covariate or weight.

The model is:
    log(E[y]) = X @ beta + log(exposed)
    Var[y] = mu   (equidispersion — NB2 relaxes this)

predict returns rate-scale predictions. The full expression is:

    rate = exp(X @ beta + log_exposed) / exp(log_exposed) = exp(X @ beta)

The log_exposed argument cancels exactly, so predictions are
exposure-independent given covariates.

When Poisson equidispersion holds, this is the efficient estimator for the
count DGP. When overdispersion is present, NB2 is preferred; Poisson
estimates remain consistent but standard errors are underestimated.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.families.links import Log

from idd_tc_mortality.distributions.base import FitResult


def fit(
    X: pd.DataFrame,
    y: np.ndarray,
    log_exposed: np.ndarray,
) -> FitResult:
    """Fit a Poisson GLM with log link and log(exposed) offset.

    Parameters
    ----------
    X:
        Design matrix. Must already include an intercept column named 'const'.
        log(exposed) must NOT be a column in X — passed separately as offset.
    y:
        Non-negative integer death counts, length n_obs.
    log_exposed:
        log(exposed) for each observation, length n_obs. Used as offset with
        coefficient fixed at 1.

    Returns
    -------
    FitResult
        params: coefficient array in X column order.
        param_names: X column names.
        fitted_values: in-sample predictions on the rate scale (exp(X @ params)).
        converged: whether the GLM IRLS converged.
        cov: None (placeholder; uncertainty module not yet built).
        meta: dict with 'n_obs', 'iterations', 'warnings'.

    Raises
    ------
    ValueError
        If X contains 'log_exposed' or 'log_exp', log_exposed contains
        non-finite values, y contains negative or non-integer values, or
        lengths are inconsistent.
    """
    _validate_inputs(X, y, log_exposed)

    model = sm.GLM(
        endog=y,
        exog=X,
        family=Poisson(link=Log()),
        offset=log_exposed,
    )

    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit()
        if not result.converged:
            converged = False

    n_iter = result.fit_history.get("iteration", None) if hasattr(result, "fit_history") else None

    # Rate-scale fitted values: exp(X @ beta). log_exposed cancels exactly.
    fitted_rates = np.exp(np.asarray(X) @ np.asarray(result.params))

    return FitResult(
        params=np.asarray(result.params),
        param_names=list(X.columns),
        fitted_values=fitted_rates,
        family="poisson",
        converged=converged,
        cov=None,
        meta={
            "n_obs": int(len(y)),
            "iterations": n_iter,
            "warnings": [str(w.message) for w in caught],
        },
    )


def predict(
    result: FitResult,
    X: pd.DataFrame,
    log_exposed: np.ndarray,
) -> np.ndarray:
    """Predict death rates from a fitted Poisson model.

    The full computation is exp(X @ params + log_exposed) / exp(log_exposed),
    which simplifies to exp(X @ params). log_exposed cancels exactly.

    Parameters
    ----------
    result:
        FitResult from poisson.fit().
    X:
        Design matrix. Columns must match result.param_names exactly.
    log_exposed:
        log(exposed) for the prediction rows. Accepted for API consistency
        with NB; cancels in the rate computation.

    Returns
    -------
    np.ndarray
        Predicted rates (strictly positive), length n_obs.
    """
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names "
            f"{result.param_names}. Call features.align_X before predicting."
        )
    # Explicit count-to-rate: exp(X @ params + log_exposed) / exp(log_exposed).
    # log_exposed cancels exactly — written out to make the intent auditable.
    eta = np.asarray(X) @ result.params + log_exposed
    predicted_counts = np.exp(eta)
    return predicted_counts / np.exp(log_exposed)


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    X: pd.DataFrame,
    y: np.ndarray,
    log_exposed: np.ndarray,
) -> None:
    forbidden = {"log_exposed", "log_exp"}
    overlap = forbidden.intersection(set(X.columns))
    if overlap:
        raise ValueError(
            f"X must not contain {overlap}. Pass log(exposed) via the "
            "log_exposed argument so the offset coefficient stays fixed at 1."
        )

    if not np.all(np.isfinite(log_exposed)):
        raise ValueError("log_exposed contains non-finite values.")

    if not np.all(y >= 0):
        raise ValueError("y must be non-negative. Found negative values.")

    if not np.all(y == np.floor(y)):
        raise ValueError("y must contain integer values. Found non-integer values.")

    if len(y) != len(X):
        raise ValueError(f"y length {len(y)} != X rows {len(X)}.")

    if len(log_exposed) != len(X):
        raise ValueError(f"log_exposed length {len(log_exposed)} != X rows {len(X)}.")
