"""
Negative Binomial component: count-outcome tail model.

Fit: Negative Binomial GLM (NB2) with log link and log(exposed) offset, coefficient
fixed at 1. y is raw death counts (integers >= 0); exposure enters only as offset,
not as a free covariate or weight.

The model is:
    log(E[y]) = X @ beta + log(exposed)
    Var[y] = mu + alpha * mu^2   (NB2 parameterization)

where alpha is the overdispersion parameter estimated alongside beta.

predict returns rate-scale predictions. The full expression is:

    rate = exp(X @ beta + log_exposed) / exp(log_exposed) = exp(X @ beta)

The log_exposed argument cancels exactly, so predictions are exposure-independent
given covariates. This is correct: the rate model should not depend on the
arbitrary exposure window, only on the covariates.

Only the mean model params (X columns) are stored in FitResult.params. The
overdispersion parameter(s) from statsmodels are stored in meta["dispersion_params"].
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from idd_tc_mortality.distributions.base import FitResult


def fit(
    X: pd.DataFrame,
    y: np.ndarray,
    log_exposed: np.ndarray,
) -> FitResult:
    """Fit a Negative Binomial GLM with log link and log(exposed) offset.

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
        params: mean model coefficient array in X column order (excludes alpha).
        param_names: X column names.
        fitted_values: in-sample predictions on the rate scale (exp(X @ params)).
        converged: whether the MLE optimizer reported convergence.
        cov: None (placeholder; uncertainty module not yet built).
        meta: dict with 'n_obs', 'dispersion_params', 'iterations', 'warnings'.

    Raises
    ------
    ValueError
        If X contains 'log_exposed' or 'log_exp', log_exposed contains non-finite
        values, y contains negative or non-integer values, or lengths are
        inconsistent.
    """
    _validate_inputs(X, y, log_exposed)

    model = sm.NegativeBinomial(
        endog=y,
        exog=X,
        offset=log_exposed,
    )

    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(disp=0)
        if hasattr(result, "mle_retvals") and result.mle_retvals is not None:
            converged = bool(result.mle_retvals.get("converged", True))

    n_iter = None
    if hasattr(result, "mle_retvals") and result.mle_retvals is not None:
        n_iter = result.mle_retvals.get("iterations", None)

    # Mean model params are the first X.shape[1] entries; remainder is alpha
    n_mean = X.shape[1]
    mean_params = np.asarray(result.params[:n_mean])
    dispersion_params = np.asarray(result.params[n_mean:])

    # Rate-scale fitted values: exp(X @ beta). statsmodels NegativeBinomial is a
    # CountModel, not a GLM — result.fittedvalues does not return predicted counts.
    # Compute directly from mean_params, consistent with predict().
    fitted_rates = np.exp(np.asarray(X) @ mean_params)

    return FitResult(
        params=mean_params,
        param_names=list(X.columns),
        fitted_values=fitted_rates,
        family="nb",
        converged=converged,
        cov=None,
        meta={
            "n_obs": int(len(y)),
            "dispersion_params": dispersion_params.tolist(),
            "iterations": n_iter,
            "warnings": [str(w.message) for w in caught],
        },
    )


def predict(
    result: FitResult,
    X: pd.DataFrame,
    log_exposed: np.ndarray,
) -> np.ndarray:
    """Predict death rates from a fitted NB model.

    The full computation is exp(X @ params + log_exposed) / exp(log_exposed),
    which simplifies to exp(X @ params). log_exposed cancels exactly: rate
    predictions are exposure-independent given covariates, as required.

    Parameters
    ----------
    result:
        FitResult from nb.fit().
    X:
        Design matrix. Columns must match result.param_names exactly
        (use features.align_X before calling).
    log_exposed:
        log(exposed) for the prediction rows. Accepted for API consistency;
        cancels in the rate computation and does not affect the result.

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
    # Explicit count-to-rate conversion: exp(X @ params + log_exposed) / exp(log_exposed)
    # The log_exposed terms cancel exactly, confirming rate predictions are
    # exposure-independent. Written out to make the intent auditable, not as an accident.
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
