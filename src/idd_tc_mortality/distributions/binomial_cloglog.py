"""
Shared helper for fitting a Binomial GLM with complementary log-log link and a
log(exposed) offset whose coefficient is fixed at 1.

Both S1 and S2 use this exact model family. The theoretical basis is:

    P(deaths = 0 | N, p) = (1 - p)^N ≈ exp(-N * p)

so the log probability of observing zero deaths scales linearly with log(N).
The cloglog link is the canonical link for this process. log(N) must be an offset
(coefficient fixed at 1), not a free covariate — estimating it freely would let the
intercept absorb a mixture of lethality and exposure effects, making coefficients
uninterpretable and predictions that do not correctly scale with population size.

Usage
-----
This module exposes two public functions: fit_binomial_cloglog and
predict_binomial_cloglog. s1.py and s2.py call them; nothing else should.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import CLogLog

from idd_tc_mortality.distributions.base import FitResult


def fit_binomial_cloglog(
    X: pd.DataFrame,
    y: np.ndarray,
    log_exposed: np.ndarray,
    family_name: str,
) -> FitResult:
    """Fit a Binomial GLM with cloglog link and log(exposed) offset.

    Parameters
    ----------
    X:
        Design matrix. Must already include an intercept column named 'const'.
        Basin dummies should have one level dropped (drop_first=True) before calling.
        log(exposed) must NOT be a column in X — it is passed separately as an offset.
    y:
        Binary outcome array (0/1), length n_obs.
    log_exposed:
        log(exposed) for each observation, length n_obs. Used as offset with
        coefficient fixed at 1. Will raise if any value is non-finite.
    family_name:
        String label for FitResult.family, e.g. 's1' or 's2'.

    Returns
    -------
    FitResult
        params: coefficient array in X column order.
        param_names: X column names.
        fitted_values: predicted probabilities (cloglog scale → probability scale),
            length n_obs.
        converged: whether statsmodels GLM iteration converged.
        cov: None (placeholder; uncertainty module not yet built).
        meta: dict with 'n_obs', 'n_events', 'iterations', 'warnings'.

    Raises
    ------
    ValueError
        If X contains a column named 'log_exposed' or 'log_exp' (would duplicate the
        offset), if log_exposed contains non-finite values, or if y is not binary.
    """
    _validate_inputs(X, y, log_exposed)

    glm = sm.GLM(
        endog=y,
        exog=X,
        family=Binomial(link=CLogLog()),
        offset=log_exposed,
    )

    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = glm.fit()
        if not result.converged:
            converged = False

    n_iter = result.fit_history.get("iteration", None) if hasattr(result, "fit_history") else None

    return FitResult(
        params=np.asarray(result.params),
        param_names=list(X.columns),
        fitted_values=np.asarray(result.fittedvalues),
        family=family_name,
        converged=converged,
        cov=None,
        meta={
            "n_obs": int(len(y)),
            "n_events": int(y.sum()),
            "iterations": n_iter,
            "warnings": [str(w.message) for w in caught],
        },
    )


def predict_binomial_cloglog(
    result: FitResult,
    X: pd.DataFrame,
    log_exposed: np.ndarray,
) -> np.ndarray:
    """Predict probabilities from a fitted cloglog Binomial model.

    Parameters
    ----------
    result:
        FitResult returned by fit_binomial_cloglog.
    X:
        Design matrix for prediction. Columns must match result.param_names exactly
        (use features.align_X before calling).
    log_exposed:
        log(exposed) offset for the prediction rows.

    Returns
    -------
    np.ndarray
        Predicted probabilities, length n_obs. Values are in (0, 1].
    """
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names "
            f"{result.param_names}. Call features.align_X before predicting."
        )
    if not np.all(np.isfinite(log_exposed)):
        raise ValueError("log_exposed contains non-finite values.")

    eta = np.asarray(X) @ result.params + log_exposed
    # cloglog inverse link: F(eta) = 1 - exp(-exp(eta))
    return 1.0 - np.exp(-np.exp(eta))


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

    unique_y = set(np.unique(y))
    if not unique_y.issubset({0, 1}):
        raise ValueError(f"y must be binary (0/1). Found values: {unique_y}.")

    if len(y) != len(X):
        raise ValueError(f"y length {len(y)} != X rows {len(X)}.")

    if len(log_exposed) != len(X):
        raise ValueError(f"log_exposed length {len(log_exposed)} != X rows {len(X)}.")
