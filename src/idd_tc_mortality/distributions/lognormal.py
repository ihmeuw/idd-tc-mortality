"""
Lognormal component: bulk death rate given deaths > 0.

Fit: WLS on log(y) with observation weights (typically exposed). This is equivalent
to a Gaussian GLM with identity link on the log scale. The residual standard deviation
sigma is stored in meta because it is required for unbiased back-transformation:

    E[y] = exp(mu + sigma² / 2)

where mu = X @ params is the predicted log-rate. Using exp(mu) alone (the median)
systematically under-predicts the mean, which matters for expected-death calculations.

sigma is estimated from the weighted residuals: sqrt(result.scale), where statsmodels
WLS sets result.scale = WRSS / (n - p).
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
    weights: np.ndarray,
) -> FitResult:
    """Fit a lognormal model via WLS on log(y).

    Parameters
    ----------
    X:
        Design matrix. Assembled by the caller; may include log_exposed as a
        free covariate column. Must already include an intercept column if one
        is wanted.
    y:
        Strictly positive continuous outcome (death rate). Length n_obs.
    weights:
        Observation weights, strictly positive. Typically the exposed population
        count for each observation.

    Returns
    -------
    FitResult
        params: coefficient array in X column order.
        param_names: X column names.
        fitted_values: in-sample predictions on the rate scale,
            exp(X @ params + sigma² / 2).
        converged: True (WLS is closed-form; always converges).
        cov: None (placeholder; uncertainty module not yet built).
        meta: dict with 'n_obs', 'sigma', 'iterations' (always 1), 'warnings'.

    Raises
    ------
    ValueError
        If y contains non-positive values, weights contains non-positive values,
        or lengths are inconsistent.
    """
    _validate_inputs(X, y, weights)

    log_y = np.log(y)

    # Normalize weights so result.scale is invariant to exposure magnitude.
    # Without normalization, result.scale ≈ sigma² * mean(weights), which inflates
    # sigma when weights = exposed (mean ~100K–1M), causing exp(sigma²/2) overflow.
    weights_norm = weights / float(np.mean(weights))
    wls = sm.WLS(endog=log_y, exog=X, weights=weights_norm)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wls.fit()

    sigma = float(np.sqrt(result.scale))
    fitted_log = np.asarray(result.fittedvalues)
    exponent = fitted_log + 0.5 * sigma**2
    # Clip at 700 to prevent float64 overflow: exp(709) ≈ 8e307, exp(710) → inf.
    # Clipping at 700 (exp(700) ≈ 1e304) leaves a safe margin.
    n_clipped = int(np.sum(exponent > 700))
    exponent_clipped = np.clip(exponent, None, 700)
    fitted_rate = np.exp(exponent_clipped)

    clip_warning = (
        [f"Clipped {n_clipped} fitted value(s): exponent > 700 would overflow float64."]
        if n_clipped > 0
        else []
    )

    return FitResult(
        params=np.asarray(result.params),
        param_names=list(X.columns),
        fitted_values=fitted_rate,
        family="lognormal",
        converged=True,
        cov=None,
        meta={
            "n_obs": int(len(y)),
            "sigma": sigma,
            "iterations": 1,
            "n_clipped": n_clipped,
            "warnings": [str(w.message) for w in caught] + clip_warning,
        },
    )


def predict(
    result: FitResult,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict rates from a fitted lognormal model (lognormal mean, not median).

    Parameters
    ----------
    result:
        FitResult from lognormal.fit().
    X:
        Design matrix. Columns must match result.param_names exactly
        (use features.align_X before calling).

    Returns
    -------
    np.ndarray
        Predicted rates on the original scale: exp(X @ params + sigma² / 2).
        Strictly positive, length n_obs.
    """
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names "
            f"{result.param_names}. Call features.align_X before predicting."
        )
    sigma = result.meta["sigma"]
    exponent = np.asarray(X) @ result.params + 0.5 * sigma**2
    return np.exp(np.clip(exponent, None, 700))


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
) -> None:
    if not np.all(y > 0):
        raise ValueError("y must be strictly positive (lognormal support). Found non-positive values.")

    if not np.all(weights > 0):
        raise ValueError("weights must be strictly positive. Found non-positive values.")

    if len(y) != len(X):
        raise ValueError(f"y length {len(y)} != X rows {len(X)}.")

    if len(weights) != len(X):
        raise ValueError(f"weights length {len(weights)} != X rows {len(X)}.")
