"""
Gamma component: bulk death rate given deaths > 0.

Fit: Gamma GLM with log link. X is assembled by the caller and may include
log_exposed as a free covariate (unlike S1/S2 where it is an offset). weights
are observation-level weights, typically the exposed population count.

The log link gives multiplicative covariate effects on the rate scale, which is
the natural parameterization for TC mortality: a unit increase in wind_speed
multiplies the expected rate by exp(beta).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log

from idd_tc_mortality.distributions.base import FitResult


def fit(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
) -> FitResult:
    """Fit a Gamma GLM with log link.

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
        fitted_values: in-sample predictions on the rate scale (exp(X @ params)).
        converged: whether statsmodels GLM iteration converged.
        cov: None (placeholder; uncertainty module not yet built).
        meta: dict with 'n_obs', 'iterations', 'warnings'.

    Raises
    ------
    ValueError
        If y contains non-positive values, weights contains non-positive values,
        or lengths are inconsistent.
    """
    _validate_inputs(X, y, weights)

    # Normalize weights so the dispersion estimate is invariant to exposure magnitude.
    # Without normalization, the dispersion parameter scales with mean(weights),
    # which inflates it when weights = exposed (mean ~100K–1M). Consistent with
    # how lognormal.fit() normalizes its WLS weights.
    weights_norm = weights / float(np.mean(weights))
    glm = sm.GLM(
        endog=y,
        exog=X,
        family=Gamma(link=Log()),
        var_weights=weights_norm,
    )

    # Provide stable starting params: intercept = log(weighted mean of y), slopes = 0.
    # The default IRLS initialisation uses mu = y directly, which produces eta = log(y).
    # When y is on the order of 1e-8 (tail excess rates) this gives eta ≈ -18, and the
    # first WLS step drives some fitted values to zero, collapsing IRLS weights.
    start_params = np.zeros(X.shape[1])
    start_params[0] = np.log(np.average(y, weights=weights))

    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = glm.fit(start_params=start_params)
        if not result.converged:
            converged = False

    n_iter = result.fit_history.get("iteration", None) if hasattr(result, "fit_history") else None

    return FitResult(
        params=np.asarray(result.params),
        param_names=list(X.columns),
        fitted_values=np.asarray(result.fittedvalues),
        family="gamma",
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
) -> np.ndarray:
    """Predict rates from a fitted Gamma log-link model.

    Parameters
    ----------
    result:
        FitResult from gamma.fit().
    X:
        Design matrix. Columns must match result.param_names exactly
        (use features.align_X before calling).

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
    return np.exp(np.asarray(X) @ result.params)


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
) -> None:
    if not np.all(y > 0):
        raise ValueError("y must be strictly positive (Gamma support). Found non-positive values.")

    if not np.all(weights > 0):
        raise ValueError("weights must be strictly positive. Found non-positive values.")

    if len(y) != len(X):
        raise ValueError(f"y length {len(y)} != X rows {len(X)}.")

    if len(weights) != len(X):
        raise ValueError(f"weights length {len(weights)} != X rows {len(X)}.")
