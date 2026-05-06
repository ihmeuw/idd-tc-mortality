"""
Scaled logit component: continuous rate strictly bounded in (0, threshold).

The model is a Gaussian GLM with ScaledLogitLink(threshold):

    g(mu) = log(mu / (threshold - mu))    [link]
    g^{-1}(eta) = threshold / (1 + exp(-eta))    [inverse link]

For Gaussian errors, the MLE with any link reduces to WLS on the working
response. With the true link, the working response equals the link-transformed
outcome exactly — so fit() uses WLS on z = log(y / (threshold - y)) with
observation weights. This is equivalent to the Gaussian GLM with ScaledLogitLink
and produces the correct weighted MLE without requiring statsmodels to accept a
custom link through its validation layer (which rejects non-approved link classes
for the Gaussian family in statsmodels 0.14).

threshold is stored in meta["threshold_rate"] because predict requires it to apply
the inverse link. A FitResult without meta["threshold_rate"] is uninterpretable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families.links import Link

from idd_tc_mortality.distributions.base import FitResult


class ScaledLogitLink(Link):
    """Logit link scaled to the interval (0, threshold).

    Link:        g(mu)        = log(mu / (threshold - mu))
    Inverse:     g^{-1}(eta)  = threshold / (1 + exp(-eta))
    Derivative:  g'(mu)       = 1/mu + 1/(threshold - mu)
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def _clean(self, p: np.ndarray) -> np.ndarray:
        return np.clip(p, 1e-10, self.threshold - 1e-10)

    def __call__(self, mu: np.ndarray) -> np.ndarray:
        mu = self._clean(np.asarray(mu, dtype=float))
        return np.log(mu / (self.threshold - mu))

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return self.threshold / (1.0 + np.exp(-np.asarray(eta, dtype=float)))

    def deriv(self, mu: np.ndarray) -> np.ndarray:
        mu = self._clean(np.asarray(mu, dtype=float))
        return 1.0 / mu + 1.0 / (self.threshold - mu)

    def deriv2(self, mu: np.ndarray) -> np.ndarray:
        mu = self._clean(np.asarray(mu, dtype=float))
        return -1.0 / mu ** 2 + 1.0 / (self.threshold - mu) ** 2

    def inverse_deriv(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        exp_neg = np.exp(-eta)
        return self.threshold * exp_neg / (1.0 + exp_neg) ** 2


def fit(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> FitResult:
    """Fit a Gaussian model with scaled logit link via WLS on the link-transformed outcome.

    Parameters
    ----------
    X:
        Design matrix. Assembled by the caller; may include log_exposed as a
        free covariate column.
    y:
        Continuous rate strictly in (0, threshold).
    weights:
        Observation weights, strictly positive. Typically the exposed population.
    threshold:
        Upper bound of the rate scale. Baked into the link; stored in
        meta["threshold_rate"] for use by predict.

    Returns
    -------
    FitResult
        params: coefficient array in X column order.
        param_names: X column names.
        fitted_values: in-sample predictions in (0, threshold).
        converged: True (WLS is closed-form).
        cov: None (placeholder; uncertainty module not yet built).
        meta: dict with 'n_obs', 'threshold', 'iterations' (always 1), 'warnings'.

    Raises
    ------
    ValueError
        If threshold <= 0, y contains values outside (0, threshold),
        weights contains non-positive values, or lengths are inconsistent.
    """
    _validate_inputs(X, y, weights, threshold)

    # Link transform: z = g(y) = log(y / (threshold - y))
    z = np.log(y / (threshold - y))

    wls = sm.WLS(endog=z, exog=X, weights=weights)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wls.fit()

    # Inverse link: g^{-1}(eta) = threshold / (1 + exp(-eta))
    fitted_z = np.asarray(result.fittedvalues)
    fitted_y = threshold / (1.0 + np.exp(-fitted_z))

    return FitResult(
        params=np.asarray(result.params),
        param_names=list(X.columns),
        fitted_values=fitted_y,
        family="scaled_logit",
        converged=True,
        cov=None,
        meta={
            "n_obs": int(len(y)),
            "threshold_rate": threshold,
            "iterations": 1,
            "warnings": [str(w.message) for w in caught],
        },
    )


def predict(
    result: FitResult,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict rates in (0, threshold) from a fitted scaled logit model.

    Parameters
    ----------
    result:
        FitResult from scaled_logit.fit().
    X:
        Design matrix. Columns must match result.param_names exactly
        (use features.align_X before calling).

    Returns
    -------
    np.ndarray
        Predicted rates in (0, threshold), length n_obs.
    """
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names "
            f"{result.param_names}. Call features.align_X before predicting."
        )
    threshold_rate = result.meta["threshold_rate"]
    eta = np.asarray(X) @ result.params
    return threshold_rate / (1.0 + np.exp(-eta))


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> None:
    if threshold <= 0:
        raise ValueError(f"threshold must be strictly positive. Got {threshold}.")

    if not np.all(y > 0):
        raise ValueError(
            f"y must be strictly in (0, threshold={threshold}). Found values <= 0."
        )
    if not np.all(y < threshold):
        raise ValueError(
            f"y must be strictly in (0, threshold={threshold}). Found values >= threshold."
        )

    if not np.all(weights > 0):
        raise ValueError("weights must be strictly positive. Found non-positive values.")

    if len(y) != len(X):
        raise ValueError(f"y length {len(y)} != X rows {len(X)}.")

    if len(weights) != len(X):
        raise ValueError(f"weights length {len(weights)} != X rows {len(X)}.")
