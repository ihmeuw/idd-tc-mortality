"""
Beta regression component: rate conditional on being in (0, threshold).

Fit: Beta regression via _WeightedBetaModel (a BetaModel subclass) with logit link
for the mean. y must already be on the (0, 1) scale — the caller divides by threshold
before calling. This module does not know about the threshold.

The mean model is: logit(mu) = X @ beta.
The precision model uses a single intercept (constant phi across observations).
Only the mean model params are stored in FitResult.params; precision params are
stored in meta["precision_params"] for inspection.

Weights are applied by scaling per-observation log-likelihoods in loglikeobs. They
are normalized to sum to n_obs so that uniform weights produce identical results to
an unweighted fit, and total log-likelihood magnitude stays comparable across sample
sizes.

predict returns fitted probabilities in (0, 1). Multiplying by threshold to recover
the rate scale is the caller's responsibility.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.othermod.betareg import BetaModel

from idd_tc_mortality.distributions.base import FitResult


class _WeightedBetaModel(BetaModel):
    """BetaModel subclass that applies per-observation weights via loglikeobs.

    Weights are normalized to sum to n_obs before scaling so that:
    - Uniform weights (all ones) produce identical results to an unweighted fit.
    - Total log-likelihood magnitude stays comparable across sample sizes.
    """

    def __init__(
        self,
        endog: np.ndarray,
        exog: pd.DataFrame,
        obs_weights: np.ndarray,
        **kwargs,
    ) -> None:
        super().__init__(endog, exog, **kwargs)
        n = len(endog)
        w = np.asarray(obs_weights, dtype=float)
        self._obs_weights = w * (n / w.sum())

    def loglikeobs(self, params: np.ndarray) -> np.ndarray:
        return super().loglikeobs(params) * self._obs_weights


def fit(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
) -> FitResult:
    """Fit a Beta regression model with logit link and observation weights.

    Parameters
    ----------
    X:
        Design matrix. Assembled by the caller; may include log_exposed as a
        free covariate column.
    y:
        Outcome in the open interval (0, 1). Exactly 0 or 1 raises. The caller
        is responsible for dividing rates by the threshold before passing y.
    weights:
        Observation weights, strictly positive. Typically the exposed population.
        Normalized to sum to n_obs internally; uniform weights give the same
        result as an unweighted fit.

    Returns
    -------
    FitResult
        params: mean model coefficient array in X column order.
        param_names: X column names.
        fitted_values: in-sample predicted means in (0, 1).
        converged: whether the MLE optimizer reported convergence.
        cov: None (placeholder; uncertainty module not yet built).
        meta: dict with 'n_obs', 'precision_params', 'iterations', 'warnings'.

    Raises
    ------
    ValueError
        If y contains values <= 0 or >= 1, weights contains non-positive values,
        or lengths are inconsistent.
    """
    _validate_inputs(X, y, weights)

    model = _WeightedBetaModel(endog=y, exog=X, obs_weights=weights)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(disp=0)

    converged = True
    if hasattr(result, "mle_retvals") and result.mle_retvals is not None:
        converged = bool(result.mle_retvals.get("converged", True))

    n_iter = None
    if hasattr(result, "mle_retvals") and result.mle_retvals is not None:
        n_iter = result.mle_retvals.get("iterations", None)

    # Mean model params are the first X.shape[1] entries; remainder are precision params
    n_mean = X.shape[1]
    mean_params = np.asarray(result.params[:n_mean])
    precision_params = np.asarray(result.params[n_mean:])

    return FitResult(
        params=mean_params,
        param_names=list(X.columns),
        fitted_values=np.asarray(result.fittedvalues),
        family="beta",
        converged=converged,
        cov=None,
        meta={
            "n_obs": int(len(y)),
            "precision_params": precision_params.tolist(),
            "iterations": n_iter,
            "warnings": [str(w.message) for w in caught],
        },
    )


def predict(
    result: FitResult,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict fitted means in (0, 1) from a fitted Beta regression model.

    Parameters
    ----------
    result:
        FitResult from beta.fit().
    X:
        Design matrix. Columns must match result.param_names exactly
        (use features.align_X before calling).

    Returns
    -------
    np.ndarray
        Predicted means in (0, 1), length n_obs. Multiply by threshold to
        recover rate scale.
    """
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names "
            f"{result.param_names}. Call features.align_X before predicting."
        )
    eta = np.asarray(X) @ result.params
    # logit inverse (expit)
    return 1.0 / (1.0 + np.exp(-eta))


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
) -> None:
    if not np.all(y > 0):
        raise ValueError(
            "y must be strictly in (0, 1). Found values <= 0. "
            "Divide rates by threshold before calling beta.fit()."
        )
    if not np.all(y < 1):
        raise ValueError(
            "y must be strictly in (0, 1). Found values >= 1. "
            "Divide rates by threshold before calling beta.fit()."
        )

    if not np.all(weights > 0):
        raise ValueError("weights must be strictly positive. Found non-positive values.")

    if len(y) != len(X):
        raise ValueError(f"y length {len(y)} != X rows {len(X)}.")

    if len(weights) != len(X):
        raise ValueError(f"weights length {len(weights)} != X rows {len(X)}.")
