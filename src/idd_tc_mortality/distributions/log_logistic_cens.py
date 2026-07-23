"""Log-logistic tail — CENSORED-at-cap MEAN variant (bounded/finite-mean selection, variant C).

The fit is IDENTICAL to the standard log-logistic tail model (:mod:`log_logistic`):
weighted BFGS MLE on the excess rate ``w = death_rate - threshold_rate`` with a shared
scalar shape ``k`` and log-linear scale ``log(alpha_i) = X_i @ beta`` (``alpha`` is the
median). The fit is re-exported unchanged, matching variant C: the censoring atom at the
cap is empty in-sample, so the censored-likelihood fit equals the untruncated fit.

:func:`predict` returns the CENSORED mean of the excess

    E[min(W, H_i)],   W ~ LogLogistic(alpha_i, k),   H_i = c_i - threshold_rate,

closed form via the Gauss hypergeometric function (``integral_0^H S(w) dw`` with the
log-logistic survival ``S(w) = 1 / (1 + (w/alpha)^k)``):

    E[min(W, H)] = H * 2F1(1, 1/k; 1 + 1/k; -(H/alpha)^k).

Checks: at k=1 this reduces to ``alpha * ln(1 + H/alpha)`` and at k=2 to
``alpha * arctan(H/alpha)`` (standard 2F1 reductions), and it is FINITE for every ``k`` —
including ``k <= 1`` where the ordinary log-logistic mean ``alpha*(pi/k)/sin(pi/k)`` is
infinite. That finiteness is the point of variant C.

``predict_component`` adds ``threshold_rate`` back (``tail_outcome='excess'``) and supplies
the per-row excess cap ``H_i`` (``needs_cap=True``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import hyp2f1

from idd_tc_mortality.distributions.base import FitResult
from idd_tc_mortality.distributions.log_logistic import fit  # noqa: F401 — re-exported: identical fit


def predict(result: FitResult, X: pd.DataFrame, excess_cap: np.ndarray) -> np.ndarray:
    """Predict log-logistic CENSORED-mean excess rates.

    Parameters
    ----------
    result:
        FitResult from :func:`log_logistic.fit` (re-exported here as ``fit``).
    X:
        Design matrix; columns must match ``result.param_names`` (call ``align_X`` first).
    excess_cap:
        Per-row excess-scale cap ``H_i = c_i - threshold_rate``, length == len(X).
    """
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names "
            f"{result.param_names}. Call features.align_X before predicting."
        )
    k = float(result.meta["shape_param"])
    alpha = np.exp(np.asarray(X, dtype=float) @ result.params)  # median = scale alpha_i
    H = np.asarray(excess_cap, dtype=float)
    if H.shape != alpha.shape:
        raise ValueError(
            f"excess_cap length {H.shape} does not match X rows {alpha.shape}."
        )
    return loglogistic_censored_mean(alpha, k, H)


def loglogistic_censored_mean(alpha: np.ndarray, k: float, H: np.ndarray) -> np.ndarray:
    """E[min(W, H)] for W ~ LogLogistic(alpha, k), elementwise over alpha and H.

    Uses ``H * 2F1(1, 1/k; 1 + 1/k; -(H/alpha)^k)``. Always finite and non-negative;
    ``H <= 0`` yields 0 (and 2F1(...,0)=1 gives 0 automatically at H=0).
    """
    alpha = np.asarray(alpha, dtype=float)
    H = np.maximum(np.asarray(H, dtype=float), 0.0)
    z = -((H / alpha) ** k)
    return H * hyp2f1(1.0, 1.0 / k, 1.0 + 1.0 / k, z)
