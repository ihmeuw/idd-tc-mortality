"""Log-logistic renormalized-TRUNCATED-likelihood tail variant (variant B). SPIKE.

Identical scheme to :mod:`gpd_trunc` but with a log-logistic renormalized onto each storm's
(0, H_i]: each storm contributes  log f_LL(w_i; alpha_i, k) - log F_LL(H_i; alpha_i, k). The
optimizer works on (beta, log_k); the reported shape is k +/- SE (delta method). Warm-started
from the untruncated log-logistic MLE.

Prediction is the truncated mean
    E[W | W <= H] = (E[min(W,H)] - H * S(H)) / (1 - S(H)),
reusing the MC-verified censored mean from :mod:`log_logistic_cens` plus the log-logistic
survival S(H) = 1 / (1 + (H/alpha)^k).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from idd_tc_mortality.distributions import log_logistic as _log_logistic
from idd_tc_mortality.distributions._trunc import fit_truncated
from idd_tc_mortality.distributions.base import FitResult
from idd_tc_mortality.distributions.log_logistic_cens import loglogistic_censored_mean


def _ll_survival(alpha: np.ndarray, k: float, H: np.ndarray) -> np.ndarray:
    """S(H) = P(W > H) = 1 / (1 + (H/alpha)^k) for W ~ LogLogistic(alpha, k)."""
    return 1.0 / (1.0 + (H / alpha) ** k)


def _neg_ll(params, X, y, H, w, n_beta):
    """Negative weighted truncated log-logistic log-likelihood and analytic gradient.

    neg_ll_i = -log f(w_i) + log F(H_i), params = [beta..., log_k]. The -log f part reuses
    the base log-logistic gradient (log_logistic.py) via t = tanh(k r / 2); the + log F(H)
    term (F(H) = sigmoid(k*rH), rH = log H - mu, S_H = 1 - F = sigmoid(-k*rH)) contributes
        d(log F)/d(beta_j) = -k * S_H * X_j,   d(log F)/d(log_k) = k * rH * S_H.
    """
    beta = params[:n_beta]
    log_k = params[n_beta]
    k = float(np.exp(log_k))
    mu = X @ beta                       # = log(alpha)
    r = np.log(y) - mu
    rH = np.log(H) - mu
    log_f = np.log(k) + (k - 1.0) * np.log(y) - k * mu - 2.0 * np.logaddexp(0.0, k * r)
    log_F = -np.logaddexp(0.0, -k * rH)   # log sigmoid(k rH) = log F(H)
    ll = log_f - log_F
    neg = -float(np.sum(w * ll))

    t = np.tanh(k * r / 2.0)              # base-family term
    S_H = expit(-k * rH)                  # 1 - F(H)
    grad_beta = X.T @ (w * (-k) * (t + S_H))          # -k*t (base) + -k*S_H (truncation)
    grad_log_k = float(np.sum(w * (-1.0 + k * r * t + k * rH * S_H)))
    return neg, np.append(grad_beta, grad_log_k)


def fit(X, y, weights, excess_cap):
    """Truncated log-logistic MLE on (0, H_i]; see :mod:`_trunc`. Reports k +/- SE in meta."""
    return fit_truncated(
        X, y, weights, excess_cap,
        base_fit=_log_logistic.fit,
        neg_ll=_neg_ll,
        x0_from_base=lambda b: np.append(b.params, np.log(b.meta["shape_param"])),  # [beta, log_k]
        shape_transform=lambda log_k, var: (float(np.exp(log_k)),
                                            float(np.exp(log_k) * np.sqrt(max(var, 0.0)))),
        family="log_logistic_trunc",
    )


def predict(result: FitResult, X: pd.DataFrame, excess_cap: np.ndarray) -> np.ndarray:
    """Truncated mean E[W | W <= H] = (E[min(W,H)] - H*S(H)) / (1 - S(H))."""
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names {result.param_names}."
        )
    k = float(result.meta["shape_param"])
    alpha = np.exp(np.asarray(X, dtype=float) @ result.params)
    H = np.maximum(np.asarray(excess_cap, dtype=float), 0.0)
    if H.shape != alpha.shape:
        raise ValueError(f"excess_cap length {H.shape} != X rows {alpha.shape}.")
    return truncated_mean(alpha, k, H)


def truncated_mean(alpha: np.ndarray, k: float, H: np.ndarray) -> np.ndarray:
    """E[W | W <= H], elementwise. Reuses the log-logistic censored mean + survival."""
    alpha = np.asarray(alpha, dtype=float)
    H = np.maximum(np.asarray(H, dtype=float), 0.0)
    cm = loglogistic_censored_mean(alpha, k, H)   # E[min(W,H)]
    S = _ll_survival(alpha, k, H)
    F = 1.0 - S
    safe = F > 1e-12
    out = np.where(safe, (cm - H * S) / np.where(safe, F, 1.0), 0.5 * H)
    return np.clip(out, 0.0, H)
