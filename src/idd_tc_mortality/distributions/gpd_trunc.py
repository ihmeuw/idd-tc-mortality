"""GPD renormalized-TRUNCATED-likelihood tail variant (variant B). SPIKE — under evaluation.

Fits a GPD renormalized onto each storm's (0, H_i], H_i = c_i - threshold_rate: each storm
contributes  log f_GPD(w_i; sigma_i, xi) - log F_GPD(H_i; sigma_i, xi), so the cap shapes the
fitted (beta, xi). Warm-started from the untruncated GPD MLE. Reports xi +/- SE from the BFGS
inverse Hessian so the Beirlant weak-identification risk (flat likelihood in xi under hard
truncation) is visible.

Prediction is the renormalized (truncated) mean
    E[W | W <= H] = (E[min(W,H)] - H * S(H)) / (1 - S(H)),
reusing the MC-verified censored mean from :mod:`gpd_cens` plus the GPD survival. Bounded in
(0, H] and finite for any xi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idd_tc_mortality.distributions import gpd as _gpd
from idd_tc_mortality.distributions._trunc import fit_truncated
from idd_tc_mortality.distributions.base import FitResult
from idd_tc_mortality.distributions.gpd_cens import gpd_censored_mean

_XI_EPS = 1e-6


def _gpd_survival(sigma: np.ndarray, xi: float, H: np.ndarray) -> np.ndarray:
    """S(H) = P(W > H) for W ~ GPD(sigma, xi). 0 beyond the finite support when xi < 0."""
    if abs(xi) < _XI_EPS:
        return np.exp(-H / sigma)
    base = 1.0 + xi * H / sigma
    base = np.where(base > 0.0, base, 0.0)  # xi<0: beyond support => S=0
    with np.errstate(divide="ignore"):
        return np.where(base > 0.0, base ** (-1.0 / xi), 0.0)


def _neg_ll(params, X, y, H, w, n_beta):
    """Negative weighted truncated GPD log-likelihood and its analytic gradient.

    neg_ll_i = -log f(w_i) + log F(H_i). The -log f part reuses the base-GPD gradient
    (gpd.py); the + log F(H) part adds the truncation term derived below (chain rule
    through sigma = exp(X beta), with zH = 1 + xi*H/sigma, S_H = zH^(-1/xi), F = 1 - S_H,
    g_H = (zH - 1)/zH):
        d(log F)/d(beta_j) = -(S_H/(xi*F)) * g_H * X_j
        d(log F)/d(xi)     = -(S_H/(xi^2*F)) * (log zH - g_H)
    with xi->0 limits d(log F)/d(beta_j) -> -(S_H/F)(H/sigma) X_j and
    d(log F)/d(xi) -> -(S_H/F)(H/sigma)^2 / 2.
    """
    beta = params[:n_beta]
    xi = params[n_beta]
    log_sigma = X @ beta
    sigma = np.exp(log_sigma)
    z = 1.0 + xi * y / sigma      # at the observation
    zH = 1.0 + xi * H / sigma     # at the cap
    if np.any(z <= 0.0) or np.any(zH <= 0.0):
        return 1e20, np.zeros_like(params)
    HS = H / sigma
    if abs(xi) < _XI_EPS:
        u = y / sigma
        S_H = np.exp(-HS)
        F = 1.0 - S_H
        if np.any(F <= 1e-15):
            return 1e20, np.zeros_like(params)
        ll = (-log_sigma - u) - np.log(F)
        gb = (1.0 - u) - (S_H / F) * HS                       # d neg_ll / d(mu_i)
        gx = (u - u * u / 2.0) - (S_H / F) * (HS * HS) / 2.0  # d neg_ll / d(xi)
    else:
        S_H = zH ** (-1.0 / xi)
        F = 1.0 - S_H
        if np.any(F <= 1e-15):
            return 1e20, np.zeros_like(params)
        ll = (-log_sigma - (1.0 / xi + 1.0) * np.log(z)) - np.log(F)
        k = (1.0 + xi) * y / (sigma * z)
        g_H = (zH - 1.0) / zH
        gb = (1.0 - k) - (S_H / (xi * F)) * g_H
        gx = (-(1.0 / xi ** 2) * np.log(z) + k / xi) - (S_H / (xi ** 2 * F)) * (np.log(zH) - g_H)
    neg = -float(np.sum(w * ll))
    grad_beta = X.T @ (w * gb)
    grad_xi = float(np.sum(w * gx))
    return neg, np.append(grad_beta, grad_xi)


def fit(X, y, weights, excess_cap):
    """Truncated GPD MLE on (0, H_i]; see :mod:`_trunc`. Reports xi +/- SE in meta."""
    return fit_truncated(
        X, y, weights, excess_cap,
        base_fit=_gpd.fit,
        neg_ll=_neg_ll,
        x0_from_base=lambda b: np.append(b.params, b.meta["shape_param"]),  # [beta, xi]
        shape_transform=lambda xi, var: (float(xi), float(np.sqrt(max(var, 0.0)))),
        family="gpd_trunc",
    )


def predict(result: FitResult, X: pd.DataFrame, excess_cap: np.ndarray) -> np.ndarray:
    """Truncated mean E[W | W <= H] = (E[min(W,H)] - H*S(H)) / (1 - S(H))."""
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names {result.param_names}."
        )
    xi = float(result.meta["shape_param"])
    sigma = np.exp(np.asarray(X, dtype=float) @ result.params)
    H = np.maximum(np.asarray(excess_cap, dtype=float), 0.0)
    if H.shape != sigma.shape:
        raise ValueError(f"excess_cap length {H.shape} != X rows {sigma.shape}.")
    return truncated_mean(sigma, xi, H)


def truncated_mean(sigma: np.ndarray, xi: float, H: np.ndarray) -> np.ndarray:
    """E[W | W <= H], elementwise. Reuses the GPD censored mean + survival."""
    sigma = np.asarray(sigma, dtype=float)
    H = np.maximum(np.asarray(H, dtype=float), 0.0)
    cm = gpd_censored_mean(sigma, xi, H)        # E[min(W,H)]
    S = _gpd_survival(sigma, xi, H)
    F = 1.0 - S
    safe = F > 1e-12
    out = np.where(safe, (cm - H * S) / np.where(safe, F, 1.0), 0.5 * H)  # H->0 limit ~ H/2
    return np.clip(out, 0.0, H)
