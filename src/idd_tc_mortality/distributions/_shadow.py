"""Shared core for the Cirillo-Taleb dual-distribution shadow-mean tail variants (variant A).

Both A families (:mod:`gpd_shadow`, :mod:`log_logistic_shadow`) use the identical
construction; they differ ONLY in which distribution is fit to the dual and how the
shadow mean of the fraction is computed from that fit. This module holds the common
machinery so the two families cannot drift apart.

Construction (each storm keeps its OWN cap c_i = P_i/E_i on BOTH legs):

  In-leg:   g_i = w_i / H_i in (0,1],  H_i = c_i - threshold_rate  (per-storm excess cap)
  Dual:     z_i = -ln(1 - g_i)  maps (0,1) -> [0, inf)
  Fit:      one distribution (GPD or log-logistic) to the pooled z (intercept-only)
  Shadow:   mu_g = E[1 - exp(-Z)] = 1 - E[exp(-Z)]  in (0,1) for ANY tail heaviness
  Out-leg:  excess mean_i = mu_g * H_i   (predict_component then adds threshold_rate)

``mu_g_from_fit`` is the only family-specific piece of the fit: it takes the fitted dual
FitResult and returns the scalar mu_g (family-specific ``1 - E[exp(-Z)]``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from idd_tc_mortality.distributions.base import FitResult

_G_EPS = 1e-9  # keep g strictly inside (0,1) so the dual z = -ln(1-g) is finite


def shadow_fit(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    excess_cap: np.ndarray,
    *,
    base_fit: Callable[..., FitResult],
    mu_g_from_fit: Callable[[FitResult], float],
    family: str,
) -> FitResult:
    """Fit the pooled dual distribution and store the shadow mean mu_g.

    Parameters
    ----------
    X:
        Design matrix. Covariate-free by construction — only its length matters (the
        pooled fit is intercept-only). Its columns become the returned param_names.
    y:
        Excess rates ``w_i = death_rate - threshold_rate`` for the tail subset (> 0).
    weights:
        Per-observation weights.
    excess_cap:
        Per-storm excess cap ``H_i = c_i - threshold_rate`` (each storm's own cap).
    base_fit:
        The distribution fit to the dual z (e.g. ``gpd.fit`` / ``log_logistic.fit``).
    mu_g_from_fit:
        Callback: fitted-dual FitResult -> scalar mu_g = 1 - E[exp(-Z)].
    family:
        Registry family name to stamp on the FitResult.
    """
    y = np.asarray(y, dtype=float)
    H = np.asarray(excess_cap, dtype=float)
    if H.shape != y.shape:
        raise ValueError(f"excess_cap length {H.shape} != y length {y.shape}.")
    if np.any(H <= 0):
        raise ValueError(
            f"{family}.fit requires strictly positive excess caps (H_i = c_i - u > 0) for "
            "every fitted tail storm; a non-positive cap means the tail event is impossible "
            "for that storm and it should not be in the tail fit subset."
        )

    g = np.clip(y / H, _G_EPS, 1.0 - _G_EPS)  # in-leg: fraction of each storm's OWN cap
    z = -np.log1p(-g)                          # dual: -ln(1 - g), (0,1) -> [0, inf)

    X_int = pd.DataFrame({"const": np.ones(len(z))})  # covariate-free pooled fit
    dual = base_fit(X_int, z, np.asarray(weights, dtype=float))
    mu_g = float(mu_g_from_fit(dual))

    return FitResult(
        params=dual.params,
        param_names=dual.param_names,
        fitted_values=mu_g * H,
        family=family,
        converged=dual.converged,
        meta={
            "shape_param": dual.meta.get("shape_param"),
            "dual_scale": float(np.exp(dual.params[0])),
            "mu_g": mu_g,
            "n_obs": int(len(y)),
        },
    )


def shadow_predict(result: FitResult, X: pd.DataFrame, excess_cap: np.ndarray) -> np.ndarray:
    """Predict shadow-mean excess rates ``mu_g * H_i`` (covariate-free).

    The out-leg multiplies the pooled shadow mean by each storm's OWN cap H_i — the same
    per-storm quantity the in-leg used to form g_i.
    """
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names "
            f"{result.param_names}. Call features.align_X before predicting."
        )
    mu_g = float(result.meta["mu_g"])
    H = np.maximum(np.asarray(excess_cap, dtype=float), 0.0)
    if H.shape[0] != len(X):
        raise ValueError(f"excess_cap length {H.shape[0]} does not match X rows {len(X)}.")
    return mu_g * H
