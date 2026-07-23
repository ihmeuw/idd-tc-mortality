"""GPD SHADOW-MEAN tail variant (bounded/finite-mean selection, variant A).

Cirillo-Taleb dual-distribution shadow mean with a GPD fit to the dual. A MARGINAL
sensitivity reference with NO covariates. The shared construction lives in
:mod:`_shadow` (in-leg g_i = w_i/H_i with each storm's own cap, dual z = -ln(1-g),
pooled intercept-only fit, back-transform mu_g * H_i). This module supplies only the
GPD-specific shadow mean of the fraction:

    mu_g = 1 - E[exp(-Z)],   Z ~ GPD(sigma, xi).

Bounded in (0,1) for ANY xi (because exp(-Z) in (0,1] for Z >= 0), so the shadow mean is
finite even when the fitted dual tail is fat (xi >= 1) — the defining requirement, since
the physical cap makes the true expected tail rate finite and an infinite estimate would
poison assembled E[rate] and every metric built on it.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.stats import genpareto

from idd_tc_mortality.distributions import gpd as _gpd
from idd_tc_mortality.distributions._shadow import shadow_fit, shadow_predict
from idd_tc_mortality.distributions.base import FitResult

_XI_EPS = 1e-8


def _mu_g_from_gpd(dual: FitResult) -> float:
    """mu_g = 1 - E[exp(-Z)] for Z ~ GPD(sigma, xi), from a fitted dual GPD."""
    xi = float(dual.meta["shape_param"])
    sigma = float(np.exp(dual.params[0]))
    if abs(xi) < _XI_EPS:
        # Z ~ Exponential(mean=sigma): E[exp(-Z)] = 1/(1+sigma).
        return sigma / (1.0 + sigma)
    upper = np.inf if xi >= 0 else -sigma / xi  # finite support sigma/|xi| when xi < 0
    e_exp, _ = integrate.quad(
        lambda z: np.exp(-z) * genpareto.pdf(z, xi, scale=sigma), 0.0, upper
    )
    return 1.0 - e_exp


def fit(X, y, weights, excess_cap):
    """Fit the pooled dual GPD and store the shadow mean (see :mod:`_shadow`)."""
    return shadow_fit(
        X, y, weights, excess_cap,
        base_fit=_gpd.fit,
        mu_g_from_fit=_mu_g_from_gpd,
        family="gpd_shadow",
    )


def predict(result, X, excess_cap):
    """Predict shadow-mean excess rates ``mu_g * H_i`` (see :mod:`_shadow`)."""
    return shadow_predict(result, X, excess_cap)


def shadow_mean_of_g(sigma: float, xi: float) -> float:
    """mu_g for a GPD(sigma, xi) dual — thin wrapper over :func:`_mu_g_from_gpd` inputs."""
    dummy = FitResult(
        params=np.array([np.log(sigma)]),
        param_names=["const"],
        fitted_values=np.array([np.nan]),
        family="gpd_shadow",
        meta={"shape_param": xi},
    )
    return _mu_g_from_gpd(dummy)
