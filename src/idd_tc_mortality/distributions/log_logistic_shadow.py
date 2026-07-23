"""Log-logistic SHADOW-MEAN tail variant (bounded/finite-mean selection, variant A).

Identical Cirillo-Taleb dual construction to :mod:`gpd_shadow` (shared core in
:mod:`_shadow`), but the pooled dual is fit with a LOG-LOGISTIC instead of a GPD. This
module supplies only the log-logistic shadow mean of the fraction:

    mu_g = 1 - E[exp(-Z)],   Z ~ LogLogistic(alpha, k),

computed by quadrature of ``exp(-z) * f_LL(z)`` (split at z=1 so the k<1 density spike at
0 is integrated cleanly). Bounded in (0,1) for any k, so finite even when the fitted dual
tail is fat (k <= 1) — the defining requirement (see :mod:`gpd_shadow`).
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from idd_tc_mortality.distributions import log_logistic as _log_logistic
from idd_tc_mortality.distributions._shadow import shadow_fit, shadow_predict
from idd_tc_mortality.distributions.base import FitResult


def _ll_pdf(z: np.ndarray, alpha: float, k: float) -> np.ndarray:
    """Log-logistic density f(z) = (k/alpha)(z/alpha)^(k-1) / (1 + (z/alpha)^k)^2."""
    za = z / alpha
    return (k / alpha) * za ** (k - 1.0) / (1.0 + za ** k) ** 2


def _mu_g_from_log_logistic(dual: FitResult) -> float:
    """mu_g = 1 - E[exp(-Z)] for Z ~ LogLogistic(alpha, k), from a fitted dual log-logistic."""
    k = float(dual.meta["shape_param"])
    alpha = float(np.exp(dual.params[0]))
    integrand = lambda z: np.exp(-z) * _ll_pdf(z, alpha, k)  # noqa: E731
    # Split at 1: for k < 1 the density diverges (integrably) at z -> 0; isolating [0,1]
    # lets the adaptive quadrature resolve the endpoint singularity.
    e_lo, _ = integrate.quad(integrand, 0.0, 1.0)
    e_hi, _ = integrate.quad(integrand, 1.0, np.inf)
    return 1.0 - (e_lo + e_hi)


def fit(X, y, weights, excess_cap):
    """Fit the pooled dual log-logistic and store the shadow mean (see :mod:`_shadow`)."""
    return shadow_fit(
        X, y, weights, excess_cap,
        base_fit=_log_logistic.fit,
        mu_g_from_fit=_mu_g_from_log_logistic,
        family="log_logistic_shadow",
    )


def predict(result, X, excess_cap):
    """Predict shadow-mean excess rates ``mu_g * H_i`` (see :mod:`_shadow`)."""
    return shadow_predict(result, X, excess_cap)


def shadow_mean_of_g(alpha: float, k: float) -> float:
    """mu_g for a LogLogistic(alpha, k) dual — for independent-target testing."""
    dummy = FitResult(
        params=np.array([np.log(alpha)]),
        param_names=["const"],
        fitted_values=np.array([np.nan]),
        family="log_logistic_shadow",
        meta={"shape_param": k},
    )
    return _mu_g_from_log_logistic(dummy)
