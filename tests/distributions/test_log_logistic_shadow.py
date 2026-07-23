"""Tests for the log-logistic SHADOW-MEAN tail variant (variant A).

Same discipline as test_gpd_shadow: the module computes mu_g = 1 - E[exp(-Z)] by
quadrature of the log-logistic density; the target here is an INDEPENDENT sample-MC of
1 - exp(-Z) over log-logistic draws (different code path). Includes k <= 1 (fat) where the
ordinary log-logistic mean is infinite but mu_g must stay in (0,1). Both-legs test uses
HETEROGENEOUS per-storm caps with an adversarial shared-cap assertion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.distributions import get_family, log_logistic_shadow
from idd_tc_mortality.distributions.log_logistic_shadow import shadow_mean_of_g


def _intercept(n):
    return pd.DataFrame({"const": np.ones(n)})


def _ll_samples(rng, alpha, k, n):
    """Inverse-CDF log-logistic draws: Z = alpha * (u/(1-u))^(1/k)."""
    u = np.clip(rng.uniform(0.0, 1.0, size=n), 1e-12, 1.0 - 1e-12)
    return alpha * (u / (1.0 - u)) ** (1.0 / k)


def _mc_mu_g_ll(rng, alpha, k, n=4_000_000):
    """INDEPENDENT target: mu_g = E[1 - exp(-Z)], Z ~ LogLogistic(alpha, k)."""
    Z = _ll_samples(rng, alpha, k, n)
    return float(np.mean(1.0 - np.exp(-Z)))


@pytest.mark.parametrize("k", [0.7, 1.0, 1.8, 3.0])
@pytest.mark.parametrize("alpha", [0.3, 1.0, 3.0])
def test_shadow_mean_matches_independent_montecarlo(k, alpha):
    rng = np.random.default_rng(abs(hash(("ll_mu_g", k, alpha))) % (2**32))
    quad_mu = shadow_mean_of_g(alpha, k)
    mc_mu = _mc_mu_g_ll(rng, alpha, k)
    assert 0.0 < quad_mu < 1.0                       # bounded even for k <= 1
    assert np.isclose(quad_mu, mc_mu, rtol=0.015, atol=2e-3), (
        f"k={k} alpha={alpha}: quad={quad_mu:.6g} mc={mc_mu:.6g}"
    )


@pytest.mark.parametrize("k0", [0.8, 1.5, 2.5])
def test_both_legs_use_per_storm_cap(k0):
    rng = np.random.default_rng(abs(hash(("ll_shadow", k0))) % (2**32))
    n = 8000
    alpha0 = 0.8
    true_mu_g = _mc_mu_g_ll(rng, alpha0, k0)

    H = 10.0 ** rng.uniform(-3.0, -1.0, size=n)
    assert H.max() / H.min() > 50.0

    Z = _ll_samples(rng, alpha0, k0, n)
    g = 1.0 - np.exp(-Z)
    w = g * H

    result = log_logistic_shadow.fit(_intercept(n), w, np.ones(n), excess_cap=H)
    mu_hat = result.meta["mu_g"]
    assert np.isclose(mu_hat, true_mu_g, rtol=0.07), f"k0={k0}: {mu_hat} vs {true_mu_g}"

    pred = log_logistic_shadow.predict(result, _intercept(n), excess_cap=H)
    assert np.allclose(pred / H, mu_hat, rtol=1e-9)
    assert np.allclose(pred, true_mu_g * H, rtol=0.07)
    assert not np.allclose(pred, mu_hat * H.mean(), rtol=0.10)


def test_registered_needs_cap():
    fam = get_family("log_logistic_shadow")
    assert fam["tail_outcome"] == "excess"
    assert fam["needs_cap"] is True
