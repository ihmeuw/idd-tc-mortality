"""Tests for the GPD SHADOW-MEAN tail variant (variant A).

Correctness (not self-consistency): the shadow mean mu_g = 1 - E[exp(-Z)] is computed in
the module by quadrature of the GPD density; every check here targets it with an
INDEPENDENT Monte-Carlo estimate mean(1 - exp(-Z)) over genpareto samples (a different
code path), so passing proves the integral is right — not merely that the code agrees with
itself. Includes xi >= 1 (fat) where the ordinary GPD mean is infinite but mu_g must stay
in (0,1).

The one subtle bug the construction can hide is treating each storm's cap as a shared
constant. Every recovery/prediction test uses HETEROGENEOUS per-storm caps (~2 orders of
magnitude) and asserts the correct prediction is provably distinguishable from a shared-cap
prediction, so a shared-H bug would fail the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

from idd_tc_mortality.distributions import get_family, gpd_shadow
from idd_tc_mortality.distributions.gpd_shadow import shadow_mean_of_g


def _intercept(n):
    return pd.DataFrame({"const": np.ones(n)})


def _heterogeneous_caps(rng, n):
    """Per-storm excess caps spanning ~2 orders of magnitude (1e-3 .. 1e-1)."""
    return 10.0 ** rng.uniform(-3.0, -1.0, size=n)


def _mc_mu_g_gpd(rng, sigma, xi, n=4_000_000):
    """INDEPENDENT target: mu_g = E[1 - exp(-Z)] by sampling Z ~ GPD(sigma, xi)."""
    Z = genpareto.rvs(xi, scale=sigma, size=n, random_state=rng)
    return float(np.mean(1.0 - np.exp(-Z)))


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.5, 1.0, 1.5, 3.0])
@pytest.mark.parametrize("sigma", [0.3, 1.0, 4.0])
def test_shadow_mean_matches_independent_montecarlo(xi, sigma):
    """Quadrature mu_g equals a sample-MC of 1 - exp(-Z) — correctness, not self-consistency."""
    rng = np.random.default_rng(abs(hash(("mu_g", xi, sigma))) % (2**32))
    quad_mu = shadow_mean_of_g(sigma, xi)
    mc_mu = _mc_mu_g_gpd(rng, sigma, xi)
    assert 0.0 < quad_mu < 1.0                       # bounded even for xi >= 1
    assert np.isclose(quad_mu, mc_mu, rtol=0.01, atol=2e-3), (
        f"xi={xi} sigma={sigma}: quad={quad_mu:.6g} mc={mc_mu:.6g}"
    )


def test_exponential_limit_closed_form():
    """xi -> 0: Z ~ Exp(mean=sigma), mu_g = sigma/(1+sigma) (independent closed form)."""
    for sigma in (0.1, 1.0, 10.0):
        assert np.isclose(shadow_mean_of_g(sigma, 0.0), sigma / (1.0 + sigma), rtol=1e-9)
    assert np.isclose(shadow_mean_of_g(1.0, 1e-6), 0.5, rtol=1e-3)  # continuity at xi->0


@pytest.mark.parametrize("xi0", [-0.2, 0.0, 0.3, 0.8])
def test_both_legs_use_per_storm_cap(xi0):
    rng = np.random.default_rng(abs(hash(("shadow", xi0))) % (2**32))
    n = 8000
    sigma0 = 0.9
    # INDEPENDENT target for mu_g (sample-MC), NOT shadow_mean_of_g (the code under test).
    true_mu_g = _mc_mu_g_gpd(rng, sigma0, xi0)

    H = _heterogeneous_caps(rng, n)
    assert H.max() / H.min() > 50.0, "caps must be heterogeneous for this test to bite"

    Z = genpareto.rvs(xi0, scale=sigma0, size=n, random_state=rng)
    g = 1.0 - np.exp(-Z)     # true fraction-of-cap in (0,1); dual z=-ln(1-g)=Z recovers the fit
    w = g * H                # in-leg: excess = g_i * own H_i

    result = gpd_shadow.fit(_intercept(n), w, np.ones(n), excess_cap=H)
    mu_hat = result.meta["mu_g"]

    # In-leg: recovered mu_g (fit + quad) matches the INDEPENDENT MC target.
    assert np.isclose(mu_hat, true_mu_g, rtol=0.06), f"xi0={xi0}: {mu_hat} vs {true_mu_g}"

    # Out-leg: predicted excess / own cap is CONSTANT across heterogeneous caps.
    pred = gpd_shadow.predict(result, _intercept(n), excess_cap=H)
    assert np.allclose(pred / H, mu_hat, rtol=1e-9)
    assert np.allclose(pred, true_mu_g * H, rtol=0.06)

    # Adversarial: the correct per-storm prediction is NOT the shared-(mean)-cap prediction,
    # so a shared-H implementation would fail this test.
    assert not np.allclose(pred, mu_hat * H.mean(), rtol=0.10)


def test_fit_rejects_nonpositive_cap():
    rng = np.random.default_rng(3)
    n = 100
    w = rng.uniform(1e-4, 1e-2, size=n)
    H = np.full(n, 0.05)
    H[0] = 0.0
    with pytest.raises(ValueError):
        gpd_shadow.fit(_intercept(n), w, np.ones(n), excess_cap=H)


def test_registered_needs_cap():
    fam = get_family("gpd_shadow")
    assert fam["tail_outcome"] == "excess"
    assert fam["needs_cap"] is True
