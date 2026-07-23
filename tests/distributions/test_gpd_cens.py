"""Tests for the GPD CENSORED-mean tail variant (variant C).

The fit is the standard untruncated GPD MLE (re-exported), so recovery is
covered by test_gpd. The novel, load-bearing piece is the censored-mean formula
E[min(W, H)] for W ~ GPD(sigma, xi): it must

  * match a Monte-Carlo censored mean across xi < 0, xi = 0, 0 < xi < 1, xi = 1,
    and xi > 1 (where the ordinary GPD mean is INFINITE but the censored mean is
    finite — the entire point of variant C),
  * be handled through predict() with a per-row excess cap,
  * and, at a very large cap, converge to the ordinary GPD mean sigma/(1-xi) for
    xi < 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.distributions import get_family, gpd, gpd_cens
from idd_tc_mortality.distributions.gpd_cens import gpd_censored_mean


def _gpd_samples(rng, sigma, xi, n):
    """Inverse-CDF GPD draws: excess W with scale sigma, shape xi."""
    u = rng.uniform(0.0, 1.0, size=n)
    if abs(xi) < 1e-12:
        return -sigma * np.log(1.0 - u)
    return (sigma / xi) * ((1.0 - u) ** (-xi) - 1.0)


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.5, 1.0, 1.5])
def test_censored_mean_matches_montecarlo(xi):
    rng = np.random.default_rng(abs(hash(("cens", xi))) % (2**32))
    sigma = 2.0
    # A spread of caps, including one below and one above the typical draw scale.
    for H in [0.5, 2.0, 8.0, 50.0]:
        samples = _gpd_samples(rng, sigma, xi, 2_000_000)
        mc = np.minimum(samples, H).mean()
        closed = float(gpd_censored_mean(np.array([sigma]), xi, np.array([H]))[0])
        assert np.isclose(closed, mc, rtol=0.02, atol=1e-6), (
            f"xi={xi} H={H}: closed={closed:.6g} mc={mc:.6g}"
        )


def test_censored_mean_finite_for_infinite_mean_shapes():
    """xi >= 1 => ordinary GPD mean is infinite; censored mean must stay finite."""
    sigma = np.array([1.0, 5.0, 0.1])
    H = np.array([3.0, 3.0, 3.0])
    for xi in (1.0, 1.5, 3.0):
        out = gpd_censored_mean(sigma, xi, H)
        assert np.all(np.isfinite(out))
        assert np.all(out > 0.0)
        assert np.all(out <= H + 1e-9)  # censored at H


def test_censored_mean_converges_to_gpd_mean_for_large_cap():
    """For xi < 1 and H -> inf, E[min(W,H)] -> E[W] = sigma/(1-xi)."""
    sigma = np.array([2.0])
    for xi in (-0.3, 0.0, 0.4):
        full_mean = sigma / (1.0 - xi)
        big = gpd_censored_mean(sigma, xi, np.array([1e7]))
        assert np.isclose(big[0], full_mean[0], rtol=1e-4)


def test_censored_mean_zero_when_cap_nonpositive():
    sigma = np.array([1.0, 2.0])
    H = np.array([0.0, -0.5])
    out = gpd_censored_mean(sigma, 0.5, H)
    assert np.allclose(out, 0.0)


def test_predict_end_to_end_with_per_row_cap():
    rng = np.random.default_rng(7)
    # Fit an untruncated GPD on synthetic excess data.
    n = 3000
    x = rng.normal(0.0, 1.0, size=n)
    X = pd.DataFrame({"const": np.ones(n), "x": x})
    sigma_i = np.exp(-9.0 + 0.3 * x)
    xi_true = 0.4
    u = rng.uniform(size=n)
    y = (sigma_i / xi_true) * ((1.0 - u) ** (-xi_true) - 1.0)
    result = gpd_cens.fit(X, y, np.ones(n))

    cap = np.full(n, 1e-3)  # excess cap
    preds = gpd_cens.predict(result, X, cap)
    assert preds.shape == (n,)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)
    assert np.all(preds <= cap + 1e-12)  # never exceeds the censoring cap


def test_predict_rejects_mismatched_cap_length():
    rng = np.random.default_rng(8)
    n = 500
    X = pd.DataFrame({"const": np.ones(n), "x": rng.normal(size=n)})
    y = rng.gamma(2.0, 1e-4, size=n)
    result = gpd_cens.fit(X, y, np.ones(n))
    with pytest.raises(ValueError):
        gpd_cens.predict(result, X, np.ones(n - 1))


def test_gpd_cens_registered_needs_cap():
    fam = get_family("gpd_cens")
    assert fam["tail_outcome"] == "excess"
    assert fam["needs_cap"] is True
    assert fam["fit"] is gpd.fit  # re-exported identical untruncated fit
