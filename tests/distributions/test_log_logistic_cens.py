"""Tests for the log-logistic CENSORED-mean tail variant (variant C).

Fit is the re-exported standard log-logistic MLE (recovery covered by test_log_logistic);
the novel piece is the censored-mean formula. Verified against:
  * a Monte-Carlo censored mean across k < 1, k = 1, k > 1 (k <= 1 is where the ordinary
    log-logistic mean is INFINITE but the censored mean must stay finite),
  * the closed-form reductions at k=1 (alpha*ln(1+H/alpha)) and k=2 (alpha*arctan(H/alpha)),
  * convergence to the full mean alpha*(pi/k)/sin(pi/k) as H -> inf for k > 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.distributions import get_family, log_logistic
from idd_tc_mortality.distributions.log_logistic_cens import loglogistic_censored_mean


def _loglogistic_samples(rng, alpha, k, n):
    """Inverse-CDF log-logistic draws: W = alpha * (u/(1-u))^(1/k)."""
    u = rng.uniform(0.0, 1.0, size=n)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return alpha * (u / (1.0 - u)) ** (1.0 / k)


@pytest.mark.parametrize("k", [0.7, 1.0, 1.5, 2.5])
def test_censored_mean_matches_montecarlo(k):
    rng = np.random.default_rng(abs(hash(("ll_cens", k))) % (2**32))
    alpha = 1.5
    for H in [0.3, 1.5, 6.0, 40.0]:
        samples = _loglogistic_samples(rng, alpha, k, 2_000_000)
        mc = np.minimum(samples, H).mean()
        closed = float(loglogistic_censored_mean(np.array([alpha]), k, np.array([H]))[0])
        assert np.isclose(closed, mc, rtol=0.02, atol=1e-6), (
            f"k={k} H={H}: closed={closed:.6g} mc={mc:.6g}"
        )


def test_closed_form_reductions():
    alpha = np.array([2.0, 0.5])
    H = np.array([3.0, 1.0])
    # k = 1 -> alpha * ln(1 + H/alpha)
    got1 = loglogistic_censored_mean(alpha, 1.0, H)
    assert np.allclose(got1, alpha * np.log1p(H / alpha), rtol=1e-6)
    # k = 2 -> alpha * arctan(H/alpha)
    got2 = loglogistic_censored_mean(alpha, 2.0, H)
    assert np.allclose(got2, alpha * np.arctan(H / alpha), rtol=1e-6)


def test_censored_mean_finite_for_infinite_mean_shapes():
    """k <= 1 => ordinary log-logistic mean is infinite; censored mean stays finite."""
    alpha = np.array([1.0, 4.0])
    H = np.array([5.0, 5.0])
    for k in (0.5, 0.8, 1.0):
        out = loglogistic_censored_mean(alpha, k, H)
        assert np.all(np.isfinite(out))
        assert np.all(out > 0.0)
        assert np.all(out <= H + 1e-9)


def test_converges_to_full_mean_for_large_cap():
    """k > 1 and H -> inf: E[min(W,H)] -> alpha * (pi/k) / sin(pi/k)."""
    alpha = np.array([2.0])
    for k in (1.5, 2.5, 4.0):
        full = alpha * (np.pi / k) / np.sin(np.pi / k)
        big = loglogistic_censored_mean(alpha, k, np.array([1e8]))
        assert np.isclose(big[0], full[0], rtol=1e-3)


def test_zero_when_cap_nonpositive():
    out = loglogistic_censored_mean(np.array([1.0, 2.0]), 1.5, np.array([0.0, -0.5]))
    assert np.allclose(out, 0.0)


def test_log_logistic_cens_registered_needs_cap():
    fam = get_family("log_logistic_cens")
    assert fam["tail_outcome"] == "excess"
    assert fam["needs_cap"] is True
    assert fam["fit"] is log_logistic.fit
