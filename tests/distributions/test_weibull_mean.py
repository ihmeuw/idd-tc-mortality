"""Tests for the Weibull MEAN tail variant (variant D).

Two things to pin:
  1. Recovery — the re-exported fit is the standard Weibull MLE, so it recovers
     the DGP scale coefficients and shape.
  2. Mean formula — predict() returns lambda * Gamma(1 + 1/k), verified both
     analytically and against a Monte-Carlo Weibull mean, and distinct from the
     median the base weibull.predict returns.
"""

from __future__ import annotations

from math import gamma

import numpy as np
import pandas as pd

from idd_tc_mortality.distributions import get_family, weibull, weibull_mean


def _weibull_excess_dgp(rng, n, b0, b1, k):
    """Sample Weibull excess data with log(lambda_i) = b0 + b1 x_i, shape k."""
    x = rng.normal(0.0, 1.0, size=n)
    X = pd.DataFrame({"const": np.ones(n), "x": x})
    lam = np.exp(b0 + b1 * x)
    # numpy weibull(k) has scale 1; scale by lambda_i.
    w = lam * rng.weibull(k, size=n)
    weights = np.ones(n)
    return X, w, weights, lam


def test_weibull_mean_recovers_dgp():
    rng = np.random.default_rng(0)
    b0, b1, k = -9.0, 0.7, 1.6
    X, w, weights, _ = _weibull_excess_dgp(rng, 6000, b0, b1, k)

    result = weibull_mean.fit(X, w, weights)

    assert result.converged
    beta = dict(zip(result.param_names, result.params))
    assert abs(beta["const"] - b0) < 0.15
    assert abs(beta["x"] - b1) < 0.10
    assert abs(result.meta["shape_param"] - k) < 0.15


def test_weibull_mean_predict_is_analytic_mean():
    rng = np.random.default_rng(1)
    b0, b1, k = -8.0, 0.5, 1.3
    X, w, weights, _ = _weibull_excess_dgp(rng, 4000, b0, b1, k)
    result = weibull_mean.fit(X, w, weights)

    preds = weibull_mean.predict(result, X)
    k_hat = result.meta["shape_param"]
    lam_hat = np.exp(np.asarray(X, dtype=float) @ result.params)
    expected = lam_hat * gamma(1.0 + 1.0 / k_hat)

    assert np.allclose(preds, expected, rtol=1e-12)
    # The mean must differ from the median the base family reports.
    median = weibull.predict(result, X)
    assert np.all(preds > median)  # mean > median for right-skewed Weibull (k here < ~3.6)


def test_weibull_mean_matches_montecarlo_mean():
    """The predicted mean equals the sample mean of Weibull draws at fixed params."""
    rng = np.random.default_rng(2)
    for lam, k in [(3.0, 0.8), (1.0, 1.5), (10.0, 2.5)]:
        samples = lam * rng.weibull(k, size=2_000_000)
        mc_mean = samples.mean()
        analytic = lam * gamma(1.0 + 1.0 / k)
        assert np.isclose(analytic, mc_mean, rtol=0.01)


def test_weibull_mean_registered_with_excess_flag():
    fam = get_family("weibull_mean")
    assert fam["tail_outcome"] == "excess"     # threshold added back downstream
    assert not fam.get("needs_cap")            # finite mean; no physical cap needed
    assert fam["fit"] is weibull.fit           # re-exported identical fit
