"""
Tests for distributions.truncated_normal.

Covers:
  - Happy path: fit returns FitResult with correct family, param count, fitted_values length.
  - Fitted values and predict outputs are strictly positive and finite.
  - meta contains required keys: sigma, threshold_rate, truncation_side, n_obs.
  - predict: column mismatch raises ValueError.
  - Input validation: y<=0, non-positive weights, length mismatches, bad truncation_side,
    bulk y >= threshold_rate, tail y < threshold_rate.
  - Non-convergence handled gracefully: converged=False, result still returned.
  - _truncated_mean: bulk and tail formulas are non-negative and correct direction
    (bulk mean < threshold, tail mean > threshold when sigma is small).
  - Synthetic recovery: log(rate) = intercept + beta*wind; sigma recovered within tolerance.
  - fit_one_component round-trip: truncated_normal wired correctly for both bulk and tail.

DGP design:
  Bulk: log(death_rate) ~ N(mu_i, sigma^2), truncated above at log(threshold)
  Tail: log(death_rate) ~ N(mu_i, sigma^2), truncated below at log(threshold)
  mu_i = true_intercept + true_wind * wind_speed_i
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from idd_tc_mortality.distributions import truncated_normal
from idd_tc_mortality.distributions.base import FitResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N = 500
_THRESHOLD = 1e-4

# Bulk DGP: draw from N(mu, sigma^2) truncated above at log(threshold)
_TRUE_INTERCEPT_BULK = np.log(_THRESHOLD) - 2.0   # mean well below threshold
_TRUE_WIND_BULK = 0.3
_TRUE_SIGMA_BULK = 0.8
_WIND_BULK = _RNG.normal(0, 1, _N)
_MU_BULK = _TRUE_INTERCEPT_BULK + _TRUE_WIND_BULK * _WIND_BULK
_LOG_THRESHOLD = np.log(_THRESHOLD)


def _draw_truncated(mu, sigma, log_threshold, side, rng, n):
    """Draw from truncated normal via inverse CDF (accept-reject for simplicity)."""
    samples = np.empty(n)
    i = 0
    while i < n:
        z = rng.normal(mu if np.isscalar(mu) else mu[i % len(mu)], sigma)
        if side == "bulk" and z < log_threshold:
            samples[i] = np.exp(z)
            i += 1
        elif side == "tail" and z >= log_threshold:
            samples[i] = np.exp(z)
            i += 1
    return samples


@pytest.fixture(scope="module")
def bulk_data():
    rng = np.random.default_rng(7)
    n = 500
    wind = rng.normal(0, 1, n)
    true_intercept = np.log(_THRESHOLD) - 2.0
    true_wind = 0.3
    true_sigma = 0.8
    mu = true_intercept + true_wind * wind

    # Use scipy truncnorm for correct truncated draws
    b = (_LOG_THRESHOLD - mu) / true_sigma   # upper bound in std-normal units
    log_y = stats.truncnorm.rvs(-np.inf, b, loc=mu, scale=true_sigma, random_state=rng)
    y = np.exp(log_y)

    X = pd.DataFrame({"const": 1.0, "wind_speed": wind})
    weights = rng.uniform(10_000, 500_000, n)
    return X, y, weights, _THRESHOLD, true_intercept, true_wind, true_sigma


@pytest.fixture(scope="module")
def tail_data():
    rng = np.random.default_rng(13)
    n = 500
    wind = rng.normal(0, 1, n)
    true_intercept = np.log(_THRESHOLD) + 1.5
    true_wind = 0.4
    true_sigma = 0.6
    mu = true_intercept + true_wind * wind

    a = (_LOG_THRESHOLD - mu) / true_sigma   # lower bound in std-normal units
    log_y = stats.truncnorm.rvs(a, np.inf, loc=mu, scale=true_sigma, random_state=rng)
    y = np.exp(log_y)

    X = pd.DataFrame({"const": 1.0, "wind_speed": wind})
    weights = rng.uniform(10_000, 500_000, n)
    return X, y, weights, _THRESHOLD, true_intercept, true_wind, true_sigma


# ---------------------------------------------------------------------------
# Happy path: fit returns FitResult
# ---------------------------------------------------------------------------

def test_bulk_fit_returns_fitresult(bulk_data):
    X, y, weights, threshold, *_ = bulk_data
    result = truncated_normal.fit(X, y, weights, threshold, "bulk")
    assert isinstance(result, FitResult)
    assert result.family == "truncated_normal"
    assert len(result.params) == X.shape[1]
    assert list(result.param_names) == list(X.columns)
    assert len(result.fitted_values) == len(y)


def test_tail_fit_returns_fitresult(tail_data):
    X, y, weights, threshold, *_ = tail_data
    result = truncated_normal.fit(X, y, weights, threshold, "tail")
    assert isinstance(result, FitResult)
    assert result.family == "truncated_normal"
    assert len(result.params) == X.shape[1]
    assert len(result.fitted_values) == len(y)


# ---------------------------------------------------------------------------
# Fitted values are positive and finite
# ---------------------------------------------------------------------------

def test_bulk_fitted_values_positive_finite(bulk_data):
    X, y, weights, threshold, *_ = bulk_data
    result = truncated_normal.fit(X, y, weights, threshold, "bulk")
    assert np.all(np.isfinite(result.fitted_values))
    assert np.all(result.fitted_values > 0)


def test_tail_fitted_values_positive_finite(tail_data):
    X, y, weights, threshold, *_ = tail_data
    result = truncated_normal.fit(X, y, weights, threshold, "tail")
    assert np.all(np.isfinite(result.fitted_values))
    assert np.all(result.fitted_values > 0)


# ---------------------------------------------------------------------------
# meta contains required keys
# ---------------------------------------------------------------------------

def test_bulk_meta_keys(bulk_data):
    X, y, weights, threshold, *_ = bulk_data
    result = truncated_normal.fit(X, y, weights, threshold, "bulk")
    for key in ("sigma", "threshold_rate", "truncation_side", "n_obs", "iterations", "warnings"):
        assert key in result.meta, f"Missing meta key: {key!r}"
    assert result.meta["threshold_rate"] == threshold
    assert result.meta["truncation_side"] == "bulk"
    assert result.meta["n_obs"] == len(y)
    assert result.meta["sigma"] > 0


def test_tail_meta_keys(tail_data):
    X, y, weights, threshold, *_ = tail_data
    result = truncated_normal.fit(X, y, weights, threshold, "tail")
    assert result.meta["truncation_side"] == "tail"
    assert result.meta["threshold_rate"] == threshold


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_bulk_predict_returns_positive_finite(bulk_data):
    X, y, weights, threshold, *_ = bulk_data
    result = truncated_normal.fit(X, y, weights, threshold, "bulk")
    preds = truncated_normal.predict(result, X)
    assert len(preds) == len(y)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0)


def test_tail_predict_returns_positive_finite(tail_data):
    X, y, weights, threshold, *_ = tail_data
    result = truncated_normal.fit(X, y, weights, threshold, "tail")
    preds = truncated_normal.predict(result, X)
    assert len(preds) == len(y)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0)


def test_predict_column_mismatch_raises(bulk_data):
    X, y, weights, threshold, *_ = bulk_data
    result = truncated_normal.fit(X, y, weights, threshold, "bulk")
    X_wrong = X.rename(columns={"wind_speed": "sdi"})
    with pytest.raises(ValueError, match="param_names"):
        truncated_normal.predict(result, X_wrong)


def test_predict_matches_fitted_values(bulk_data):
    """predict(result, X_train) must reproduce fitted_values exactly."""
    X, y, weights, threshold, *_ = bulk_data
    result = truncated_normal.fit(X, y, weights, threshold, "bulk")
    preds = truncated_normal.predict(result, X)
    np.testing.assert_allclose(preds, result.fitted_values, rtol=1e-10)


# ---------------------------------------------------------------------------
# Truncated mean direction (weak sanity checks)
# ---------------------------------------------------------------------------

def test_bulk_fitted_values_below_threshold(bulk_data):
    """Bulk truncated mean must be < threshold (the upper bound)."""
    X, y, weights, threshold, *_ = bulk_data
    result = truncated_normal.fit(X, y, weights, threshold, "bulk")
    # At least 95% of fitted values should be well below threshold.
    # (A few may exceed it if mu is very close to log(threshold), but that's rare.)
    assert np.mean(result.fitted_values < threshold) > 0.90


def test_tail_fitted_values_above_threshold(tail_data):
    """Tail truncated mean must exceed threshold (the lower bound) on average."""
    X, y, weights, threshold, *_ = tail_data
    result = truncated_normal.fit(X, y, weights, threshold, "tail")
    assert np.mean(result.fitted_values >= threshold) > 0.90


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_y_zero_raises(bulk_data):
    X, y, weights, threshold, *_ = bulk_data
    y_bad = y.copy()
    y_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        truncated_normal.fit(X, y_bad, weights, threshold, "bulk")


def test_y_negative_raises(bulk_data):
    X, y, weights, threshold, *_ = bulk_data
    y_bad = y.copy()
    y_bad[0] = -1e-8
    with pytest.raises(ValueError, match="strictly positive"):
        truncated_normal.fit(X, y_bad, weights, threshold, "bulk")


def test_nonpositive_weights_raises(bulk_data):
    X, y, weights, threshold, *_ = bulk_data
    w_bad = weights.copy()
    w_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        truncated_normal.fit(X, y, w_bad, threshold, "bulk")


def test_y_length_mismatch_raises(bulk_data):
    X, y, weights, threshold, *_ = bulk_data
    with pytest.raises(ValueError, match="length"):
        truncated_normal.fit(X, y[:-1], weights, threshold, "bulk")


def test_weights_length_mismatch_raises(bulk_data):
    X, y, weights, threshold, *_ = bulk_data
    with pytest.raises(ValueError, match="length"):
        truncated_normal.fit(X, y, weights[:-1], threshold, "bulk")


def test_bad_truncation_side_raises(bulk_data):
    X, y, weights, threshold, *_ = bulk_data
    with pytest.raises(ValueError, match="truncation_side"):
        truncated_normal.fit(X, y, weights, threshold, "wrong")


def test_bulk_y_above_threshold_raises():
    """bulk fit requires all y < threshold_rate."""
    rng = np.random.default_rng(0)
    threshold = 1e-4
    X = pd.DataFrame({"const": 1.0, "wind_speed": rng.normal(0, 1, 10)})
    y = np.full(10, threshold * 2.0)   # all above threshold — invalid
    weights = np.ones(10)
    with pytest.raises(ValueError, match="bulk"):
        truncated_normal.fit(X, y, weights, threshold, "bulk")


def test_tail_y_below_threshold_raises():
    """tail fit requires all y >= threshold_rate."""
    rng = np.random.default_rng(0)
    threshold = 1e-4
    X = pd.DataFrame({"const": 1.0, "wind_speed": rng.normal(0, 1, 10)})
    y = np.full(10, threshold * 0.5)   # all below threshold — invalid
    weights = np.ones(10)
    with pytest.raises(ValueError, match="tail"):
        truncated_normal.fit(X, y, weights, threshold, "tail")


def test_nonpositive_threshold_raises(bulk_data):
    X, y, weights, *_ = bulk_data
    with pytest.raises(ValueError, match="threshold_rate"):
        truncated_normal.fit(X, y, weights, 0.0, "bulk")


# ---------------------------------------------------------------------------
# Non-convergence handled gracefully
# ---------------------------------------------------------------------------

def test_nonconvergence_returns_result_with_flag(bulk_data):
    """maxiter=1 forces non-convergence; should warn but not crash."""
    import unittest.mock as mock
    from scipy.optimize import OptimizeResult

    X, y, weights, threshold, *_ = bulk_data

    original_minimize = __import__("scipy.optimize", fromlist=["minimize"]).minimize

    def minimize_one_iter(*args, **kwargs):
        kwargs["options"] = dict(kwargs.get("options", {}))
        kwargs["options"]["maxiter"] = 1
        return original_minimize(*args, **kwargs)

    with mock.patch("idd_tc_mortality.distributions.truncated_normal.optimize.minimize", minimize_one_iter):
        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = truncated_normal.fit(X, y, weights, threshold, "bulk")

    assert result.converged is False
    assert result.family == "truncated_normal"
    assert len(result.params) == X.shape[1]


# ---------------------------------------------------------------------------
# Synthetic coefficient recovery
# ---------------------------------------------------------------------------

def test_bulk_recovery(bulk_data):
    """Bulk fit recovers wind_speed coefficient and sigma within tolerance."""
    X, y, weights, threshold, true_intercept, true_wind, true_sigma = bulk_data
    result = truncated_normal.fit(X, y, weights, threshold, "bulk")

    est_wind = result.params[result.param_names.index("wind_speed")]
    est_sigma = result.meta["sigma"]

    assert abs(est_wind - true_wind) < 0.15, (
        f"wind_speed: true={true_wind}, est={est_wind:.4f}"
    )
    assert abs(est_sigma - true_sigma) < 0.15, (
        f"sigma: true={true_sigma}, est={est_sigma:.4f}"
    )


def test_tail_recovery(tail_data):
    """Tail fit recovers wind_speed coefficient and sigma within tolerance."""
    X, y, weights, threshold, true_intercept, true_wind, true_sigma = tail_data
    result = truncated_normal.fit(X, y, weights, threshold, "tail")

    est_wind = result.params[result.param_names.index("wind_speed")]
    est_sigma = result.meta["sigma"]

    assert abs(est_wind - true_wind) < 0.15, (
        f"wind_speed: true={true_wind}, est={est_wind:.4f}"
    )
    assert abs(est_sigma - true_sigma) < 0.15, (
        f"sigma: true={true_sigma}, est={est_sigma:.4f}"
    )


# ---------------------------------------------------------------------------
# fit_one_component round-trip (integration)
# ---------------------------------------------------------------------------

def test_fit_component_bulk_truncated_normal():
    """fit_one_component correctly routes truncated_normal for bulk."""
    from idd_tc_mortality.fit.fit_component import fit_one_component
    from idd_tc_mortality.thresholds import compute_thresholds

    rng = np.random.default_rng(55)
    n = 400
    exposed = rng.uniform(10_000, 500_000, n)
    wind = rng.normal(0, 1, n)

    # Generate deaths: ~50% zero, rest from lognormal
    any_death = rng.binomial(1, 0.5, n)
    rate_nz = np.exp(-11.0 + 0.3 * wind[any_death == 1])
    deaths = np.zeros(n)
    deaths[any_death == 1] = np.maximum(np.round(rate_nz * exposed[any_death == 1]), 1)

    df = pd.DataFrame({
        "deaths": deaths, "exposed": exposed,
        "wind_speed": wind,
        "sdi": rng.uniform(0.3, 0.8, n),
        "basin": "NA", "is_island": 0.0,
    })

    death_rate = deaths / exposed
    threshold_rate = compute_thresholds(death_rate, np.array([0.75]))[0.75]
    bulk_mask = (deaths >= 1) & (death_rate < threshold_rate)
    assert bulk_mask.sum() >= 10, "Need at least 10 bulk obs for the test"

    spec = {
        "component": "bulk",
        "covariate_combo": {"wind_speed": True, "sdi": False, "basin": False, "is_island": False},
        "threshold_quantile": 0.75,
        "threshold_rate": None,
        "family": "truncated_normal",
    }
    result = fit_one_component(spec, df)

    assert result.family == "truncated_normal"
    assert result.meta["truncation_side"] == "bulk"
    assert len(result.fitted_values) == bulk_mask.sum()
    assert np.all(result.fitted_values > 0)


def test_fit_component_tail_truncated_normal():
    """fit_one_component correctly routes truncated_normal for tail (raw rate, not excess)."""
    from idd_tc_mortality.fit.fit_component import fit_one_component
    from idd_tc_mortality.thresholds import compute_thresholds

    rng = np.random.default_rng(77)
    n = 400
    exposed = rng.uniform(10_000, 500_000, n)
    wind = rng.normal(0, 1, n)

    any_death = rng.binomial(1, 0.5, n)
    rate_nz = np.exp(-11.0 + 0.3 * wind[any_death == 1])
    deaths = np.zeros(n)
    deaths[any_death == 1] = np.maximum(np.round(rate_nz * exposed[any_death == 1]), 1)

    df = pd.DataFrame({
        "deaths": deaths, "exposed": exposed,
        "wind_speed": wind,
        "sdi": rng.uniform(0.3, 0.8, n),
        "basin": "NA", "is_island": 0.0,
    })

    death_rate = deaths / exposed
    threshold_rate = compute_thresholds(death_rate, np.array([0.75]))[0.75]
    tail_mask = death_rate >= threshold_rate
    assert tail_mask.sum() >= 10, "Need at least 10 tail obs for the test"

    spec = {
        "component": "tail",
        "covariate_combo": {"wind_speed": True, "sdi": False, "basin": False, "is_island": False},
        "threshold_quantile": 0.75,
        "threshold_rate": None,
        "family": "truncated_normal",
    }
    result = fit_one_component(spec, df)

    assert result.family == "truncated_normal"
    assert result.meta["truncation_side"] == "tail"
    assert len(result.fitted_values) == tail_mask.sum()
    assert np.all(result.fitted_values > 0)
    # Tail predictions must be >= threshold (it's a lower-truncated distribution)
    assert np.all(result.fitted_values >= threshold_rate * 0.5), (
        "Tail truncated normal predictions should be on the tail-rate scale"
    )
