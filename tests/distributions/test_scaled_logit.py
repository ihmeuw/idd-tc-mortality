"""
Tests for distributions.scaled_logit.

Covers:
  - Happy path: predictions in (0, threshold), family='scaled_logit',
    meta["threshold_rate"] matches what was passed.
  - y at or outside (0, threshold) raises. Non-positive threshold raises.
    Length mismatches raise. Predict column mismatch raises.
  - Weighted synthetic recovery: exposure-proportional variance DGP, weighted
    fit recovers true coefficients better than unweighted — same proof pattern
    as test_beta.py. If weights are ignored both fits are identical and the test
    fails.

Fixture design rationale
------------------------
threshold=0.05 (5% death rate cap). phi_i = exposed_i / phi_scale gives 100x
precision variation (phi in [5, 500]) so large-N storms dominate when weights
are applied. DGP: z_i = logit(y_i/threshold) = true_intercept + true_wind * x_i
+ epsilon_i where Var(epsilon_i) = sigma^2 / phi_i.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.distributions import scaled_logit

THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_scaled_logit_data():
    """200 observations from a scaled logit DGP, y well inside (0, threshold)."""
    rng = np.random.default_rng(55)
    n = 200
    x = rng.normal(0, 1, n)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x})
    # eta centered near -1 → mu centered near threshold * expit(-1) ≈ 0.012
    eta = -1.0 + 0.4 * x
    mu = THRESHOLD / (1.0 + np.exp(-eta))
    sigma = 0.003
    y = np.clip(mu + rng.normal(0, sigma, n), 1e-6, THRESHOLD - 1e-6)
    weights = np.ones(n)
    return X, y, weights


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_fit_returns_fitresult(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    result = scaled_logit.fit(X, y, weights, threshold=THRESHOLD)
    assert result.family == "scaled_logit"
    assert len(result.params) == X.shape[1]
    assert list(result.param_names) == list(X.columns)
    assert len(result.fitted_values) == len(y)
    assert result.cov is None


def test_fitted_values_in_bounds(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    result = scaled_logit.fit(X, y, weights, threshold=THRESHOLD)
    assert np.all(result.fitted_values > 0)
    assert np.all(result.fitted_values < THRESHOLD)


def test_predict_in_bounds(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    result = scaled_logit.fit(X, y, weights, threshold=THRESHOLD)
    preds = scaled_logit.predict(result, X)
    assert len(preds) == len(y)
    assert np.all(preds > 0)
    assert np.all(preds < THRESHOLD)


def test_meta_threshold_stored(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    result = scaled_logit.fit(X, y, weights, threshold=THRESHOLD)
    assert result.meta["threshold_rate"] == THRESHOLD
    assert result.meta["n_obs"] == len(y)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_y_equal_zero_raises(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    y_bad = y.copy()
    y_bad[0] = 0.0
    with pytest.raises(ValueError, match="<= 0"):
        scaled_logit.fit(X, y_bad, weights, threshold=THRESHOLD)


def test_y_equal_threshold_raises(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    y_bad = y.copy()
    y_bad[0] = THRESHOLD
    with pytest.raises(ValueError, match=">= threshold"):
        scaled_logit.fit(X, y_bad, weights, threshold=THRESHOLD)


def test_y_above_threshold_raises(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    y_bad = y.copy()
    y_bad[0] = THRESHOLD + 0.01
    with pytest.raises(ValueError, match=">= threshold"):
        scaled_logit.fit(X, y_bad, weights, threshold=THRESHOLD)


def test_nonpositive_threshold_raises(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    with pytest.raises(ValueError, match="strictly positive"):
        scaled_logit.fit(X, y, weights, threshold=0.0)


def test_negative_threshold_raises(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    with pytest.raises(ValueError, match="strictly positive"):
        scaled_logit.fit(X, y, weights, threshold=-1.0)


def test_nonpositive_weights_raises(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    weights_bad = weights.copy()
    weights_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        scaled_logit.fit(X, y, weights_bad, threshold=THRESHOLD)


def test_y_length_mismatch_raises(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    with pytest.raises(ValueError, match="length"):
        scaled_logit.fit(X, y[:-1], weights, threshold=THRESHOLD)


def test_weights_length_mismatch_raises(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    with pytest.raises(ValueError, match="length"):
        scaled_logit.fit(X, y, weights[:-1], threshold=THRESHOLD)


def test_predict_column_mismatch_raises(small_scaled_logit_data):
    X, y, weights = small_scaled_logit_data
    result = scaled_logit.fit(X, y, weights, threshold=THRESHOLD)
    X_wrong = X.rename(columns={"wind_speed": "sdi"})
    with pytest.raises(ValueError, match="param_names"):
        scaled_logit.predict(result, X_wrong)


# ---------------------------------------------------------------------------
# Synthetic coefficient recovery  (TC-realistic design)
# ---------------------------------------------------------------------------
#
# Scaled_logit is used for the BULK component.  y ∈ (0, threshold).
# DGP: z_i = logit(y_i / threshold) = intercept + wind·wind_speed + sdi·sdi
#                                      + logexp·log_exposed + epsilon_i
#      epsilon_i ~ N(0, sigma²/phi_i),  phi_i = exposed_i / phi_scale
#      y_i = threshold / (1 + exp(-z_i)),   weights = exposed
# log_exposed is a FREE covariate in X — the pipeline always includes it.
# threshold = 2e-5 (representative TC bulk/tail split rate).

def test_recovery_two_active_covariates():
    """Both wind_speed and sdi have non-zero effects; all four params recovered within ±0.15."""
    rng = np.random.default_rng(42)
    n = 2000
    true_intercept = -2.5
    true_wind      =  0.3
    true_sdi       = -0.2
    true_logexp    =  0.15
    sigma          =  0.5
    threshold      =  THRESHOLD

    wind    = rng.normal(0, 1, n)
    sdi_val = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    exposed = np.exp(log_exp)
    phi     = exposed / 500.0

    z = (true_intercept + true_wind * wind + true_sdi * sdi_val
         + true_logexp * log_exp + rng.normal(0, sigma / np.sqrt(phi)))
    y = np.clip(threshold / (1.0 + np.exp(-z)), 1e-7, threshold - 1e-7)

    X = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val, "log_exposed": log_exp})
    result_w  = scaled_logit.fit(X, y, weights=exposed, threshold=threshold)
    result_uw = scaled_logit.fit(X, y, weights=np.ones(n), threshold=threshold)

    tol = 0.15
    est = {name: result_w.params[result_w.param_names.index(name)]
           for name in ["const", "wind_speed", "sdi", "log_exposed"]}

    assert abs(est["const"]       - true_intercept) < tol, f"intercept: true={true_intercept}, est={est['const']:.3f}"
    assert abs(est["wind_speed"]  - true_wind)      < tol, f"wind_speed: true={true_wind}, est={est['wind_speed']:.3f}"
    assert abs(est["sdi"]         - true_sdi)       < tol, f"sdi: true={true_sdi}, est={est['sdi']:.3f}"
    assert abs(est["log_exposed"] - true_logexp)    < tol, f"log_exposed: true={true_logexp}, est={est['log_exposed']:.3f}"

    # Exposure weighting guard
    true_vals = np.array([true_intercept, true_wind, true_sdi, true_logexp])
    sse_w  = float(np.sum((result_w.params  - true_vals) ** 2))
    sse_uw = float(np.sum((result_uw.params - true_vals) ** 2))
    assert sse_w < sse_uw, (
        f"Weighted SSE {sse_w:.6f} should be < unweighted {sse_uw:.6f}. Weights may be silently ignored."
    )

    # Bounded link guarantee: predictions stay in (0, threshold) by construction.
    preds = scaled_logit.predict(result_w, X)
    assert np.all(preds > 0) and np.all(preds < threshold), "Predictions out of (0, threshold)"


def test_recovery_null_covariate():
    """SDI with true coefficient 0 is not spuriously detected; wind and log_exposed still recovered."""
    rng = np.random.default_rng(17)
    n = 2000
    true_intercept = -2.5
    true_wind      =  0.3
    true_sdi       =  0.0   # null
    true_logexp    =  0.15
    sigma          =  0.5
    threshold      =  THRESHOLD

    wind    = rng.normal(0, 1, n)
    sdi_val = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    exposed = np.exp(log_exp)
    phi     = exposed / 500.0

    z = (true_intercept + true_wind * wind
         + true_logexp * log_exp + rng.normal(0, sigma / np.sqrt(phi)))
    y = np.clip(threshold / (1.0 + np.exp(-z)), 1e-7, threshold - 1e-7)

    X      = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val, "log_exposed": log_exp})
    result = scaled_logit.fit(X, y, weights=exposed, threshold=threshold)

    tol = 0.15
    est = {name: result.params[result.param_names.index(name)]
           for name in ["wind_speed", "sdi", "log_exposed"]}

    assert abs(est["wind_speed"]  - true_wind)   < tol, f"wind_speed: true={true_wind}, est={est['wind_speed']:.3f}"
    assert abs(est["log_exposed"] - true_logexp) < tol, f"log_exposed: true={true_logexp}, est={est['log_exposed']:.3f}"
    assert abs(est["sdi"])                       < tol, f"sdi should be near 0: est={est['sdi']:.3f}"
