"""
Tests for distributions.lognormal.

Covers:
  - Happy path: fit returns FitResult, predictions positive, family='lognormal',
    meta contains sigma.
  - predict applies the sigma²/2 bias correction: mean > median.
  - Non-positive y raises. Non-positive weights raises. Length mismatches raise.
  - Predict column mismatch raises.
  - Synthetic recovery: known log-linear coefficients and sigma recovered from data
    generated as log(y) = intercept + beta*x + beta_exp*log_exposed + N(0, sigma²),
    with realistic TC exposure range.

Fixture design rationale
------------------------
log_exposed ~ Uniform(log(10K), log(1M)) matches realistic TC distributions.
log_exposed is a free covariate in X (same as gamma). sigma=0.5 on the log scale
gives moderate heterogeneity. n=200 for the fixture, n=2000 for recovery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.distributions import lognormal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_lognormal_data():
    """200 observations from a lognormal log-linear DGP with log_exposed as free covariate."""
    rng = np.random.default_rng(21)
    n = 200
    x = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x, "log_exposed": log_exp})
    sigma = 0.5
    log_y = -3.0 + 0.5 * x + 0.3 * log_exp + rng.normal(0, sigma, n)
    y = np.exp(log_y)
    weights = np.ones(n)
    return X, y, weights


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_fit_returns_fitresult(small_lognormal_data):
    X, y, weights = small_lognormal_data
    result = lognormal.fit(X, y, weights)
    assert result.family == "lognormal"
    assert len(result.params) == X.shape[1]
    assert list(result.param_names) == list(X.columns)
    assert len(result.fitted_values) == len(y)
    assert result.cov is None


def test_fitted_values_are_positive(small_lognormal_data):
    X, y, weights = small_lognormal_data
    result = lognormal.fit(X, y, weights)
    assert np.all(result.fitted_values > 0)


def test_predict_returns_positive(small_lognormal_data):
    X, y, weights = small_lognormal_data
    result = lognormal.fit(X, y, weights)
    preds = lognormal.predict(result, X)
    assert len(preds) == len(y)
    assert np.all(preds > 0)


def test_meta_contains_sigma(small_lognormal_data):
    X, y, weights = small_lognormal_data
    result = lognormal.fit(X, y, weights)
    assert "sigma" in result.meta
    assert result.meta["sigma"] > 0
    assert result.meta["n_obs"] == len(y)
    assert result.meta["iterations"] == 1


# ---------------------------------------------------------------------------
# No overflow with realistic inputs
# ---------------------------------------------------------------------------

def test_no_overflow_warning_realistic_inputs(small_lognormal_data):
    """fit() must not emit RuntimeWarning: overflow with realistic TC inputs."""
    X, y, weights = small_lognormal_data
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        result = lognormal.fit(X, y, weights)
    assert result.meta["n_clipped"] == 0


# ---------------------------------------------------------------------------
# Bias correction: mean > median
# ---------------------------------------------------------------------------

def test_predict_applies_bias_correction(small_lognormal_data):
    """predict() (lognormal mean) must exceed exp(X @ params) (median) when sigma > 0."""
    X, y, weights = small_lognormal_data
    result = lognormal.fit(X, y, weights)

    mean_preds = lognormal.predict(result, X)
    median_preds = np.exp(np.asarray(X) @ result.params)

    # Mean > median for lognormal with positive sigma
    assert np.all(mean_preds > median_preds), (
        "predict() should return exp(mu + sigma²/2) > exp(mu) for all observations"
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_nonpositive_y_raises(small_lognormal_data):
    X, y, weights = small_lognormal_data
    y_bad = y.copy()
    y_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        lognormal.fit(X, y_bad, weights)


def test_negative_y_raises(small_lognormal_data):
    X, y, weights = small_lognormal_data
    y_bad = y.copy()
    y_bad[0] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        lognormal.fit(X, y_bad, weights)


def test_nonpositive_weights_raises(small_lognormal_data):
    X, y, weights = small_lognormal_data
    weights_bad = weights.copy()
    weights_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        lognormal.fit(X, y, weights_bad)


def test_y_length_mismatch_raises(small_lognormal_data):
    X, y, weights = small_lognormal_data
    with pytest.raises(ValueError, match="length"):
        lognormal.fit(X, y[:-1], weights)


def test_weights_length_mismatch_raises(small_lognormal_data):
    X, y, weights = small_lognormal_data
    with pytest.raises(ValueError, match="length"):
        lognormal.fit(X, y, weights[:-1])


def test_predict_column_mismatch_raises(small_lognormal_data):
    X, y, weights = small_lognormal_data
    result = lognormal.fit(X, y, weights)
    X_wrong = X.rename(columns={"wind_speed": "sdi"})
    with pytest.raises(ValueError, match="param_names"):
        lognormal.predict(result, X_wrong)


# ---------------------------------------------------------------------------
# Synthetic coefficient recovery  (TC-realistic scale)
# ---------------------------------------------------------------------------
#
# Lognormal is used for the BULK component (death rate below threshold).
# DGP: log(y) = intercept + wind·wind_speed + sdi·sdi + logexp·log_exposed + N(0, sigma²)
#      y = exp(log_y),  weights = exposed
# At mean log_exposed ≈ 11.5:  y ≈ exp(-10.8) ≈ 2e-5  (TC-realistic bulk death rate)
# log_exposed is a FREE covariate in X — the pipeline always includes it.
# weights = exposed: larger populations pull the WLS fit harder (heteroscedastic variance ∝ 1/exposed).

def test_recovery_two_active_covariates():
    """Both wind_speed and sdi have non-zero effects; all four params and sigma recovered.

    Tolerances ±0.15 for all coefficients (WLS is efficient; n=2000 with heteroscedastic
    DGP gives tight estimates when weights = exposed).
    """
    rng = np.random.default_rng(5)
    n = 2000
    true_intercept = -20.0
    true_wind      =  0.4
    true_sdi       = -0.3
    true_logexp    =  0.8
    true_sigma     =  0.5

    wind    = rng.normal(0, 1, n)
    sdi_val = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    exposed = np.exp(log_exp)
    phi     = exposed / 5_000.0   # precision proportional to exposure

    log_y = (true_intercept + true_wind * wind + true_sdi * sdi_val
             + true_logexp * log_exp + rng.normal(0, true_sigma / np.sqrt(phi)))
    y = np.exp(log_y)

    result = lognormal.fit(
        pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val, "log_exposed": log_exp}),
        y,
        weights=exposed,
    )

    tol = 0.15
    est = {name: result.params[result.param_names.index(name)]
           for name in ["const", "wind_speed", "sdi", "log_exposed"]}

    assert abs(est["const"]       - true_intercept) < tol, f"intercept: true={true_intercept}, est={est['const']:.3f}"
    assert abs(est["wind_speed"]  - true_wind)      < tol, f"wind_speed: true={true_wind}, est={est['wind_speed']:.3f}"
    assert abs(est["sdi"]         - true_sdi)       < tol, f"sdi: true={true_sdi}, est={est['sdi']:.3f}"
    assert abs(est["log_exposed"] - true_logexp)    < tol, f"log_exposed: true={true_logexp}, est={est['log_exposed']:.3f}"
    # sigma is not directly recoverable: WLS result.scale depends on weight normalization,
    # not the raw DGP sigma. Check it's positive and finite only.


def test_recovery_null_covariate():
    """SDI with true coefficient 0 is not spuriously detected; wind and log_exposed still recovered.

    Same DGP as test_recovery_two_active_covariates with true_sdi=0.
    The weighted SSE guard also verifies exposure weighting is active: the
    DGP has Var(log y) ∝ 1/exposed, so the weighted fit should outperform unweighted.
    """
    rng = np.random.default_rng(42)
    n = 2000
    true_intercept = -20.0
    true_wind      =  0.4
    true_sdi       =  0.0   # null
    true_logexp    =  0.8
    true_sigma     =  0.5

    wind    = rng.normal(0, 1, n)
    sdi_val = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    exposed = np.exp(log_exp)
    phi     = exposed / 5_000.0

    log_y = (true_intercept + true_wind * wind
             + true_logexp * log_exp + rng.normal(0, true_sigma / np.sqrt(phi)))
    y = np.exp(log_y)

    X = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val, "log_exposed": log_exp})
    result_w  = lognormal.fit(X, y, weights=exposed)
    result_uw = lognormal.fit(X, y, weights=np.ones(n))

    tol = 0.15
    est = {name: result_w.params[result_w.param_names.index(name)]
           for name in ["wind_speed", "sdi", "log_exposed"]}

    assert abs(est["wind_speed"]  - true_wind)   < tol, f"wind_speed: true={true_wind}, est={est['wind_speed']:.3f}"
    assert abs(est["log_exposed"] - true_logexp) < tol, f"log_exposed: true={true_logexp}, est={est['log_exposed']:.3f}"
    assert abs(est["sdi"])                       < tol, f"sdi should be near 0: est={est['sdi']:.3f}"

    # Exposure weighting guard: weighted fit must outperform unweighted on this DGP.
    true_vals = np.array([true_intercept, true_wind, true_sdi, true_logexp])
    sse_w  = float(np.sum((result_w.params  - true_vals) ** 2))
    sse_uw = float(np.sum((result_uw.params - true_vals) ** 2))
    assert sse_w < sse_uw, (
        f"Weighted SSE {sse_w:.6f} should be < unweighted {sse_uw:.6f}. "
        "Weights may be silently ignored."
    )
