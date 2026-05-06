"""
Tests for distributions.gamma.

Covers:
  - Happy path: fit returns FitResult with correct shape, positive predictions, family='gamma'.
  - Weights are used: uniform vs. highly unequal weights on the same data give different params.
  - Non-positive y raises.
  - Non-positive weights raises.
  - Length mismatches raise.
  - Predict column mismatch raises.
  - Synthetic recovery: known Gamma log-linear coefficients recovered from data with
    log_exposed as a free covariate, at realistic TC exposure scale.

Fixture design rationale
------------------------
log_exposed ~ Uniform(log(10K), log(1M)) matches realistic TC distributions.
log_exposed is a free covariate in X (not an offset) — this is the key structural
difference from S1/S2. The intercept is a rate intercept; no centering trick needed.
Shape parameter = 5 gives moderate Gamma variance (CV = 1/sqrt(5) ≈ 0.45).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.distributions import gamma


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_gamma_data():
    """200 observations from a Gamma log-linear DGP with log_exposed as free covariate."""
    rng = np.random.default_rng(17)
    n = 200
    x = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x, "log_exposed": log_exp})
    mu = np.exp(-3.0 + 0.5 * x + 0.3 * log_exp)
    shape = 5.0
    y = rng.gamma(shape=shape, scale=mu / shape)
    weights = np.ones(n)
    return X, y, weights


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_fit_returns_fitresult(small_gamma_data):
    X, y, weights = small_gamma_data
    result = gamma.fit(X, y, weights)
    assert result.family == "gamma"
    assert len(result.params) == X.shape[1]
    assert list(result.param_names) == list(X.columns)
    assert len(result.fitted_values) == len(y)
    assert result.cov is None


def test_fitted_values_are_positive(small_gamma_data):
    X, y, weights = small_gamma_data
    result = gamma.fit(X, y, weights)
    assert np.all(result.fitted_values > 0)


def test_predict_returns_positive(small_gamma_data):
    X, y, weights = small_gamma_data
    result = gamma.fit(X, y, weights)
    preds = gamma.predict(result, X)
    assert len(preds) == len(y)
    assert np.all(preds > 0)


def test_meta_contains_expected_keys(small_gamma_data):
    X, y, weights = small_gamma_data
    result = gamma.fit(X, y, weights)
    assert "n_obs" in result.meta
    assert result.meta["n_obs"] == len(y)


# ---------------------------------------------------------------------------
# Weights are used
# ---------------------------------------------------------------------------

def test_exposure_weights_change_params(small_gamma_data):
    """Exposure-proportional weights must produce different estimates than uniform.

    This is a change-detection guard only. The recovery tests below verify
    that the weighted fit recovers the correct direction.
    """
    X, y, _ = small_gamma_data
    n = len(y)
    rng = np.random.default_rng(99)
    exposure_weights = rng.uniform(10_000, 1_000_000, n)

    result_uniform = gamma.fit(X, y, np.ones(n))
    result_exposure = gamma.fit(X, y, exposure_weights)

    assert not np.allclose(result_uniform.params, result_exposure.params, atol=1e-6), (
        "Uniform and exposure-proportional weights produced identical params — weights are not being used."
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_nonpositive_y_raises(small_gamma_data):
    X, y, weights = small_gamma_data
    y_bad = y.copy()
    y_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        gamma.fit(X, y_bad, weights)


def test_negative_y_raises(small_gamma_data):
    X, y, weights = small_gamma_data
    y_bad = y.copy()
    y_bad[0] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        gamma.fit(X, y_bad, weights)


def test_nonpositive_weights_raises(small_gamma_data):
    X, y, weights = small_gamma_data
    weights_bad = weights.copy()
    weights_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        gamma.fit(X, y, weights_bad)


def test_y_length_mismatch_raises(small_gamma_data):
    X, y, weights = small_gamma_data
    with pytest.raises(ValueError, match="length"):
        gamma.fit(X, y[:-1], weights)


def test_weights_length_mismatch_raises(small_gamma_data):
    X, y, weights = small_gamma_data
    with pytest.raises(ValueError, match="length"):
        gamma.fit(X, y, weights[:-1])


def test_predict_column_mismatch_raises(small_gamma_data):
    X, y, weights = small_gamma_data
    result = gamma.fit(X, y, weights)
    X_wrong = X.rename(columns={"wind_speed": "sdi"})
    with pytest.raises(ValueError, match="param_names"):
        gamma.predict(result, X_wrong)


# ---------------------------------------------------------------------------
# Synthetic coefficient recovery  (TC-realistic scale)
# ---------------------------------------------------------------------------
#
# Gamma is used for the TAIL component only (excess death rate above threshold).
# DGP: mu = exp(intercept + wind·wind_speed + sdi·sdi + logexp·log_exposed)
#      y ~ Gamma(shape=5, scale=mu/5),  weights = exposed
# At mean log_exposed ≈ 11.5:  mu ≈ exp(-11.25) ≈ 1.3e-5  (TC-realistic tail excess rate)
# log_exposed is a FREE covariate in X (not an offset) — the pipeline always includes it.
# weights = exposed so large-population storms dominate the fit.

def test_recovery_two_active_covariates():
    """Both wind_speed and sdi have non-zero effects; all four params recovered within ±0.2.

    Verifies: correct X design (const + wind + sdi + log_exposed), TC-realistic y scale,
    exposure weighting applied in the correct direction.
    """
    rng = np.random.default_rng(3)
    n = 2000
    true_intercept = -17.0
    true_wind      =  0.4
    true_sdi       = -0.3
    true_logexp    =  0.5

    wind    = rng.normal(0, 1, n)
    sdi_val = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    exposed = np.exp(log_exp)

    X = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val, "log_exposed": log_exp})
    mu = np.exp(true_intercept + true_wind * wind + true_sdi * sdi_val + true_logexp * log_exp)
    y = rng.gamma(shape=5.0, scale=mu / 5.0)

    result = gamma.fit(X, y, weights=exposed)

    tol = 0.2
    est = {name: result.params[result.param_names.index(name)]
           for name in ["const", "wind_speed", "sdi", "log_exposed"]}

    assert abs(est["const"]       - true_intercept) < tol, f"intercept: true={true_intercept}, est={est['const']:.3f}"
    assert abs(est["wind_speed"]  - true_wind)      < tol, f"wind_speed: true={true_wind}, est={est['wind_speed']:.3f}"
    assert abs(est["sdi"]         - true_sdi)       < tol, f"sdi: true={true_sdi}, est={est['sdi']:.3f}"
    assert abs(est["log_exposed"] - true_logexp)    < tol, f"log_exposed: true={true_logexp}, est={est['log_exposed']:.3f}"


def test_recovery_null_covariate():
    """SDI with true coefficient 0 is not spuriously detected; wind and log_exposed still recovered.

    Verifies: the model doesn't inflate a zero-effect covariate when the design matrix
    is correctly specified with all four columns.
    """
    rng = np.random.default_rng(7)
    n = 2000
    true_intercept = -17.0
    true_wind      =  0.4
    true_sdi       =  0.0   # null — must not be spuriously detected
    true_logexp    =  0.5

    wind    = rng.normal(0, 1, n)
    sdi_val = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    exposed = np.exp(log_exp)

    X = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val, "log_exposed": log_exp})
    mu = np.exp(true_intercept + true_wind * wind + true_logexp * log_exp)
    y = rng.gamma(shape=5.0, scale=mu / 5.0)

    result = gamma.fit(X, y, weights=exposed)

    tol = 0.2
    est = {name: result.params[result.param_names.index(name)]
           for name in ["wind_speed", "sdi", "log_exposed"]}

    assert abs(est["wind_speed"]  - true_wind)   < tol, f"wind_speed: true={true_wind}, est={est['wind_speed']:.3f}"
    assert abs(est["log_exposed"] - true_logexp) < tol, f"log_exposed: true={true_logexp}, est={est['log_exposed']:.3f}"
    assert abs(est["sdi"])                       < tol, f"sdi should be near 0: est={est['sdi']:.3f}"
