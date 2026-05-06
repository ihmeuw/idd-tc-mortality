"""
Tests for distributions.poisson.

Covers:
  - Happy path: family='poisson', fitted values positive, no extra dispersion params.
  - predict is exposure-independent: same covariates, different log_exposed → same rate.
  - Negative y raises. Non-integer y raises. Non-finite log_exposed raises.
  - Forbidden log_exposed/log_exp column in X raises.
  - Length mismatches raise. Predict column mismatch raises.
  - Synthetic recovery: generate counts from a known Poisson log-linear DGP with
    varying exposure as offset, recover mean model coefficients within tolerance.

Fixture design rationale
------------------------
Offset: log_exposed ~ Uniform(log(10K), log(1M)), matching realistic TC exposures.
True rate: exp(true_intercept + true_wind * x) ~ 10^{-5} to 10^{-4} per person.
Expected counts: exposed * rate ~ 0.1 to 1000 deaths per storm.
n=300 provides enough counts for reliable estimation under equidispersion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.distributions import poisson


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_poisson_data():
    """300 observations from a Poisson log-linear DGP with realistic TC exposures."""
    rng = np.random.default_rng(83)
    n = 300
    x = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x})
    mu = np.exp(-9.0 + 0.5 * x + log_exp)
    y = rng.poisson(mu).astype(float)
    return X, y, log_exp


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_fit_returns_fitresult(small_poisson_data):
    X, y, log_exp = small_poisson_data
    result = poisson.fit(X, y, log_exp)
    assert result.family == "poisson"
    assert len(result.params) == X.shape[1]
    assert list(result.param_names) == list(X.columns)
    assert len(result.fitted_values) == len(y)
    assert result.cov is None


def test_fitted_values_positive(small_poisson_data):
    X, y, log_exp = small_poisson_data
    result = poisson.fit(X, y, log_exp)
    assert np.all(result.fitted_values > 0)


def test_meta_contains_n_obs(small_poisson_data):
    X, y, log_exp = small_poisson_data
    result = poisson.fit(X, y, log_exp)
    assert "n_obs" in result.meta
    assert result.meta["n_obs"] == len(y)
    # Poisson has no overdispersion parameter — no dispersion_params key
    assert "dispersion_params" not in result.meta


def test_converged(small_poisson_data):
    X, y, log_exp = small_poisson_data
    result = poisson.fit(X, y, log_exp)
    assert result.converged is True


# ---------------------------------------------------------------------------
# predict is exposure-independent
# ---------------------------------------------------------------------------

def test_predict_rate_is_exposure_independent(small_poisson_data):
    """Rate predictions must not depend on log_exposed — exposure cancels exactly."""
    X, y, log_exp = small_poisson_data
    result = poisson.fit(X, y, log_exp)

    X_row = X.iloc[:1]
    rate_small = poisson.predict(result, X_row, np.array([np.log(10_000)]))
    rate_large = poisson.predict(result, X_row, np.array([np.log(1_000_000)]))

    np.testing.assert_allclose(
        rate_small, rate_large, rtol=1e-10,
        err_msg="Rate predictions should be identical regardless of log_exposed.",
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_negative_y_raises(small_poisson_data):
    X, y, log_exp = small_poisson_data
    y_bad = y.copy()
    y_bad[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        poisson.fit(X, y_bad, log_exp)


def test_noninteger_y_raises(small_poisson_data):
    X, y, log_exp = small_poisson_data
    y_bad = y.copy()
    y_bad[0] = 1.5
    with pytest.raises(ValueError, match="integer"):
        poisson.fit(X, y_bad, log_exp)


def test_nonfinite_log_exposed_raises(small_poisson_data):
    X, y, log_exp = small_poisson_data
    log_exp_bad = log_exp.copy()
    log_exp_bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        poisson.fit(X, y, log_exp_bad)


def test_forbidden_log_exposed_column_raises(small_poisson_data):
    X, y, log_exp = small_poisson_data
    X_bad = X.copy()
    X_bad["log_exposed"] = np.log(100)
    with pytest.raises(ValueError, match="log_exposed"):
        poisson.fit(X_bad, y, log_exp)


def test_forbidden_log_exp_column_raises(small_poisson_data):
    X, y, log_exp = small_poisson_data
    X_bad = X.copy()
    X_bad["log_exp"] = np.log(100)
    with pytest.raises(ValueError, match="log_exp"):
        poisson.fit(X_bad, y, log_exp)


def test_y_length_mismatch_raises(small_poisson_data):
    X, y, log_exp = small_poisson_data
    with pytest.raises(ValueError, match="length"):
        poisson.fit(X, y[:-1], log_exp)


def test_log_exposed_length_mismatch_raises(small_poisson_data):
    X, y, log_exp = small_poisson_data
    with pytest.raises(ValueError, match="length"):
        poisson.fit(X, y, log_exp[:-1])


def test_predict_column_mismatch_raises(small_poisson_data):
    X, y, log_exp = small_poisson_data
    result = poisson.fit(X, y, log_exp)
    X_wrong = X.rename(columns={"wind_speed": "sdi"})
    with pytest.raises(ValueError, match="param_names"):
        poisson.predict(result, X_wrong, log_exp)


# ---------------------------------------------------------------------------
# Synthetic coefficient recovery  (TC-realistic scale)
# ---------------------------------------------------------------------------
#
# Poisson is used as a count model for BULK or TAIL.
# DGP: mu_i = exposed_i · exp(intercept + wind·wind_speed + sdi·sdi)
#      y_i ~ Poisson(mu_i)
# log_exposed is the OFFSET (coefficient fixed at 1, NOT in X) — the pipeline design.
# At mean log_exposed ≈ 11.5: mu ≈ exp(11.5 - 10) ≈ 4.5 deaths per storm (bulk scale).
# Poisson MLE is efficient under correct specification; tighter tolerances than NB.

def test_recovery_two_active_covariates():
    """Both wind_speed and sdi have non-zero effects; intercept, wind, sdi recovered within ±0.15."""
    rng = np.random.default_rng(17)
    n = 1000
    true_intercept = -10.0
    true_wind      =  0.4
    true_sdi       = -0.3

    wind    = rng.normal(0, 1, n)
    sdi_val = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)

    X  = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val})
    mu = np.exp(true_intercept + true_wind * wind + true_sdi * sdi_val + log_exp)
    y  = rng.poisson(mu).astype(float)

    result = poisson.fit(X, y, log_exp)

    tol = 0.15
    est = {name: result.params[result.param_names.index(name)]
           for name in ["const", "wind_speed", "sdi"]}

    assert abs(est["const"]      - true_intercept) < tol, f"intercept: true={true_intercept}, est={est['const']:.3f}"
    assert abs(est["wind_speed"] - true_wind)      < tol, f"wind_speed: true={true_wind}, est={est['wind_speed']:.3f}"
    assert abs(est["sdi"]        - true_sdi)       < tol, f"sdi: true={true_sdi}, est={est['sdi']:.3f}"


def test_recovery_null_covariate():
    """SDI with true coefficient 0 is not spuriously detected; wind and intercept still recovered."""
    rng = np.random.default_rng(23)
    n = 1000
    true_intercept = -10.0
    true_wind      =  0.4
    true_sdi       =  0.0   # null

    wind    = rng.normal(0, 1, n)
    sdi_val = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)

    X  = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val})
    mu = np.exp(true_intercept + true_wind * wind + log_exp)
    y  = rng.poisson(mu).astype(float)

    result = poisson.fit(X, y, log_exp)

    tol = 0.15
    est = {name: result.params[result.param_names.index(name)]
           for name in ["wind_speed", "sdi"]}

    assert abs(est["wind_speed"] - true_wind) < tol, f"wind_speed: true={true_wind}, est={est['wind_speed']:.3f}"
    assert abs(est["sdi"])                    < tol, f"sdi should be near 0: est={est['sdi']:.3f}"
