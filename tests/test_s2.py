"""
Tests for s2.

s2 fits a binary GLM (logit or cloglog) with log_exposed as a free covariate in X.
Tests cover:
  - Both link functions produce valid probabilities.
  - meta["threshold_rate"] matches the value passed to fit.
  - meta["link"] matches the family argument.
  - family field on FitResult matches the family argument.
  - Coefficient recovery: log_exposed coefficient is positive (more exposure
    increases P(tail) when the DGP includes a positive log_exposed effect).
  - Invalid family raises ValueError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality import s2


@pytest.fixture
def simulated_s2_data():
    """250 observations from a cloglog DGP with log_exposed as free covariate."""
    rng = np.random.default_rng(13)
    n = 250
    wind = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    # log_exposed coefficient = 0.5, wind = 0.8, intercept = -8
    eta = -8.0 + 0.8 * wind + 0.5 * log_exp
    p = 1.0 - np.exp(-np.exp(eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"const": 1.0, "wind_speed": wind, "log_exposed": log_exp})
    return X, y


@pytest.mark.parametrize("family", ["cloglog", "logit"])
def test_fit_family_label(simulated_s2_data, family):
    X, y = simulated_s2_data
    result = s2.fit(X, y, family=family, threshold=1e-4)
    assert result.family == family


@pytest.mark.parametrize("family", ["cloglog", "logit"])
def test_meta_link_matches_family(simulated_s2_data, family):
    X, y = simulated_s2_data
    result = s2.fit(X, y, family=family, threshold=1e-4)
    assert result.meta["link"] == family


@pytest.mark.parametrize("family", ["cloglog", "logit"])
def test_meta_threshold_rate_stored(simulated_s2_data, family):
    X, y = simulated_s2_data
    threshold = 2.5e-5
    result = s2.fit(X, y, family=family, threshold=threshold)
    assert result.meta["threshold_rate"] == threshold


@pytest.mark.parametrize("family", ["cloglog", "logit"])
def test_fitted_values_are_probabilities(simulated_s2_data, family):
    X, y = simulated_s2_data
    result = s2.fit(X, y, family=family, threshold=1e-4)
    assert np.all(result.fitted_values >= 0)
    assert np.all(result.fitted_values <= 1)


@pytest.mark.parametrize("family", ["cloglog", "logit"])
def test_predict_returns_probabilities(simulated_s2_data, family):
    X, y = simulated_s2_data
    result = s2.fit(X, y, family=family, threshold=1e-4)
    preds = s2.predict(result, X)
    assert preds.shape == (len(y),)
    assert np.all(preds >= 0)
    assert np.all(preds <= 1)


def test_log_exposed_coefficient_positive(simulated_s2_data):
    """DGP has positive log_exposed effect; estimated coefficient should be positive."""
    X, y = simulated_s2_data
    result = s2.fit(X, y, family="cloglog", threshold=1e-4)
    log_exp_idx = result.param_names.index("log_exposed")
    assert result.params[log_exp_idx] > 0


def test_invalid_family_raises(simulated_s2_data):
    X, y = simulated_s2_data
    with pytest.raises(ValueError, match="family must be"):
        s2.fit(X, y, family="gamma", threshold=1e-4)


def test_predict_column_mismatch_raises(simulated_s2_data):
    X, y = simulated_s2_data
    result = s2.fit(X, y, family="logit", threshold=1e-4)
    X_bad = X.rename(columns={"wind_speed": "wrong_col"})
    with pytest.raises(ValueError, match="param_names"):
        s2.predict(result, X_bad)


def test_synthetic_recovery_cloglog_free():
    """Recover known S2 coefficients: cloglog + free DGP (log_exposed as free covariate)."""
    rng = np.random.default_rng(21)
    n = 3000
    true_intercept = -8.0
    true_wind      =  0.8
    true_log_exp   =  0.5

    wind = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    eta = true_intercept + true_wind * wind + true_log_exp * log_exp
    p = 1.0 - np.exp(-np.exp(eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"const": 1.0, "wind_speed": wind, "log_exposed": log_exp})

    result = s2.fit(X, y, family="cloglog", threshold=1e-4)

    assert result.meta["link"] == "cloglog"
    assert abs(result.params[result.param_names.index("const")]       - true_intercept) < 0.3
    assert abs(result.params[result.param_names.index("wind_speed")]  - true_wind)      < 0.3
    assert abs(result.params[result.param_names.index("log_exposed")] - true_log_exp)   < 0.3


def test_synthetic_recovery_logit_free():
    """Recover known S2 coefficients: logit + free DGP (log_exposed as free covariate)."""
    rng = np.random.default_rng(34)
    n = 3000
    true_intercept = -4.0
    true_wind      =  0.7
    true_log_exp   =  0.4

    wind = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    eta = true_intercept + true_wind * wind + true_log_exp * log_exp
    p = 1.0 / (1.0 + np.exp(-eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"const": 1.0, "wind_speed": wind, "log_exposed": log_exp})

    result = s2.fit(X, y, family="logit", threshold=1e-4)

    assert result.meta["link"] == "logit"
    assert abs(result.params[result.param_names.index("const")]       - true_intercept) < 0.3
    assert abs(result.params[result.param_names.index("wind_speed")]  - true_wind)      < 0.3
    assert abs(result.params[result.param_names.index("log_exposed")] - true_log_exp)   < 0.3


def test_synthetic_recovery_cloglog_excluded():
    """Recover known S2 coefficients: cloglog + excluded DGP (no exposure in model)."""
    rng = np.random.default_rng(56)
    n = 3000
    true_intercept = -1.5
    true_wind      =  0.9

    wind = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    eta = true_intercept + true_wind * wind
    p = 1.0 - np.exp(-np.exp(eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"const": 1.0, "wind_speed": wind})

    result = s2.fit(X, y, family="cloglog", threshold=1e-4)

    assert result.meta["link"] == "cloglog"
    assert "log_exposed" not in result.param_names
    assert abs(result.params[result.param_names.index("const")]      - true_intercept) < 0.3
    assert abs(result.params[result.param_names.index("wind_speed")] - true_wind)      < 0.3


def test_synthetic_recovery_logit_excluded():
    """Recover known S2 coefficients: logit + excluded DGP (no exposure in model)."""
    rng = np.random.default_rng(78)
    n = 3000
    true_intercept = -1.0
    true_wind      =  0.8

    wind = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    eta = true_intercept + true_wind * wind
    p = 1.0 / (1.0 + np.exp(-eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"const": 1.0, "wind_speed": wind})

    result = s2.fit(X, y, family="logit", threshold=1e-4)

    assert result.meta["link"] == "logit"
    assert "log_exposed" not in result.param_names
    assert abs(result.params[result.param_names.index("const")]      - true_intercept) < 0.3
    assert abs(result.params[result.param_names.index("wind_speed")] - true_wind)      < 0.3
