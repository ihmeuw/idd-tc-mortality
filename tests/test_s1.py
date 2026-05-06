"""
Tests for s1.

s1.fit and s1.predict are thin wrappers around distributions.binomial_cloglog. Tests here focus on:
  - Correct family label ('s1') in FitResult.
  - fit/predict roundtrip produces probabilities in (0, 1].
  - Offset is active: higher exposed → higher probability at identical covariates.
  - Synthetic recovery: known cloglog parameters recoverable from data generated with
    realistic TC-scale exposures and a sufficiently negative intercept.

Exhaustive input validation is tested in test_binomial_cloglog.py and not duplicated here.

Fixture design rationale
------------------------
log_exp ~ Uniform(log(10K), log(1M)) matches realistic TC exposure distributions.
Intercept = -11.0 keeps eta centered near 0. Saturated probabilities (p = 1.0) are
valid for large-exposure events, so the range assertion is (0, 1], not (0, 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality import s1
from idd_tc_mortality.distributions.base import FitResult


@pytest.fixture
def simulated_s1_data():
    """300 observations from a known S1 DGP with realistic TC-scale exposures."""
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x})
    # Intercept -11.0 centers eta given mean(log_exp) ~ 11.5
    eta = -11.0 + 1.2 * x + log_exp
    p = 1.0 - np.exp(-np.exp(eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)
    return X, y, log_exp


def test_fitresult_length_mismatch_raises():
    with pytest.raises(ValueError, match="param_names"):
        FitResult(
            params=np.array([1.0, 2.0]),
            param_names=["const"],
            fitted_values=np.array([0.5]),
            family="s1",
        )


def test_fit_family_label(simulated_s1_data):
    X, y, log_exp = simulated_s1_data
    result = s1.fit(X, y, log_exp)
    assert result.family == "s1"


def test_fit_fitted_values_are_probabilities(simulated_s1_data):
    X, y, log_exp = simulated_s1_data
    result = s1.fit(X, y, log_exp)
    assert np.all(result.fitted_values >= 0)
    assert np.all(result.fitted_values <= 1)
    # At least some non-trivial probabilities (not all 0 or all 1)
    assert np.any((result.fitted_values > 0.01) & (result.fitted_values < 0.99))


def test_predict_returns_probabilities(simulated_s1_data):
    X, y, log_exp = simulated_s1_data
    result = s1.fit(X, y, log_exp)
    preds = s1.predict(result, X, log_exp)
    assert len(preds) == len(y)
    assert np.all(preds >= 0)
    assert np.all(preds <= 1)


def test_predict_scales_with_exposed(simulated_s1_data):
    """Holding covariates fixed, much larger exposed increases P(deaths>=1)."""
    X, y, log_exp = simulated_s1_data
    result = s1.fit(X, y, log_exp)

    X_row = X.iloc[:1]
    p_low = s1.predict(result, X_row, np.array([np.log(10_000)]))
    p_high = s1.predict(result, X_row, np.array([np.log(1_000_000)]))

    assert p_high[0] > p_low[0]


def test_synthetic_recovery_cloglog_offset():
    """Recover known S1 coefficients: cloglog + offset DGP (log_exposed coeff fixed at 1)."""
    rng = np.random.default_rng(99)
    n = 3000
    true_intercept = -11.0
    true_wind = 1.0

    x = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x})
    eta = true_intercept + true_wind * x + log_exp
    p = 1.0 - np.exp(-np.exp(eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)

    result = s1.fit(X, y, log_exp, family="cloglog", exposure_mode="offset")

    assert abs(result.params[result.param_names.index("const")]      - true_intercept) < 0.3
    assert abs(result.params[result.param_names.index("wind_speed")] - true_wind)      < 0.3


def test_synthetic_recovery_cloglog_free():
    """Recover known S1 coefficients: cloglog + free DGP (log_exposed coeff estimated, != 1)."""
    rng = np.random.default_rng(42)
    n = 3000
    true_intercept  = -10.0
    true_wind       =  1.0
    true_log_exp    =  0.7   # deliberately != 1 to distinguish from offset mode

    x = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    eta = true_intercept + true_wind * x + true_log_exp * log_exp
    p = 1.0 - np.exp(-np.exp(eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x, "log_exposed": log_exp})

    result = s1.fit(X, y, log_exp, family="cloglog", exposure_mode="free")

    assert abs(result.params[result.param_names.index("const")]       - true_intercept) < 0.3
    assert abs(result.params[result.param_names.index("wind_speed")]  - true_wind)      < 0.3
    assert abs(result.params[result.param_names.index("log_exposed")] - true_log_exp)   < 0.3


def test_synthetic_recovery_cloglog_excluded():
    """Recover known S1 coefficients: cloglog + excluded DGP (no exposure in model)."""
    rng = np.random.default_rng(17)
    n = 3000
    true_intercept = -1.0
    true_wind      =  1.0

    x = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    eta = true_intercept + true_wind * x   # no log_exposed term
    p = 1.0 - np.exp(-np.exp(eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x})

    result = s1.fit(X, y, log_exp, family="cloglog", exposure_mode="excluded")

    assert "log_exposed" not in result.param_names
    assert abs(result.params[result.param_names.index("const")]      - true_intercept) < 0.3
    assert abs(result.params[result.param_names.index("wind_speed")] - true_wind)      < 0.3


def test_synthetic_recovery_logit_free():
    """Recover known S1 coefficients: logit + free DGP."""
    rng = np.random.default_rng(55)
    n = 3000
    true_intercept = -6.0   # centers mean eta near 0: -6 + 0.5*mean(log_exp≈11.5) ≈ -0.25
    true_wind      =  1.0
    true_log_exp   =  0.5

    x = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    eta = true_intercept + true_wind * x + true_log_exp * log_exp
    p = 1.0 / (1.0 + np.exp(-eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x, "log_exposed": log_exp})

    result = s1.fit(X, y, log_exp, family="logit", exposure_mode="free")

    assert result.meta["link"] == "logit"
    assert abs(result.params[result.param_names.index("const")]       - true_intercept) < 0.3
    assert abs(result.params[result.param_names.index("wind_speed")]  - true_wind)      < 0.3
    assert abs(result.params[result.param_names.index("log_exposed")] - true_log_exp)   < 0.3


def test_synthetic_recovery_logit_excluded():
    """Recover known S1 coefficients: logit + excluded DGP (no exposure in model)."""
    rng = np.random.default_rng(88)
    n = 3000
    true_intercept = -1.0
    true_wind      =  1.0

    x = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    eta = true_intercept + true_wind * x
    p = 1.0 / (1.0 + np.exp(-eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x})

    result = s1.fit(X, y, log_exp, family="logit", exposure_mode="excluded")

    assert result.meta["link"] == "logit"
    assert "log_exposed" not in result.param_names
    assert abs(result.params[result.param_names.index("const")]      - true_intercept) < 0.3
    assert abs(result.params[result.param_names.index("wind_speed")] - true_wind)      < 0.3
