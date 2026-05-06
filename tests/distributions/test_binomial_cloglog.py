"""
Tests for distributions.binomial_cloglog.

Covers:
  - Happy path: fit returns FitResult with correct shape and valid probability range.
  - Offset is active: higher exposed → higher probability at fixed covariates.
  - Forbidden column guard: X with 'log_exposed' or 'log_exp' raises.
  - Non-finite log_exposed raises.
  - Non-binary y raises.
  - Length mismatch raises.
  - Predict column mismatch raises.
  - Synthetic recovery: known cloglog coefficients recoverable from data generated
    with realistic TC-scale exposures and a sufficiently negative intercept.

Fixture design rationale
------------------------
log_exp is drawn from Uniform(log(10K), log(1M)) to match realistic TC exposure
distributions. The true intercept is set to -11.0 so that eta = intercept + beta*x
+ log_exp is centered near 0, keeping probabilities away from 0 and 1 and making
coefficients identifiable. Strict < 1 is not asserted: saturated probabilities
(p = 1.0) are physically valid for large-exposure, high-wind events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.distributions.binomial_cloglog import (
    fit_binomial_cloglog,
    predict_binomial_cloglog,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_data():
    """200 observations with realistic TC exposure range and non-degenerate probabilities."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, n)
    # Realistic TC exposures: 10K to 1M people
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x})
    # Intercept chosen so eta centers near 0: mean(log_exp) ~ 11.5, intercept = -11.0
    eta = -11.0 + 0.8 * x + log_exp
    p = 1.0 - np.exp(-np.exp(eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)
    return X, y, log_exp


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_fit_returns_fitresult(small_data):
    X, y, log_exp = small_data
    result = fit_binomial_cloglog(X, y, log_exp, "test")
    assert result.family == "test"
    assert len(result.params) == X.shape[1]
    assert list(result.param_names) == list(X.columns)
    assert len(result.fitted_values) == len(y)
    assert np.all(result.fitted_values >= 0)
    assert np.all(result.fitted_values <= 1)
    assert result.cov is None  # placeholder until uncertainty module


def test_fit_converged_flag(small_data):
    X, y, log_exp = small_data
    result = fit_binomial_cloglog(X, y, log_exp, "test")
    assert result.converged is True


def test_meta_contains_expected_keys(small_data):
    X, y, log_exp = small_data
    result = fit_binomial_cloglog(X, y, log_exp, "test")
    assert "n_obs" in result.meta
    assert "n_events" in result.meta
    assert result.meta["n_obs"] == len(y)
    assert result.meta["n_events"] == int(y.sum())


# ---------------------------------------------------------------------------
# Offset correctness: same covariates, higher exposed → higher predicted prob
# ---------------------------------------------------------------------------

def test_predictions_scale_with_exposure(small_data):
    """At identical covariates, larger exposed must yield higher P(death>=1)."""
    X, y, log_exp = small_data
    result = fit_binomial_cloglog(X, y, log_exp, "test")

    X_row = pd.DataFrame({"const": [1.0], "wind_speed": [0.0]})
    p_small = predict_binomial_cloglog(result, X_row, np.array([np.log(10_000)]))
    p_large = predict_binomial_cloglog(result, X_row, np.array([np.log(1_000_000)]))

    assert p_large[0] > p_small[0], (
        "Larger exposed population should yield higher P(deaths>=1) at same covariates"
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_forbidden_log_exposed_column_raises(small_data):
    X, y, log_exp = small_data
    X_bad = X.copy()
    X_bad["log_exposed"] = np.log(100)
    with pytest.raises(ValueError, match="log_exposed"):
        fit_binomial_cloglog(X_bad, y, log_exp, "test")


def test_forbidden_log_exp_column_raises(small_data):
    X, y, log_exp = small_data
    X_bad = X.copy()
    X_bad["log_exp"] = np.log(100)
    with pytest.raises(ValueError, match="log_exp"):
        fit_binomial_cloglog(X_bad, y, log_exp, "test")


def test_nonfinite_log_exposed_raises(small_data):
    X, y, log_exp = small_data
    log_exp_bad = log_exp.copy()
    log_exp_bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_binomial_cloglog(X, y, log_exp_bad, "test")


def test_nonbinary_y_raises(small_data):
    X, y, log_exp = small_data
    y_bad = y.copy()
    y_bad[0] = 2.0
    with pytest.raises(ValueError, match="binary"):
        fit_binomial_cloglog(X, y_bad, log_exp, "test")


def test_y_length_mismatch_raises(small_data):
    X, y, log_exp = small_data
    with pytest.raises(ValueError, match="length"):
        fit_binomial_cloglog(X, y[:-1], log_exp, "test")


def test_log_exposed_length_mismatch_raises(small_data):
    X, y, log_exp = small_data
    with pytest.raises(ValueError, match="length"):
        fit_binomial_cloglog(X, y, log_exp[:-1], "test")


def test_predict_column_mismatch_raises(small_data):
    X, y, log_exp = small_data
    result = fit_binomial_cloglog(X, y, log_exp, "test")
    X_wrong = X.rename(columns={"wind_speed": "sdi"})
    with pytest.raises(ValueError, match="param_names"):
        predict_binomial_cloglog(result, X_wrong, log_exp)


def test_predict_nonfinite_log_exposed_raises(small_data):
    X, y, log_exp = small_data
    result = fit_binomial_cloglog(X, y, log_exp, "test")
    log_exp_bad = log_exp.copy()
    log_exp_bad[5] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        predict_binomial_cloglog(result, X, log_exp_bad)


# ---------------------------------------------------------------------------
# Synthetic recovery test
# ---------------------------------------------------------------------------

def test_synthetic_coefficient_recovery():
    """Fit on data generated from known cloglog parameters; recover them within tolerance.

    Exposure drawn from Uniform(log(10K), log(1M)) to match realistic TC scale.
    Intercept set to -11.0 so eta centers near 0 and coefficients are identifiable.
    Tolerances are ±0.3 — testing that the link function and offset are correct,
    not benchmarking estimation precision.
    """
    rng = np.random.default_rng(0)
    n = 2000
    true_intercept = -11.0
    true_beta = 0.8

    x = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x})

    eta = true_intercept + true_beta * x + log_exp
    p = 1.0 - np.exp(-np.exp(eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = rng.binomial(1, p).astype(float)

    result = fit_binomial_cloglog(X, y, log_exp, "test")

    est_intercept = result.params[result.param_names.index("const")]
    est_beta = result.params[result.param_names.index("wind_speed")]

    assert abs(est_intercept - true_intercept) < 0.3, (
        f"Intercept recovery failed: true={true_intercept}, est={est_intercept:.3f}"
    )
    assert abs(est_beta - true_beta) < 0.3, (
        f"Beta recovery failed: true={true_beta}, est={est_beta:.3f}"
    )
