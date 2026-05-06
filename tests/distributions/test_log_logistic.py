"""
Tests for distributions.log_logistic.

Covers:
  - Happy path: family='log_logistic', fitted values positive, meta contains
    shape_param and hess_inv.
  - hess_inv is a dense ndarray, shape (n_params, n_params).
  - predict returns exp(X @ beta) = median = scale; column mismatch raises.
  - Median is exactly the scale parameter (theoretical property of log-logistic).
  - y <= 0 raises. Non-positive weights raise. Length mismatches raise.
  - Non-convergence handled gracefully: converged=False, result still returned.
  - Gradient correctness: analytic gradient matches finite-difference to 1e-5.
  - Synthetic recovery: generate excess rates from a known DGP, recover betas and k.
  - fit_one_component / predict_one_component round-trip: uses generic excess-rate path,
    predict_component adds threshold_rate back via tail_outcome:'excess'.

DGP design:
  log(alpha_i) = true_intercept + true_wind * wind_i
  y_i ~ LogLogistic(alpha_i, k)  (y = excess rate, positive)
  scipy.stats.fisk(c=k, scale=alpha) is the log-logistic distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisk  # fisk == log-logistic

from idd_tc_mortality.distributions import log_logistic
from idd_tc_mortality.distributions.base import FitResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_ll_data():
    """300 observations from a log-logistic log-linear DGP."""
    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(0, 1, n)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x})
    mu = -10.0 + 0.3 * x     # log(alpha); alpha ~ exp(-10) ≈ 4.5e-5
    k_true = 1.5
    y = fisk.rvs(c=k_true, scale=np.exp(mu), random_state=rng)
    weights = np.ones(n)
    return X, y, weights


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_fit_returns_fitresult(small_ll_data):
    X, y, weights = small_ll_data
    result = log_logistic.fit(X, y, weights)
    assert result.family == "log_logistic"
    assert len(result.params) == X.shape[1]
    assert list(result.param_names) == list(X.columns)
    assert len(result.fitted_values) == len(y)
    assert result.cov is None


def test_fitted_values_positive_finite(small_ll_data):
    X, y, weights = small_ll_data
    result = log_logistic.fit(X, y, weights)
    assert np.all(np.isfinite(result.fitted_values))
    assert np.all(result.fitted_values > 0)


def test_meta_contains_required_keys(small_ll_data):
    X, y, weights = small_ll_data
    result = log_logistic.fit(X, y, weights)
    for key in ("shape_param", "hess_inv", "n_obs", "iterations", "warnings"):
        assert key in result.meta, f"Missing meta key: {key!r}"
    assert result.meta["n_obs"] == len(y)
    assert result.meta["shape_param"] > 0


# ---------------------------------------------------------------------------
# hess_inv is a dense ndarray
# ---------------------------------------------------------------------------

def test_hess_inv_is_dense_ndarray(small_ll_data):
    X, y, weights = small_ll_data
    result = log_logistic.fit(X, y, weights)
    hess_inv = result.meta["hess_inv"]
    assert isinstance(hess_inv, np.ndarray), (
        f"hess_inv must be np.ndarray, got {type(hess_inv)}."
    )
    n_params = X.shape[1] + 1
    assert hess_inv.shape == (n_params, n_params)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_returns_positive_finite(small_ll_data):
    X, y, weights = small_ll_data
    result = log_logistic.fit(X, y, weights)
    preds = log_logistic.predict(result, X)
    assert len(preds) == len(y)
    assert np.all(preds > 0)
    assert np.all(np.isfinite(preds))


def test_predict_matches_fitted_values(small_ll_data):
    X, y, weights = small_ll_data
    result = log_logistic.fit(X, y, weights)
    preds = log_logistic.predict(result, X)
    np.testing.assert_allclose(preds, result.fitted_values, rtol=1e-10)


def test_predict_equals_exp_x_beta(small_ll_data):
    """Log-logistic median = exp(X @ beta); predict must return this exactly."""
    X, y, weights = small_ll_data
    result = log_logistic.fit(X, y, weights)
    expected = np.exp(np.asarray(X) @ result.params)
    np.testing.assert_allclose(log_logistic.predict(result, X), expected, rtol=1e-12)


def test_predict_column_mismatch_raises(small_ll_data):
    X, y, weights = small_ll_data
    result = log_logistic.fit(X, y, weights)
    X_wrong = X.rename(columns={"wind_speed": "sdi"})
    with pytest.raises(ValueError, match="param_names"):
        log_logistic.predict(result, X_wrong)


# ---------------------------------------------------------------------------
# Median is the scale parameter (theoretical check)
# ---------------------------------------------------------------------------

def test_median_is_scale_parameter():
    """For LogLogistic(alpha, k), the median is exactly alpha = exp(mu).

    This verifies the log-logistic median formula analytically:
    CDF(y) = (y/alpha)^k / (1 + (y/alpha)^k). At y=alpha: CDF = 0.5.
    """
    mu = np.array([-12.0, -11.0, -10.0])
    alpha = np.exp(mu)
    k = 2.0
    # Confirm CDF(alpha) = 0.5 via scipy
    cdf_at_alpha = fisk.cdf(alpha, c=k, scale=alpha)
    np.testing.assert_allclose(cdf_at_alpha, 0.5, atol=1e-12)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_y_zero_raises(small_ll_data):
    X, y, weights = small_ll_data
    y_bad = y.copy()
    y_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        log_logistic.fit(X, y_bad, weights)


def test_y_negative_raises(small_ll_data):
    X, y, weights = small_ll_data
    y_bad = y.copy()
    y_bad[0] = -1e-8
    with pytest.raises(ValueError, match="strictly positive"):
        log_logistic.fit(X, y_bad, weights)


def test_nonpositive_weights_raises(small_ll_data):
    X, y, weights = small_ll_data
    w_bad = weights.copy()
    w_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        log_logistic.fit(X, y, w_bad)


def test_y_length_mismatch_raises(small_ll_data):
    X, y, weights = small_ll_data
    with pytest.raises(ValueError, match="length"):
        log_logistic.fit(X, y[:-1], weights)


def test_weights_length_mismatch_raises(small_ll_data):
    X, y, weights = small_ll_data
    with pytest.raises(ValueError, match="length"):
        log_logistic.fit(X, y, weights[:-1])


# ---------------------------------------------------------------------------
# Gradient correctness
# ---------------------------------------------------------------------------

def test_analytic_gradient_matches_finite_difference():
    """Analytic gradient must match central FD to 1e-5 relative error."""
    from idd_tc_mortality.distributions.log_logistic import _neg_loglik_and_grad

    rng = np.random.default_rng(7)
    n, p = 80, 3
    X = np.column_stack([np.ones(n), rng.normal(0, 1, (n, p - 1))])
    true_beta = np.array([-11.0, 0.4, -0.3])
    true_k = 1.5
    y = fisk.rvs(c=true_k, scale=np.exp(X @ true_beta), random_state=rng)
    w = rng.uniform(1, 2, n)
    params = np.append(true_beta + rng.normal(0, 0.1, p), np.log(true_k + 0.05))

    _, g_analytic = _neg_loglik_and_grad(params, X, y, w, p)

    eps = 1e-6
    g_fd = np.zeros(len(params))
    for i in range(len(params)):
        p_plus = params.copy(); p_plus[i] += eps
        p_minus = params.copy(); p_minus[i] -= eps
        f_plus, _ = _neg_loglik_and_grad(p_plus, X, y, w, p)
        f_minus, _ = _neg_loglik_and_grad(p_minus, X, y, w, p)
        g_fd[i] = (f_plus - f_minus) / (2 * eps)

    rel_err = np.abs(g_analytic - g_fd) / (np.abs(g_fd) + 1e-10)
    assert np.max(rel_err) < 1e-5, (
        f"Max relative gradient error {np.max(rel_err):.2e} > 1e-5. "
        f"Analytic: {g_analytic}, FD: {g_fd}"
    )


# ---------------------------------------------------------------------------
# Non-convergence handled gracefully
# ---------------------------------------------------------------------------

def test_nonconvergence_returns_result_with_flag(small_ll_data):
    import unittest.mock as mock

    X, y, weights = small_ll_data
    original_minimize = __import__("scipy.optimize", fromlist=["minimize"]).minimize

    def minimize_one_iter(*args, **kwargs):
        kwargs["options"] = dict(kwargs.get("options", {}))
        kwargs["options"]["maxiter"] = 1
        return original_minimize(*args, **kwargs)

    with mock.patch("idd_tc_mortality.distributions.log_logistic.optimize.minimize", minimize_one_iter):
        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = log_logistic.fit(X, y, weights)

    assert result.converged is False
    assert result.family == "log_logistic"
    assert len(result.params) == X.shape[1]


# ---------------------------------------------------------------------------
# Synthetic coefficient recovery
# ---------------------------------------------------------------------------

def test_recovery_two_active_covariates():
    """Scale betas and shape k recovered within tolerance from a known DGP."""
    rng = np.random.default_rng(7)
    n = 1000
    true_intercept = -17.0
    true_wind = 0.4
    true_sdi = -0.3
    true_logexp = 0.5
    k_true = 1.5

    wind = rng.normal(0, 1, n)
    sdi_val = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    exposed = np.exp(log_exp)

    mu = true_intercept + true_wind * wind + true_sdi * sdi_val + true_logexp * log_exp
    y = fisk.rvs(c=k_true, scale=np.exp(mu), random_state=rng)

    X = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val, "log_exposed": log_exp})
    result = log_logistic.fit(X, y, weights=exposed)

    tol_beta = 0.3
    tol_k = 0.2
    est = {name: result.params[result.param_names.index(name)]
           for name in ["wind_speed", "sdi", "log_exposed"]}

    assert abs(est["wind_speed"] - true_wind) < tol_beta, (
        f"wind_speed: true={true_wind}, est={est['wind_speed']:.3f}"
    )
    assert abs(est["sdi"] - true_sdi) < tol_beta, (
        f"sdi: true={true_sdi}, est={est['sdi']:.3f}"
    )
    assert abs(est["log_exposed"] - true_logexp) < tol_beta, (
        f"log_exposed: true={true_logexp}, est={est['log_exposed']:.3f}"
    )
    assert abs(result.meta["shape_param"] - k_true) < tol_k, (
        f"k: true={k_true}, est={result.meta['shape_param']:.4f}"
    )


def test_recovery_null_covariate():
    """SDI with true coefficient 0 is not spuriously detected."""
    rng = np.random.default_rng(13)
    n = 1000
    true_intercept = -17.0
    true_wind = 0.4
    true_logexp = 0.5
    k_true = 1.2

    wind = rng.normal(0, 1, n)
    sdi_val = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    exposed = np.exp(log_exp)

    mu = true_intercept + true_wind * wind + true_logexp * log_exp
    y = fisk.rvs(c=k_true, scale=np.exp(mu), random_state=rng)

    X = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val, "log_exposed": log_exp})
    result = log_logistic.fit(X, y, weights=exposed)

    tol = 0.3
    est = {name: result.params[result.param_names.index(name)]
           for name in ["wind_speed", "sdi", "log_exposed"]}

    assert abs(est["wind_speed"] - true_wind) < tol
    assert abs(est["log_exposed"] - true_logexp) < tol
    assert abs(est["sdi"]) < tol, f"sdi should be near 0: est={est['sdi']:.3f}"


# ---------------------------------------------------------------------------
# fit_one_component / predict_one_component round-trip
# ---------------------------------------------------------------------------

def test_fit_component_tail_log_logistic():
    """fit_one_component routes log_logistic through the generic excess-rate tail path."""
    from idd_tc_mortality.fit.fit_component import fit_one_component
    from idd_tc_mortality.thresholds import compute_thresholds

    rng = np.random.default_rng(55)
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
    assert tail_mask.sum() >= 10

    spec = {
        "component": "tail",
        "covariate_combo": {"wind_speed": True, "sdi": False, "basin": False, "is_island": False},
        "threshold_quantile": 0.75,
        "threshold_rate": None,
        "family": "log_logistic",
    }
    result = fit_one_component(spec, df)

    assert result.family == "log_logistic"
    assert len(result.fitted_values) == tail_mask.sum()
    assert np.all(result.fitted_values > 0)
    assert np.all(np.isfinite(result.fitted_values))


def test_predict_component_tail_log_logistic_adds_threshold():
    """predict_one_component adds threshold_rate for log_logistic (tail_outcome:'excess')."""
    from idd_tc_mortality.fit.fit_component import fit_one_component
    from idd_tc_mortality.evaluate.predict_component import predict_one_component
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

    spec = {
        "component": "tail",
        "covariate_combo": {"wind_speed": True, "sdi": False, "basin": False, "is_island": False},
        "threshold_quantile": 0.75,
        "threshold_rate": float(threshold_rate),
        "family": "log_logistic",
    }
    result = fit_one_component(spec, df)
    preds = predict_one_component(spec, result, df)

    assert np.all(preds.values >= threshold_rate * 0.99), (
        "Log-logistic tail predictions after predict_component must be >= threshold_rate."
    )
    assert np.all(np.isfinite(preds.values))
