"""
Tests for distributions.weibull.

Covers:
  - Happy path: family='weibull', fitted values positive, meta contains
    shape_param and hess_inv.
  - hess_inv is a dense ndarray (not an implicit operator), shape (n_params, n_params).
  - predict returns positive values; column mismatch raises.
  - y <= 0 raises. Non-positive weights raise. Length mismatches raise.
  - Non-convergence handled gracefully: converged=False, result still returned.
  - Gradient correctness: analytic gradient matches finite-difference to 1e-5 relative
    error across a random parameter vector.
  - Synthetic recovery: generate excess rates from a known Weibull log-linear DGP,
    recover scale betas and shape k within tolerance.
  - fit_one_component round-trip: weibull wired as tail family, uses excess-rate path
    (no special case in fit_component.py), predict_component adds threshold_rate back.

DGP design:
  log(lambda_i) = true_intercept + true_wind * wind_speed_i
  y_i ~ Weibull(k, lambda_i)   (y = excess rate, positive)
  At mean log_exposed ≈ 11.5: lambda ≈ exp(-11.25) ≈ 1.3e-5 (TC-realistic tail scale)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import weibull_min

from idd_tc_mortality.distributions import weibull
from idd_tc_mortality.distributions.base import FitResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_weibull_data():
    """300 observations from a Weibull log-linear DGP."""
    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(0, 1, n)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x})
    # log(lambda) = -10 + 0.3*x → lambda ~ exp(-10) ≈ 4.5e-5
    mu = -10.0 + 0.3 * x
    lam = np.exp(mu)
    k_true = 1.5
    # scipy weibull_min(c=k, scale=lambda): y = lambda * U^(1/k) where U ~ Exp(1)
    y = weibull_min.rvs(c=k_true, scale=lam, random_state=rng)
    weights = np.ones(n)
    return X, y, weights


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_fit_returns_fitresult(small_weibull_data):
    X, y, weights = small_weibull_data
    result = weibull.fit(X, y, weights)
    assert result.family == "weibull"
    assert len(result.params) == X.shape[1]
    assert list(result.param_names) == list(X.columns)
    assert len(result.fitted_values) == len(y)
    assert result.cov is None


def test_fitted_values_positive_finite(small_weibull_data):
    X, y, weights = small_weibull_data
    result = weibull.fit(X, y, weights)
    assert np.all(np.isfinite(result.fitted_values))
    assert np.all(result.fitted_values > 0)


def test_meta_contains_required_keys(small_weibull_data):
    X, y, weights = small_weibull_data
    result = weibull.fit(X, y, weights)
    for key in ("shape_param", "hess_inv", "n_obs", "iterations", "warnings"):
        assert key in result.meta, f"Missing meta key: {key!r}"
    assert result.meta["n_obs"] == len(y)
    assert result.meta["shape_param"] > 0


# ---------------------------------------------------------------------------
# hess_inv is a dense ndarray
# ---------------------------------------------------------------------------

def test_hess_inv_is_dense_ndarray(small_weibull_data):
    """BFGS hess_inv must be a dense ndarray, not an implicit L-BFGS-B operator."""
    X, y, weights = small_weibull_data
    result = weibull.fit(X, y, weights)
    hess_inv = result.meta["hess_inv"]

    assert isinstance(hess_inv, np.ndarray), (
        f"hess_inv must be np.ndarray, got {type(hess_inv)}. "
        "If BFGS was replaced by L-BFGS-B, hess_inv becomes an implicit operator."
    )
    n_params = X.shape[1] + 1   # beta params + log_k
    assert hess_inv.shape == (n_params, n_params), (
        f"Expected hess_inv shape {(n_params, n_params)}, got {hess_inv.shape}."
    )


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_returns_positive_finite(small_weibull_data):
    X, y, weights = small_weibull_data
    result = weibull.fit(X, y, weights)
    preds = weibull.predict(result, X)
    assert len(preds) == len(y)
    assert np.all(preds > 0)
    assert np.all(np.isfinite(preds))


def test_predict_matches_fitted_values(small_weibull_data):
    """predict(result, X_train) must reproduce fitted_values exactly."""
    X, y, weights = small_weibull_data
    result = weibull.fit(X, y, weights)
    preds = weibull.predict(result, X)
    np.testing.assert_allclose(preds, result.fitted_values, rtol=1e-10)


def test_predict_column_mismatch_raises(small_weibull_data):
    X, y, weights = small_weibull_data
    result = weibull.fit(X, y, weights)
    X_wrong = X.rename(columns={"wind_speed": "sdi"})
    with pytest.raises(ValueError, match="param_names"):
        weibull.predict(result, X_wrong)


# ---------------------------------------------------------------------------
# Median formula: exp(mu + log(ln2)/k) is always finite and positive
# ---------------------------------------------------------------------------

def test_median_formula_large_k():
    """Median is finite and positive for large k (thin tail)."""
    mu = np.array([-11.0, -10.0, -9.0])
    k = 10.0
    med = weibull._weibull_median(mu, k)
    assert np.all(np.isfinite(med))
    assert np.all(med > 0)


def test_median_formula_small_k():
    """Median is finite and positive for small k (heavy tail)."""
    mu = np.array([-11.0, -10.0, -9.0])
    k = 0.1
    med = weibull._weibull_median(mu, k)
    assert np.all(np.isfinite(med))
    assert np.all(med > 0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_y_zero_raises(small_weibull_data):
    X, y, weights = small_weibull_data
    y_bad = y.copy()
    y_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        weibull.fit(X, y_bad, weights)


def test_y_negative_raises(small_weibull_data):
    X, y, weights = small_weibull_data
    y_bad = y.copy()
    y_bad[0] = -1e-8
    with pytest.raises(ValueError, match="strictly positive"):
        weibull.fit(X, y_bad, weights)


def test_nonpositive_weights_raises(small_weibull_data):
    X, y, weights = small_weibull_data
    w_bad = weights.copy()
    w_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        weibull.fit(X, y, w_bad)


def test_y_length_mismatch_raises(small_weibull_data):
    X, y, weights = small_weibull_data
    with pytest.raises(ValueError, match="length"):
        weibull.fit(X, y[:-1], weights)


def test_weights_length_mismatch_raises(small_weibull_data):
    X, y, weights = small_weibull_data
    with pytest.raises(ValueError, match="length"):
        weibull.fit(X, y, weights[:-1])


# ---------------------------------------------------------------------------
# Gradient correctness (finite difference check)
# ---------------------------------------------------------------------------

def test_analytic_gradient_matches_finite_difference():
    """Analytic gradient must match central finite differences to 1e-5 relative error."""
    from idd_tc_mortality.distributions.weibull import _neg_loglik_and_grad

    rng = np.random.default_rng(99)
    n, p = 80, 3
    X = np.column_stack([np.ones(n), rng.normal(0, 1, (n, p - 1))])
    true_beta = np.array([-11.0, 0.4, -0.3])
    true_k = 1.5
    y = weibull_min.rvs(c=true_k, scale=np.exp(X @ true_beta), random_state=rng)
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

def test_nonconvergence_returns_result_with_flag(small_weibull_data):
    """maxiter=1 forces non-convergence; should warn but not crash."""
    import unittest.mock as mock

    X, y, weights = small_weibull_data
    original_minimize = __import__("scipy.optimize", fromlist=["minimize"]).minimize

    def minimize_one_iter(*args, **kwargs):
        kwargs["options"] = dict(kwargs.get("options", {}))
        kwargs["options"]["maxiter"] = 1
        return original_minimize(*args, **kwargs)

    with mock.patch("idd_tc_mortality.distributions.weibull.optimize.minimize", minimize_one_iter):
        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = weibull.fit(X, y, weights)

    assert result.converged is False
    assert result.family == "weibull"
    assert len(result.params) == X.shape[1]


# ---------------------------------------------------------------------------
# Synthetic coefficient recovery  (TC-realistic scale)
# ---------------------------------------------------------------------------

def test_recovery_two_active_covariates():
    """Scale betas and shape k are recovered within tolerance from a known DGP.

    DGP mirrors the GPD recovery test: log(lambda_i) = intercept + wind*wind_speed + ...
    At mean log_exposed ≈ 11.5, lambda ≈ exp(-11.25) ≈ 1.3e-5 (TC tail scale).
    """
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
    lam = np.exp(mu)
    y = weibull_min.rvs(c=k_true, scale=lam, random_state=rng)

    X = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val, "log_exposed": log_exp})
    result = weibull.fit(X, y, weights=exposed)

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
    y = weibull_min.rvs(c=k_true, scale=np.exp(mu), random_state=rng)

    X = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val, "log_exposed": log_exp})
    result = weibull.fit(X, y, weights=exposed)

    tol = 0.3
    est = {name: result.params[result.param_names.index(name)]
           for name in ["wind_speed", "sdi", "log_exposed"]}

    assert abs(est["wind_speed"] - true_wind) < tol, (
        f"wind_speed: true={true_wind}, est={est['wind_speed']:.3f}"
    )
    assert abs(est["log_exposed"] - true_logexp) < tol, (
        f"log_exposed: true={true_logexp}, est={est['log_exposed']:.3f}"
    )
    assert abs(est["sdi"]) < tol, f"sdi should be near 0: est={est['sdi']:.3f}"


# ---------------------------------------------------------------------------
# fit_one_component / predict_one_component round-trip (integration)
# ---------------------------------------------------------------------------

def test_fit_component_tail_weibull():
    """fit_one_component routes weibull through the generic excess-rate tail path.

    Confirms: no special case needed in fit_component.py; tail_outcome:'excess'
    means predict_component will add threshold_rate back.
    """
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
        "family": "weibull",
    }
    result = fit_one_component(spec, df)

    assert result.family == "weibull"
    assert len(result.fitted_values) == tail_mask.sum()
    assert np.all(result.fitted_values > 0)
    assert np.all(np.isfinite(result.fitted_values))
    # Fitted values are excess rates — they should be small positive numbers,
    # well below threshold_rate (Weibull is fit on death_rate - threshold).
    assert np.all(result.fitted_values < threshold_rate * 10), (
        "Fitted excess rates should be smaller than 10x threshold_rate"
    )


def test_predict_component_tail_weibull_adds_threshold():
    """predict_one_component adds threshold_rate for weibull (tail_outcome:'excess')."""
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
        "family": "weibull",
    }
    result = fit_one_component(spec, df)
    preds = predict_one_component(spec, result, df)

    # predict_component adds threshold_rate, so all predictions must be >= threshold_rate
    assert np.all(preds.values >= threshold_rate * 0.99), (
        "Weibull tail predictions after predict_component must be >= threshold_rate. "
        "If they are not, tail_outcome:'excess' is missing from the registry."
    )
    assert np.all(np.isfinite(preds.values))
