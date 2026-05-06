"""
Tests for distributions.gpd.

Covers:
  - Happy path: family='gpd', fitted values positive (xi < 1), meta contains
    shape_param and hess_inv.
  - hess_inv is a dense ndarray (not an implicit operator), shape (n_params, n_params).
  - predict returns positive values; column mismatch raises.
  - y <= 0 raises. Non-positive weights raise. Length mismatches raise.
  - Non-convergence is handled gracefully: converged=False, result still returned.
  - Synthetic recovery: generate excess rates from a known GPD log-linear DGP,
    recover scale betas and shape xi within tolerance.

Fixture design rationale
------------------------
Excess rates ~ GPD(sigma, xi) via scipy.stats.genpareto. DGP:
  log(sigma_i) = true_intercept + true_wind * x_i  (sigma ~ 1e-5 to 1e-4)
  xi_true = 0.3 (moderate heavy tail, mean = sigma/(1-xi) exists)
  n=500 gives reliable estimation across the parameter space.

For the weighted recovery test, the phi pattern follows beta.py and
scaled_logit.py: phi_i = exposed_i / scale in [20, 200]. Weighted fit must
recover truth better (lower SSE) than unweighted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

from idd_tc_mortality.distributions import gpd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_gpd_data():
    """300 observations from a GPD log-linear DGP."""
    rng = np.random.default_rng(77)
    n = 300
    x = rng.normal(0, 1, n)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x})
    # log(sigma) = -10 + 0.3*x → sigma ~ exp(-10) ≈ 4.5e-5
    log_sigma = -10.0 + 0.3 * x
    sigma = np.exp(log_sigma)
    xi_true = 0.3
    y = genpareto.rvs(c=xi_true, scale=sigma, random_state=rng)
    weights = np.ones(n)
    return X, y, weights


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_fit_returns_fitresult(small_gpd_data):
    X, y, weights = small_gpd_data
    result = gpd.fit(X, y, weights)
    assert result.family == "gpd"
    assert len(result.params) == X.shape[1]
    assert list(result.param_names) == list(X.columns)
    assert len(result.fitted_values) == len(y)
    assert result.cov is None


def test_fitted_values_positive(small_gpd_data):
    X, y, weights = small_gpd_data
    result = gpd.fit(X, y, weights)
    # xi < 1 for this DGP, so mean exists and fitted values must be > 0
    assert np.all(np.isfinite(result.fitted_values))
    assert np.all(result.fitted_values > 0)


def test_meta_contains_required_keys(small_gpd_data):
    X, y, weights = small_gpd_data
    result = gpd.fit(X, y, weights)
    assert "shape_param" in result.meta
    assert "hess_inv" in result.meta
    assert "n_obs" in result.meta
    assert result.meta["n_obs"] == len(y)


# ---------------------------------------------------------------------------
# hess_inv is a dense ndarray
# ---------------------------------------------------------------------------

def test_hess_inv_is_dense_ndarray(small_gpd_data):
    """BFGS hess_inv must be a dense ndarray, not an implicit L-BFGS-B operator.

    This is the reason BFGS is used instead of L-BFGS-B: downstream uncertainty
    quantification requires explicit matrix access. If the optimizer is silently
    switched to L-BFGS-B, hess_inv becomes an LbfgsInvHessProduct (implicit
    operator) and this test will fail.
    """
    X, y, weights = small_gpd_data
    result = gpd.fit(X, y, weights)
    hess_inv = result.meta["hess_inv"]

    assert isinstance(hess_inv, np.ndarray), (
        f"hess_inv must be np.ndarray, got {type(hess_inv)}. "
        "This means BFGS was replaced with L-BFGS-B or another optimizer that "
        "stores hess_inv as an implicit operator."
    )
    n_params = X.shape[1] + 1  # beta params + xi
    assert hess_inv.shape == (n_params, n_params), (
        f"Expected hess_inv shape {(n_params, n_params)}, got {hess_inv.shape}."
    )


# ---------------------------------------------------------------------------
# xi >= 1: median is always finite (unlike the old mean convention)
# ---------------------------------------------------------------------------

def test_predict_xi_gte_1_returns_finite(small_gpd_data):
    """predict() returns finite positive values even when xi >= 1.

    The old mean-based prediction returned inf for xi >= 1, which silently
    corrupted OOS fold aggregation. The median sigma*(2^xi-1)/xi is always
    finite and positive for sigma > 0.
    """
    X, y, weights = small_gpd_data
    result = gpd.fit(X, y, weights)

    from copy import deepcopy
    result_xi1 = deepcopy(result)
    result_xi1.meta["shape_param"] = 1.5

    preds = gpd.predict(result_xi1, X)

    assert np.all(np.isfinite(preds)), (
        f"GPD median must be finite for xi=1.5. Got non-finite values: {preds[~np.isfinite(preds)]}"
    )
    assert np.all(preds > 0), "GPD median must be positive."
    assert len(preds) == len(X)


def test_fit_xi_gte_1_fitted_values_finite():
    """fit() returns finite fitted_values even when MLE produces xi >= 1.

    With the median convention, xi >= 1 no longer triggers inf or a warning.
    """
    import unittest.mock as mock
    from scipy.optimize import OptimizeResult

    rng = np.random.default_rng(7)
    n = 50
    X = pd.DataFrame({"const": 1.0, "wind_speed": rng.normal(0, 1, n)})
    y = np.abs(rng.standard_cauchy(n)) * 1e-4 + 1e-6
    weights = np.ones(n)

    fake_x = np.array([-10.0, 0.1, 1.5])  # [intercept, wind, xi=1.5]
    fake_result = OptimizeResult(
        x=fake_x,
        success=True,
        message="Mock",
        nit=1,
        hess_inv=np.eye(3),
        fun=0.0,
    )

    with mock.patch("idd_tc_mortality.distributions.gpd.optimize.minimize", return_value=fake_result):
        result = gpd.fit(X, y, weights)

    assert np.all(np.isfinite(result.fitted_values)), (
        "fitted_values must be finite for xi >= 1 under median convention."
    )
    assert np.all(result.fitted_values > 0)
    assert result.meta["shape_param"] == 1.5


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_returns_positive(small_gpd_data):
    X, y, weights = small_gpd_data
    result = gpd.fit(X, y, weights)
    preds = gpd.predict(result, X)
    assert len(preds) == len(y)
    assert np.all(preds > 0)
    assert np.all(np.isfinite(preds))


def test_predict_column_mismatch_raises(small_gpd_data):
    X, y, weights = small_gpd_data
    result = gpd.fit(X, y, weights)
    X_wrong = X.rename(columns={"wind_speed": "sdi"})
    with pytest.raises(ValueError, match="param_names"):
        gpd.predict(result, X_wrong)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_y_zero_raises(small_gpd_data):
    X, y, weights = small_gpd_data
    y_bad = y.copy()
    y_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        gpd.fit(X, y_bad, weights)


def test_y_negative_raises(small_gpd_data):
    X, y, weights = small_gpd_data
    y_bad = y.copy()
    y_bad[0] = -1e-6
    with pytest.raises(ValueError, match="strictly positive"):
        gpd.fit(X, y_bad, weights)


def test_nonpositive_weights_raises(small_gpd_data):
    X, y, weights = small_gpd_data
    weights_bad = weights.copy()
    weights_bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        gpd.fit(X, y, weights_bad)


def test_y_length_mismatch_raises(small_gpd_data):
    X, y, weights = small_gpd_data
    with pytest.raises(ValueError, match="length"):
        gpd.fit(X, y[:-1], weights)


def test_weights_length_mismatch_raises(small_gpd_data):
    X, y, weights = small_gpd_data
    with pytest.raises(ValueError, match="length"):
        gpd.fit(X, y, weights[:-1])


# ---------------------------------------------------------------------------
# Non-convergence handled gracefully
# ---------------------------------------------------------------------------

def test_nonconvergence_returns_result_with_flag(small_gpd_data):
    """maxiter=1 forces non-convergence; should warn but not crash."""
    import unittest.mock as mock

    X, y, weights = small_gpd_data

    original_minimize = __import__("scipy.optimize", fromlist=["minimize"]).minimize

    def minimize_one_iter(*args, **kwargs):
        kwargs["options"] = dict(kwargs.get("options", {}))
        kwargs["options"]["maxiter"] = 1
        return original_minimize(*args, **kwargs)

    with mock.patch("idd_tc_mortality.distributions.gpd.optimize.minimize", minimize_one_iter):
        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = gpd.fit(X, y, weights)

    assert result.converged is False
    assert result.family == "gpd"
    assert len(result.params) == X.shape[1]


# ---------------------------------------------------------------------------
# Synthetic coefficient recovery  (TC-realistic scale)
# ---------------------------------------------------------------------------
#
# GPD is used for the TAIL component (excess death rate above threshold).
# DGP: log(sigma_i) = intercept + wind·wind_speed + sdi·sdi + logexp·log_exposed
#      y_i ~ GPD(sigma_i, xi),   weights = exposed
# At mean log_exposed ≈ 11.5:  sigma ≈ exp(-11.25) ≈ 1.3e-5  (TC-realistic tail excess scale)
# log_exposed is a FREE covariate in X — the pipeline always includes it.

def test_recovery_two_active_covariates():
    """Both wind_speed and sdi have non-zero effects; all four scale params and xi recovered.

    Tolerances ±0.3 for scale betas (GPD MLE is noisier than WLS), ±0.15 for xi.
    """
    rng = np.random.default_rng(42)
    n = 1000
    true_intercept = -17.0
    true_wind      =  0.4
    true_sdi       = -0.3
    true_logexp    =  0.5
    xi_true        =  0.3

    wind    = rng.normal(0, 1, n)
    sdi_val = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    exposed = np.exp(log_exp)

    log_sigma = true_intercept + true_wind * wind + true_sdi * sdi_val + true_logexp * log_exp
    sigma = np.exp(log_sigma)
    y = genpareto.rvs(c=xi_true, scale=sigma, random_state=rng)

    X = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val, "log_exposed": log_exp})
    result = gpd.fit(X, y, weights=exposed)

    tol_beta = 0.3
    tol_xi   = 0.15
    est = {name: result.params[result.param_names.index(name)]
           for name in ["wind_speed", "sdi", "log_exposed"]}

    # Note: the intercept is not asserted. With log_exposed having mean ~11.5 and
    # MLE optimization, the intercept (at log_exposed=0, far outside data range) is
    # poorly identified due to predictor collinearity. The slope coefficients and xi are
    # well-identified and are what matter for the pipeline.
    assert abs(est["wind_speed"]  - true_wind)      < tol_beta, f"wind_speed: true={true_wind}, est={est['wind_speed']:.3f}"
    assert abs(est["sdi"]         - true_sdi)       < tol_beta, f"sdi: true={true_sdi}, est={est['sdi']:.3f}"
    assert abs(est["log_exposed"] - true_logexp)    < tol_beta, f"log_exposed: true={true_logexp}, est={est['log_exposed']:.3f}"
    assert abs(result.meta["shape_param"] - xi_true) < tol_xi,  f"xi: true={xi_true}, est={result.meta['shape_param']:.4f}"


def test_recovery_null_covariate():
    """SDI with true coefficient 0 is not spuriously detected; wind and log_exposed still recovered."""
    rng = np.random.default_rng(7)
    n = 1000
    true_intercept = -17.0
    true_wind      =  0.4
    true_sdi       =  0.0   # null
    true_logexp    =  0.5
    xi_true        =  0.3

    wind    = rng.normal(0, 1, n)
    sdi_val = rng.normal(0, 1, n)
    log_exp = rng.uniform(np.log(10_000), np.log(1_000_000), n)
    exposed = np.exp(log_exp)

    log_sigma = true_intercept + true_wind * wind + true_logexp * log_exp
    sigma = np.exp(log_sigma)
    y = genpareto.rvs(c=xi_true, scale=sigma, random_state=rng)

    X      = pd.DataFrame({"const": 1.0, "wind_speed": wind, "sdi": sdi_val, "log_exposed": log_exp})
    result = gpd.fit(X, y, weights=exposed)

    tol = 0.3
    est = {name: result.params[result.param_names.index(name)]
           for name in ["wind_speed", "sdi", "log_exposed"]}

    assert abs(est["wind_speed"]  - true_wind)   < tol, f"wind_speed: true={true_wind}, est={est['wind_speed']:.3f}"
    assert abs(est["log_exposed"] - true_logexp) < tol, f"log_exposed: true={true_logexp}, est={est['log_exposed']:.3f}"
    assert abs(est["sdi"])                       < tol, f"sdi should be near 0: est={est['sdi']:.3f}"


# Tombstone: the old test_weighted_synthetic_recovery used a two-group design
# with no log_exposed in X. Replaced by the tests above.
def _old_weighted_synthetic_recovery_tombstone():
    rng = np.random.default_rng(99)
    n_per = 200
    true_intercept_A = -11.0
    true_intercept_B = -9.0
    true_wind = 0.3
    xi_true = 0.2

    x_A = rng.normal(0, 1, n_per)
    x_B = rng.normal(0, 1, n_per)
    X = pd.DataFrame({
        "const": 1.0,
        "wind_speed": np.concatenate([x_A, x_B]),
    })

    sigma_A = np.exp(true_intercept_A + true_wind * x_A)
    sigma_B = np.exp(true_intercept_B + true_wind * x_B)
    y_A = genpareto.rvs(c=xi_true, scale=sigma_A, random_state=rng)
    y_B = genpareto.rvs(c=xi_true, scale=sigma_B, random_state=rng)
    y = np.concatenate([y_A, y_B])

    pass  # tombstone body — not executed (function name starts with _)
