"""
GPD (Generalized Pareto Distribution) tail model.

Fit: Weighted MLE for GPD with log-linear scale and scalar shape xi.

Outcome y is the *excess* rate: death_rate - threshold. The caller subtracts
the threshold before calling fit. This module never sees or stores the threshold.

Model:
    log(sigma_i) = X_i @ beta          (log-linear scale model)
    y_i | x_i ~ GPD(sigma_i, xi)       (shared scalar shape xi)

Log-likelihood (xi != 0):
    ll_i = -log(sigma_i) - (1/xi + 1) * log(1 + xi * y_i / sigma_i)
Support: 1 + xi * y_i / sigma_i > 0; for xi < 0 this imposes y_i < sigma_i / |xi|.

Log-likelihood (xi = 0, exponential limit):
    ll_i = -log(sigma_i) - y_i / sigma_i

Predictions are GPD medians: sigma_i * (2^xi - 1) / xi for xi != 0,
sigma_i * ln(2) for xi = 0 (exponential limit). The median is always finite,
unlike the mean which is undefined for xi >= 1. Using the median is consistent
with the convention adopted for Weibull and log-logistic.

BFGS is used instead of L-BFGS-B because scipy BFGS stores hess_inv as a dense
ndarray, which is required for downstream uncertainty quantification. L-BFGS-B
stores hess_inv as an implicit LbfgsInvHessProduct operator.

Analytic gradient is provided to avoid finite-difference approximation, which
causes precision loss in BFGS when the log-likelihood magnitude is large (as it
is when log(sigma) << 0, giving -log(sigma) >> 1 per observation).

The gradient for beta (from d(-ll)/d(beta_j)):
    k_i = (1 + xi) * y_i / (sigma_i * z_i),  z_i = 1 + xi * y_i / sigma_i
    grad_beta_j = sum_i w_i * X_ij * (1 - k_i)

For xi = 0:  k_i = y_i / sigma_i   (exponential limit)

The gradient for xi (d(-ll)/d(xi)):
    grad_xi_i = -(1/xi^2) * log(z_i) + k_i / xi   [xi != 0]
    grad_xi_i = u_i - u_i^2 / 2,  u_i = y_i/sigma_i   [xi = 0, Taylor limit]

Only the scale model params (X columns) are stored in FitResult.params. The shape
parameter xi is stored in meta["shape_param"], and the full (n_params x n_params)
inverse Hessian covering [beta, xi] is stored in meta["hess_inv"].
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize

from idd_tc_mortality.distributions.base import FitResult

_XI_EPS = 1e-8  # threshold for xi=0 (exponential) special case in gradient


def fit(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
) -> FitResult:
    """Fit a GPD tail model via weighted BFGS MLE.

    Parameters
    ----------
    X:
        Design matrix. Must already include an intercept column named 'const'.
    y:
        Excess rates (death_rate - threshold), strictly positive, length n_obs.
        The caller subtracts the threshold; this function never sees it.
    weights:
        Positive per-observation weights, length n_obs. Typically proportional
        to exposure (person-years at risk).

    Returns
    -------
    FitResult
        params: scale model coefficients (beta), length X.shape[1].
        param_names: X column names.
        fitted_values: GPD means (sigma_i / (1 - xi)); np.inf if xi >= 1.
        converged: True if BFGS reported success.
        cov: None (placeholder; uncertainty module not yet built).
        meta: dict with 'n_obs', 'shape_param' (xi), 'hess_inv' (dense ndarray,
              shape n_params x n_params where n_params = X.shape[1] + 1),
              'iterations', 'warnings'.

    Raises
    ------
    ValueError
        If y contains non-positive values, weights are non-positive, or lengths
        are inconsistent.

    Warns
    -----
    RuntimeWarning
        If BFGS does not converge. A result is still returned with converged=False.
        If xi >= 1, mean is undefined; fitted_values and predict return np.inf.
    """
    _validate_inputs(X, y, weights)

    n_mean = X.shape[1]
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(weights, dtype=float)

    # Initialization: OLS on log(y) gives plausible scale betas; xi=0 (exponential)
    beta_init, _, _, _ = np.linalg.lstsq(X_arr, np.log(y_arr), rcond=None)
    x0 = np.append(beta_init, 0.0)

    caught_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt_result = optimize.minimize(
            _neg_loglik_and_grad,
            x0,
            args=(X_arr, y_arr, w_arr, n_mean),
            method="BFGS",
            jac=True,  # function returns (f, grad) — avoids finite-difference noise
            options={"maxiter": 1000, "disp": False},
        )
    caught_warnings = [str(w.message) for w in caught]

    converged = bool(opt_result.success)
    if not converged:
        warnings.warn(
            f"GPD MLE did not converge: {opt_result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    beta_hat = opt_result.x[:n_mean]
    xi_hat = float(opt_result.x[n_mean])

    # hess_inv from BFGS is a dense ndarray; L-BFGS-B would give an implicit operator
    hess_inv = np.asarray(opt_result.hess_inv)

    sigma_hat = np.exp(X_arr @ beta_hat)
    fitted_values = _gpd_median(sigma_hat, xi_hat)

    return FitResult(
        params=beta_hat,
        param_names=list(X.columns),
        fitted_values=fitted_values,
        family="gpd",
        converged=converged,
        cov=None,
        meta={
            "n_obs": int(len(y_arr)),
            "shape_param": xi_hat,
            "hess_inv": hess_inv,
            "iterations": int(opt_result.nit),
            "warnings": caught_warnings,
        },
    )


def predict(
    result: FitResult,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict GPD median excess rates from a fitted GPD model.

    Returns the GPD median: sigma_i * (2^xi - 1) / xi for xi != 0,
    sigma_i * ln(2) for xi = 0 (exponential limit). Always finite and positive.

    Parameters
    ----------
    result:
        FitResult from gpd.fit().
    X:
        Design matrix. Columns must match result.param_names exactly
        (use features.align_X before calling).

    Returns
    -------
    np.ndarray
        Predicted median excess rates, length n_obs. Strictly positive.
    """
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names "
            f"{result.param_names}. Call features.align_X before predicting."
        )
    xi = result.meta["shape_param"]
    sigma = np.exp(np.asarray(X, dtype=float) @ result.params)
    return _gpd_median(sigma, xi)


# ---------------------------------------------------------------------------
# GPD median helper
# ---------------------------------------------------------------------------

def _gpd_median(sigma: np.ndarray, xi: float) -> np.ndarray:
    """Return GPD median: sigma * (2^xi - 1) / xi, with ln(2) limit at xi=0.

    Always finite and strictly positive for sigma > 0. This is the canonical
    prediction convention for all MLE tail distributions in this pipeline.
    """
    if abs(xi) < _XI_EPS:
        # Exponential limit: median = sigma * ln(2)
        return sigma * np.log(2.0)
    return sigma * (2.0**xi - 1.0) / xi


# ---------------------------------------------------------------------------
# Negative log-likelihood with analytic gradient
# ---------------------------------------------------------------------------

def _neg_loglik_and_grad(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_mean: int,
) -> tuple[float, np.ndarray]:
    """Negative weighted GPD log-likelihood and its analytic gradient.

    Returns (neg_ll, gradient) for use with scipy.optimize.minimize jac=True.

    params = [beta_0, ..., beta_k, xi]

    Gradient derivation:
        Let z_i = 1 + xi * y_i / sigma_i,  k_i = (1 + xi) * y_i / (sigma_i * z_i)
        d(-ll)/d(beta_j) = sum_i w_i * X_ij * (1 - k_i)
        d(-ll)/d(xi)     = sum_i w_i * [-(1/xi^2)*log(z_i) + k_i/xi]   [xi != 0]
        d(-ll)/d(xi)     = sum_i w_i * [u_i - u_i^2/2],  u_i=y_i/sigma_i  [xi = 0 limit]
    """
    beta = params[:n_mean]
    xi = params[n_mean]

    log_sigma = X @ beta
    sigma = np.exp(log_sigma)
    z = 1.0 + xi * y / sigma

    if np.any(z <= 0):
        # Outside the support — return a large finite value so BFGS steps away
        return 1e20, np.zeros_like(params)

    if abs(xi) < _XI_EPS:
        # Exponential limit (xi -> 0)
        u = y / sigma
        ll_obs = -log_sigma - u
        k = u  # (1+0)*y/(sigma*1) = u in the limit
        # Taylor limit: d(-ll_i)/d(xi) = u_i - u_i^2/2
        grad_xi_obs = u - u**2 / 2.0
    else:
        log_z = np.log(z)
        k = (1.0 + xi) * y / (sigma * z)
        ll_obs = -log_sigma - (1.0 / xi + 1.0) * log_z
        # d(-ll_i)/d(xi) = -(1/xi^2)*log(z_i) + k_i/xi
        grad_xi_obs = -(1.0 / xi**2) * log_z + k / xi

    neg_ll = -float(np.sum(weights * ll_obs))

    # grad_beta_j = X.T @ (weights * (1 - k))
    grad_beta = X.T @ (weights * (1.0 - k))
    grad_xi = float(np.sum(weights * grad_xi_obs))

    return neg_ll, np.append(grad_beta, grad_xi)


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
) -> None:
    if not np.all(y > 0):
        raise ValueError(
            "y must be strictly positive (excess rates). "
            "Found values <= 0. The caller must subtract the threshold before calling fit."
        )

    if not np.all(weights > 0):
        raise ValueError("weights must be strictly positive. Found values <= 0.")

    if len(y) != len(X):
        raise ValueError(f"y length {len(y)} != X rows {len(X)}.")

    if len(weights) != len(X):
        raise ValueError(f"weights length {len(weights)} != X rows {len(X)}.")
