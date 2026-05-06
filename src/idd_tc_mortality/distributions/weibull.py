"""
Weibull AFT tail component: models excess_rate (death_rate - threshold) as Weibull.

Model:
    log(lambda_i) = X_i @ beta          (log-linear scale)
    y_i | x_i ~ Weibull(k, lambda_i)    (shared scalar shape k)

    y = excess_rate = death_rate - threshold_rate. The caller subtracts the threshold
    before calling fit. This module never sees or stores the threshold.

Log-likelihood:
    ll_i = log(k) + (k-1)*log(y_i) - k*log(lambda_i) - (y_i/lambda_i)^k
         = log(k) - k*r_i + log(y_i) + u_i * (-1)    [sign corrected below]

    where r_i = log(y_i) - X_i @ beta,  u_i = exp(k * r_i) = (y_i / lambda_i)^k

    -ll_i = -log(k) - k*r_i + log(y_i) + u_i

Prediction — Weibull median:
    median_i = lambda_i * (ln 2)^(1/k) = exp(X_i @ beta + log(ln 2) / k)

    The median is always finite and positive for lambda > 0, k > 0. Consistent
    with the convention used by GPD and log-logistic in this pipeline.

Analytic gradient (used with BFGS, jac=True):
    Let r_i = log(y_i) - X_i @ beta,  u_i = exp(k * r_i)

    d(-ll_i)/d(beta_j) = k * X_ij * (1 - u_i)
    d(-ll_i)/d(log_k)  = -1 + k * r_i * (u_i - 1)

    (using chain rule: d/d(log_k) = k * d/d(k), and d(u_i)/d(k) = r_i * u_i)

Both formulas verified analytically and by finite-difference check in tests.

Shape parameter k is stored in meta["shape_param"]. Inverse Hessian covering all
params (beta + log_k) is stored in meta["hess_inv"] as a dense ndarray (BFGS,
not L-BFGS-B), consistent with GPD.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize

from idd_tc_mortality.distributions.base import FitResult


def fit(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
) -> FitResult:
    """Fit a Weibull AFT tail model via weighted BFGS MLE.

    Parameters
    ----------
    X:
        Design matrix. Must already include an intercept column named 'const'.
        log_exposed is included as a free covariate (not an offset).
    y:
        Excess rates (death_rate - threshold_rate), strictly positive, length n_obs.
        The caller subtracts the threshold; this function never sees it.
    weights:
        Positive per-observation weights, length n_obs. Typically the exposed
        population count. Normalized internally so the shape estimate is invariant
        to weight scale.

    Returns
    -------
    FitResult
        params: scale model coefficients (beta), length X.shape[1].
        param_names: X column names.
        fitted_values: Weibull median excess rates (lambda_i * (ln2)^(1/k)).
        converged: True if BFGS reported success.
        cov: None (placeholder; uncertainty module not yet built).
        meta: dict with 'n_obs', 'shape_param' (k), 'hess_inv' (dense ndarray,
              shape (n_params x n_params) where n_params = X.shape[1] + 1),
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
    """
    _validate_inputs(X, y, weights)

    n_beta = X.shape[1]
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    # Normalize weights so shape estimate is invariant to exposure magnitude.
    weights_norm = np.asarray(weights, dtype=float) / float(np.mean(weights))

    # Initialization: OLS on log(y) gives plausible scale betas; k=1 (exponential).
    beta_init, _, _, _ = np.linalg.lstsq(X_arr, np.log(y_arr), rcond=None)
    x0 = np.append(beta_init, 0.0)   # log_k = 0 → k = 1

    caught_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt_result = optimize.minimize(
            _neg_loglik_and_grad,
            x0,
            args=(X_arr, y_arr, weights_norm, n_beta),
            method="BFGS",
            jac=True,
            options={"maxiter": 2000, "disp": False},
        )
    caught_warnings = [str(w.message) for w in caught]

    converged = bool(opt_result.success)
    if not converged:
        warnings.warn(
            f"Weibull MLE did not converge: {opt_result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    beta_hat = opt_result.x[:n_beta]
    k_hat = float(np.exp(opt_result.x[n_beta]))
    hess_inv = np.asarray(opt_result.hess_inv)

    fitted_values = _weibull_median(X_arr @ beta_hat, k_hat)

    return FitResult(
        params=beta_hat,
        param_names=list(X.columns),
        fitted_values=fitted_values,
        family="weibull",
        converged=converged,
        cov=None,
        meta={
            "n_obs": int(len(y_arr)),
            "shape_param": k_hat,
            "hess_inv": hess_inv,
            "iterations": int(opt_result.nit),
            "warnings": caught_warnings,
        },
    )


def predict(
    result: FitResult,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict Weibull median excess rates from a fitted Weibull model.

    Returns lambda_i * (ln 2)^(1/k) = exp(X @ beta + log(ln 2) / k).
    Always finite and positive.

    Parameters
    ----------
    result:
        FitResult from weibull.fit().
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
    k = result.meta["shape_param"]
    mu = np.asarray(X, dtype=float) @ result.params
    return _weibull_median(mu, k)


# ---------------------------------------------------------------------------
# Weibull median helper
# ---------------------------------------------------------------------------

def _weibull_median(mu: np.ndarray, k: float) -> np.ndarray:
    """Return Weibull median: exp(mu + log(ln2) / k).

    Equivalent to lambda * (ln 2)^(1/k) where lambda = exp(mu). Always finite
    and strictly positive. Consistent with the median convention used by GPD
    and log-logistic in this pipeline.
    """
    return np.exp(mu + np.log(np.log(2.0)) / k)


# ---------------------------------------------------------------------------
# Negative log-likelihood with analytic gradient
# ---------------------------------------------------------------------------

def _neg_loglik_and_grad(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_beta: int,
) -> tuple[float, np.ndarray]:
    """Negative weighted Weibull log-likelihood and analytic gradient.

    Returns (neg_ll, gradient) for use with scipy.optimize.minimize jac=True.

    params = [beta_0, ..., beta_{p-1}, log_k]

    Model: log(lambda_i) = X_i @ beta,  y_i ~ Weibull(k, lambda_i)
    Negative log-likelihood per obs:
        -ll_i = -log(k) - k*r_i + log(y_i) + u_i
    where r_i = log(y_i) - X_i @ beta,  u_i = exp(k * r_i)

    Gradients:
        d(-ll_i)/d(beta_j) = k * X_ij * (1 - u_i)
        d(-ll_i)/d(log_k)  = -1 + k * r_i * (u_i - 1)
    """
    beta = params[:n_beta]
    log_k = params[n_beta]
    k = float(np.exp(log_k))

    mu = X @ beta
    r = np.log(y) - mu      # log residuals: log(y_i) - X_i @ beta

    # Guard against overflow: k*r_i > 700 would overflow float64.
    # Clipping the exponent is safe here — the BFGS penalty from the clipped
    # region is still large and finite, steering the optimizer away.
    kr = np.clip(k * r, -500.0, 700.0)
    u = np.exp(kr)           # u_i = (y_i / lambda_i)^k

    # Negative log-likelihood
    neg_ll_obs = -np.log(k) - k * r + np.log(y) + u
    neg_ll = float(np.sum(weights * neg_ll_obs))

    # Gradient w.r.t. beta: sum_i w_i * k * X_ij * (1 - u_i)
    grad_beta = X.T @ (weights * k * (1.0 - u))

    # Gradient w.r.t. log_k: sum_i w_i * (-1 + k * r_i * (u_i - 1))
    grad_log_k = float(np.sum(weights * (-1.0 + k * r * (u - 1.0))))

    return neg_ll, np.append(grad_beta, grad_log_k)


# ---------------------------------------------------------------------------
# Input validation
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
