"""
Log-logistic AFT tail component: models excess_rate (death_rate - threshold) as
log-logistic.

Model:
    log(alpha_i) = X_i @ beta           (log-linear scale)
    y_i | x_i ~ LogLogistic(alpha_i, k) (shared scalar shape k)

    y = excess_rate = death_rate - threshold_rate. The caller subtracts the threshold
    before calling fit. This module never sees or stores the threshold.

PDF:
    f(y; alpha, k) = (k/alpha) * (y/alpha)^(k-1) / (1 + (y/alpha)^k)^2

Log-likelihood:
    ll_i = log(k) + (k-1)*log(y_i) - k*log(alpha_i) - 2*log(1 + (y_i/alpha_i)^k)

    Rewriting with r_i = log(y_i) - X_i @ beta:
    -ll_i = -log(k) + log(y_i) - k*r_i + 2*log(1 + exp(k*r_i))

Prediction — log-logistic median:
    The CDF is F(y) = (y/alpha)^k / (1 + (y/alpha)^k).
    F(y) = 0.5 iff y = alpha, so the median is exactly alpha_i = exp(X_i @ beta).

    This is the simplest median formula of all the tail families: exp(X @ params).
    Always finite and positive. Consistent with the median convention used by GPD
    and Weibull in this pipeline.

Analytic gradient (used with BFGS, jac=True):
    Let r_i = log(y_i) - X_i @ beta,  t_i = tanh(k * r_i / 2) = (v_i-1)/(v_i+1)
    where v_i = exp(k * r_i).

    d(-ll_i)/d(beta_j) = -k * X_ij * t_i       [i.e. k * X_ij * (1-v_i)/(1+v_i)]
    d(-ll_i)/d(log_k)  = -1 + k * r_i * t_i

    Using tanh for the gradient (instead of v_i directly) avoids exp overflow for
    large k*r_i, since tanh is bounded in (-1, 1) for any finite input and
    log(1 + exp(k*r_i)) = logaddexp(0, k*r_i) is computed stably via numpy.

Gradient verified analytically and by finite-difference check in tests.

Shape parameter k is stored in meta["shape_param"]. Inverse Hessian covering all
params (beta + log_k) is stored in meta["hess_inv"] as a dense ndarray (BFGS,
not L-BFGS-B), consistent with GPD and Weibull.
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
    """Fit a log-logistic AFT tail model via weighted BFGS MLE.

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
        fitted_values: log-logistic median excess rates (exp(X @ beta)).
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
    weights_norm = np.asarray(weights, dtype=float) / float(np.mean(weights))

    # Initialization: OLS on log(y) gives plausible scale betas; k=1 (exponential-like).
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
            f"Log-logistic MLE did not converge: {opt_result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    beta_hat = opt_result.x[:n_beta]
    k_hat = float(np.exp(opt_result.x[n_beta]))
    hess_inv = np.asarray(opt_result.hess_inv)

    fitted_values = np.exp(X_arr @ beta_hat)   # median = alpha = exp(mu)

    return FitResult(
        params=beta_hat,
        param_names=list(X.columns),
        fitted_values=fitted_values,
        family="log_logistic",
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
    """Predict log-logistic median excess rates from a fitted log-logistic model.

    Returns exp(X @ beta) = alpha_i, the log-logistic median. The median equals
    the scale parameter exactly (CDF = 0.5 at y = alpha). Always finite and positive.

    Parameters
    ----------
    result:
        FitResult from log_logistic.fit().
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
    return np.exp(np.asarray(X, dtype=float) @ result.params)


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
    """Negative weighted log-logistic log-likelihood and analytic gradient.

    Returns (neg_ll, gradient) for use with scipy.optimize.minimize jac=True.

    params = [beta_0, ..., beta_{p-1}, log_k]

    Model: log(alpha_i) = X_i @ beta,  y_i ~ LogLogistic(alpha_i, k)
    Negative log-likelihood per obs:
        -ll_i = -log(k) + log(y_i) - k*r_i + 2*log(1 + exp(k*r_i))
    where r_i = log(y_i) - X_i @ beta

    Gradients (using t_i = tanh(k*r_i/2) = (v_i-1)/(v_i+1), v_i = exp(k*r_i)):
        d(-ll_i)/d(beta_j) = -k * X_ij * t_i
        d(-ll_i)/d(log_k)  = -1 + k * r_i * t_i

    tanh is used instead of exp(k*r_i) directly for numerical stability: tanh is
    bounded in (-1,1) for any finite input, so no overflow risk from large k*r_i.
    The log-likelihood uses logaddexp(0, k*r_i) = log(1+exp(k*r_i)) for the same reason.
    """
    beta = params[:n_beta]
    log_k = params[n_beta]
    k = float(np.exp(log_k))

    mu = X @ beta
    r = np.log(y) - mu     # log(y_i / alpha_i)
    kr = k * r

    # Stable computation: log(1 + exp(k*r))
    log1p_v = np.logaddexp(0.0, kr)

    # Negative log-likelihood
    neg_ll_obs = -np.log(k) + np.log(y) - kr + 2.0 * log1p_v
    neg_ll = float(np.sum(weights * neg_ll_obs))

    # t_i = tanh(k*r_i/2) = (v_i-1)/(v_i+1); bounded in (-1,1)
    t = np.tanh(kr / 2.0)

    # Gradient w.r.t. beta: sum_i w_i * (-k) * X_ij * t_i
    grad_beta = X.T @ (weights * (-k) * t)

    # Gradient w.r.t. log_k: sum_i w_i * (-1 + k*r_i*t_i)
    grad_log_k = float(np.sum(weights * (-1.0 + kr * t)))

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
