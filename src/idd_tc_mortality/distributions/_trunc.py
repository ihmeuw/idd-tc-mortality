"""Shared driver for the renormalized-TRUNCATED-likelihood tail variants (variant B).

*** SPIKE — under evaluation, not a committed pipeline variant. ***

B fits the distribution renormalized onto each storm's admissible interval (0, H_i],
H_i = c_i - threshold_rate (the per-storm excess cap). Unlike variant C (which fits the
UNtruncated distribution and censors at predict), B puts the bound INSIDE the likelihood:
each storm contributes  log f(w_i) - log F(H_i)  so the fitted parameters are shaped by the
truncation. This is the generalization of the Aban-Meerschaert-Panorska truncated-Pareto MLE
to a covariate scale model + family sweep.

The Beirlant risk (flagged in the design): under hard truncation the tail index can be weakly
identified — the likelihood is near-flat in the shape parameter — so a fit can "converge" to a
shape with a huge standard error. This driver therefore records shape ± SE (from the BFGS
inverse Hessian) so identifiability is visible, not inferred from the convergence flag.

Prediction (the renormalized/truncated mean E[W | W <= H]) is NOT here — each family computes
it from its censored-mean helper as (E[min(W,H)] - H*S(H)) / (1 - S(H)); see gpd_trunc /
log_logistic_trunc.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy import optimize

from idd_tc_mortality.distributions.base import FitResult


def fit_truncated(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    excess_cap: np.ndarray,
    *,
    base_fit: Callable[..., FitResult],
    neg_ll: Callable,
    x0_from_base: Callable[[FitResult], np.ndarray],
    shape_transform: Callable[[float, float], tuple[float, float]],
    family: str,
) -> FitResult:
    """Weighted truncated-MLE fit, warm-started from the untruncated base fit.

    Parameters
    ----------
    X, y, weights:
        Design matrix, excess rates (>0), per-obs weights (as for the base family).
    excess_cap:
        Per-storm excess cap H_i = c_i - threshold_rate (each storm's own cap, > 0).
    base_fit:
        The untruncated family fit (gpd.fit / log_logistic.fit) used for the warm start.
    neg_ll:
        Negative weighted truncated log-likelihood WITH analytic gradient:
        ``neg_ll(params, X_arr, y, H, w_norm, n_beta) -> (float, grad_array)``.
        params = [beta..., shape_raw]. Used with scipy ``jac=True``.
    x0_from_base:
        Build the optimizer start vector from the base FitResult (append the shape on its
        optimization scale, e.g. xi or log_k).
    shape_transform:
        ``(shape_raw, var_raw) -> (shape_natural, shape_se_natural)`` — converts the last
        parameter and its inverse-Hessian variance to the reported shape ± SE (delta method
        where the optimizer works on a transformed scale, e.g. log_k).
    family:
        Registry family name to stamp on the FitResult.
    """
    y = np.asarray(y, dtype=float)
    H = np.asarray(excess_cap, dtype=float)
    if H.shape != y.shape:
        raise ValueError(f"excess_cap length {H.shape} != y length {y.shape}.")
    if np.any(H <= 0):
        raise ValueError(
            f"{family}.fit requires strictly positive excess caps (H_i = c_i - u > 0) "
            "for every fitted tail storm."
        )

    X_arr = np.asarray(X, dtype=float)
    n_beta = X_arr.shape[1]
    w = np.asarray(weights, dtype=float)
    w_mean = float(np.mean(w))
    w_norm = w / w_mean  # mean-1 for numerical conditioning (as in the base gpd/ll fits)

    base = base_fit(X, y, weights)          # untruncated warm start
    x0 = np.asarray(x0_from_base(base), dtype=float)

    caught: list[str] = []
    with warnings.catch_warnings(record=True) as cw:
        warnings.simplefilter("always")
        opt = optimize.minimize(
            neg_ll, x0, args=(X_arr, y, H, w_norm, n_beta),
            method="BFGS", jac=True, options={"maxiter": 2000, "disp": False},
        )
    caught = [str(m.message) for m in cw]

    beta = opt.x[:n_beta]
    shape_raw = float(opt.x[n_beta])
    # SE for the IDENTIFIABILITY read (the Beirlant question) uses the mean-1-weight inverse
    # Hessian directly: normalizing the exposure weights to mean 1 makes the total weight = n
    # tail storms, so this SE reflects the information the n storms carry about the shape. We
    # deliberately do NOT rescale by w_mean (the exposure magnitude ~1e6) — that would give the
    # asymptotic weighted-MLE covariance, which is dominated by the arbitrary exposure scale and
    # deflates the SE ~1000x, masking a flat likelihood as "tightly identified".
    hess_inv = np.asarray(opt.hess_inv)  # mean-1-weight scale (total weight = n storms)
    var_raw = float(hess_inv[n_beta, n_beta])
    shape, shape_se = shape_transform(shape_raw, var_raw)

    return FitResult(
        params=beta,
        param_names=list(X.columns),
        fitted_values=np.full(len(y), np.nan),  # not needed for the spike
        family=family,
        converged=bool(opt.success),
        meta={
            "shape_param": shape,
            "shape_se": shape_se,          # <-- identifiability signal (Beirlant risk)
            "shape_raw": shape_raw,
            "hess_inv": hess_inv,
            "n_obs": int(len(y)),
            "iterations": int(opt.nit),
            "opt_message": str(opt.message),
            "warnings": caught,
        },
    )
