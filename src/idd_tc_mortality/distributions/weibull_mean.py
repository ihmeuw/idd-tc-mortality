"""Weibull tail — MEAN variant (bounded/finite-mean selection, variant D).

Fit is IDENTICAL to the standard Weibull tail model (:mod:`weibull`): weighted
BFGS MLE on the excess rate ``w = death_rate - threshold_rate`` with a shared
scalar shape ``k`` and log-linear scale ``log(lambda_i) = X_i @ beta``. The fit
function is re-exported unchanged so the two families share one implementation.

What differs is :func:`predict`. The standard Weibull tail reports the MEDIAN
``lambda_i * (ln 2)^(1/k)``; this variant reports the analytic Weibull MEAN

    E[W] = lambda_i * Gamma(1 + 1/k) = exp(X_i @ beta) * Gamma(1 + 1/k)

on the excess scale. ``predict_component`` then adds ``threshold_rate`` back
(``tail_outcome='excess'``) to recover the full-rate mean.

Variant D is the untruncated-mean baseline: it measures how much the bounding
correction (variants C/B) buys relative to simply reporting an honest mean.
Weibull's mean is finite for every ``k > 0`` (``Gamma(1 + 1/k) < inf``), so unlike
GPD / log-logistic it needs no physical cap — it is the finite-mean *check*
family. No ``needs_cap`` flag; predict takes the standard ``(result, X)`` signature.
"""

from __future__ import annotations

from math import gamma as _gamma

import numpy as np
import pandas as pd

from idd_tc_mortality.distributions.base import FitResult
from idd_tc_mortality.distributions.weibull import fit  # noqa: F401 — re-exported: identical fit


def predict(result: FitResult, X: pd.DataFrame) -> np.ndarray:
    """Predict Weibull MEAN excess rates from a fitted Weibull model.

    Returns ``exp(X @ beta) * Gamma(1 + 1/k)`` — the analytic Weibull mean using
    the fitted shape ``k`` (from ``result.meta['shape_param']``), computed on the
    excess scale. Always finite and positive.

    Parameters
    ----------
    result:
        FitResult from :func:`weibull.fit` (re-exported here as ``fit``).
    X:
        Design matrix; columns must match ``result.param_names`` exactly
        (call ``features.align_X`` first).
    """
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names "
            f"{result.param_names}. Call features.align_X before predicting."
        )
    k = result.meta["shape_param"]
    lam = np.exp(np.asarray(X, dtype=float) @ result.params)  # scale lambda_i = exp(X @ beta)
    return lam * _gamma(1.0 + 1.0 / k)
