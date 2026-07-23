"""GPD tail — CENSORED-at-cap MEAN variant (bounded/finite-mean selection, variant C).

The fit is IDENTICAL to the standard GPD tail model (:mod:`gpd`): untruncated
weighted MLE on the excess rate ``w = death_rate - threshold_rate``. This is
correct for variant C ("censoring at the cap"): the interior density is the
untruncated family and the mass above the cap becomes an atom at ``c_i``. Because
no in-sample storm sits at its physical cap (``max death_rate << c_i``), the atom
is empty in-sample and the censored-likelihood fit equals the untruncated fit.
The fit function is therefore re-exported from :mod:`gpd` unchanged.

What differs is :func:`predict`. Instead of the GPD median, it returns the
CENSORED mean of the excess

    E[min(W, H_i)],   W ~ GPD(sigma_i, xi),   H_i = c_i - threshold_rate,

where ``sigma_i = exp(X_i @ beta)``, ``xi = result.meta['shape_param']`` and the
per-row excess cap ``H_i`` is supplied by ``predict_component`` (``needs_cap=True``
in the registry). ``predict_component`` then adds ``threshold_rate`` back
(``tail_outcome='excess'``), so the full-rate central estimate is
``E[min(rate, c_i)]``.

The censored mean is FINITE for any tail heaviness — including ``xi >= 1`` where
the ordinary GPD mean is infinite — which is the entire point of variant C: an
honest, physically bounded expected tail rate that makes assembled ``E[rate]`` a
true expectation.

Closed form (E[min(W,H)] = integral_0^H S(w) dw, S the GPD survival):

    xi == 0 :  sigma * (1 - exp(-H/sigma))                              (exponential limit)
    xi == 1 :  sigma * log(1 + H/sigma)                                 (removable singularity)
    else    :  (sigma / (1 - xi)) * (1 - (1 + xi*H/sigma)^(1 - 1/xi))

For ``xi < 0`` the GPD has finite support ``sigma/|xi|``; ``H`` is clamped to that
support (beyond it ``min(W,H) = W`` almost surely, so the censored mean equals
the full mean ``sigma/(1-xi)``). ``H <= 0`` (cap at or below threshold — the tail
event is impossible) yields ``0`` excess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idd_tc_mortality.distributions.base import FitResult
from idd_tc_mortality.distributions.gpd import fit  # noqa: F401 — re-exported: identical untruncated fit

_XI_EPS = 1e-6   # |xi| below this uses the exponential limit
_XI1_EPS = 1e-6  # |xi - 1| below this uses the log limit


def predict(result: FitResult, X: pd.DataFrame, excess_cap: np.ndarray) -> np.ndarray:
    """Predict GPD CENSORED-mean excess rates.

    Parameters
    ----------
    result:
        FitResult from :func:`gpd.fit` (re-exported here as ``fit``).
    X:
        Design matrix; columns must match ``result.param_names`` exactly
        (call ``features.align_X`` first).
    excess_cap:
        Per-row excess-scale cap ``H_i = c_i - threshold_rate``, length == len(X).
        Supplied by ``predict_component`` via ``tail_cap.excess_cap``.

    Returns
    -------
    np.ndarray
        Censored excess-rate means ``E[min(W, H_i)]``, length n_obs. Finite and
        non-negative for every row and every shape ``xi``.
    """
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names "
            f"{result.param_names}. Call features.align_X before predicting."
        )
    xi = float(result.meta["shape_param"])
    sigma = np.exp(np.asarray(X, dtype=float) @ result.params)
    H = np.asarray(excess_cap, dtype=float)
    if H.shape != sigma.shape:
        raise ValueError(
            f"excess_cap length {H.shape} does not match X rows {sigma.shape}."
        )
    return gpd_censored_mean(sigma, xi, H)


def gpd_censored_mean(sigma: np.ndarray, xi: float, H: np.ndarray) -> np.ndarray:
    """E[min(W, H)] for W ~ GPD(sigma, xi), elementwise over sigma and H.

    ``sigma`` and ``H`` are arrays of equal length; ``xi`` is a shared scalar.
    Always finite and non-negative. See the module docstring for the branches.
    """
    sigma = np.asarray(sigma, dtype=float)
    H = np.maximum(np.asarray(H, dtype=float), 0.0)  # cap at/below threshold -> 0 excess

    if abs(xi) < _XI_EPS:
        # Exponential limit: integral_0^H exp(-w/sigma) dw
        return sigma * (1.0 - np.exp(-H / sigma))

    if abs(xi - 1.0) < _XI1_EPS:
        # xi == 1 removable singularity: integral_0^H (1 + w/sigma)^{-1} dw
        return sigma * np.log1p(H / sigma)

    if xi < 0.0:
        # Finite upper support sigma/|xi|; beyond it min(W,H)=W a.s.
        H = np.minimum(H, -sigma / xi)

    base = np.maximum(1.0 + xi * H / sigma, 0.0)  # guard fp negatives at the support edge
    return (sigma / (1.0 - xi)) * (1.0 - base ** (1.0 - 1.0 / xi))
