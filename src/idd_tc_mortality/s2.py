"""
S2 component: P(death_rate >= threshold | deaths >= 1, covariates, exposed).

Fit: Binary GLM (logit or cloglog link). log(exposed) is a free covariate in X,
not a constrained offset. The cloglog + fixed-offset derivation that motivates S1
does not carry over to S2 cleanly: S2 is routing between bulk and tail conditional
on deaths already having occurred, and there is no first-principles reason the
log(N) coefficient should be 1. Freeing it lets the data determine whether exposure
matters for routing, and the OOS comparison is the right arbiter between link choices.

The family ("logit" or "cloglog") is specified by the caller and stored in
FitResult.family and meta["link"] so predict() can apply the correct inverse link.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import CLogLog, Logit

from idd_tc_mortality.distributions.base import FitResult


def fit(
    X: pd.DataFrame,
    y: np.ndarray,
    family: str,
    threshold: float,
) -> FitResult:
    """Fit S2: P(death_rate >= threshold | deaths >= 1).

    Parameters
    ----------
    X:
        Design matrix for the deaths >= 1 subset. Must include a 'const' column
        and a 'log_exposed' column (log_exposed is a free covariate, not an offset).
    y:
        Binary array: 1 if death_rate >= threshold, 0 otherwise. Length n_obs.
    family:
        Link function to use: "logit" or "cloglog".
    threshold:
        Rate-scale threshold used by the caller to construct y. Stored in meta
        for provenance only — not used in computation.

    Returns
    -------
    FitResult with family set to the link name ("logit" or "cloglog") and
    meta["threshold_rate"] set to the supplied threshold value.
    """
    _validate_inputs(X, y, family)

    link = CLogLog() if family == "cloglog" else Logit()
    glm = sm.GLM(endog=y, exog=X, family=Binomial(link=link))

    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = glm.fit()
        if not result.converged:
            converged = False

    n_iter = result.fit_history.get("iteration", None) if hasattr(result, "fit_history") else None

    fit_result = FitResult(
        params=np.asarray(result.params),
        param_names=list(X.columns),
        fitted_values=np.asarray(result.fittedvalues),
        family=family,
        converged=converged,
        cov=None,
        meta={
            "n_obs": int(len(y)),
            "n_events": int(y.sum()),
            "link": family,
            "threshold_rate": threshold,
            "iterations": n_iter,
            "warnings": [str(w.message) for w in caught],
        },
    )
    return fit_result


def predict(
    result: FitResult,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict P(death_rate >= threshold) for new observations.

    Parameters
    ----------
    result:
        FitResult from s2.fit(). meta["link"] must be set to "logit" or "cloglog".
    X:
        Design matrix. Columns must match result.param_names (use features.align_X).
        Must include a 'log_exposed' column — it is a free covariate, not passed
        separately.

    Returns
    -------
    np.ndarray
        Predicted probabilities in (0, 1], length n_obs.
    """
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names "
            f"{result.param_names}. Call features.align_X before predicting."
        )
    eta = np.asarray(X) @ result.params
    link = result.meta.get("link", "cloglog")
    if link == "cloglog":
        return 1.0 - np.exp(-np.exp(eta))
    else:
        return 1.0 / (1.0 + np.exp(-eta))


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    X: pd.DataFrame,
    y: np.ndarray,
    family: str,
) -> None:
    if family not in ("logit", "cloglog"):
        raise ValueError(
            f"family must be 'logit' or 'cloglog'. Got {family!r}."
        )

    unique_y = set(np.unique(y))
    if not unique_y.issubset({0, 1}):
        raise ValueError(f"y must be binary (0/1). Found values: {unique_y}.")

    if len(y) != len(X):
        raise ValueError(f"y length {len(y)} != X rows {len(X)}.")
