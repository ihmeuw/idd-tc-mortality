"""
S1 component: P(deaths >= 1 | covariates, exposed).

Three supported (family, exposure_mode) combinations:
  cloglog + offset  — theoretically motivated: P(deaths=0|N,p) = exp(-N*p), so
                      log hazard scales linearly with log(N). Offset coeff fixed at 1.
  cloglog + free    — cloglog link, log(exposed) as free covariate in X (coeff estimated).
  cloglog + excluded — cloglog link, no exposure information.
  logit   + free    — logit link, log(exposed) as free covariate in X.
  logit   + excluded — logit link, no exposure information.

family and exposure_mode are stored in meta["link"] and meta["exposure_mode"] so
predict() can apply the correct inverse link and eta construction.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import CLogLog, Logit

from idd_tc_mortality.distributions.base import FitResult
from idd_tc_mortality.distributions.binomial_cloglog import (
    fit_binomial_cloglog,
    predict_binomial_cloglog,
)

_VALID_FAMILIES       = {"logit", "cloglog"}
_VALID_EXPOSURE_MODES = {"offset", "free", "excluded"}


def fit(
    X: pd.DataFrame,
    y: np.ndarray,
    log_exposed: np.ndarray,
    family: str = "cloglog",
    exposure_mode: str = "offset",
) -> FitResult:
    """Fit S1: P(deaths >= 1).

    Parameters
    ----------
    X:
        Design matrix for the full dataset. Must include a 'const' column.
        For exposure_mode='free', must include a 'log_exposed' column.
        For exposure_mode='offset' or 'excluded', must NOT include 'log_exposed'.
    y:
        Binary array: 1 if deaths >= 1, 0 otherwise. Length n_obs.
    log_exposed:
        log(exposed) for each row. Used as offset when exposure_mode='offset';
        otherwise accepted for API consistency and ignored.
    family:
        Link function: 'cloglog' or 'logit'. Only 'cloglog' supports 'offset' mode.
    exposure_mode:
        How log(exposed) enters the model: 'offset', 'free', or 'excluded'.

    Returns
    -------
    FitResult with family='s1', meta['link']=family, meta['exposure_mode']=exposure_mode.
    """
    if family not in _VALID_FAMILIES:
        raise ValueError(f"family must be one of {_VALID_FAMILIES}. Got {family!r}.")
    if exposure_mode not in _VALID_EXPOSURE_MODES:
        raise ValueError(f"exposure_mode must be one of {_VALID_EXPOSURE_MODES}. Got {exposure_mode!r}.")
    if exposure_mode == "offset" and family != "cloglog":
        raise ValueError("exposure_mode='offset' is only valid for family='cloglog'.")

    if exposure_mode == "offset":
        result = fit_binomial_cloglog(X, y, log_exposed, family_name="s1")
    else:
        link = CLogLog() if family == "cloglog" else Logit()
        glm = sm.GLM(endog=y, exog=X, family=Binomial(link=link))
        converged = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            glm_result = glm.fit()
            if not glm_result.converged:
                converged = False
        n_iter = glm_result.fit_history.get("iteration", None) if hasattr(glm_result, "fit_history") else None
        result = FitResult(
            params=np.asarray(glm_result.params),
            param_names=list(X.columns),
            fitted_values=np.asarray(glm_result.fittedvalues),
            family="s1",
            converged=converged,
            cov=None,
            meta={
                "n_obs":     int(len(y)),
                "n_events":  int(y.sum()),
                "iterations": n_iter,
                "warnings":  [str(w.message) for w in caught],
            },
        )

    result.meta["link"]          = family
    result.meta["exposure_mode"] = exposure_mode
    return result


def predict(
    result: FitResult,
    X: pd.DataFrame,
    log_exposed: np.ndarray,
) -> np.ndarray:
    """Predict P(deaths >= 1) for new observations.

    Parameters
    ----------
    result:
        FitResult from s1.fit().
    X:
        Design matrix. Columns must match result.param_names.
        Use features.align_X before calling.
    log_exposed:
        log(exposed) for the prediction rows. Used only when
        meta['exposure_mode'] == 'offset'; otherwise ignored.

    Returns
    -------
    np.ndarray
        Predicted probabilities in (0, 1], length n_obs.
    """
    exposure_mode = result.meta.get("exposure_mode", "offset")

    if exposure_mode == "offset":
        return predict_binomial_cloglog(result, X, log_exposed)

    # free or excluded: eta = X @ params only (no offset addition)
    if list(X.columns) != result.param_names:
        raise ValueError(
            f"X columns {list(X.columns)} do not match fitted param_names "
            f"{result.param_names}. Call features.align_X before predicting."
        )
    eta = np.asarray(X) @ result.params
    link = result.meta.get("link", "cloglog")
    if link == "cloglog":
        return 1.0 - np.exp(-np.exp(eta))
    return 1.0 / (1.0 + np.exp(-eta))
