"""
Component prediction: predict_one_component(spec, result, df).

Takes a fitted component spec dict, a FitResult, and the full DataFrame. Returns a
pd.Series of rate-scale or probability-scale predictions for **every row of df**:

    s1   — P(deaths >= 1) for every row
    s2   — P(rate >= threshold | deaths >= 1) for every row (model evaluated at row covariates)
    bulk — E[rate | deaths >= 1, rate < threshold] for every row
    tail — E[rate | rate >= threshold] for every row (>= threshold_rate when family fits on excess)

Predictions are made for every row in df, not just the row's "applicable subset" at fit
time. The fitted model is a function of covariates and applies to any row with valid
covariates; subsetting (for in-stage metrics that compare to observed outcomes on the
applicable subset) happens at the call site, not here. Predicting on all rows is what
the unconditional double-hurdle assembly E[rate] = p_s1 * (p_s2*rate_tail + (1-p_s2)*rate_bulk)
requires.

All predictions are on the rate or probability scale as appropriate for the component.
Count-scale families (nb, poisson) return rates — the log_exposed offset cancels in
their predict implementations.

Post-processing applied after the registry predict call:
    beta  (bulk) — multiply by threshold_rate: beta.predict returns (0, 1), not rate scale.
    gpd   (tail) — add threshold_rate: gpd.predict returns mean excess rate, not full rate.

align_X is always called before predicting to handle column mismatches (e.g. basin
levels absent from prediction data that were present at fit time).

threshold_rate is read from spec["threshold_rate"] when non-None (set by orchestrate.py),
and computed from the data otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idd_tc_mortality.distributions import get_family
from idd_tc_mortality.distributions.base import FitResult
from idd_tc_mortality.features import align_X, build_X
from idd_tc_mortality.thresholds import compute_thresholds
from idd_tc_mortality import s1 as s1_mod
from idd_tc_mortality import s2 as s2_mod


def _get_threshold_rate(spec: dict, death_rate: np.ndarray) -> float:
    """Return threshold_rate from spec when available, else compute from data."""
    if spec.get("threshold_rate") is not None:
        return float(spec["threshold_rate"])
    return compute_thresholds(
        death_rate, quantile_levels=np.array([spec["threshold_quantile"]])
    )[spec["threshold_quantile"]]


def predict_one_component(
    spec: dict,
    result: FitResult,
    df: pd.DataFrame,
) -> pd.Series:
    """Predict one model component for the applicable rows of df.

    Parameters
    ----------
    spec:
        Component spec dict (from enumerate_component_specs or manifest).
        Must contain 'component', 'covariate_combo', 'threshold_quantile',
        'threshold_rate' (or None), and 'family' (or None for s1/s2).
    result:
        Fitted FitResult for this component.
    df:
        Full DataFrame. Must contain 'deaths', 'exposed', and the covariates
        in spec['covariate_combo']. Index is preserved in the returned Series.

    Returns
    -------
    pd.Series
        Rate-scale (or probability-scale for s1/s2) predictions for every row of df.
        Length == len(df), index == df.index.

    Raises
    ------
    ValueError
        If spec['component'] is not one of 's1', 's2', 'bulk', 'tail'.
    """
    component = spec["component"]
    covariates = spec["covariate_combo"]

    if component not in ("s1", "s2", "bulk", "tail"):
        raise ValueError(
            f"Unknown component type: {component!r}. "
            "Expected one of 's1', 's2', 'bulk', 'tail'."
        )

    # threshold_rate is needed for bulk/tail post-processing (beta multiply, excess add).
    threshold_rate: float | None = None
    if component in ("bulk", "tail"):
        death_rate = df["deaths"].values / df["exposed"].values
        threshold_rate = _get_threshold_rate(spec, death_rate)

    # ---- Build design matrix and align to fitted param_names ----------------
    if component in ("s1", "s2"):
        include_log_exposed = (spec["exposure_mode"] == "free")
    else:
        family_info = get_family(spec["family"])
        if family_info["log_exposed"]:
            # Count model (nb, poisson): log_exposed is an offset, never in X.
            include_log_exposed = False
        else:
            # Rate model: determined by exposure_mode.
            # Default 'free+weight' matches original behaviour (log_exposed in X).
            exposure_mode = spec.get("exposure_mode", "free+weight")
            include_log_exposed = exposure_mode in ("free", "free+weight")

    X = build_X(df, covariates, include_log_exposed=include_log_exposed)
    X = align_X(X, result.param_names)

    log_exposed = np.log(df["exposed"].values)

    # ---- Call the appropriate predict function --------------------------------
    if component == "s1":
        preds = s1_mod.predict(result, X, log_exposed)

    elif component == "s2":
        preds = s2_mod.predict(result, X)

    else:
        family_info = get_family(spec["family"])
        pred_fn = family_info["predict"]

        if family_info["log_exposed"]:
            preds = pred_fn(result, X, log_exposed)
        else:
            preds = pred_fn(result, X)

        # Post-process to rate scale where needed.
        if spec["family"] == "beta":
            # beta.predict returns values in (0, 1); multiply to recover rate scale.
            preds = preds * threshold_rate
        elif component == "tail" and family_info.get("tail_outcome") == "excess":
            # Distribution was fit on excess rate (death_rate - threshold_rate).
            # predict() returns excess rates; add threshold_rate to recover full rate.
            # Applies to: gamma, lognormal, gpd, and any future excess-rate tail family.
            preds = preds + threshold_rate

    return pd.Series(preds, index=df.index, name=component)
