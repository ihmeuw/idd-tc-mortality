"""
Component prediction: predict_one_component(spec, result, df).

Takes a fitted component spec dict, a FitResult, and the full DataFrame. Returns a
pd.Series of rate-scale predictions indexed to the rows of df that this component
applies to:

    s1   — all rows (probabilities, in [0, 1])
    s2   — deaths >= 1 rows (probabilities, in [0, 1])
    bulk — deaths >= 1 AND death_rate < threshold_rate rows (rates, >= 0)
    tail — death_rate >= threshold_rate rows (rates, >= threshold_rate for gpd)

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
        Rate-scale (or probability-scale for s1/s2) predictions, indexed to
        the applicable subset of df. Length is strictly <= len(df).

        s1:   length == len(df),  index == df.index
        s2:   index == df.index[deaths >= 1]
        bulk: index == df.index[bulk_mask]
        tail: index == df.index[tail_mask]

    Raises
    ------
    ValueError
        If spec['component'] is not one of 's1', 's2', 'bulk', 'tail'.
    """
    component = spec["component"]
    covariates = spec["covariate_combo"]

    death_rate = df["deaths"].values / df["exposed"].values

    # ---- Determine applicable subset ----------------------------------------
    if component == "s1":
        mask = np.ones(len(df), dtype=bool)
    elif component == "s2":
        mask = df["deaths"].values >= 1
    elif component in ("bulk", "tail"):
        threshold_rate = _get_threshold_rate(spec, death_rate)
        if component == "bulk":
            mask = (df["deaths"].values >= 1) & (death_rate < threshold_rate)
        else:
            mask = death_rate >= threshold_rate
    else:
        raise ValueError(
            f"Unknown component type: {component!r}. "
            "Expected one of 's1', 's2', 'bulk', 'tail'."
        )

    sub_index = df.index[mask]
    df_sub = df.loc[sub_index]

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

    X = build_X(df_sub, covariates, include_log_exposed=include_log_exposed)
    X = align_X(X, result.param_names)

    log_exposed = np.log(df_sub["exposed"].values)

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

    return pd.Series(preds, index=sub_index, name=component)
