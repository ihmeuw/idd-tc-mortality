"""
Component fitting: fit_one_component(spec, df).

Maps a component spec dict (from enumerate_component_specs) to a fitted FitResult
by subsetting the data, constructing the outcome on the correct scale, building the
design matrix, and calling the appropriate fit function.

This is a pure function — no CLI, no caching, no I/O side effects.

Component types and their data subsets
---------------------------------------
s1:   Full dataset. y = (deaths >= 1). Offset: log(exposed). X excludes log_exposed.
s2:   deaths >= 1 rows. y = (death_rate >= threshold_rate). Offset: log(exposed).
      X excludes log_exposed. threshold_rate stored in FitResult.meta.
bulk: deaths >= 1 AND death_rate < threshold_rate rows.
      Rate models: y = death_rate; X includes log_exposed; weights = exposed.
      Count models (nb/poisson): y = deaths; X excludes log_exposed; offset = log(exposed).
      Special bulk cases: beta uses y = death_rate/threshold_rate; scaled_logit passes
      threshold_rate as extra argument.
tail: death_rate >= threshold_rate rows.
      Rate models: y = death_rate - threshold_rate (excess); X includes log_exposed; weights = exposed.
      Count models (nb/poisson): y = deaths; X excludes log_exposed; offset = log(exposed).

threshold_rate is the computed rate-scale value (e.g. 2.1e-5 deaths/person). It is
distinct from threshold_quantile (e.g. 0.75), which is the quantile level stored in the
spec dict for identification purposes. If spec["threshold_rate"] is already populated
(non-None), it is used directly; otherwise it is computed from the data and threshold_quantile.

The log_exposed flag in the distributions registry distinguishes offset models (nb, poisson)
from free-covariate models (gamma, lognormal, beta, scaled_logit, gpd).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idd_tc_mortality.distributions import get_family
from idd_tc_mortality.distributions.base import FitResult
from idd_tc_mortality.features import build_X
from idd_tc_mortality.thresholds import compute_thresholds
from idd_tc_mortality import s1 as s1_mod
from idd_tc_mortality import s2 as s2_mod


def _compute_threshold_rate(death_rate: np.ndarray, quantile: float) -> float:
    """Return the rate-scale threshold at a single quantile level from positive death rates."""
    return compute_thresholds(death_rate, quantile_levels=np.array([quantile]))[quantile]


def fit_one_component(spec: dict, df: pd.DataFrame) -> FitResult:
    """Fit one model component from a spec dict and the full training DataFrame.

    Parameters
    ----------
    spec:
        Component spec dict as returned by enumerate_component_specs. Keys:
        'component' ('s1', 's2', 'bulk', or 'tail'),
        'covariate_combo' (bool flag dict),
        'threshold_quantile' (float quantile level, or None for s1),
        'threshold_rate' (float rate-scale threshold, or None — computed from data if None),
        'family' (family name string, or None for s1/s2).
    df:
        Full cleaned training DataFrame. Must contain columns: 'deaths',
        'exposed', and any covariates flagged True in spec['covariate_combo'].

    Returns
    -------
    FitResult
        Fitted result from the appropriate distribution module. If fitting raises
        an exception, returns a non-converged FitResult with NaN params and the
        error message stored in meta["fit_error"].

    Raises
    ------
    ValueError
        If spec['component'] is not one of 's1', 's2', 'bulk', 'tail'.
    ValueError
        If compute_thresholds finds no positive death rates (no events in data).
    """
    component = spec["component"]
    if component not in ("s1", "s2", "bulk", "tail"):
        raise ValueError(
            f"Unknown component type: {component!r}. "
            "Expected one of 's1', 's2', 'bulk', 'tail'."
        )

    try:
        return _fit_one_component(spec, df)
    except Exception as exc:
        family = spec.get("family") or component
        return FitResult(
            params=np.array([np.nan]),
            param_names=["__failed__"],
            fitted_values=np.array([np.nan]),
            family=family,
            converged=False,
            meta={"fit_error": str(exc)},
        )


def _fit_one_component(spec: dict, df: pd.DataFrame) -> FitResult:
    component = spec["component"]
    covariates = spec["covariate_combo"]

    # Compute death_rate from first principles to ensure consistency.
    death_rate = df["deaths"].values / df["exposed"].values

    if component == "s1":
        family       = spec["family"]
        exposure_mode = spec["exposure_mode"]
        include_log_exposed = (exposure_mode == "free")
        X = build_X(df, covariates, include_log_exposed=include_log_exposed)
        y = (df["deaths"].values >= 1).astype(float)
        log_exposed = np.log(df["exposed"].values)
        return s1_mod.fit(X, y, log_exposed, family=family, exposure_mode=exposure_mode)

    # S2, bulk, tail all require a threshold_rate.
    # Use the pre-computed value from the spec when available; compute from data otherwise.
    threshold_rate: float = spec.get("threshold_rate") or _compute_threshold_rate(
        death_rate, spec["threshold_quantile"]
    )

    if component == "s2":
        exposure_mode = spec["exposure_mode"]
        include_log_exposed = (exposure_mode == "free")
        mask = df["deaths"].values >= 1
        df_sub = df[mask].reset_index(drop=True)
        X = build_X(df_sub, covariates, include_log_exposed=include_log_exposed)
        y = (death_rate[mask] >= threshold_rate).astype(float)
        return s2_mod.fit(X, y, spec["family"], threshold_rate)

    if component not in ("bulk", "tail"):
        raise ValueError(
            f"Unknown component type: {component!r}. "
            "Expected one of 's1', 's2', 'bulk', 'tail'."
        )

    # Bulk and tail: look up family and determine design matrix and weight construction.
    family_info = get_family(spec["family"])
    uses_offset = family_info["log_exposed"]  # True only for count families (nb, poisson)

    # exposure_mode governs how log(exposed) enters rate-scale models.
    # Count families always use 'offset' regardless of this field.
    # Default 'free+weight' preserves the original behaviour (log_exposed in X + exposed weights).
    exposure_mode = spec.get("exposure_mode", "free+weight")

    if uses_offset:
        # Count model: log_exposed is a fixed offset argument, never in X.
        include_log_exposed_in_X = False
    else:
        # Rate model: include log_exposed in X only for 'free' and 'free+weight' modes.
        include_log_exposed_in_X = exposure_mode in ("free", "free+weight")

    if component == "bulk":
        mask = (df["deaths"].values >= 1) & (death_rate < threshold_rate)
        df_sub = df[mask].reset_index(drop=True)
        X = build_X(df_sub, covariates, include_log_exposed=include_log_exposed_in_X)

        if uses_offset:
            y = df_sub["deaths"].values.astype(float)
            log_exposed = np.log(df_sub["exposed"].values)
            return family_info["fit"](X, y, log_exposed)

        y_rate = death_rate[mask]
        family_name = spec["family"]
        # 'weight' and 'free+weight' use exposed as IRLS var_weights; others use uniform.
        weights = (
            df_sub["exposed"].values
            if exposure_mode in ("weight", "free+weight")
            else np.ones(len(df_sub))
        )

        if family_name == "beta":
            # beta.fit expects y in (0, 1): normalise by threshold_rate.
            # beta.fit signature is (X, y, weights) — threshold_rate is NOT passed.
            return family_info["fit"](X, y_rate / threshold_rate, weights)
        elif family_name == "scaled_logit":
            # scaled_logit.fit expects y in (0, threshold_rate) and needs threshold_rate explicitly
            return family_info["fit"](X, y_rate, weights, threshold_rate)
        elif family_name == "truncated_normal":
            # truncated_normal fits on raw log(death_rate) with upper truncation at log(threshold).
            return family_info["fit"](X, y_rate, weights, threshold_rate, "bulk")
        else:
            return family_info["fit"](X, y_rate, weights)

    if component == "tail":
        mask = death_rate >= threshold_rate
        df_sub = df[mask].reset_index(drop=True)
        X = build_X(df_sub, covariates, include_log_exposed=include_log_exposed_in_X)

        if uses_offset:
            y = df_sub["deaths"].values.astype(float)
            log_exposed = np.log(df_sub["exposed"].values)
            return family_info["fit"](X, y, log_exposed)

        family_name = spec["family"]
        weights = (
            df_sub["exposed"].values
            if exposure_mode in ("weight", "free+weight")
            else np.ones(len(df_sub))
        )

        # Routing contract for tail rate families:
        # ─ truncated_normal is the ONLY family that fits on raw log(death_rate).
        #   It takes (X, y_raw, weights, threshold_rate, "tail") and stores
        #   threshold_rate in meta. predict_component does NOT add threshold_rate
        #   back (no tail_outcome:"excess" flag in the registry).
        # ─ Every other rate-scale tail family fits on excess_rate = death_rate - threshold.
        #   They take (X, y_excess, weights) and return excess-rate predictions.
        #   predict_component adds threshold_rate back via the tail_outcome:"excess" flag.
        # When adding a new tail distribution: if it fits on excess_rate, add
        #   tail_outcome:"excess" to its registry entry and do nothing here — the
        #   else-branch below handles it. Only touch this name check if you are
        #   deliberately adding another raw-rate tail family (rare).
        if family_name == "truncated_normal":
            y_raw = death_rate[mask]
            return family_info["fit"](X, y_raw, weights, threshold_rate, "tail")

        y_excess = death_rate[mask] - threshold_rate
        # Clip to a positive floor: np.quantile can land exactly on a data point,
        # producing y_excess == 0 for the boundary row. Gamma / lognormal require
        # strictly positive y. Use half the minimum positive excess as the floor so
        # the clipped value is on the same order of magnitude as the rest of the data
        # (vs. a machine-epsilon floor, which would be a 14-order-of-magnitude outlier
        # and destabilise the IRLS regardless of the clip).
        _pos = y_excess[y_excess > 0]
        _floor = _pos.min() / 2 if len(_pos) > 0 else threshold_rate * 1e-6
        y_excess = np.maximum(y_excess, _floor)
        return family_info["fit"](X, y_excess, weights)

    # Unreachable: guard above already raises for unknown component types.
    raise AssertionError(f"Unreachable: component={component!r}")
