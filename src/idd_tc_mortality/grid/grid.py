"""
Component spec enumeration for preliminary and refined grid fits.

enumerate_component_specs(mode, ...) returns a flat list of dicts, one per
component fit to run. Each dict uniquely identifies a component and is passed
as the spec argument to cache.component_id / cache.save_result.

Preliminary mode uses fixed defaults for covariate sets, thresholds, and family
lists. Refined mode requires all four parameters explicitly — no defaults exist
because the refined grid is defined by what survived the preliminary cull.

Spec dict keys:
    component          — 's1', 's2', 'bulk', or 'tail'
    covariate_combo    — bool flag dict (wind_speed, sdi, basin, is_island)
    threshold_quantile — float quantile level, or None for S1
    family             — family name string, or None for S1/S2
    exposure_mode      — how log(exposed) enters the model (bulk/tail only):
                         'free'       log_exposed as free covariate in X, weights=1
                         'weight'     exposed as IRLS var_weight, log_exposed not in X
                         'free+weight' log_exposed in X AND exposed as var_weight (default)
                         'excluded'   no exposure information in the model
                         'offset'     log_exposed as fixed offset (count families only)
                         S1/S2 specs do not have this key.
    fold_tag           — 'is' for in-sample; 's{seed}_f{fold}' for OOS folds

All values are JSON-serializable (Python floats, not numpy floats).

Exposure mode notes:
  - COUNT_FAMILIES (nb, poisson) always use 'offset'; other modes are not generated.
  - Rate-scale families get one spec per exposure mode in RATE_EXPOSURE_MODES.
  - S1 always uses log_exposed as a fixed offset (not configurable).
  - S2 always uses log_exposed as a free covariate (not configurable; offset variants
    were evaluated and dropped from the grid).
"""

from __future__ import annotations

import random

from idd_tc_mortality.constants import QUANTILE_LEVELS


# ---------------------------------------------------------------------------
# Covariate sets
# ---------------------------------------------------------------------------

PRELIMINARY_COVARIATE_SETS: list[dict[str, bool]] = [
    {"wind_speed": False, "sdi": False, "basin": False, "is_island": False},  # intercept only
    {"wind_speed": True,  "sdi": True,  "basin": False, "is_island": False},  # wind + SDI
    {"wind_speed": True,  "sdi": True,  "basin": True,  "is_island": True},   # all four
]


# ---------------------------------------------------------------------------
# Family and mode constants
# ---------------------------------------------------------------------------

BULK_RATE_FAMILIES: list[str] = ["lognormal", "beta", "scaled_logit", "gamma"]
BULK_COUNT_FAMILIES: list[str] = ["nb", "poisson"]
TAIL_RATE_FAMILIES: list[str] = [
    "gamma", "lognormal", "gpd", "truncated_normal", "weibull", "log_logistic"
]
TAIL_COUNT_FAMILIES: list[str] = ["nb", "poisson"]

# Backwards-compatible aggregates (used by tests and downstream code)
BULK_FAMILIES: list[str] = BULK_RATE_FAMILIES + BULK_COUNT_FAMILIES
TAIL_FAMILIES: list[str] = TAIL_RATE_FAMILIES + TAIL_COUNT_FAMILIES

S2_FAMILIES: list[str] = ["logit", "cloglog"]
S2_EXPOSURE_MODES: list[str] = ["free", "excluded"]

# S1 family → allowed exposure modes. cloglog gets offset (theoretically required)
# plus free/excluded for grid search; logit gets free/excluded only.
S1_FAMILY_MODES: dict[str, list[str]] = {
    "cloglog": ["offset", "free", "excluded"],
    "logit":   ["free", "excluded"],
}

# Count families always receive the 'offset' mode; all others receive RATE_EXPOSURE_MODES.
COUNT_FAMILIES: frozenset[str] = frozenset(["nb", "poisson"])
RATE_EXPOSURE_MODES: list[str] = ["free", "weight", "free+weight", "excluded"]


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def enumerate_component_specs(
    mode: str,
    thresholds: list[float] | None = None,
    covariate_combos: list[dict[str, bool]] | None = None,
    s1_family_modes: dict[str, list[str]] | None = None,
    s2_families: list[str] | None = None,
    s2_exposure_modes: list[str] | None = None,
    bulk_families: list[str] | None = None,
    tail_families: list[str] | None = None,
    count_families: frozenset[str] | None = None,
    rate_exposure_modes: list[str] | None = None,
    sample: int | None = None,
) -> list[dict]:
    """Return a flat list of component spec dicts for a grid fit run.

    Parameters
    ----------
    mode:
        'preliminary' — fixed defaults for covariate sets, thresholds, and
        family lists. Any default may be overridden by passing the argument.
        'refined' — thresholds, covariate_combos, s2_families, bulk_families,
        and tail_families are required; count_families and rate_exposure_modes
        have sensible defaults.
    thresholds:
        List of threshold quantile levels (floats in [0, 1]).
        Preliminary default: QUANTILE_LEVELS (0.70, 0.75, ..., 0.95).
        Required in refined mode.
    covariate_combos:
        List of covariate flag dicts.
        Preliminary default: PRELIMINARY_COVARIATE_SETS (3 sets).
        Required in refined mode.
    s2_families:
        List of S2 link function names ("logit", "cloglog").
        Preliminary default: S2_FAMILIES.
        Required in refined mode.
    bulk_families:
        List of ALL bulk family names (rate + count).
        Preliminary default: BULK_FAMILIES.
        Required in refined mode.
    tail_families:
        List of ALL tail family names (rate + count).
        Preliminary default: TAIL_FAMILIES.
        Required in refined mode.
    count_families:
        Set of family names that use offset mode only (log_exposed as fixed offset).
        Default: COUNT_FAMILIES (nb, poisson). These families always get a single
        spec with exposure_mode='offset', regardless of rate_exposure_modes.
    rate_exposure_modes:
        Ordered list of exposure modes for non-count rate families.
        Default: RATE_EXPOSURE_MODES ['free', 'weight', 'free+weight', 'excluded'].
    sample:
        If provided, randomly sample this many specs per component type
        (s1, s2, bulk, tail) from the full enumerated list. Useful for
        smoke-testing pipeline speed without submitting the full grid.

    Returns
    -------
    list[dict]
        Flat list of spec dicts. Counts for preliminary run with defaults:
            S1:         n_combos × n_s1_modes (5)                          =   15
            S2:         n_combos × n_thresh × n_s2 × n_s2_modes            =   72
            Bulk rate:  n_combos × n_thresh × n_bulk_rate × n_rate_modes   =  288
            Bulk count: n_combos × n_thresh × n_bulk_count                 =   36
            Tail rate:  n_combos × n_thresh × n_tail_rate × n_rate_modes   =  432
            Tail count: n_combos × n_thresh × n_tail_count                 =   36
            Total                                                           =  879

    Raises
    ------
    ValueError
        If mode is not 'preliminary' or 'refined', or if mode='refined' and
        any required parameter is not provided.
    """
    # Defaults for both modes
    count_families = count_families if count_families is not None else COUNT_FAMILIES
    rate_exposure_modes = rate_exposure_modes if rate_exposure_modes is not None else RATE_EXPOSURE_MODES

    if mode == "preliminary":
        covariate_combos  = covariate_combos  if covariate_combos  is not None else PRELIMINARY_COVARIATE_SETS
        thresholds        = thresholds        if thresholds        is not None else [float(q) for q in QUANTILE_LEVELS]
        s1_family_modes   = s1_family_modes   if s1_family_modes   is not None else S1_FAMILY_MODES
        s2_families       = s2_families       if s2_families       is not None else S2_FAMILIES
        s2_exposure_modes = s2_exposure_modes if s2_exposure_modes is not None else S2_EXPOSURE_MODES
        bulk_families     = bulk_families     if bulk_families     is not None else BULK_FAMILIES
        tail_families     = tail_families     if tail_families     is not None else TAIL_FAMILIES
    elif mode == "refined":
        missing = [
            name
            for name, val in [
                ("covariate_combos", covariate_combos),
                ("thresholds",       thresholds),
                ("s1_family_modes",  s1_family_modes),
                ("s2_families",      s2_families),
                ("s2_exposure_modes",s2_exposure_modes),
                ("bulk_families",    bulk_families),
                ("tail_families",    tail_families),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                f"mode='refined' requires explicit values for: {missing}. "
                "Refined runs have no defaults — all parameters must be specified."
            )
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. Must be 'preliminary' or 'refined'."
        )

    # Normalise to plain Python floats so spec dicts are JSON-serializable
    thresholds = [float(q) for q in thresholds]

    specs: list[dict] = []

    for combo in covariate_combos:
        # S1: one per (covariate combo, family, exposure_mode)
        for family, modes in s1_family_modes.items():
            for em in modes:
                specs.append({
                    "component":          "s1",
                    "covariate_combo":    combo,
                    "threshold_quantile": None,
                    "threshold_rate":     None,
                    "family":             family,
                    "exposure_mode":      em,
                    "fold_tag":           "is",
                })

        for q in thresholds:
            # S2: one per (covariate combo, threshold, family, exposure_mode)
            for family in s2_families:
                for em in s2_exposure_modes:
                    specs.append({
                        "component":          "s2",
                        "covariate_combo":    combo,
                        "threshold_quantile": q,
                        "threshold_rate":     None,
                        "family":             family,
                        "exposure_mode":      em,
                        "fold_tag":           "is",
                    })

            # Bulk: one per (covariate combo, threshold, family, exposure_mode)
            for family in bulk_families:
                modes = ["offset"] if family in count_families else rate_exposure_modes
                for em in modes:
                    specs.append({
                        "component":          "bulk",
                        "covariate_combo":    combo,
                        "threshold_quantile": q,
                        "threshold_rate":     None,
                        "family":             family,
                        "exposure_mode":      em,
                        "fold_tag":           "is",
                    })

            # Tail: one per (covariate combo, threshold, family, exposure_mode)
            for family in tail_families:
                modes = ["offset"] if family in count_families else rate_exposure_modes
                for em in modes:
                    specs.append({
                        "component":          "tail",
                        "covariate_combo":    combo,
                        "threshold_quantile": q,
                        "threshold_rate":     None,
                        "family":             family,
                        "exposure_mode":      em,
                        "fold_tag":           "is",
                    })

    if sample is not None:
        by_component: dict[str, list[dict]] = {}
        for spec in specs:
            by_component.setdefault(spec["component"], []).append(spec)
        specs = []
        for component_specs in by_component.values():
            n = min(sample, len(component_specs))
            specs.extend(random.sample(component_specs, n))

    return specs


