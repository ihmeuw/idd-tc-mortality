"""
Tests for grid/grid.py.

Covers:
  - preliminary mode produces exactly 879 specs.
  - S1 specs: threshold_quantile=None, family in S1_FAMILY_MODES, valid exposure_mode.
  - S2 specs: valid family, valid threshold, exposure_mode in S2_EXPOSURE_MODES.
  - Bulk specs: valid family; rate families have all 4 exposure modes,
    count families have only 'offset'.
  - Tail specs: same exposure_mode pattern as bulk.
  - No duplicate component IDs in preliminary output.
  - refined mode raises ValueError if any required parameter is missing.
  - refined mode with explicit inputs produces correct counts.
  - Every spec can be passed to cache.component_id without error.
  - exposure_mode field: rate families enumerate all RATE_EXPOSURE_MODES,
    count families only enumerate 'offset'.

Preliminary counts (3 covariate combos, 6 thresholds):
  S1:         3 × 5 (family/mode combos)                          =   15
  S2:         3 × 6 × 2 families × 2 modes                        =   72
  Bulk rate:  3 × 6 × 4 families × 4 modes                        =  288
  Bulk count: 3 × 6 × 2 families × 1 mode                         =   36
  Tail rate:  3 × 6 × 6 families × 4 modes                        =  432
  Tail count: 3 × 6 × 2 families × 1 mode                         =   36
  Total                                                             =  879
"""

from __future__ import annotations

import pytest

from idd_tc_mortality.cache import component_id
from idd_tc_mortality.constants import QUANTILE_LEVELS
from idd_tc_mortality.grid.grid import (
    BULK_COUNT_FAMILIES,
    BULK_FAMILIES,
    BULK_RATE_FAMILIES,
    COUNT_FAMILIES,
    PRELIMINARY_COVARIATE_SETS,
    RATE_EXPOSURE_MODES,
    S1_FAMILY_MODES,
    S2_EXPOSURE_MODES,
    S2_FAMILIES,
    TAIL_COUNT_FAMILIES,
    TAIL_FAMILIES,
    TAIL_RATE_FAMILIES,
    enumerate_component_specs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _by_component(specs: list[dict], component: str) -> list[dict]:
    return [s for s in specs if s["component"] == component]


# ---------------------------------------------------------------------------
# Preliminary mode — total count
# ---------------------------------------------------------------------------

def test_preliminary_total_count():
    specs = enumerate_component_specs("preliminary")
    assert len(specs) == 879


def test_preliminary_s1_count():
    specs = enumerate_component_specs("preliminary")
    n_s1_combos = sum(len(modes) for modes in S1_FAMILY_MODES.values())
    assert len(_by_component(specs, "s1")) == len(PRELIMINARY_COVARIATE_SETS) * n_s1_combos


def test_preliminary_s2_count():
    specs = enumerate_component_specs("preliminary")
    n_thresh = len(QUANTILE_LEVELS)
    assert len(_by_component(specs, "s2")) == (
        len(PRELIMINARY_COVARIATE_SETS) * n_thresh * len(S2_FAMILIES) * len(S2_EXPOSURE_MODES)
    )


def test_preliminary_bulk_count():
    """Bulk = rate families × 4 modes + count families × 1 mode."""
    specs = enumerate_component_specs("preliminary")
    n_combos = len(PRELIMINARY_COVARIATE_SETS)
    n_thresh = len(QUANTILE_LEVELS)
    expected = (
        n_combos * n_thresh * len(BULK_RATE_FAMILIES) * len(RATE_EXPOSURE_MODES)
        + n_combos * n_thresh * len(BULK_COUNT_FAMILIES)
    )
    assert len(_by_component(specs, "bulk")) == expected


def test_preliminary_tail_count():
    """Tail = rate families × 4 modes + count families × 1 mode."""
    specs = enumerate_component_specs("preliminary")
    n_combos = len(PRELIMINARY_COVARIATE_SETS)
    n_thresh = len(QUANTILE_LEVELS)
    expected = (
        n_combos * n_thresh * len(TAIL_RATE_FAMILIES) * len(RATE_EXPOSURE_MODES)
        + n_combos * n_thresh * len(TAIL_COUNT_FAMILIES)
    )
    assert len(_by_component(specs, "tail")) == expected


# ---------------------------------------------------------------------------
# Preliminary mode — spec field correctness
# ---------------------------------------------------------------------------

def test_s1_specs_have_no_threshold_valid_family_valid_exposure_mode():
    specs = enumerate_component_specs("preliminary")
    for s in _by_component(specs, "s1"):
        assert s["threshold_quantile"] is None, f"S1 spec has threshold: {s}"
        assert s["family"] in S1_FAMILY_MODES, f"S1 spec has unknown family: {s['family']}"
        assert s["exposure_mode"] in S1_FAMILY_MODES[s["family"]], (
            f"S1 spec has invalid exposure_mode {s['exposure_mode']!r} for family {s['family']!r}"
        )


def test_s2_specs_have_valid_family_and_valid_exposure_mode():
    specs = enumerate_component_specs("preliminary")
    for s in _by_component(specs, "s2"):
        assert s["family"] in S2_FAMILIES, f"S2 spec has unknown family: {s['family']}"
        assert s["exposure_mode"] in S2_EXPOSURE_MODES, (
            f"S2 spec has invalid exposure_mode: {s['exposure_mode']!r}"
        )


def test_s2_specs_have_valid_threshold():
    valid_quantiles = set(float(q) for q in QUANTILE_LEVELS)
    specs = enumerate_component_specs("preliminary")
    for s in _by_component(specs, "s2"):
        assert s["threshold_quantile"] in valid_quantiles


def test_bulk_specs_have_valid_family():
    specs = enumerate_component_specs("preliminary")
    for s in _by_component(specs, "bulk"):
        assert s["family"] in BULK_FAMILIES, f"Unknown bulk family: {s['family']}"


def test_tail_specs_have_valid_family():
    specs = enumerate_component_specs("preliminary")
    for s in _by_component(specs, "tail"):
        assert s["family"] in TAIL_FAMILIES, f"Unknown tail family: {s['family']}"


# ---------------------------------------------------------------------------
# exposure_mode correctness
# ---------------------------------------------------------------------------

def test_bulk_rate_families_get_all_rate_modes():
    """Each rate family appears with every exposure mode in RATE_EXPOSURE_MODES."""
    specs = enumerate_component_specs("preliminary")
    bulk_specs = _by_component(specs, "bulk")
    for family in BULK_RATE_FAMILIES:
        modes_seen = {s["exposure_mode"] for s in bulk_specs if s["family"] == family}
        assert modes_seen == set(RATE_EXPOSURE_MODES), (
            f"Bulk rate family {family!r} has exposure modes {modes_seen}, "
            f"expected {set(RATE_EXPOSURE_MODES)}"
        )


def test_bulk_count_families_get_only_offset():
    """Count families only appear with exposure_mode='offset'."""
    specs = enumerate_component_specs("preliminary")
    bulk_specs = _by_component(specs, "bulk")
    for family in BULK_COUNT_FAMILIES:
        modes_seen = {s["exposure_mode"] for s in bulk_specs if s["family"] == family}
        assert modes_seen == {"offset"}, (
            f"Count family {family!r} has exposure modes {modes_seen}, expected {{'offset'}}"
        )


def test_tail_rate_families_get_all_rate_modes():
    specs = enumerate_component_specs("preliminary")
    tail_specs = _by_component(specs, "tail")
    for family in TAIL_RATE_FAMILIES:
        modes_seen = {s["exposure_mode"] for s in tail_specs if s["family"] == family}
        assert modes_seen == set(RATE_EXPOSURE_MODES), (
            f"Tail rate family {family!r} has exposure modes {modes_seen}"
        )


def test_tail_count_families_get_only_offset():
    specs = enumerate_component_specs("preliminary")
    tail_specs = _by_component(specs, "tail")
    for family in TAIL_COUNT_FAMILIES:
        modes_seen = {s["exposure_mode"] for s in tail_specs if s["family"] == family}
        assert modes_seen == {"offset"}, (
            f"Count family {family!r} has exposure modes {modes_seen}"
        )


def test_all_specs_always_have_exposure_mode():
    """Every spec has an exposure_mode key."""
    specs = enumerate_component_specs("preliminary")
    for s in specs:
        assert "exposure_mode" in s, f"Missing exposure_mode in {s}"


# ---------------------------------------------------------------------------
# No duplicates
# ---------------------------------------------------------------------------

def test_no_duplicate_component_ids():
    specs = enumerate_component_specs("preliminary")
    ids = [component_id(s) for s in specs]
    assert len(ids) == len(set(ids)), (
        f"Duplicate component IDs: {len(ids) - len(set(ids))} collision(s) in {len(ids)} specs"
    )


# ---------------------------------------------------------------------------
# All specs are cache-hashable
# ---------------------------------------------------------------------------

def test_all_specs_hashable_via_component_id():
    specs = enumerate_component_specs("preliminary")
    for s in specs:
        cid = component_id(s)
        assert isinstance(cid, str) and len(cid) == 32


# ---------------------------------------------------------------------------
# refined mode — raises on missing parameters
# ---------------------------------------------------------------------------

_MINIMAL_COMBOS   = [{"wind_speed": True, "sdi": False, "basin": False, "is_island": False}]
_MINIMAL_THRESH   = [0.85]
_MINIMAL_S1_FM    = {"cloglog": ["offset"]}
_MINIMAL_S2_FAM   = ["logit"]
_MINIMAL_S2_EM    = ["free"]
_MINIMAL_BULK_FAM = ["gamma"]
_MINIMAL_TAIL_FAM = ["gpd"]


@pytest.mark.parametrize("missing_param", [
    "covariate_combos",
    "thresholds",
    "s1_family_modes",
    "s2_families",
    "s2_exposure_modes",
    "bulk_families",
    "tail_families",
])
def test_refined_raises_if_param_missing(missing_param):
    kwargs = {
        "covariate_combos":  _MINIMAL_COMBOS,
        "thresholds":        _MINIMAL_THRESH,
        "s1_family_modes":   _MINIMAL_S1_FM,
        "s2_families":       _MINIMAL_S2_FAM,
        "s2_exposure_modes": _MINIMAL_S2_EM,
        "bulk_families":     _MINIMAL_BULK_FAM,
        "tail_families":     _MINIMAL_TAIL_FAM,
    }
    kwargs[missing_param] = None
    with pytest.raises(ValueError, match=missing_param):
        enumerate_component_specs("refined", **kwargs)


def test_refined_raises_if_all_params_missing():
    with pytest.raises(ValueError):
        enumerate_component_specs("refined")


# ---------------------------------------------------------------------------
# refined mode — correct counts with explicit inputs
# ---------------------------------------------------------------------------

def test_refined_correct_counts():
    """Refined mode with a mix of rate and count families produces the right counts."""
    combos     = _MINIMAL_COMBOS * 2          # 2 covariate combos
    thresholds = [0.80, 0.90]                  # 2 thresholds
    s1_fm      = {"cloglog": ["offset", "free"], "logit": ["free"]}  # 3 S1 combos
    s2_fam     = ["logit", "cloglog"]          # 2 S2 families
    s2_em      = ["free", "excluded"]          # 2 S2 modes
    bulk_fam   = ["lognormal", "nb"]           # 1 rate + 1 count
    tail_fam   = ["gpd", "poisson"]            # 1 rate + 1 count

    specs = enumerate_component_specs(
        "refined",
        covariate_combos=combos,
        thresholds=thresholds,
        s1_family_modes=s1_fm,
        s2_families=s2_fam,
        s2_exposure_modes=s2_em,
        bulk_families=bulk_fam,
        tail_families=tail_fam,
    )

    n_combos  = len(combos)
    n_thresh  = len(thresholds)
    n_modes   = len(RATE_EXPOSURE_MODES)
    n_s1      = sum(len(m) for m in s1_fm.values())

    assert len(_by_component(specs, "s1"))   == n_combos * n_s1
    assert len(_by_component(specs, "s2"))   == n_combos * n_thresh * len(s2_fam) * len(s2_em)
    # bulk: 1 rate × 4 modes + 1 count × 1 mode = 5 per (combo, thresh)
    assert len(_by_component(specs, "bulk")) == n_combos * n_thresh * (1 * n_modes + 1)
    # tail: same pattern
    assert len(_by_component(specs, "tail")) == n_combos * n_thresh * (1 * n_modes + 1)


# ---------------------------------------------------------------------------
# Unknown mode raises
# ---------------------------------------------------------------------------

def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown mode"):
        enumerate_component_specs("bogus")
