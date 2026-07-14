"""Unit tests for the final-grid IS spec builder."""

from collections import Counter

from idd_tc_mortality.grid.build_final_specs import (
    BULK_COVS,
    BULK_EXPOSURES,
    S1_COVS,
    S2_COVS,
    TAIL_COVS,
    TAIL_FAMILY_EXPOSURES,
    build_specs,
)


def test_per_component_counts():
    specs = build_specs()
    counts = Counter(s["component"] for s in specs)
    assert counts["s1"] == 2          # 2 cov sets
    assert counts["s2"] == 2          # 2 cov sets × 1 threshold
    assert counts["bulk"] == 8        # 2 exposures × 4 cov sets
    assert counts["tail"] == 8        # 4 family/exposure pairs × 2 cov sets
    assert len(specs) == 20


def test_total_dh_config_count_is_256():
    # The grid is the full cartesian of the per-stage option sets.
    n = (
        len(S1_COVS)
        * len(S2_COVS)
        * (len(BULK_EXPOSURES) * len(BULK_COVS))
        * (len(TAIL_FAMILY_EXPOSURES) * len(TAIL_COVS))
    )
    assert n == 256


def test_threshold_pinned_to_070():
    specs = build_specs()
    s1_thr = {s["threshold_quantile"] for s in specs if s["component"] == "s1"}
    rest_thr = {s["threshold_quantile"] for s in specs if s["component"] != "s1"}
    assert s1_thr == {None}
    assert rest_thr == {0.70}


def test_tail_additions_present():
    specs = build_specs()
    tail = [s for s in specs if s["component"] == "tail"]
    pairs = {(s["family"], s["exposure_mode"]) for s in tail}
    assert ("gpd", "weight") in pairs            # 2026-06-15 addition
    assert pairs == {
        ("log_logistic", "free+weight"),
        ("log_logistic", "weight"),
        ("weibull", "free"),
        ("gpd", "weight"),
    }
    # tail covariate axis gained sdi: cov sets are {(none), sdi}.
    sdi_on = {s["covariate_combo"]["sdi"] for s in tail}
    assert sdi_on == {True, False}
    n_island_or_basin = {
        s["covariate_combo"]["basin"] or s["covariate_combo"]["is_island"]
        or s["covariate_combo"]["wind_speed"]
        for s in tail
    }
    assert n_island_or_basin == {False}          # tail covs only ever (none) or sdi


def test_specs_are_distinct():
    specs = build_specs()
    keyed = {
        (s["component"], s["family"], s["exposure_mode"],
         s["threshold_quantile"], tuple(sorted(s["covariate_combo"].items())))
        for s in specs
    }
    assert len(keyed) == len(specs)
