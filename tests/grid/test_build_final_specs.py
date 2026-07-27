"""Unit tests for the final-grid IS spec builder (20260714 grid)."""

from collections import Counter

from idd_tc_mortality.grid.build_final_specs import (
    BULK_COVS,
    BULK_EXPOSURES,
    S1_COVS,
    S2_COVS,
    TAIL_COVS,
    TAIL_FAMILY_EXPOSURES,
    THRESHOLDS,
    build_specs,
)


def _on_axes(cov: dict) -> frozenset:
    return frozenset(axis for axis, on in cov.items() if on)


def test_per_component_counts():
    specs = build_specs()
    counts = Counter(s["component"] for s in specs)
    assert counts["s1"] == 2           # 2 cov sets, threshold-free
    assert counts["s2"] == 6           # 2 cov sets × 3 thresholds
    assert counts["bulk"] == 36        # 2 exposures × 6 cov sets × 3 thresholds
    assert counts["tail"] == 96        # 4 family/exposure pairs × 8 cov sets × 3 thresholds
    assert len(specs) == 140


def test_total_dh_config_count_is_4608():
    # The grid is the full cartesian of the per-stage option sets × thresholds.
    n = (
        len(THRESHOLDS)
        * len(S1_COVS)
        * len(S2_COVS)
        * (len(BULK_EXPOSURES) * len(BULK_COVS))
        * (len(TAIL_FAMILY_EXPOSURES) * len(TAIL_COVS))
    )
    assert n == 4608


def test_thresholds():
    specs = build_specs()
    s1_thr = {s["threshold_quantile"] for s in specs if s["component"] == "s1"}
    rest_thr = {s["threshold_quantile"] for s in specs if s["component"] != "s1"}
    assert s1_thr == {None}
    assert rest_thr == {0.70, 0.75, 0.85}


def test_tail_pairs_and_cov_sets():
    specs = build_specs()
    tail = [s for s in specs if s["component"] == "tail"]
    pairs = {(s["family"], s["exposure_mode"]) for s in tail}
    assert pairs == {
        ("log_logistic", "free+weight"),
        ("log_logistic", "weight"),
        ("weibull", "free"),
        ("gpd", "weight"),
    }
    # 20260714 grid: tail cov axis opened to 8 sets.
    cov_sets = {_on_axes(s["covariate_combo"]) for s in tail}
    assert cov_sets == {
        frozenset(),
        frozenset({"wind_speed"}),
        frozenset({"sdi"}),
        frozenset({"sdi", "is_island"}),
        frozenset({"sdi", "basin"}),
        frozenset({"wind_speed", "sdi", "basin"}),
        frozenset({"wind_speed", "sdi", "is_island"}),
        frozenset({"wind_speed", "sdi", "basin", "is_island"}),
    }


def test_bulk_cov_sets():
    # 20260714 grid: is_island enters bulk via the two new sets.
    cov_sets = {_on_axes(c) for c in BULK_COVS}
    assert cov_sets == {
        frozenset({"sdi"}),
        frozenset({"sdi", "basin"}),
        frozenset({"wind_speed", "sdi"}),
        frozenset({"wind_speed", "sdi", "basin"}),
        frozenset({"wind_speed", "sdi", "is_island"}),
        frozenset({"wind_speed", "sdi", "basin", "is_island"}),
    }


def test_specs_are_distinct():
    specs = build_specs()
    keyed = {
        (s["component"], s["family"], s["exposure_mode"],
         s["threshold_quantile"], tuple(sorted(s["covariate_combo"].items())))
        for s in specs
    }
    assert len(keyed) == len(specs)
