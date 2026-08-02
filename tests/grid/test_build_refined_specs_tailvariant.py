from collections import Counter

from idd_tc_mortality.constants import QUANTILE_LEVELS
from idd_tc_mortality.grid.build_refined_specs_post2000 import snap_to_quantile_levels
from idd_tc_mortality.grid.build_refined_specs_tailvariant import (
    DEFAULT_THRESHOLDS,
    HONEST_TAIL_FAMILIES,
    TAIL_EXPOSURE_MODES,
    build_specs,
)

SPEC_KEYS = {
    "component", "covariate_combo", "threshold_quantile", "threshold_rate",
    "family", "exposure_mode", "fold_tag",
}


def _default_specs() -> list[dict]:
    return build_specs(snap_to_quantile_levels(list(DEFAULT_THRESHOLDS)))


def test_spec_counts():
    specs = _default_specs()
    assert len(specs) == 1696  # 16 s1 + 3 x (16 s2 + 32 bulk + 512 tail)
    by_comp = Counter(s["component"] for s in specs)
    assert by_comp == {"s1": 16, "s2": 48, "bulk": 96, "tail": 1536}


def test_spec_shape_matches_enumerator_output():
    specs = _default_specs()
    assert all(set(s) == SPEC_KEYS for s in specs)
    assert all(s["fold_tag"] == "is" for s in specs)
    assert all(s["threshold_rate"] is None for s in specs)


def test_tail_is_full_honest_cartesian_per_threshold():
    specs = _default_specs()
    expected = {(f, em) for f in HONEST_TAIL_FAMILIES for em in TAIL_EXPOSURE_MODES}
    for q in {s["threshold_quantile"] for s in specs if s["component"] == "tail"}:
        got = {
            (s["family"], s["exposure_mode"])
            for s in specs
            if s["component"] == "tail" and s["threshold_quantile"] == q
        }
        assert got == expected


def test_no_median_or_dropped_families():
    dropped = {
        "gpd", "weibull", "log_logistic",            # median-reporting bases
        "nb", "poisson",                              # count tails
        "gpd_shadow", "log_logistic_shadow",          # A families, culled
    }
    families = {s["family"] for s in _default_specs()}
    assert families.isdisjoint(dropped)


def test_thresholds_are_canonical_and_decided():
    specs = _default_specs()
    qs = {s["threshold_quantile"] for s in specs if s["component"] != "s1"}
    canonical = {
        float(q) for q in QUANTILE_LEVELS
        if any(abs(float(q) - d) < 1e-6 for d in (0.70, 0.85, 0.95))
    }
    assert qs == canonical
    assert all(s["threshold_quantile"] is None for s in specs if s["component"] == "s1")


def test_all_16_cov_combos_per_component():
    specs = _default_specs()
    for comp in ("s1", "s2", "bulk", "tail"):
        combos = {
            tuple(sorted(s["covariate_combo"].items()))
            for s in specs
            if s["component"] == comp
        }
        assert len(combos) == 16
