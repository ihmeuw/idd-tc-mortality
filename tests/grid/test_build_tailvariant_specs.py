"""Tests for the honest-mean preliminary spec generator."""

from __future__ import annotations

import json
from collections import Counter

from idd_tc_mortality.grid.build_tailvariant_specs import build_specs


def test_total_and_per_component_counts():
    specs = build_specs()
    by_comp = Counter(s["component"] for s in specs)
    # s1 15, s2 72, bulk 324 (unchanged); tail 636 = 8 std-rate x 72 + 2 count x 18 + A 24.
    assert by_comp == {"s1": 15, "s2": 72, "bulk": 324, "tail": 636}
    assert len(specs) == 1047


def test_median_tail_families_dropped():
    tail_fams = {s["family"] for s in build_specs() if s["component"] == "tail"}
    for dropped in ("gpd", "weibull", "log_logistic"):
        assert dropped not in tail_fams, f"median family {dropped} must not be a tail candidate"
    # honest set present
    for keep in ("gamma", "lognormal", "truncated_normal", "weibull_mean",
                 "gpd_cens", "log_logistic_cens", "gpd_trunc", "log_logistic_trunc",
                 "gpd_shadow", "log_logistic_shadow", "nb", "poisson"):
        assert keep in tail_fams


def test_shadow_families_are_intercept_only_and_two_exposures():
    specs = build_specs()
    for fam in ("gpd_shadow", "log_logistic_shadow"):
        rows = [s for s in specs if s["family"] == fam]
        assert len(rows) == 12, f"{fam}: expected 6 thr x 2 exposure = 12, got {len(rows)}"
        assert all(not any(s["covariate_combo"].values()) for s in rows), "A must be intercept-only"
        assert {s["exposure_mode"] for s in rows} == {"weight", "excluded"}


def test_specs_are_json_serializable():
    json.dumps(build_specs())  # floats not numpy, etc.
