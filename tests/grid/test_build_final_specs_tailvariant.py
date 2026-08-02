from collections import Counter

import pytest

from idd_tc_mortality.grid.build_final_specs_tailvariant import (
    GRIDS,
    _covs,
    build_specs,
    n_configs,
)


def test_cov_pattern_conversion():
    combos = _covs("", "basin+sdi")
    assert combos[0] == {"wind_speed": False, "sdi": False, "basin": False, "is_island": False}
    assert combos[1] == {"wind_speed": False, "sdi": True, "basin": True, "is_island": False}
    with pytest.raises(ValueError, match="Unknown"):
        _covs("basin+bogus")


def test_implied_config_counts_match_notebooks():
    assert n_configs("v2000") == 158_400
    assert n_configs("v1985") == 54_432


@pytest.mark.parametrize("vintage,expected", [
    # s1 + n_thresholds x (s2 + bulk_exp*bulk_cov + tail_pairs*tail_cov)
    ("v2000", 8 + 2 * (5 + 2 * 11 + 10 * 9)),
    ("v1985", 6 + 2 * (3 + 2 * 9 + 7 * 12)),
])
def test_is_spec_counts(vintage, expected):
    specs = build_specs(vintage)
    assert len(specs) == expected
    assert all(s["fold_tag"] == "is" for s in specs)


def test_thresholds_are_canonical():
    for vintage in GRIDS:
        specs = build_specs(vintage)
        qs = {s["threshold_quantile"] for s in specs if s["component"] != "s1"}
        assert len(qs) == 2
        for q in qs:
            assert any(abs(q - t) < 1e-6 for t in GRIDS[vintage].THRESHOLDS)


def test_grids_differ_where_notebooks_did():
    assert ("gpd_cens", "excluded") in GRIDS["v2000"].TAIL_FAMILY_EXPOSURES
    assert all(f != "gpd_cens" for f, _ in GRIDS["v1985"].TAIL_FAMILY_EXPOSURES)
    assert ("gpd_trunc", "free+weight") in GRIDS["v1985"].TAIL_FAMILY_EXPOSURES
    assert 0.95 in GRIDS["v2000"].THRESHOLDS and 0.85 in GRIDS["v1985"].THRESHOLDS
