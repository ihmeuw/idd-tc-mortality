"""Counting rules: category pivot, worst-category aggregation, basin keying, IBTrACS table."""

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.viz.cmip6_ibtracs_comparison import storms as st
from idd_tc_mortality.viz.cmip6_ibtracs_comparison.constants import CAT_COLS, NO_BASIN


@pytest.fixture
def hier():
    return pd.DataFrame({
        "location_id": [1, 4, 64, 5, 100, 6, 102, 22],
        "level": [0, 1, 1, 2, 2, 3, 3, 3],
        "region_id": [-1, -1, -1, 5, 100, 5, 100, 5],
        "super_region_id": [-1, 4, 64, 4, 64, 4, 64, 4],
        "location_name": ["Global", "SEA", "HI", "East Asia", "HI NA", "China", "USA", "Fiji"],
    })


def test_counts_by_category_pivots_and_drops_code_zero():
    df = pd.DataFrame({"year": [1, 1, 1, 2, 2], "cat": [0, 1, 6, 4, 4]})
    out = st.counts_by_category(df, ["year"]).set_index("year")
    assert list(out.columns) == CAT_COLS
    assert out.loc[1].tolist() == [1, 0, 0, 0, 0, 1]
    assert out.loc[2].tolist() == [0, 0, 0, 2, 0, 0]


def test_counts_by_category_empty():
    out = st.counts_by_category(pd.DataFrame({"year": [1], "cat": [0]}), ["year"])
    assert out.empty and list(out.columns) == ["year"] + CAT_COLS


def test_storm_counts_by_level_worst_category_rule(hier):
    # storm A hits China (Cat1, 35) and USA (Cat4, 60); storm B hits Fiji (TS, 20)
    # storm C is the same storm_id as A but in another basin → distinct storm
    rows = pd.DataFrame({
        "year": [2000] * 4, "basin": ["WP", "WP", "SP", "NA"], "storm_id": [1, 1, 2, 1],
        "location_id": [6, 102, 22, 102], "max_wind_speed": [35.0, 60.0, 20.0, 45.0],
    })
    out = st.storm_counts_by_level(rows, hier, keys=[])
    tab = out.set_index(["level", "geo"])[CAT_COLS]
    assert tab.loc[("country", "6")].tolist() == [0, 1, 0, 0, 0, 0]
    # USA: storm A at Cat4 (60) and storm C at Cat2 (45) — same storm_id, different basin
    assert tab.loc[("country", "102")].tolist() == [0, 0, 1, 0, 1, 0]
    # global: A once at its worst (Cat4), B (TS), C (Cat2) → three storms
    assert tab.loc[("global", "1")].tolist() == [1, 0, 1, 0, 1, 0]
    # super-region 4 (SEA): A at worst within SR (China, Cat1) + B (TS)
    assert tab.loc[("super_region", "4")].tolist() == [1, 1, 0, 0, 0, 0]
    # basin level: WP has A at Cat4; NA has C; SP has B
    assert tab.loc[("basin", "WP")].tolist() == [0, 0, 0, 0, 1, 0]
    assert tab.loc[("basin", "NA")].tolist() == [0, 0, 1, 0, 0, 0]


def test_storm_counts_by_level_no_basin_counts_everywhere_but_basin(hier):
    rows = pd.DataFrame({"year": [2001], "basin": [NO_BASIN], "storm_id": [9], "location_id": [22],
                         "max_wind_speed": [55.0]})
    out = st.storm_counts_by_level(rows, hier, keys=[])
    assert set(out["level"]) == {"country", "region", "super_region", "global"}


def test_storm_counts_by_level_rejects_non_admin0(hier):
    rows = pd.DataFrame({"year": [2001], "basin": ["NA"], "storm_id": [9], "location_id": [100],
                         "max_wind_speed": [55.0]})
    with pytest.raises(ValueError, match="admin0"):
        st.storm_counts_by_level(rows, hier, keys=[])


def test_check_member_grid_rejects_missing_draws():
    from idd_tc_mortality.viz.cmip6_ibtracs_comparison.constants import RUNS
    df = pd.DataFrame({"run": list(RUNS), "draw": [0] * 8})
    with pytest.raises(RuntimeError, match="draws per run"):
        st.check_member_grid(df)


def test_ibtracs_all_counts_fallback_and_spur(monkeypatch):
    storms = pd.DataFrame({
        "storm_id": ["a", "b", "c", "d"], "year": [1990, 1990, 1991, 1991],
        "track_type": ["main", "spur-split", "main", "main"],
        "tc_basin": ["NA", "NA", NO_BASIN, "WP"], "ib_basin": ["NA", "NA", "SA", "WP"],
        "wind_1min": [60.0, 60.0, 40.0, np.nan], "wind_source": ["usa", "usa", "wmo_10min_adj", "none"],
    })
    out = st.ibtracs_all_counts(storms, (1990, 1991)).set_index(["level", "geo", "year"])[CAT_COLS]
    assert out.loc[("global", "1", 1990)].tolist() == [0, 0, 0, 0, 1, 0]   # spur excluded
    assert out.loc[("global", "1", 1991)].tolist() == [0, 1, 0, 0, 0, 0]   # c (40 m/s, Cat1) via wmo; d unclassified
    assert ("basin", NO_BASIN, 1991) not in out.index
    cov = st.ibtracs_wind_coverage(storms, (1990, 1991)).set_index(["ib_basin", "decade"])
    assert cov.loc[("WP", 1990), "none"] == 1 and cov.loc[("NA", 1990), "n"] == 1


def test_splice_filter_requires_overlap():
    with pytest.raises(ValueError):
        st.splice_filter("scenario", "year", (2030, 2040))
    assert st.splice_filter("scenario", "year", (1970, 2025)) is not None
