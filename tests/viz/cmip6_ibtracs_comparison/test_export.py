"""JSON shaping, overview rows, basin boxes and the checked writers."""

import json

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.viz.cmip6_ibtracs_comparison import export as ex
from idd_tc_mortality.viz.cmip6_ibtracs_comparison import io as cio
from idd_tc_mortality.viz.cmip6_ibtracs_comparison.stats import (
    CEIL_COLS,
    METRIC_COLS,
    RAW_BAND_START,
)


def test_num_rounding_and_nan():
    assert ex._num(np.nan) is None and ex._num(1.23456789) == 1.2346 and ex._nums([1, np.inf]) == [1.0, None]
    assert ex._num(None) is None and ex._num(np.float32(2.5)) == 2.5


def test_keys():
    assert ex.stat_key("obs_vs_model", "ens", "mean", "ma5", True, 1985) == "obs_vs_model|ens|mean|ma5|d|1985"
    assert ex.band_key("model", "ma10", False) == f"model|ma10|r|{RAW_BAND_START}"
    assert ex.band_key("ibt", "ma5", True, 1990) == "ibt|ma5|d|1990"
    assert ex.ceil_key("ma5", True, 1985, "mean") == "ma5|d|1985|mean"


def _group():
    years = np.array([2000, 2001])
    key = {"domain": "storm", "source": "land", "level": "global", "geo": "1", "outcome": "total"}
    series = pd.DataFrame([dict(key, series="obs", stat="raw", year=y, value=v) for y, v in zip(years, [1.0, 2.0])]
                          + [dict(key, series="ens", stat="mean", year=y, value=v) for y, v in zip(years, [3.0, 4.0])])
    band = pd.DataFrame([dict(key, band="model", window="ma1", detrended=False, start_year=RAW_BAND_START, start=y, end=y,
                              q05=0, q25=1, q50=2, q75=3, q95=4, obs=1.0, obs_pct=0.5) for y in years])
    stats = pd.DataFrame([dict(key, start_year=2000, pair="obs_vs_model", series="ens", stat="mean", window="ma1",
                               detrended=False, **{c: 0.5 for c in METRIC_COLS}),
                          dict(key, start_year=2001, pair="obs_vs_model", series="ens", stat="mean", window="ma1",
                               detrended=True, **{c: np.nan for c in METRIC_COLS})])
    ceiling = pd.DataFrame([dict(key, start_year=2000, window="ma1", detrended=False, stat="mean", **{c: 0.25 for c in CEIL_COLS})])
    return stats, band, series, ceiling, years


def test_shape_cell_group():
    stats, band, series, ceiling, years = _group()
    out = ex.shape_cell_group(stats, band, series, ceiling, {"1": "Global"}, years)
    g = out["geos"]["1"]
    assert g["name"] == "Global" and g["obs"] == [1.0, 2.0] and g["series"]["ens"]["mean"] == [3.0, 4.0]
    bk = ex.band_key("model", "ma1", False)
    assert g["band"][bk]["q"][2] == [2.0, 2.0] and g["band"][bk]["start"] == [2000, 2001]
    assert out["stat_keys"] == ["obs_vs_model|ens|mean|ma1|d|2001", "obs_vs_model|ens|mean|ma1|r|2000"]
    assert g["stats"][1] == [0.5] * len(METRIC_COLS) and g["stats"][0] == [None] * len(METRIC_COLS)
    assert g["ceiling"]["ma1|r|2000|mean"] == [0.25] * len(CEIL_COLS) and out["ceil_cols"] == CEIL_COLS
    json.dumps(out)


def test_shape_cell_group_without_stats_or_ceiling():
    stats, band, series, ceiling, years = _group()
    out = ex.shape_cell_group(stats.iloc[0:0], band.iloc[0:0], series, ceiling.iloc[0:0], {}, years)
    assert out["stat_keys"] == [] and "stats" not in out["geos"]["1"] and out["geos"]["1"]["name"] == "1"


def test_overview_rows_medians_and_significance():
    base = {"domain": "storm", "source": "land", "level": "country", "outcome": "total", "pair": "obs_vs_model",
            "series": "ens", "stat": "mean", "window": "ma5", "detrended": False, "start_year": 1985}
    stats = pd.DataFrame([dict(base, geo="6", spearman=0.2, kendall=0.1, hit_rate=0.5, updown=0.5, ratio=1.0, pct_mean=0.5, sur_pct=0.6, sur_p=0.8),
                          dict(base, geo="7", spearman=0.6, kendall=0.3, hit_rate=0.5, updown=0.5, ratio=3.0, pct_mean=0.5, sur_pct=0.99, sur_p=0.02)])
    ceiling = pd.DataFrame([{"domain": "storm", "source": "land", "level": "country", "outcome": "total", "geo": g,
                             "window": "ma5", "detrended": False, "start_year": 1985, "stat": "mean", "ceil_med": v}
                            for g, v in (("6", 0.3), ("7", 0.5))])
    rows = ex.overview_rows(stats, ceiling)
    c = {k: i for i, k in enumerate(ex.OVERVIEW_COLS)}
    assert len(rows) == 1 and rows[0][c["n_geos"]] == 2 and rows[0][c["spearman"]] == pytest.approx(0.4)
    assert rows[0][c["detrended"]] == "r" and rows[0][c["start_year"]] == 1985
    assert rows[0][c["n_sig"]] == 1 and rows[0][c["ceil_med"]] == pytest.approx(0.4) and rows[0][c["sur_pct"]] == pytest.approx(0.795)
    assert len(rows[0]) == len(ex.OVERVIEW_COLS)
    assert ex.overview_rows(stats, None)[0][c["ceil_med"]] is None


def test_box_geojson_has_seven_basins():
    fc = ex.box_geojson()
    assert len(fc["features"]) == 7 and {f["properties"]["id"] for f in fc["features"]} == {"AU", "EP", "NA", "NI", "SI", "SP", "WP"}


def test_write_parquet_checked_refuses_empty_and_writes(tmp_path):
    with pytest.raises(RuntimeError, match="zero rows"):
        cio.write_parquet_checked(pd.DataFrame({"a": []}), tmp_path / "x.parquet")
    p = cio.write_parquet_checked(pd.DataFrame({"a": [1, 2]}), tmp_path / "x.parquet", {"a": "int16"})
    assert pd.read_parquet(p)["a"].dtype == "int16" and not (tmp_path / "x.parquet.tmp").exists()


def test_write_json_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        cio.write_json({"a": float("nan")}, tmp_path / "a.json")
    cio.write_json({"a": None}, tmp_path / "a.json")
    assert json.loads((tmp_path / "a.json").read_text()) == {"a": None}


def test_spearman_index_shape():
    stats = pd.DataFrame([{"geo": "6", "outcome": "total", "pair": "obs_vs_model", "series": "ens", "stat": "mean",
                           "window": "ma5", "detrended": True, "start_year": 1985, "spearman": 0.25},
                          {"geo": "6", "outcome": "cat1", "pair": "obs_vs_model", "series": "ens", "stat": "mean",
                           "window": "ma5", "detrended": False, "start_year": 1985, "spearman": np.nan}])
    out = ex.spearman_index(stats)
    assert out["keys"] == ["obs_vs_model|ens|mean|ma5|d|1985", "obs_vs_model|ens|mean|ma5|r|1985"]
    assert out["geos"]["6"]["total"] == [0.25, None]
    assert out["geos"]["6"]["cat1"] == [None, None]
