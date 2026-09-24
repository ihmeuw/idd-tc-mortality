"""Pure helpers of the historical blend: parent detection, blend rule, roll-up, ratios."""

import json

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.predict import blend_historical as bh


@pytest.fixture
def hier():
    # 1 = Global; 4 = super-region; 100 = region; 6, 7 = countries; 60, 61 = admin-1 of country 6
    rows = [
        (1, "1", 0), (4, "1,4", 1), (100, "1,4,100", 2),
        (6, "1,4,100,6", 3), (7, "1,4,100,7", 3),
        (60, "1,4,100,6,60", 4), (61, "1,4,100,6,61", 4),
    ]
    return pd.DataFrame(rows, columns=["location_id", "path_to_top_parent", "level"]).assign(super_region_id=4)


def _md():
    base = {"storm_draw": 1, "ssp_scenario": "historical", "year_id": 2000}
    return pd.DataFrame([
        {**base, "location_id": 6, "deaths": 10.0, "frame": "a0"},
        {**base, "location_id": 7, "deaths": 3.0, "frame": "a0"},
        {**base, "location_id": 60, "deaths": 4.0, "frame": "a1"},
        {**base, "location_id": 61, "deaths": 5.0, "frame": "a1"},
    ])


def test_subnationalized_parents(hier):
    assert bh.subnationalized_parents(hier, [60, 61]) == [6]
    assert bh.subnationalized_parents(hier, []) == []


def test_blend_keeps_a1_units_and_non_subnationalized_a0(hier):
    blend, parents = bh.blend_a1_a0(_md(), hier)
    assert parents == [6]
    assert sorted(blend["location_id"]) == [7, 60, 61]
    assert "frame" not in blend.columns
    assert blend["deaths"].sum() == pytest.approx(12.0)  # 3 + 4 + 5, country 6's own 10 dropped


def test_rollup_sums_into_every_ancestor(hier):
    blend, _ = bh.blend_a1_a0(_md(), hier)
    up = bh.rollup(blend, hier).set_index("location_id")["deaths"]
    assert up[1] == pytest.approx(12.0) and up[4] == pytest.approx(12.0)
    assert up[6] == pytest.approx(9.0)  # admin-1 units roll into their country
    assert up[7] == pytest.approx(3.0) and up[60] == pytest.approx(4.0)
    assert str(bh.rollup(blend, hier)["location_id"].dtype) == "int64"


def test_shipped_ratios_reads_int_keys(tmp_path):
    p = tmp_path / "model.json"
    p.write_text(json.dumps({"ratios": {"4": 0.5, "31": 0.0032}}))
    r = bh.shipped_ratios(p)
    assert list(r.index) == [4, 31] and r[4] == 0.5


def test_write_dtypes(tmp_path):
    df = pd.DataFrame({"storm_draw": [1], "ssp_scenario": ["historical"], "year_id": [2000],
                       "location_id": [1], "deaths": [np.float32(2.5)], "extra": ["drop me"]})
    bh._write(df, tmp_path / "x.parquet")
    back = pd.read_parquet(tmp_path / "x.parquet")
    assert list(back.columns) == list(bh.DEATHS_DTYPES)
    assert str(back["location_id"].dtype) == "int64" and str(back["deaths"].dtype) == "float64"


def test_cli_help():
    from click.testing import CliRunner

    r = CliRunner().invoke(bh.main, ["--help"])
    assert r.exit_code == 0 and "--years" in r.output
