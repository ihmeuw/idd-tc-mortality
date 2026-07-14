"""Tests for scripts/ingest/01_prepare_input_data.py.

The script filename starts with a digit, so it is loaded by path rather
than imported as a module.
"""

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "ingest" / "01_prepare_input_data.py"
)
_spec = importlib.util.spec_from_file_location("prepare_input_data", _SCRIPT)
prepare_input_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_input_data)


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})


def test_write_versioned_explicit_vintage(tmp_path, df):
    out = prepare_input_data.write_versioned(df, tmp_path, vintage="20990101_v2")
    assert out == tmp_path / "20990101_v2" / "input.parquet"
    assert out.exists()
    assert (tmp_path / "current").resolve().name == "20990101_v2"
    pd.testing.assert_frame_equal(pd.read_parquet(out), df)


def test_write_versioned_defaults_to_today(tmp_path, df):
    out = prepare_input_data.write_versioned(df, tmp_path)
    assert out.parent.name == date.today().strftime("%Y%m%d")
    assert out.exists()


def test_write_versioned_repoints_existing_current(tmp_path, df):
    prepare_input_data.write_versioned(df, tmp_path, vintage="20990101")
    prepare_input_data.write_versioned(df, tmp_path, vintage="20990102")
    assert (tmp_path / "current").resolve().name == "20990102"
    # the first vintage dir is untouched
    assert (tmp_path / "20990101" / "input.parquet").exists()


def _raw_frame() -> pd.DataFrame:
    """Minimal raw-CSV frame for prepare()'s level_filter='all' path.

    Row 0: clean. Row 1: no_wind_exposure=True. Row 2: low_exposure_flag=1
    (always dropped).
    """
    n = 3
    return pd.DataFrame({
        "year":               [2000, 2001, 2002],
        "person_storm_hours": [10.0, 20.0, 30.0],
        "low_exposure_flag":  [0, 0, 1],
        "no_wind_exposure":   [False, True, False],
        "total_deaths":       [5, 6, 7],
        "max_wind_speed":     [50.0] * n,
        "basins_standard":    ["EP"] * n,
        "sdi":                [0.5] * n,
        "is_island":          [0] * n,
        "storm_id":           [f"s{i}" for i in range(n)],
        "location_id":        [101, 102, 103],
        "location_name":      ["A", "B", "C"],
        "path_to_top_parent": ["1,2,3,101"] * n,
        "super_region_name":  ["SR"] * n,
        "region_name":        ["R"] * n,
    })


def test_prepare_keeps_no_wind_exposure_rows_by_default():
    out = prepare_input_data.prepare(_raw_frame())
    assert sorted(out["storm_id"]) == ["s0", "s1"]  # only low_exposure row dropped


def test_prepare_drop_no_wind_exposure():
    out = prepare_input_data.prepare(_raw_frame(), drop_no_wind_exposure=True)
    assert list(out["storm_id"]) == ["s0"]


def test_prepare_drop_no_wind_exposure_rejects_non_bool():
    raw = _raw_frame()
    raw["no_wind_exposure"] = raw["no_wind_exposure"].astype(object)
    with pytest.raises(ValueError, match="no_wind_exposure"):
        prepare_input_data.prepare(raw, drop_no_wind_exposure=True)
