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
