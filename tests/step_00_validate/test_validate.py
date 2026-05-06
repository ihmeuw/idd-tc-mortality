"""
Tests for step_00_validate.validate.

Covers:
  - check_basins counts canonical values correctly.
  - check_basins counts NaN as '<NaN>' in noncanonical.
  - check_basins counts empty string in noncanonical.
  - check_basins counts unknown codes in noncanonical.
  - check_basins returns empty noncanonical when all basins are clean.
  - check_basins never modifies the input DataFrame.
  - recode_noncanonical_basins replaces NaN with 'NA'.
  - recode_noncanonical_basins replaces '' with 'NA'.
  - recode_noncanonical_basins replaces unknown codes with 'NA'.
  - recode_noncanonical_basins does not touch canonical rows.
  - recode_noncanonical_basins returns a new DataFrame (does not modify input).
  - recode_noncanonical_basins is a no-op on all-canonical data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.constants import BASIN_LEVELS
from idd_tc_mortality.step_00_validate.validate import check_basins, recode_noncanonical_basins


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(basins: list) -> pd.DataFrame:
    """Minimal DataFrame with just a basin column."""
    return pd.DataFrame({"basin": basins, "value": range(len(basins))})


# ---------------------------------------------------------------------------
# check_basins
# ---------------------------------------------------------------------------

def test_check_basins_canonical_counts():
    df = _make_df(["EP", "EP", "NA", "WP"])
    result = check_basins(df)
    assert result["canonical"]["EP"] == 2
    assert result["canonical"]["NA"] == 1
    assert result["canonical"]["WP"] == 1


def test_check_basins_nan_in_noncanonical():
    df = _make_df(["EP", np.nan, np.nan])
    result = check_basins(df)
    assert "<NaN>" in result["noncanonical"]
    assert result["noncanonical"]["<NaN>"] == 2


def test_check_basins_empty_string_in_noncanonical():
    df = _make_df(["EP", "", ""])
    result = check_basins(df)
    assert "" in result["noncanonical"]
    assert result["noncanonical"][""] == 2


def test_check_basins_unknown_code_in_noncanonical():
    df = _make_df(["EP", "XX", "ZZ"])
    result = check_basins(df)
    assert "XX" in result["noncanonical"]
    assert "ZZ" in result["noncanonical"]
    assert result["noncanonical"]["XX"] == 1
    assert result["noncanonical"]["ZZ"] == 1


def test_check_basins_mixed_canonical_and_noncanonical():
    df = _make_df(["EP", "NA", np.nan, "", "UNKNOWN", "WP"])
    result = check_basins(df)
    assert result["canonical"]["EP"] == 1
    assert result["canonical"]["NA"] == 1
    assert result["canonical"]["WP"] == 1
    assert result["noncanonical"]["<NaN>"] == 1
    assert result["noncanonical"][""] == 1
    assert result["noncanonical"]["UNKNOWN"] == 1


def test_check_basins_empty_noncanonical_when_all_clean():
    df = _make_df(["EP", "NA", "WP", "SI", "SP", "NI", "SA"])
    result = check_basins(df)
    assert result["noncanonical"] == {}
    assert len(result["canonical"]) == 7


def test_check_basins_does_not_modify_input():
    df = _make_df(["EP", np.nan, "XX"])
    original_basin = df["basin"].tolist()
    check_basins(df)
    assert df["basin"].tolist() == original_basin


def test_check_basins_all_levels_present():
    """All BASIN_LEVELS can appear and are all counted as canonical."""
    df = _make_df(BASIN_LEVELS)
    result = check_basins(df)
    for level in BASIN_LEVELS:
        assert level in result["canonical"]
        assert result["canonical"][level] == 1
    assert result["noncanonical"] == {}


# ---------------------------------------------------------------------------
# recode_noncanonical_basins
# ---------------------------------------------------------------------------

def test_recode_replaces_nan_with_na():
    df = _make_df(["EP", np.nan, np.nan])
    result = recode_noncanonical_basins(df)
    assert result["basin"].tolist() == ["EP", "NA", "NA"]


def test_recode_replaces_empty_string_with_na():
    df = _make_df(["EP", "", "WP", ""])
    result = recode_noncanonical_basins(df)
    assert result["basin"].tolist() == ["EP", "NA", "WP", "NA"]


def test_recode_replaces_unknown_code_with_na():
    df = _make_df(["EP", "XX", "UNKNOWN"])
    result = recode_noncanonical_basins(df)
    assert result["basin"].tolist() == ["EP", "NA", "NA"]


def test_recode_replaces_all_noncanonical_types():
    """NaN, '', and unknown code all replaced in a single call."""
    df = _make_df(["EP", np.nan, "", "FOO", "NA"])
    result = recode_noncanonical_basins(df)
    assert result["basin"].tolist() == ["EP", "NA", "NA", "NA", "NA"]


def test_recode_does_not_touch_canonical_rows():
    """No canonical row is modified."""
    df = _make_df(BASIN_LEVELS + ["XX"])
    result = recode_noncanonical_basins(df)
    for i, level in enumerate(BASIN_LEVELS):
        assert result.iloc[i]["basin"] == level
    assert result.iloc[-1]["basin"] == "NA"


def test_recode_returns_new_dataframe():
    """Input DataFrame must not be modified."""
    df = _make_df(["EP", np.nan, "XX"])
    original_basin = df["basin"].copy()
    result = recode_noncanonical_basins(df)
    # result is a different object
    assert result is not df
    # input basin column is unchanged
    pd.testing.assert_series_equal(df["basin"], original_basin)


def test_recode_noop_on_all_canonical():
    """All-canonical input: returned DataFrame equals original (basin unchanged)."""
    df = _make_df(BASIN_LEVELS)
    result = recode_noncanonical_basins(df)
    pd.testing.assert_series_equal(result["basin"].reset_index(drop=True),
                                   df["basin"].reset_index(drop=True))


def test_recode_preserves_other_columns():
    """Non-basin columns are unchanged after recoding."""
    df = pd.DataFrame({
        "basin": ["EP", np.nan, "XX"],
        "death_rate": [0.001, 0.002, 0.003],
        "exposed": [100_000, 200_000, 300_000],
    })
    result = recode_noncanonical_basins(df)
    pd.testing.assert_series_equal(result["death_rate"], df["death_rate"])
    pd.testing.assert_series_equal(result["exposed"], df["exposed"])
