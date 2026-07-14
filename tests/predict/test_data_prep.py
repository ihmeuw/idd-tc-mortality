"""
Tests for predict.data_prep.load_sdi_df year-bin boundary handling.

The SDI source has two files joined at a hard boundary:
  - past SDI:   year_id 1980-2021
  - future SDI: year_id 2022-2100  (has a `scenario` dim that gets squeezed)

The branching logic in load_sdi_df must pick the right combination for any
year_bin. Regression coverage targets the bug found 2026-05-16 where bins
with year_start in {2020, 2021} and year_end >= 2022 fell into the
future-only branch and silently dropped years 2020 and 2021 from the
returned frame.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from idd_tc_mortality.predict.data_prep import load_sdi_df


PAST_SDI_VALUE   = 0.5
FUTURE_SDI_VALUE = 0.7


@pytest.fixture
def sdi_paths(tmp_path):
    """Write tiny past + future SDI netCDFs matching the production layout."""
    past_years   = np.arange(1980, 2022)
    future_years = np.arange(2022, 2101)
    locations    = np.array([1, 2, 3])
    draws        = np.array([0, 1])

    past = xr.Dataset(
        {
            "draws": (
                ("year_id", "location_id", "draw"),
                np.full((len(past_years), len(locations), len(draws)), PAST_SDI_VALUE),
            )
        },
        coords={"year_id": past_years, "location_id": locations, "draw": draws},
    )
    past_path = tmp_path / "past_sdi.nc"
    past.to_netcdf(past_path)

    future = xr.Dataset(
        {
            "draws": (
                ("year_id", "location_id", "draw", "scenario"),
                np.full((len(future_years), len(locations), len(draws), 1), FUTURE_SDI_VALUE),
            )
        },
        coords={
            "year_id":     future_years,
            "location_id": locations,
            "draw":        draws,
            "scenario":    ["ssp245"],
        },
    )
    future_path = tmp_path / "future_sdi.nc"
    future.to_netcdf(future_path)

    return str(past_path), str(future_path)


def test_year_bin_past_only(sdi_paths):
    past, future = sdi_paths
    df = load_sdi_df("1990-1994", old_path=past, new_path=future)
    assert sorted(df["year"].unique().tolist()) == [1990, 1991, 1992, 1993, 1994]
    assert (df["sdi"] == PAST_SDI_VALUE).all()


def test_year_bin_ending_exactly_at_2021_uses_past_only(sdi_paths):
    past, future = sdi_paths
    df = load_sdi_df("2017-2021", old_path=past, new_path=future)
    assert sorted(df["year"].unique().tolist()) == [2017, 2018, 2019, 2020, 2021]
    assert (df["sdi"] == PAST_SDI_VALUE).all()


def test_year_bin_future_only(sdi_paths):
    past, future = sdi_paths
    df = load_sdi_df("2025-2029", old_path=past, new_path=future)
    assert sorted(df["year"].unique().tolist()) == [2025, 2026, 2027, 2028, 2029]
    assert (df["sdi"] == FUTURE_SDI_VALUE).all()


def test_year_bin_cross_boundary_2018_2022(sdi_paths):
    """Bin that straddles 2021/2022 boundary — must stitch past + future."""
    past, future = sdi_paths
    df = load_sdi_df("2018-2022", old_path=past, new_path=future)
    yrs = sorted(df["year"].unique().tolist())
    assert yrs == [2018, 2019, 2020, 2021, 2022]
    assert (df.loc[df["year"] <= 2021, "sdi"] == PAST_SDI_VALUE).all()
    assert (df.loc[df["year"] >= 2022, "sdi"] == FUTURE_SDI_VALUE).all()


def test_year_bin_2020_2023_regression(sdi_paths):
    """Regression: year_start=2020 must use stitch path (was falling to future-only)."""
    past, future = sdi_paths
    df = load_sdi_df("2020-2023", old_path=past, new_path=future)
    yrs = sorted(df["year"].unique().tolist())
    assert yrs == [2020, 2021, 2022, 2023], (
        f"bin 2020-2023 must stitch past + future to include all 4 years; got {yrs}. "
        "If 2020 and 2021 are missing, load_sdi_df's cross-boundary elif "
        "condition has regressed (must be `year_start <= 2021`, not `< 2020`)."
    )
    assert (df.loc[df["year"] <= 2021, "sdi"] == PAST_SDI_VALUE).all()
    assert (df.loc[df["year"] >= 2022, "sdi"] == FUTURE_SDI_VALUE).all()


def test_year_bin_2021_2025_regression(sdi_paths):
    """Regression: year_start=2021 must also use stitch path."""
    past, future = sdi_paths
    df = load_sdi_df("2021-2025", old_path=past, new_path=future)
    yrs = sorted(df["year"].unique().tolist())
    assert yrs == [2021, 2022, 2023, 2024, 2025]
    assert (df.loc[df["year"] == 2021,  "sdi"] == PAST_SDI_VALUE).all()
    assert (df.loc[df["year"] >= 2022,  "sdi"] == FUTURE_SDI_VALUE).all()


def test_year_bin_pre_1980_backfill(sdi_paths):
    """year_start < 1980 backfills the 1970s by duplicating 1980."""
    past, future = sdi_paths
    df = load_sdi_df("1975-1979", old_path=past, new_path=future)
    assert sorted(df["year"].unique().tolist()) == [1975, 1976, 1977, 1978, 1979]
    assert (df["sdi"] == PAST_SDI_VALUE).all()


@pytest.mark.parametrize("year_bin", ["2020-2023", "2021-2025", "2018-2022"])
def test_cross_boundary_bins_include_full_year_range(sdi_paths, year_bin):
    """Parametrized regression: any cross-boundary bin returns every year inclusive."""
    past, future = sdi_paths
    year_start, year_end = map(int, year_bin.split("-"))
    df = load_sdi_df(year_bin, old_path=past, new_path=future)
    yrs = sorted(df["year"].unique().tolist())
    expected = list(range(year_start, year_end + 1))
    assert yrs == expected, f"bin {year_bin}: missing years {set(expected) - set(yrs)}"
