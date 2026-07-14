"""Tests for observed-deaths loading/aggregation in predict.postprocess.

Covers:
  - aggregate_observed_deaths: storm-dim collapse + hierarchy rollup.
  - filter_source_observations: the ingest-mirroring row filters + optional
    min_year floor, on a raw ibtracs+deaths-shaped frame.
"""

from __future__ import annotations

import pandas as pd

from idd_tc_mortality.predict.postprocess import (
    aggregate_observed_deaths,
    filter_source_observations,
)

# Global(1) over super-region A(100) with one country child (101).
ANCESTOR_MAP = pd.DataFrame(
    {
        "location_id": [1, 100, 101, 100, 101, 101],
        "ancestor":    [1, 100, 101, 1,   1,   100],
    }
)


def test_aggregate_observed_deaths_collapses_storms_and_rolls_up():
    # Country 101, year 2000: two storms (3 + 4) collapse to 7; year 2001: 5.
    obs_raw = pd.DataFrame(
        {
            "location_id": [101, 101, 101],
            "year":        [2000, 2000, 2001],
            "deaths":      [3, 4, 5],
            "extra":       ["x", "y", "z"],  # ignored (only 3 cols used)
        }
    )
    out = aggregate_observed_deaths(obs_raw, ANCESTOR_MAP).set_index(["location_id", "year"])["deaths"]

    # 101 rolls up to itself, its super-region 100, and Global 1.
    assert out.loc[(101, 2000)] == 7
    assert out.loc[(100, 2000)] == 7
    assert out.loc[(1, 2000)] == 7
    assert out.loc[(101, 2001)] == 5
    assert out.loc[(1, 2001)] == 5


def _raw_row(year, psh, low_flag, deaths, loc=101.0):
    return {
        "location_id": loc,
        "year": year,
        "person_storm_hours": psh,
        "low_exposure_flag": low_flag,
        "total_deaths": deaths,
    }


def test_filter_source_observations_applies_ingest_filters():
    raw = pd.DataFrame([
        _raw_row(1980, 5.0, 0, 10),    # keep (early year, valid)
        _raw_row(2023, 5.0, 0, 20),    # keep (last valid year)
        _raw_row(2024, 5.0, 0, 30),    # drop: year > 2023
        _raw_row(2005, 0.0, 0, 40),    # drop: person_storm_hours < 1
        _raw_row(2005, 5.0, 1, 50),    # drop: low_exposure_flag == 1
    ])

    out = filter_source_observations(raw)  # no min_year -> full history
    assert sorted(out["year"]) == [1980, 2023]
    assert set(out.columns) == {"location_id", "year", "deaths"}
    assert out["location_id"].dtype.kind == "i"          # int, not the float input
    assert out.loc[out["year"] == 1980, "deaths"].iloc[0] == 10


def test_filter_source_observations_min_year_floor():
    raw = pd.DataFrame([
        _raw_row(1980, 5.0, 0, 10),
        _raw_row(2000, 5.0, 0, 20),
        _raw_row(2010, 5.0, 0, 30),
    ])
    out = filter_source_observations(raw, min_year=2000)
    assert sorted(out["year"]) == [2000, 2010]   # 1980 dropped by the floor
