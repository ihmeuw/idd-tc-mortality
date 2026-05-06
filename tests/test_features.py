"""
Tests for features.build_X and features.align_X.

Covers:
  - build_X produces correct columns for every combination of covariate flags.
  - include_log_exposed=True adds log_exposed as the last column.
  - include_log_exposed=False never adds log_exposed, regardless of other flags.
  - Basin dummies: EP is always the dropped reference level (from constants.BASIN_REFERENCE),
    regardless of which basins are present in the data.
  - Column order is deterministic: const | wind_speed | sdi | basin dummies
    | is_island | log_exposed.
  - align_X adds missing basin columns as zeros.
  - align_X reorders columns to match param_names.
  - align_X raises ValueError if const is missing from X or param_names.
  - Values: const=1.0, log_exposed = log(exposed), basin dummies 0/1.

Fixture
-------
DataFrame with wind_speed, sdi, basin (NA/WP/SP — three real TC basin codes,
EP deliberately absent to test that EP dummy is still dropped), is_island, exposed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.constants import BASIN_LEVELS, BASIN_REFERENCE
from idd_tc_mortality.features import align_X, build_X

ALL_COVS = {"wind_speed": True, "sdi": True, "basin": True, "is_island": True}
NO_COVS = {"wind_speed": False, "sdi": False, "basin": False, "is_island": False}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """10-row DataFrame with three real TC basin codes (NA, WP, SP); EP absent."""
    rng = np.random.default_rng(0)
    n = 10
    return pd.DataFrame({
        "wind_speed": rng.normal(50, 10, n),
        "sdi":        rng.uniform(0, 1, n),
        "basin":      (["NA"] * 4) + (["WP"] * 3) + (["SP"] * 3),
        "is_island":  rng.integers(0, 2, n).astype(float),
        "exposed":    rng.uniform(10_000, 1_000_000, n),
    })


# ---------------------------------------------------------------------------
# const column
# ---------------------------------------------------------------------------

def test_const_always_present_no_covariates(sample_df):
    X = build_X(sample_df, NO_COVS, include_log_exposed=False)
    assert "const" in X.columns
    assert np.all(X["const"] == 1.0)


def test_const_always_present_all_covariates(sample_df):
    X = build_X(sample_df, ALL_COVS, include_log_exposed=True)
    assert "const" in X.columns
    assert np.all(X["const"] == 1.0)


def test_const_is_first_column(sample_df):
    X = build_X(sample_df, ALL_COVS, include_log_exposed=True)
    assert X.columns[0] == "const"


# ---------------------------------------------------------------------------
# include_log_exposed
# ---------------------------------------------------------------------------

def test_log_exposed_present_when_requested(sample_df):
    X = build_X(sample_df, NO_COVS, include_log_exposed=True)
    assert "log_exposed" in X.columns


def test_log_exposed_is_last_column(sample_df):
    X = build_X(sample_df, ALL_COVS, include_log_exposed=True)
    assert X.columns[-1] == "log_exposed"


def test_log_exposed_values_correct(sample_df):
    X = build_X(sample_df, NO_COVS, include_log_exposed=True)
    np.testing.assert_allclose(X["log_exposed"].values, np.log(sample_df["exposed"].values))


def test_log_exposed_absent_when_not_requested_no_covariates(sample_df):
    X = build_X(sample_df, NO_COVS, include_log_exposed=False)
    assert "log_exposed" not in X.columns


def test_log_exposed_absent_when_not_requested_all_covariates(sample_df):
    """log_exposed must be absent even when all four covariate flags are True."""
    X = build_X(sample_df, ALL_COVS, include_log_exposed=False)
    assert "log_exposed" not in X.columns


@pytest.mark.parametrize("covs", [
    {"wind_speed": True, "sdi": False, "basin": False, "is_island": False},
    {"wind_speed": False, "sdi": True,  "basin": False, "is_island": False},
    {"wind_speed": False, "sdi": False, "basin": True,  "is_island": False},
    {"wind_speed": False, "sdi": False, "basin": False, "is_island": True},
    ALL_COVS,
    NO_COVS,
])
def test_log_exposed_never_present_when_false(sample_df, covs):
    """log_exposed absent for every covariate combination when include_log_exposed=False."""
    X = build_X(sample_df, covs, include_log_exposed=False)
    assert "log_exposed" not in X.columns, (
        f"log_exposed found in X.columns={list(X.columns)} with covs={covs} "
        "and include_log_exposed=False. The binomial_cloglog validator "
        "relies on this column being absent for S1/S2."
    )


# ---------------------------------------------------------------------------
# Individual covariate flags
# ---------------------------------------------------------------------------

def test_wind_speed_included_when_flagged(sample_df):
    X = build_X(sample_df, {"wind_speed": True}, include_log_exposed=False)
    assert "wind_speed" in X.columns
    np.testing.assert_allclose(X["wind_speed"].values, sample_df["wind_speed"].values)


def test_wind_speed_excluded_when_not_flagged(sample_df):
    X = build_X(sample_df, {"wind_speed": False}, include_log_exposed=False)
    assert "wind_speed" not in X.columns


def test_sdi_included_when_flagged(sample_df):
    X = build_X(sample_df, {"sdi": True}, include_log_exposed=False)
    assert "sdi" in X.columns


def test_sdi_excluded_when_not_flagged(sample_df):
    X = build_X(sample_df, {"sdi": False}, include_log_exposed=False)
    assert "sdi" not in X.columns


def test_is_island_included_when_flagged(sample_df):
    X = build_X(sample_df, {"is_island": True}, include_log_exposed=False)
    assert "is_island" in X.columns


def test_is_island_excluded_when_not_flagged(sample_df):
    X = build_X(sample_df, {"is_island": False}, include_log_exposed=False)
    assert "is_island" not in X.columns


# ---------------------------------------------------------------------------
# Basin dummies
# ---------------------------------------------------------------------------

def test_basin_dummies_present_when_flagged(sample_df):
    X = build_X(sample_df, {"basin": True}, include_log_exposed=False)
    basin_cols = [c for c in X.columns if c.startswith("basin_")]
    assert len(basin_cols) > 0


def test_basin_reference_level_dropped(sample_df):
    """BASIN_REFERENCE (EP) must be absent from the dummies regardless of whether
    EP appears in the data. The fixture has no EP rows — EP dummy must still be dropped."""
    X = build_X(sample_df, {"basin": True}, include_log_exposed=False)
    assert f"basin_{BASIN_REFERENCE}" not in X.columns, (
        f"basin_{BASIN_REFERENCE} should always be the dropped reference level"
    )


def test_basin_all_non_reference_levels_present(sample_df):
    """All BASIN_LEVELS except BASIN_REFERENCE should appear as dummy columns."""
    X = build_X(sample_df, {"basin": True}, include_log_exposed=False)
    expected = {f"basin_{b}" for b in BASIN_LEVELS if b != BASIN_REFERENCE}
    actual = {c for c in X.columns if c.startswith("basin_")}
    assert actual == expected, (
        f"Expected basin dummies {sorted(expected)}, got {sorted(actual)}"
    )


def test_basin_dummy_count(sample_df):
    """Number of basin dummies = len(BASIN_LEVELS) - 1."""
    X = build_X(sample_df, {"basin": True}, include_log_exposed=False)
    basin_cols = [c for c in X.columns if c.startswith("basin_")]
    assert len(basin_cols) == len(BASIN_LEVELS) - 1


def test_basin_dummy_values_correct(sample_df):
    """basin_NA == 1 iff the original basin is 'NA'."""
    X = build_X(sample_df, {"basin": True}, include_log_exposed=False)
    expected_NA = (sample_df["basin"] == "NA").astype(float).values
    np.testing.assert_array_equal(X["basin_NA"].values, expected_NA)


def test_basin_ep_rows_produce_all_zero_dummies():
    """Rows with basin=EP (the reference level) produce all-zero dummy rows."""
    rng = np.random.default_rng(1)
    n = 5
    df = pd.DataFrame({
        "wind_speed": rng.normal(50, 10, n),
        "sdi":        rng.uniform(0, 1, n),
        "basin":      ["EP"] * n,
        "is_island":  np.zeros(n),
        "exposed":    rng.uniform(10_000, 1_000_000, n),
    })
    X = build_X(df, {"basin": True}, include_log_exposed=False)
    basin_cols = [c for c in X.columns if c.startswith("basin_")]
    assert np.all(X[basin_cols].values == 0.0), (
        "EP rows should produce all-zero dummy rows (EP is the reference level)"
    )


def test_basin_excluded_when_not_flagged(sample_df):
    X = build_X(sample_df, {"basin": False}, include_log_exposed=False)
    basin_cols = [c for c in X.columns if c.startswith("basin_")]
    assert len(basin_cols) == 0


# ---------------------------------------------------------------------------
# Column ordering
# ---------------------------------------------------------------------------

def test_column_order_all_covariates(sample_df):
    """Canonical order: const, wind_speed, sdi, basin dummies, is_island, log_exposed."""
    X = build_X(sample_df, ALL_COVS, include_log_exposed=True)
    cols = list(X.columns)
    assert cols[0] == "const"
    assert cols[-1] == "log_exposed"
    # wind_speed before sdi
    assert cols.index("wind_speed") < cols.index("sdi")
    # sdi before basin dummies
    basin_idx = [i for i, c in enumerate(cols) if c.startswith("basin_")]
    assert cols.index("sdi") < min(basin_idx)
    # basin dummies before is_island
    assert max(basin_idx) < cols.index("is_island")
    # is_island before log_exposed
    assert cols.index("is_island") < cols.index("log_exposed")


def test_column_order_is_deterministic(sample_df):
    """Two calls with the same arguments return the same column order."""
    X1 = build_X(sample_df, ALL_COVS, include_log_exposed=True)
    X2 = build_X(sample_df, ALL_COVS, include_log_exposed=True)
    assert list(X1.columns) == list(X2.columns)


def test_index_preserved(sample_df):
    df_reindexed = sample_df.copy()
    df_reindexed.index = range(100, 110)
    X = build_X(df_reindexed, ALL_COVS, include_log_exposed=True)
    assert list(X.index) == list(df_reindexed.index)


# ---------------------------------------------------------------------------
# align_X
# ---------------------------------------------------------------------------

def test_align_x_reorders_to_param_names(sample_df):
    X = build_X(sample_df, ALL_COVS, include_log_exposed=True)
    param_names = list(reversed(X.columns))
    X_aligned = align_X(X, param_names)
    assert list(X_aligned.columns) == param_names


def test_align_x_fills_missing_basin_level_with_zeros(sample_df):
    """A basin level absent from the prediction data is filled with zeros.

    The fixture has NA/WP/SP rows but no SI rows. A model fitted on data with
    SI should still produce a valid aligned matrix for this prediction set.
    """
    X = build_X(sample_df, ALL_COVS, include_log_exposed=True)
    # param_names from a hypothetical fitted model that also saw SI
    param_names = list(X.columns) + ["basin_SI"]
    X_aligned = align_X(X, param_names)
    assert "basin_SI" in X_aligned.columns
    assert np.all(X_aligned["basin_SI"] == 0.0)


def test_align_x_drops_extra_columns(sample_df):
    X = build_X(sample_df, ALL_COVS, include_log_exposed=True)
    param_names = ["const", "wind_speed"]
    X_aligned = align_X(X, param_names)
    assert list(X_aligned.columns) == param_names


def test_align_x_missing_const_in_x_raises(sample_df):
    X = build_X(sample_df, NO_COVS, include_log_exposed=False)
    X_no_const = X.drop(columns=["const"])
    with pytest.raises(ValueError, match="const"):
        align_X(X_no_const, ["const", "wind_speed"])


def test_align_x_missing_const_in_param_names_raises(sample_df):
    X = build_X(sample_df, NO_COVS, include_log_exposed=False)
    with pytest.raises(ValueError, match="const"):
        align_X(X, ["wind_speed"])


def test_align_x_values_preserved(sample_df):
    """Values in shared columns are not modified by alignment."""
    X = build_X(sample_df, {"wind_speed": True}, include_log_exposed=True)
    param_names = ["const", "wind_speed", "log_exposed"]
    X_aligned = align_X(X, param_names)
    np.testing.assert_array_equal(X_aligned["wind_speed"].values, X["wind_speed"].values)
    np.testing.assert_array_equal(X_aligned["log_exposed"].values, X["log_exposed"].values)


def test_align_x_only_const(sample_df):
    X = build_X(sample_df, ALL_COVS, include_log_exposed=True)
    X_aligned = align_X(X, ["const"])
    assert list(X_aligned.columns) == ["const"]
    assert np.all(X_aligned["const"] == 1.0)
