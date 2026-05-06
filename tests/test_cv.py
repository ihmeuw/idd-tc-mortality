"""
Tests for idd_tc_mortality.cv — compute_fold_assignments.

Covers:
  - Output shape and column names.
  - Values in [0, n_folds).
  - Basin stratification: each fold contains rows from every basin present.
  - Approximate balance: max fold size - min fold size <= 1 within each basin.
  - Different seeds produce different assignments (statistically almost certain
    for any non-trivial dataset).
  - Missing 'basin' column raises ValueError.
  - n_seeds < 1 raises ValueError.
  - n_folds < 2 raises ValueError.
  - Index from df is preserved in the output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.cv import compute_fold_assignments


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_multi_basin():
    """DataFrame with 3 basins and enough rows to test stratification."""
    rng = np.random.default_rng(0)
    n = 90  # 30 per basin → divides evenly into 5 folds
    basins = ["EP"] * 30 + ["NA"] * 30 + ["WP"] * 30
    return pd.DataFrame({
        "basin":   basins,
        "deaths":  rng.integers(0, 10, n),
        "exposed": rng.uniform(1e4, 1e6, n),
    })


@pytest.fixture
def df_unequal_basins():
    """DataFrame with basins of unequal size."""
    rng = np.random.default_rng(1)
    n = 70
    basins = ["EP"] * 10 + ["NA"] * 40 + ["WP"] * 20
    return pd.DataFrame({
        "basin":   basins,
        "deaths":  rng.integers(0, 5, n),
        "exposed": np.ones(n) * 1e5,
    })


# ---------------------------------------------------------------------------
# Shape and column names
# ---------------------------------------------------------------------------

def test_output_shape(df_multi_basin):
    result = compute_fold_assignments(df_multi_basin, n_seeds=5, n_folds=5)
    assert result.shape == (len(df_multi_basin), 5)


def test_column_names(df_multi_basin):
    result = compute_fold_assignments(df_multi_basin, n_seeds=3, n_folds=4)
    assert list(result.columns) == ["seed_0", "seed_1", "seed_2"]


def test_custom_n_seeds_n_folds(df_multi_basin):
    result = compute_fold_assignments(df_multi_basin, n_seeds=2, n_folds=3)
    assert result.shape == (len(df_multi_basin), 2)
    assert result.values.min() == 0
    assert result.values.max() == 2


# ---------------------------------------------------------------------------
# Value range
# ---------------------------------------------------------------------------

def test_values_in_range(df_multi_basin):
    n_folds = 5
    result = compute_fold_assignments(df_multi_basin, n_seeds=5, n_folds=n_folds)
    assert int(result.values.min()) >= 0
    assert int(result.values.max()) < n_folds


def test_all_folds_represented_in_each_seed(df_multi_basin):
    n_folds = 5
    result = compute_fold_assignments(df_multi_basin, n_seeds=3, n_folds=n_folds)
    for col in result.columns:
        assert set(result[col].unique()) == set(range(n_folds)), (
            f"Column {col} does not contain all folds 0..{n_folds - 1}."
        )


# ---------------------------------------------------------------------------
# Basin stratification
# ---------------------------------------------------------------------------

def test_each_fold_contains_every_basin(df_multi_basin):
    """Every fold must contain rows from every basin present in df."""
    n_folds = 5
    result = compute_fold_assignments(df_multi_basin, n_seeds=1, n_folds=n_folds)
    basins = df_multi_basin["basin"].values
    col = result["seed_0"]
    for basin in np.unique(basins):
        basin_folds = set(col[basins == basin].unique())
        assert basin_folds == set(range(n_folds)), (
            f"Basin {basin!r} not present in all folds. Found folds: {basin_folds}"
        )


def test_fold_balance_within_basin(df_multi_basin):
    """Within each basin, fold sizes should differ by at most 1."""
    n_folds = 5
    result = compute_fold_assignments(df_multi_basin, n_seeds=1, n_folds=n_folds)
    basins = df_multi_basin["basin"].values
    col = result["seed_0"].values
    for basin in np.unique(basins):
        basin_col = col[basins == basin]
        sizes = [int((basin_col == f).sum()) for f in range(n_folds)]
        assert max(sizes) - min(sizes) <= 1, (
            f"Basin {basin!r}: fold sizes {sizes} differ by more than 1."
        )


def test_fold_balance_unequal_basins(df_unequal_basins):
    """Balance check holds for basins with non-multiples of n_folds."""
    n_folds = 5
    result = compute_fold_assignments(df_unequal_basins, n_seeds=1, n_folds=n_folds)
    basins = df_unequal_basins["basin"].values
    col = result["seed_0"].values
    for basin in np.unique(basins):
        basin_col = col[basins == basin]
        sizes = [int((basin_col == f).sum()) for f in range(n_folds)]
        assert max(sizes) - min(sizes) <= 1, (
            f"Basin {basin!r}: fold sizes {sizes} differ by more than 1."
        )


# ---------------------------------------------------------------------------
# Seed independence
# ---------------------------------------------------------------------------

def test_different_seeds_produce_different_assignments(df_multi_basin):
    """At least two seed columns must differ (true for any non-trivial dataset)."""
    result = compute_fold_assignments(df_multi_basin, n_seeds=5, n_folds=5)
    cols = [result[f"seed_{i}"].values for i in range(5)]
    # At least one pair of seeds must differ.
    any_differ = any(
        not np.array_equal(cols[i], cols[j])
        for i in range(5)
        for j in range(i + 1, 5)
    )
    assert any_differ, "All seeds produced identical assignments — seeds are not independent."


def test_deterministic(df_multi_basin):
    """Same df and same n_seeds/n_folds always produces the same result."""
    r1 = compute_fold_assignments(df_multi_basin, n_seeds=3, n_folds=5)
    r2 = compute_fold_assignments(df_multi_basin, n_seeds=3, n_folds=5)
    pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Index preservation
# ---------------------------------------------------------------------------

def test_index_preserved():
    """Output index matches the input df index."""
    df = pd.DataFrame(
        {"basin": ["EP", "NA", "WP"] * 10},
        index=pd.RangeIndex(start=100, stop=130),
    )
    result = compute_fold_assignments(df, n_seeds=2, n_folds=3)
    assert list(result.index) == list(df.index)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_missing_basin_column_raises():
    df = pd.DataFrame({"deaths": [1, 2, 3]})
    with pytest.raises(ValueError, match="basin"):
        compute_fold_assignments(df)


def test_n_seeds_lt_1_raises(df_multi_basin):
    with pytest.raises(ValueError, match="n_seeds"):
        compute_fold_assignments(df_multi_basin, n_seeds=0)


def test_n_folds_lt_2_raises(df_multi_basin):
    with pytest.raises(ValueError, match="n_folds"):
        compute_fold_assignments(df_multi_basin, n_folds=1)
