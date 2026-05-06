"""
Basin-stratified k-fold cross-validation assignment.

Public function:
    compute_fold_assignments(df, n_seeds=5, n_folds=5) -> pd.DataFrame

Returns a DataFrame of shape (len(df), n_seeds) with integer fold indices
[0, n_folds) for each seed. Fold assignment is stratified by the 'basin'
column: rows within each basin are shuffled independently and then assigned
to folds in round-robin order, so each fold contains roughly 1/n_folds of
each basin's rows.

A fold_tag "s{seed}_f{fold}" identifies a unique held-out set. The training
set for that fold is all rows where the corresponding column value != fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_fold_assignments(
    df: pd.DataFrame,
    n_seeds: int = 5,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Basin-stratified k-fold assignment for cross-validation.

    For each seed, rows within each basin are independently shuffled and then
    assigned to folds [0, n_folds) in round-robin order. Different seeds produce
    different shuffles of the same stratified structure.

    Parameters
    ----------
    df:
        DataFrame with a 'basin' column used for stratification.
        Index is preserved in the returned DataFrame.
    n_seeds:
        Number of independent random seeds (columns in the output).
    n_folds:
        Number of folds per seed.

    Returns
    -------
    pd.DataFrame
        Shape (len(df), n_seeds). Columns are named 'seed_0', 'seed_1', ...,
        'seed_{n_seeds-1}'. Values are integers in [0, n_folds).
    """
    if "basin" not in df.columns:
        raise ValueError("df must contain a 'basin' column for stratified fold assignment.")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}.")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")

    folds = np.empty((len(df), n_seeds), dtype=np.int32)
    basins = df["basin"].values
    unique_basins = np.unique(basins)

    for seed_idx in range(n_seeds):
        rng = np.random.default_rng(seed_idx)
        col = np.empty(len(df), dtype=np.int32)
        for basin in unique_basins:
            basin_positions = np.where(basins == basin)[0]
            shuffled = rng.permutation(len(basin_positions))
            col[basin_positions] = shuffled % n_folds
        folds[:, seed_idx] = col

    columns = [f"seed_{i}" for i in range(n_seeds)]
    return pd.DataFrame(folds, index=df.index, columns=columns)
