"""Admin-frame preparation: load island_cov + sdi, merge, filter."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def load_island_cov(path: str) -> pd.DataFrame:
    """Read island_cov.parquet, dedupe by location_id, return (location_id, is_island)."""
    df = pd.read_parquet(path).drop_duplicates('location_id')
    df = df.rename(columns={'mean_value': 'is_island'})
    df['is_island'] = df['is_island'].astype(int)
    return df[['location_id', 'is_island']].copy()


def load_sdi_df(
    year_bin: str,
    location_ids: Iterable[int] | None = None,
    *,
    old_path: str,
    new_path: str,
) -> pd.DataFrame:
    """Mean-across-draws SDI for one year_bin.

    Returns columns: location_id, year, sdi. With location_ids=None (default),
    returns SDI for every location in the dataset — caller does the merge and
    NaN-filter downstream. This is the cheap path when many tc_draws inside
    one task share the same (year_bin) but have overlapping location sets:
    load once, merge N times. Pass an explicit list to restrict and emit a
    warning for missing locations.
    """
    year_start, year_end = map(int, year_bin.split('-'))

    if year_end <= 2021:
        ds = xr.open_dataset(old_path).sel(year_id=slice(year_start, max(year_end, 1980)))
        if year_start < 1980:
            # Old SDI dataset starts at 1980 — backfill 1970s by duplicating 1980.
            ds_1980  = ds.sel(year_id=1980)
            ds_1970s = ds_1980.expand_dims(
                year_id=np.arange(year_start, 1980),
            ).assign_coords(year_id=np.arange(year_start, 1980))
            ds = xr.concat([ds_1970s, ds], dim='year_id').sel(
                year_id=slice(year_start, year_end),
            )
    elif year_start <= 2021:
        # Cross-boundary year bin: stitch past + future SDI.
        # Past SDI covers 1980-2021; future SDI covers 2022-2100. Any bin whose
        # year_start is in {2020, 2021} and ends >= 2022 needs both halves —
        # the prior condition `year_start < 2020` left 2020/2021 missing from
        # the future-only branch (which slices 2022-2100), causing those years'
        # rows to be dropped downstream via NaN-sdi filtering.
        ds_old = xr.open_dataset(old_path).sel(year_id=slice(year_start, year_end))
        ds_new = xr.open_dataset(new_path).sel(
            year_id=slice(year_start, year_end),
        ).squeeze('scenario', drop=True)
        ds = xr.concat([ds_old, ds_new], dim='year_id').sortby('year_id')
    else:
        ds = xr.open_dataset(new_path).sel(
            year_id=slice(year_start, year_end),
        ).squeeze('scenario', drop=True)

    if location_ids is not None:
        available = set(int(x) for x in ds.location_id.values)
        requested = list(location_ids)
        present   = [lid for lid in requested if int(lid) in available]
        missing   = [lid for lid in requested if int(lid) not in available]
        if missing:
            logger.warning(
                "SDI dataset missing %d/%d location_ids (first 5: %s).",
                len(missing), len(requested), missing[:5],
            )
        ds = ds.sel(location_id=present)

    sdi_mean = ds['draws'].mean(dim='draw')
    return (
        sdi_mean.to_dataframe(name='sdi').reset_index()
        .rename(columns={'year_id': 'year'})
    )


def prep_admin_df(
    admin_df: pd.DataFrame,
    island_cov: pd.DataFrame,
    sdi_df: pd.DataFrame,
    basin: str,
    exposure_col: str = 'person_storm_hours',
) -> pd.DataFrame:
    """Merge covariates, rename to model-expected names, filter to predictable rows.

    Drops rows with exposed<=0 or total_population<=0 (storms with no exposure
    can't produce deaths) and rows with NaN sdi / is_island (the linear
    predictor would propagate NaN into the binomial flip and crash). Logs the
    drop counts so silent attrition is visible.

    `basin` stamps `df['basin'] = basin` — all rows get that basin. AU is a
    first-class fitted basin (basins_standard ingest), so every basin is
    scored directly with its own coefficient; no lon-split reassignment.

    `exposure_col` selects which climada column to use as the exposure measure.
    Default is 'person_storm_hours' (standard). Pass 'total_population_exposed'
    for experimental population-exposed runs.
    """
    df = admin_df.merge(island_cov, on='location_id', how='left')
    df = df.rename(columns={
        'max_wind_speed': 'wind_speed',
        exposure_col:     'exposed',
    })
    df = df.merge(sdi_df, on=['location_id', 'year'], how='left')
    df['basin'] = basin

    before = len(df)
    df = df[(df['exposed'] > 0) & (df['total_population'] > 0)].copy()
    n_zero_exposed = before - len(df)

    missing_cov = df['sdi'].isna() | df['is_island'].isna()
    if missing_cov.any():
        dropped_locs = sorted(df.loc[missing_cov, 'location_id'].unique().tolist())
        logger.warning(
            "Dropping %d/%d rows with missing sdi or is_island "
            "(zero-exposed dropped: %d). Unique location_ids dropped (%d): %s",
            int(missing_cov.sum()), len(df), n_zero_exposed,
            len(dropped_locs), dropped_locs,
        )
        df = df[~missing_cov].copy()
    return df
