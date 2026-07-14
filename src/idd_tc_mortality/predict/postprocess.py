"""Predict-pipeline postprocess: build the consumer-facing wide parquets.

Runs as the 5th-tier jobmon node after `aggregate_storm_draw` completes, or
as a standalone CLI via `run-predict-postprocess`.

Inputs:
  - <draw_base>/storm_draw_*/admin_level_exposure_deaths_mean.parquet  (one per finished sd)
  - hierarchy parquet                                                  (GBD location hierarchy)
  - FHS population parquet                                             (scenario-agnostic for now)
  - observed-deaths training parquet                                   (the file fit was run on)

Outputs (all in `<draw_base>/`):
  - `deaths_c{c}_s{s}_o{o}_b{b}_draws.parquet`   x16
      (scenario, location_id, year, mean, lower, upper, draw_00..draw_99)
  - `population.parquet`
      (location_id, year, population)
  - `observed_deaths.parquet`
      (location_id, year, deaths)

Public helpers (importable for notebooks):
  - `build_ancestor_map(hierarchy_df)`
  - `build_grid(hierarchy_df, years_by_scenario, max_level)`
  - `aggregate_storm_draws_up_hierarchy(draw_base, ancestor_map, deaths_cols)`
  - `long_to_wide_with_summary(long_df, deaths_col, grid, all_draw_cols, finished_draw_cols)`
  - `build_population(pop_path, hierarchy_df, ancestor_map, all_location_ids)`
  - `build_observed_deaths(obs_path, ancestor_map)`
  - `write_postprocess_outputs(...)` — the driver that calls all of the above.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd

from idd_tc_mortality.predict.paths import (
    STORM_DRAW_DIR,
    atomic_write_parquet,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEATHS_COLS = [
    f'deaths_c{c}_s{s}_o{o}_b{b}'
    for c in (0, 1) for s in (0, 1) for o in (0, 1) for b in (0, 1)
]
N_DRAWS       = 100
ALL_DRAW_COLS = [f'draw_{i:02d}' for i in range(N_DRAWS)]

DEFAULT_YEARS_BY_SCENARIO: dict[str, np.ndarray] = {
    'historical': np.arange(1980, 2015),
    'ssp126':     np.arange(2015, 2101),
    'ssp245':     np.arange(2015, 2101),
    'ssp585':     np.arange(2015, 2101),
}

# Default external-input paths. Override via CLI; the *_PATH on POP is the
# temporary FHS reference source and should be swapped when scenario-specific
# population becomes available.
DEFAULT_HIERARCHY_PATH = '/mnt/team/idd/pub/forecast-mbp/01-raw_data/gbd/fhs_2023_modeling_hierarchy.parquet'
DEFAULT_POP_PATH       = '/mnt/team/rapidresponse/pub/tropical-storms/fhs_population_2021_all_years.parquet'
DEFAULT_OBS_PATH       = '/mnt/team/idd/pub/idd_tc_mortality/00-data/current/input.parquet'

MAX_HIERARCHY_LEVEL = 3  # levels 0..3 (Global, super-region, region, country) — no subnationals


# ---------------------------------------------------------------------------
# Hierarchy + grid
# ---------------------------------------------------------------------------

def build_ancestor_map(hierarchy_df: pd.DataFrame) -> pd.DataFrame:
    """Explode `path_to_top_parent` so each (location_id, ancestor) pair is a row.

    A subsequent merge onto a most-detailed frame fans each input row out into
    one row per ancestor (including the input location itself), enabling the
    standard sum-aggregate-up pattern.
    """
    return (
        hierarchy_df[['location_id', 'path_to_top_parent']]
        .assign(ancestor=lambda d: d['path_to_top_parent'].str.split(','))
        .explode('ancestor')[['location_id', 'ancestor']]
        .astype({'ancestor': int})
    )


def build_grid(
    hierarchy_df: pd.DataFrame,
    years_by_scenario: dict[str, np.ndarray] = DEFAULT_YEARS_BY_SCENARIO,
    max_level: int = MAX_HIERARCHY_LEVEL,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (all_location_ids, grid_df).

    grid_df has one row per (scenario, location_id, year) at every level up to
    `max_level`. Years per scenario are scenario-specific.
    """
    all_locs = np.sort(
        hierarchy_df.loc[hierarchy_df['level'] <= max_level, 'location_id'].unique()
    )
    parts: list[pd.DataFrame] = []
    for s, yrs in years_by_scenario.items():
        idx = pd.MultiIndex.from_product(
            [[s], all_locs, yrs],
            names=['scenario', 'location_id', 'year'],
        )
        parts.append(idx.to_frame(index=False))
    return all_locs, pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-draw aggregation up the hierarchy
# ---------------------------------------------------------------------------

def aggregate_storm_draws_up_hierarchy(
    draw_base: Path,
    ancestor_map: pd.DataFrame,
    deaths_cols: list[str] = DEATHS_COLS,
) -> tuple[pd.DataFrame, list[str]]:
    """Load every finished storm_draw's mean parquet, aggregate up the hierarchy.

    Returns (long_df, finished_draw_cols):
        long_df has columns (scenario, location_id, year, draw, <deaths_cols>...).
        finished_draw_cols is the list of `draw_NN` names for storm_draws that
        produced output (e.g. 87 of the 100 expected).
    """
    frames: list[pd.DataFrame] = []
    finished: list[str] = []
    valid_locs = set(ancestor_map['location_id'].unique())

    for d in sorted(p for p in Path(draw_base).iterdir()
                    if p.is_dir() and p.name.startswith('storm_draw_')):
        mean_path = d / 'admin_level_exposure_deaths_mean.parquet'
        if not mean_path.exists():
            continue
        sd = int(d.name.split('_')[-1])
        draw_col = f'draw_{sd - 1:02d}'

        df = pd.read_parquet(
            mean_path,
            columns=['scenario', 'location_id', 'year'] + deaths_cols,
        )
        missing = set(df['location_id'].unique()) - valid_locs
        if missing:
            raise ValueError(f'storm_draw {sd}: location_ids not in hierarchy: {missing}')

        df = (
            df.merge(ancestor_map, on='location_id')
              .drop(columns='location_id')
              .rename(columns={'ancestor': 'location_id'})
              .groupby(['scenario', 'location_id', 'year'], as_index=False)[deaths_cols]
              .sum()
        )
        df['draw'] = draw_col
        frames.append(df)
        finished.append(draw_col)

    if not frames:
        raise FileNotFoundError(
            f'No storm_draw_*/admin_level_exposure_deaths_mean.parquet under {draw_base}'
        )

    long_df = pd.concat(frames, ignore_index=True)
    return long_df, finished


# ---------------------------------------------------------------------------
# Wide reshape + summary stats
# ---------------------------------------------------------------------------

def long_to_wide_with_summary(
    long_df: pd.DataFrame,
    deaths_col: str,
    grid: pd.DataFrame,
    all_draw_cols: list[str] = ALL_DRAW_COLS,
    finished_draw_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Pivot one deaths_col into draw_00..draw_99, expand to grid, summarize.

    Returns a DataFrame with columns:
        scenario, location_id, year, mean, lower, upper, draw_00..draw_99.

    For storm_draws that ran (`finished_draw_cols`), NaN cells after the grid
    join are filled with 0 (interpreting "ran but no contribution" as zero
    deaths). For storm_draws that didn't run, those columns stay NaN; mean /
    lower / upper are computed across non-NaN draws via nanmean / nanpercentile.
    """
    if finished_draw_cols is None:
        finished_draw_cols = list(all_draw_cols)

    wide = (
        long_df.pivot(
            index=['scenario', 'location_id', 'year'],
            columns='draw',
            values=deaths_col,
        )
        .reindex(columns=all_draw_cols)
        .reset_index()
    )
    wide.columns.name = None
    wide = grid.merge(wide, on=['scenario', 'location_id', 'year'], how='left')
    wide[finished_draw_cols] = wide[finished_draw_cols].fillna(0.0)

    arr = wide[all_draw_cols].to_numpy()
    wide['mean']  = np.nanmean(arr, axis=1)
    wide['lower'] = np.nanpercentile(arr, 2.5,  axis=1)
    wide['upper'] = np.nanpercentile(arr, 97.5, axis=1)
    return wide[['scenario', 'location_id', 'year', 'mean', 'lower', 'upper'] + all_draw_cols]


# ---------------------------------------------------------------------------
# Population (scenario-agnostic for now)
# ---------------------------------------------------------------------------

def build_population(
    pop_path: str,
    hierarchy_df: pd.DataFrame,
    ancestor_map: pd.DataFrame,
    all_location_ids: np.ndarray,
) -> pd.DataFrame:
    """Load FHS population, filter to all-age + both-sex, aggregate level-3 country pops up.

    The source file has uneven coverage by hierarchy level (levels 0/1/2 past-only,
    level 4 future-only, level 3 1970-2100). Aggregating from level 3 only gives
    consistent 1970-2100 coverage at every requested level.
    """
    pop_raw = pd.read_parquet(pop_path)
    if 'age_group_id' in pop_raw.columns:
        pop_raw = pop_raw[pop_raw['age_group_id'] == 22]
    if 'sex_id' in pop_raw.columns:
        pop_raw = pop_raw[pop_raw['sex_id'] == 3]
    pop_raw = pop_raw.rename(columns={'year_id': 'year'})[['location_id', 'year', 'population']]

    country_pop = (
        pop_raw
        .merge(hierarchy_df[['location_id', 'level']], on='location_id')
        .query('level == 3')
        .drop(columns='level')
    )
    return (
        country_pop
        .merge(ancestor_map, on='location_id')
        .drop(columns='location_id')
        .rename(columns={'ancestor': 'location_id'})
        .groupby(['location_id', 'year'], as_index=False)['population'].sum()
        .query('location_id in @all_location_ids')
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Observed deaths (training data)
# ---------------------------------------------------------------------------

def aggregate_observed_deaths(obs_raw: pd.DataFrame,
                              ancestor_map: pd.DataFrame) -> pd.DataFrame:
    """Collapse the storm dim and roll observed deaths up the hierarchy.

    ``obs_raw`` needs (location_id, year, deaths) — one row per
    (location, year, storm). Sums storms within (location, year), then
    ancestor-aggregates to roll up to region / super-region / Global. Output is
    (location_id, year, deaths).
    """
    return (
        obs_raw[['location_id', 'year', 'deaths']]
        .groupby(['location_id', 'year'], as_index=False)['deaths'].sum()
        .merge(ancestor_map, on='location_id')
        .drop(columns='location_id')
        .rename(columns={'ancestor': 'location_id'})
        .groupby(['location_id', 'year'], as_index=False)['deaths'].sum()
    )


def build_observed_deaths(obs_path: str, ancestor_map: pd.DataFrame) -> pd.DataFrame:
    """Load training-data observed deaths (input.parquet) and aggregate up.

    Thin reader over :func:`aggregate_observed_deaths`; the parquet is the model's
    filtered training data, so its year range is whatever the ingest produced
    (the current vintage is 2000-2023).
    """
    obs_raw = pd.read_parquet(obs_path, columns=['location_id', 'year', 'deaths'])
    return aggregate_observed_deaths(obs_raw, ancestor_map)


# Row filters that define a modeling observation, mirroring
# scripts/ingest/01_prepare_input_data.py (kept in sync by hand; see that script).
_OBS_MAX_YEAR = 2023      # exclude 2024+ (incomplete data)
_OBS_MIN_EXPOSED = 1      # drop zero/missing exposure


def filter_source_observations(raw: pd.DataFrame,
                               min_year: int | None = None) -> pd.DataFrame:
    """Apply the ingest's row filters to a raw ibtracs+deaths frame.

    Mirrors ``01_prepare_input_data.py``: year<=2023, person_storm_hours>=1,
    low_exposure_flag!=1, plus an optional ``min_year`` floor (None = no floor,
    i.e. the full source history). Returns (location_id, year, deaths) with an
    integer location_id, ready for :func:`aggregate_observed_deaths`.
    """
    mask = (
        (raw['year'] <= _OBS_MAX_YEAR)
        & (raw['person_storm_hours'] >= _OBS_MIN_EXPOSED)
        & (raw['low_exposure_flag'] != 1)
    )
    if min_year is not None:
        mask &= raw['year'] >= min_year
    out = raw.loc[mask].rename(columns={'total_deaths': 'deaths'})[
        ['location_id', 'year', 'deaths']].copy()
    out['location_id'] = out['location_id'].astype(int)
    return out


def build_observed_deaths_from_source_csv(source_csv: str, ancestor_map: pd.DataFrame,
                                          min_year: int | None = None) -> pd.DataFrame:
    """Full-history observed deaths straight from the raw ibtracs+deaths CSV.

    Reads the source CSV with the ingest's NA-safe settings, applies
    :func:`filter_source_observations` (default ``min_year=None`` keeps the whole
    source history, e.g. 1980-2023), and aggregates up the hierarchy. Lets a plot
    show pre-training-window years WITHOUT rebuilding the model's input.parquet.
    """
    raw = pd.read_csv(source_csv, keep_default_na=False, na_values=[''],
                      dtype={'basins_standard': str})
    return aggregate_observed_deaths(filter_source_observations(raw, min_year),
                                     ancestor_map)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def write_postprocess_outputs(
    draw_base: Path,
    hierarchy_path: str = DEFAULT_HIERARCHY_PATH,
    pop_path: str = DEFAULT_POP_PATH,
    obs_path: str = DEFAULT_OBS_PATH,
    years_by_scenario: dict[str, np.ndarray] = DEFAULT_YEARS_BY_SCENARIO,
    max_hierarchy_level: int = MAX_HIERARCHY_LEVEL,
) -> dict[str, Path]:
    """Build everything and write 18 parquets to <draw_base>/.

    Returns {name: out_path} for the 16 deaths files + population + observed.
    """
    draw_base = Path(draw_base)
    outputs: dict[str, Path] = {}

    logger.info('Loading hierarchy from %s', hierarchy_path)
    hierarchy_df = pd.read_parquet(hierarchy_path)
    ancestor_map = build_ancestor_map(hierarchy_df)
    all_locs, grid = build_grid(
        hierarchy_df, years_by_scenario=years_by_scenario, max_level=max_hierarchy_level,
    )
    logger.info('Grid: %d rows (%d locations × scenario-specific years).',
                len(grid), len(all_locs))

    logger.info('Aggregating storm_draws up the hierarchy ...')
    t0 = time.monotonic()
    long_df, finished = aggregate_storm_draws_up_hierarchy(
        draw_base, ancestor_map, DEATHS_COLS,
    )
    logger.info('Loaded + aggregated %d storm_draws in %.1fs (%d long-form rows).',
                len(finished), time.monotonic() - t0, len(long_df))

    for col in DEATHS_COLS:
        t0 = time.monotonic()
        wide = long_to_wide_with_summary(long_df, col, grid, ALL_DRAW_COLS, finished)
        out_path = draw_base / f'{col}_draws.parquet'
        atomic_write_parquet(wide, out_path)
        outputs[col] = out_path
        logger.info('%s: %d rows -> %s (%.1fs)',
                    col, len(wide), out_path.name, time.monotonic() - t0)

    logger.info('Building population ...')
    pop = build_population(pop_path, hierarchy_df, ancestor_map, all_locs)
    pop_out = draw_base / 'population.parquet'
    atomic_write_parquet(pop, pop_out)
    outputs['population'] = pop_out
    logger.info('Population: %d rows -> %s', len(pop), pop_out.name)

    logger.info('Building observed deaths ...')
    obs = build_observed_deaths(obs_path, ancestor_map)
    obs_out = draw_base / 'observed_deaths.parquet'
    atomic_write_parquet(obs, obs_out)
    outputs['observed_deaths'] = obs_out
    logger.info('Observed deaths: %d rows -> %s', len(obs), obs_out.name)

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option('--draw-base',     type=click.Path(path_type=Path), default=str(STORM_DRAW_DIR),
              show_default=True, help='Storm-draw output root.')
@click.option('--hierarchy-path', type=str, default=DEFAULT_HIERARCHY_PATH, show_default=True)
@click.option('--pop-path',      type=str, default=DEFAULT_POP_PATH, show_default=True,
              help='*** TEMPORARY *** — swap when scenario-specific population is available.')
@click.option('--obs-path',      type=str, default=DEFAULT_OBS_PATH, show_default=True)
def main(draw_base, hierarchy_path, pop_path, obs_path):
    """Build the 16 wide deaths parquets + population + observed-deaths frames."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    write_postprocess_outputs(
        draw_base=draw_base,
        hierarchy_path=hierarchy_path,
        pop_path=pop_path,
        obs_path=obs_path,
    )


if __name__ == '__main__':
    main()
