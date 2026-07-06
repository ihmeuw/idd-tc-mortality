"""Predict-pipeline deep-dive helpers.

Functions for tracing a `(storm_draw, location_id, year, deaths_col)` anchor
through the full chain:
    wide deaths parquet (postprocess output)
      -> scenario_mean (tier 4 → 3 aggregation step)
      -> year_bin_mean (tier 2)
      -> basin_mean (tier 1)
      -> per-tc parquets (climada inputs)

Plus helpers for finding extreme cells (e.g. "what location/year has the
largest predicted deaths") and diagnostic plots.

This module is consumed by `notebooks/predict_deep_dive.ipynb` (and any
ad-hoc analysis). It deliberately exposes building blocks, not a single
fixed flow — each diagnostic question wants a slightly different chain.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Re-export load_sdi_df from data_prep so deep-dive notebooks can `from
# idd_tc_mortality.predict.diagnostics import load_sdi_df` without needing
# to know its actual home.
from idd_tc_mortality.predict.data_prep import load_sdi_df  # noqa: F401
from idd_tc_mortality.predict.paths import (
    BASIN_LEVELS,
    MEAN_FILE_NAME,
    STORM_DRAW_TABLE_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wide-frame loading + extreme-finding
# ---------------------------------------------------------------------------

def load_wide_deaths(
    deaths_col: str,
    draws_dir: Path,
    hierarchy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load `<draws_dir>/<deaths_col>_draws.parquet`; optionally join hierarchy.

    With hierarchy_df provided, adds `location_name` and `level` columns.
    """
    df = pd.read_parquet(Path(draws_dir) / f'{deaths_col}_draws.parquet')
    if hierarchy_df is not None:
        df = df.merge(
            hierarchy_df[['location_id', 'location_name', 'level']],
            on='location_id', how='left',
        )
    return df


def find_extreme_locations(
    df: pd.DataFrame,
    year_max: int | None = None,
    year_min: int | None = None,
    level: int | None = 3,
    n: int = 10,
    value_col: str = 'mean',
    ascending: bool = False,
) -> pd.DataFrame:
    """Top-N (location, year) rows by `value_col`, optionally filtered by level / year range.

    - level: GBD hierarchy level (3 = country). Pass None to include all.
    - ascending=True returns the smallest rows instead of largest.
    """
    sub = df
    if year_max is not None:
        sub = sub[sub['year'] <= year_max]
    if year_min is not None:
        sub = sub[sub['year'] >= year_min]
    if level is not None:
        if 'level' not in sub.columns:
            raise KeyError(
                "df has no 'level' column — call load_wide_deaths(..., hierarchy_df=...) "
                "or pass level=None to skip the filter."
            )
        sub = sub[sub['level'] == level]
    return sub.sort_values(value_col, ascending=ascending).head(n)


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_location_draws(
    df: pd.DataFrame,
    location_id: int,
    draw_col: str = 'draw_00',
    scenarios: list[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot year vs `draw_col` for one location, one line per scenario.

    Useful for spot-checking one storm_draw's prediction trajectory at a
    specific location (e.g. "is sd=1's Mozambique going haywire?").
    """
    sub = df[df['location_id'] == location_id]
    if sub.empty:
        raise ValueError(f"location_id={location_id} not present in df")
    if scenarios is not None:
        sub = sub[sub['scenario'].isin(scenarios)]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    for scenario, grp in sub.groupby('scenario'):
        ax.plot(grp['year'], grp[draw_col], label=scenario, lw=1.5)

    ax.set_xlabel('Year')
    ax.set_ylabel(draw_col)
    if title is None:
        loc_name = sub['location_name'].iloc[0] if 'location_name' in sub.columns else f'loc {location_id}'
        title = f'{loc_name} ({location_id}) — {draw_col}'
    ax.set_title(title)
    ax.legend(loc='best', frameon=False)
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Climada per-tc file resolution
# ---------------------------------------------------------------------------

def get_tc_draw_suffix(draw: int) -> str:
    """'' for draw 0, '_e{N-1}' for N>0 — matches climada's filename convention."""
    return '' if draw == 0 else f'_e{draw - 1}'


def get_admin_path(
    admin_df_path: Path,
    model: str,
    variant: str,
    scenario: str,
    year_bin: str,
    basin: str,
    draw: int,
) -> Path:
    """Compose the climada stage4 per-tc admin parquet path.

    Mirrors `idd_tc_mortality.predict.paths.input_admin_path` but takes the
    `admin_df_path` (the `<basin>/tc_risk_draw_N/admin_level_exposure/` dir)
    directly. Useful when you already have the directory in hand.
    """
    suffix = get_tc_draw_suffix(draw)
    year_start, year_end = map(int, year_bin.split('-'))
    return (
        Path(admin_df_path)
        / f'admin_level_exposure_{basin}_{model}_{variant}_{scenario}'
          f'_{year_start}01_{year_end}12{suffix}.parquet'
    )


# ---------------------------------------------------------------------------
# Find which basin a location appears in
# ---------------------------------------------------------------------------

def find_basin_for_location(
    location_id: int,
    storm_draw: int,
    scenario: str,
    year_bin: str,
    draw_base: Path,
    basin_levels: tuple[str, ...] = BASIN_LEVELS,
) -> list[str]:
    """Return the basin(s) whose basin_mean contains this location for (sd, sc, yb).

    Reads the per-basin `admin_level_exposure_deaths_mean.parquet` files
    (cheap; metadata-only lookup). A location can show up in multiple basins
    (e.g. Mexico in NA + EP); the return list reflects that.
    """
    found: list[str] = []
    for basin in basin_levels:
        path = (
            Path(draw_base) / f'storm_draw_{storm_draw}' / scenario / year_bin / basin
            / MEAN_FILE_NAME
        )
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=['location_id'])
        if (df['location_id'] == location_id).any():
            found.append(basin)
    return found


# ---------------------------------------------------------------------------
# Stage4 per-storm-event loading
# ---------------------------------------------------------------------------

def load_per_tc_admin_data(
    location_id: int,
    year: int,
    *,
    model: str,
    variant: str,
    scenario: str,
    year_bin: str,
    basin: str,
    climada_base: Path,
) -> pd.DataFrame:
    """Walk every `tc_risk_draw_N` for one (sd-mapped, sc, yb, basin),
    pull the (location_id, year) rows, return stacked.

    Used to inspect per-storm-event behaviour at one (location, year):
    multiple rows = multiple storms hit that (loc, year) within that tc_draw.
    """
    climada_base = Path(climada_base)
    bin_dir = climada_base / model / variant / scenario / year_bin / basin
    if not bin_dir.exists():
        raise FileNotFoundError(f"climada output dir missing: {bin_dir}")

    tc_dirs = sorted(p for p in bin_dir.iterdir() if p.is_dir() and p.name.startswith('tc_risk_draw_'))
    frames: list[pd.DataFrame] = []
    for tc_dir in tc_dirs:
        tc_draw = int(tc_dir.name.split('_')[-1])
        admin_path = get_admin_path(
            admin_df_path=tc_dir / 'admin_level_exposure',
            model=model, variant=variant, scenario=scenario,
            year_bin=year_bin, basin=basin, draw=tc_draw,
        )
        if not admin_path.exists():
            logger.debug("missing %s, skipping", admin_path)
            continue
        df = pd.read_parquet(admin_path)
        df = df[(df['location_id'] == location_id) & (df['year'] == year)]
        if df.empty:
            continue
        df = df.copy()
        df['tc_draw'] = tc_draw
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# storm_draw → (model, variant) lookup
# ---------------------------------------------------------------------------

def lookup_model_variant(
    storm_draw: int,
    storm_draw_table_path: str = STORM_DRAW_TABLE_PATH,
) -> tuple[str, str]:
    """Return (model, variant) for a given storm_draw id, from the canonical CSV."""
    tbl = pd.read_csv(storm_draw_table_path, keep_default_na=False, na_values=[''])
    row = tbl.loc[tbl['storm_draw'] == storm_draw]
    if row.empty:
        raise ValueError(f"storm_draw={storm_draw} not in {storm_draw_table_path}")
    return str(row['source_id'].iloc[0]), str(row['variant_label'].iloc[0])
