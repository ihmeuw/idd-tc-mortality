"""Path constants and path-builder helpers for the predict pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

STAGE4_DIR      = Path('/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4_v2')
TRACKS_DIR      = Path('/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6')
DIRECT_RISK_DIR = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk')
STORM_DRAW_DIR  = DIRECT_RISK_DIR / 'storm_draws'

DRAWS_DIR       = Path('/mnt/team/idd/pub/idd_tc_mortality/03-draws/20260514/topsis_winner_v1')

NEW_SDI_PATH    = '/mnt/share/forecasting/data/32/future/sdi/future_sdi_s130v66/sdi.nc'
OLD_SDI_PATH    = '/mnt/share/forecasting/data/16/past/sdi/past_sdi_s130v66/sdi.nc'
ISLAND_COV_PATH = '/mnt/team/idd/pub/idd_tc_mortality/00-data/current/is_island.parquet'
STORM_DRAW_TABLE_PATH = '/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv'
TIME_BINS_PATH  = '/mnt/team/rapidresponse/pub/tropical-storms/tempestextremes/outputs/cmip6/bayespoisson_time_bins_max_bin_5.csv'

# FHS all-age both-sex population (sex_id=3, age_group_id=22) for death-rate
# denominators — the deliverable rate uses these (they cover admin-1, unlike the
# country-and-up build_population). Past spliced with future at 2024.
FHS_PAST_POP_PATH   = '/mnt/share/forecasting/data/16/past/population/20250603_etl_run_id_417/population.nc'
FHS_FUTURE_POP_PATH = '/mnt/share/forecasting/data/32/future/population/future_population_s130v41/summary/summary.nc'

SCENARIOS    = ('historical', 'ssp126', 'ssp245', 'ssp585')
BASIN_LEVELS = ('AU', 'EP', 'NA', 'NI', 'SI', 'SP', 'WP')
SEED         = 42

MEAN_FILE_NAME = 'admin_level_exposure_deaths_mean.parquet'


def atomic_write_parquet(df: pd.DataFrame, out_path: Path, **kwargs) -> None:
    """Write df to out_path via a .tmp staging file + os.replace.

    Guarantees out_path either does not exist or is a fully-written parquet.
    A task killed mid-write leaves a .tmp file behind (which is ignored by
    skip-checks) and never a half-written final file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
    df.to_parquet(tmp_path, index=False, **kwargs)
    os.replace(tmp_path, out_path)


def lookup_year_bins(
    storm_draw: int,
    scenario: str,
    storm_draw_table_path: str = STORM_DRAW_TABLE_PATH,
    time_bins_path: str = TIME_BINS_PATH,
) -> list[str]:
    """Return the deduped year_bin strings for one (storm_draw, scenario).

    Uses the canonical CSVs — no filesystem scan. Matches orchestrate.py's
    enumeration (filter start_year >= 1970, drop bin_idx duplicates).
    """
    sd_table = pd.read_csv(storm_draw_table_path)
    row = sd_table.loc[sd_table['storm_draw'] == storm_draw]
    if row.empty:
        raise ValueError(f"storm_draw={storm_draw} not in {storm_draw_table_path}")
    model   = str(row['source_id'].values[0])
    variant = str(row['variant_label'].values[0])

    tb = pd.read_csv(time_bins_path)
    tb = tb[(tb['start_year'] >= 1970)
            & (tb['model'] == model)
            & (tb['variant'] == variant)
            & (tb['scenario'] == scenario)]
    tb = tb[['bin_idx', 'start_year', 'end_year']].drop_duplicates()

    seen: set[str] = set()
    out: list[str] = []
    for _, r in tb.iterrows():
        yb = f"{int(r['start_year'])}-{int(r['end_year'])}"
        if yb not in seen:
            out.append(yb)
            seen.add(yb)
    return out


def tc_draw_suffix(tc_draw: int) -> str:
    """Filename suffix matching climada's convention: '' for draw 0, '_e{N-1}' otherwise."""
    return '' if tc_draw == 0 else f'_e{tc_draw - 1}'


def year_bin_to_months(year_bin: str) -> tuple[int, int]:
    """'1970-1974' -> (1970, 1974)."""
    a, b = year_bin.split('-')
    return int(a), int(b)


def input_admin_path(
    stage4_dir: Path, model: str, variant: str, scenario: str,
    year_bin: str, basin: str, tc_draw: int,
) -> Path:
    year_start, year_end = year_bin_to_months(year_bin)
    suffix = tc_draw_suffix(tc_draw)
    return (
        stage4_dir / model / variant / scenario / year_bin / basin
        / f'tc_risk_draw_{tc_draw}' / 'admin_level_exposure'
        / f'admin_level_exposure_{basin}_{model}_{variant}_{scenario}'
          f'_{year_start}01_{year_end}12{suffix}.parquet'
    )


def input_track_path(
    tracks_dir: Path, model: str, variant: str, scenario: str,
    year_bin: str, basin: str, tc_draw: int,
) -> Path:
    # Track filename uses model_scenario_variant ordering — note the swap vs.
    # admin_level_exposure filenames which use model_variant_scenario. The
    # directory layout is identical.
    year_start, year_end = year_bin_to_months(year_bin)
    suffix = tc_draw_suffix(tc_draw)
    return (
        tracks_dir / model / variant / scenario / year_bin / basin
        / f'tracks_{basin}_{model}_{scenario}_{variant}'
          f'_{year_start}01_{year_end}12{suffix}.nc'
    )


def storm_draw_root(storm_draw_dir: Path, storm_draw: int) -> Path:
    return storm_draw_dir / f'storm_draw_{storm_draw}'


def basin_folder(
    storm_draw_dir: Path, storm_draw: int, scenario: str, year_bin: str, basin: str,
) -> Path:
    return storm_draw_root(storm_draw_dir, storm_draw) / scenario / year_bin / basin


def predict_output_path(
    storm_draw_dir: Path, storm_draw: int, scenario: str,
    year_bin: str, basin: str, tc_draw: int,
) -> Path:
    return basin_folder(storm_draw_dir, storm_draw, scenario, year_bin, basin) \
        / f'admin_level_exposure_deaths{tc_draw_suffix(tc_draw)}.parquet'


def basin_mean_path(
    storm_draw_dir: Path, storm_draw: int, scenario: str, year_bin: str, basin: str,
) -> Path:
    return basin_folder(storm_draw_dir, storm_draw, scenario, year_bin, basin) / MEAN_FILE_NAME


def year_bin_mean_path(
    storm_draw_dir: Path, storm_draw: int, scenario: str, year_bin: str,
) -> Path:
    return storm_draw_root(storm_draw_dir, storm_draw) / scenario / year_bin / MEAN_FILE_NAME


def scenario_mean_path(
    storm_draw_dir: Path, storm_draw: int, scenario: str,
) -> Path:
    return storm_draw_root(storm_draw_dir, storm_draw) / scenario / MEAN_FILE_NAME


def storm_draw_mean_path(storm_draw_dir: Path, storm_draw: int) -> Path:
    return storm_draw_root(storm_draw_dir, storm_draw) / MEAN_FILE_NAME
