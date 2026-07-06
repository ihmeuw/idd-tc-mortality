"""Basin-level aggregation: concat 100 tc_draw predict outputs, mean by (location_id, year).

Sum is divided by 100 always (even when fewer tc_draws are present), so missing
tc_draws contribute zero — the right behaviour when a tc_draw represents a
stochastic storm-season rollout in which the basin produced no exposure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from idd_tc_mortality.predict.done_manifest import (
    TIER_PREDICT_BASIN,
    append_done,
)
from idd_tc_mortality.predict.paths import (
    STORM_DRAW_DIR,
    atomic_write_parquet,
    basin_folder,
    basin_mean_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIVIDE_BY = 100  # always /100, even when fewer tc_draw parquets exist


def aggregate_basin(
    *,
    storm_draw: int,
    scenario: str,
    year_bin: str,
    basin: str,
    storm_draw_dir: Path,
) -> Path:
    out_path = basin_mean_path(storm_draw_dir, storm_draw, scenario, year_bin, basin)
    if out_path.exists():
        logger.info("basin-mean already present at %s — skipping.", out_path)
        return out_path

    in_dir = basin_folder(storm_draw_dir, storm_draw, scenario, year_bin, basin)
    files = sorted(
        p for p in in_dir.glob('admin_level_exposure_deaths*.parquet')
        if p.name != 'admin_level_exposure_deaths_mean.parquet'
    )
    if not files:
        raise FileNotFoundError(f"No predict outputs in {in_dir}")

    frames = [pd.read_parquet(p) for p in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        logger.warning("All %d predict outputs empty in %s — writing empty mean.", len(files), in_dir)
        atomic_write_parquet(
            pd.DataFrame(columns=['location_id', 'year', 'basin', 'exposed_mean', 'total_population']),
            out_path,
        )
        append_done(storm_draw_dir, TIER_PREDICT_BASIN,
                    storm_draw=storm_draw, scenario=scenario,
                    year_bin=year_bin, basin=basin)
        return out_path

    combined = pd.concat(frames, axis=0, ignore_index=True)
    deaths_cols = [c for c in combined.columns if c.startswith('deaths_')]

    grouped = combined.groupby(['location_id', 'year'], as_index=False).agg(
        **{c: (c, 'sum') for c in deaths_cols},
        exposed_sum     =('exposed',           'sum'),
        total_population=('total_population', 'first'),
        basin           =('basin',             'first'),
    )
    grouped[deaths_cols]   = grouped[deaths_cols] / DIVIDE_BY
    grouped['exposed_mean'] = grouped.pop('exposed_sum') / DIVIDE_BY

    out_cols = ['location_id', 'year', 'basin', 'exposed_mean', 'total_population'] + deaths_cols
    atomic_write_parquet(grouped[out_cols], out_path)
    append_done(storm_draw_dir, TIER_PREDICT_BASIN,
                storm_draw=storm_draw, scenario=scenario,
                year_bin=year_bin, basin=basin)
    return out_path


@click.command()
@click.option('--storm-draw',     type=int, required=True)
@click.option('--scenario',       type=str, required=True)
@click.option('--year-bin',       type=str, required=True)
@click.option('--basin',          type=str, required=True)
@click.option('--storm-draw-dir', type=click.Path(path_type=Path), default=str(STORM_DRAW_DIR), show_default=True)
def main(storm_draw, scenario, year_bin, basin, storm_draw_dir):
    out_path = aggregate_basin(
        storm_draw=storm_draw, scenario=scenario, year_bin=year_bin, basin=basin,
        storm_draw_dir=Path(storm_draw_dir),
    )
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
