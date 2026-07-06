"""Year-bin aggregation: concat 7 basin means, re-sum by (location_id, year).

A single admin location can be exposed to storms from multiple basins, so the
basin frames must be summed (not just concatenated) at the (location_id, year)
level. Drops the `basin` column — once basins are combined, it no longer makes
sense to label a row with a single basin.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from idd_tc_mortality.predict.done_manifest import (
    TIER_AGG_YEAR_BIN,
    append_done,
)
from idd_tc_mortality.predict.paths import (
    BASIN_LEVELS,
    STORM_DRAW_DIR,
    atomic_write_parquet,
    basin_mean_path,
    year_bin_mean_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def aggregate_year_bin(
    *,
    storm_draw: int,
    scenario: str,
    year_bin: str,
    storm_draw_dir: Path,
) -> Path:
    out_path = year_bin_mean_path(storm_draw_dir, storm_draw, scenario, year_bin)
    if out_path.exists():
        logger.info("year_bin-mean already present at %s — skipping.", out_path)
        return out_path

    missing = [
        basin for basin in BASIN_LEVELS
        if not basin_mean_path(storm_draw_dir, storm_draw, scenario, year_bin, basin).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"sd={storm_draw} {scenario} {year_bin}: basin-mean missing for "
            f"{missing}. Upstream predict_basin task(s) reported success but did "
            "not produce the expected output — fail loudly."
        )

    frames = []
    for basin in BASIN_LEVELS:
        p = basin_mean_path(storm_draw_dir, storm_draw, scenario, year_bin, basin)
        df = pd.read_parquet(p)
        if not df.empty:
            frames.append(df)
    logger.info("sd=%d %s %s: %d/%d basin means present, %d non-empty.",
                storm_draw, scenario, year_bin, len(BASIN_LEVELS), len(BASIN_LEVELS), len(frames))

    if not frames:
        atomic_write_parquet(
            pd.DataFrame(columns=['location_id', 'year', 'exposed_mean', 'total_population']),
            out_path,
        )
        logger.warning("All basin means empty for %s/%s/%s — wrote empty.",
                       storm_draw, scenario, year_bin)
        append_done(storm_draw_dir, TIER_AGG_YEAR_BIN,
                    storm_draw=storm_draw, scenario=scenario, year_bin=year_bin)
        return out_path

    combined = pd.concat(frames, axis=0, ignore_index=True)
    deaths_cols = [c for c in combined.columns if c.startswith('deaths_')]

    grouped = combined.groupby(['location_id', 'year'], as_index=False).agg(
        **{c: (c, 'sum') for c in deaths_cols},
        exposed_mean    =('exposed_mean',     'sum'),
        total_population=('total_population', 'first'),
    )

    out_cols = ['location_id', 'year', 'exposed_mean', 'total_population'] + deaths_cols
    atomic_write_parquet(grouped[out_cols], out_path)
    append_done(storm_draw_dir, TIER_AGG_YEAR_BIN,
                storm_draw=storm_draw, scenario=scenario, year_bin=year_bin)
    return out_path


@click.command()
@click.option('--storm-draw',     type=int, required=True)
@click.option('--scenario',       type=str, required=True)
@click.option('--year-bin',       type=str, required=True)
@click.option('--storm-draw-dir', type=click.Path(path_type=Path), default=str(STORM_DRAW_DIR), show_default=True)
def main(storm_draw, scenario, year_bin, storm_draw_dir):
    out_path = aggregate_year_bin(
        storm_draw=storm_draw, scenario=scenario, year_bin=year_bin,
        storm_draw_dir=Path(storm_draw_dir),
    )
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
