"""Scenario aggregation: concat year_bin means for one (storm_draw, scenario).

Year bins are disjoint time partitions, so no further grouping — just stack.
Year bins are looked up from the canonical time_bins CSV (via lookup_year_bins);
no filesystem scan of the storm_draw output tree.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from idd_tc_mortality.predict.done_manifest import (
    TIER_AGG_SCENARIO,
    append_done,
)
from idd_tc_mortality.predict.paths import (
    STORM_DRAW_DIR,
    STORM_DRAW_TABLE_PATH,
    TIME_BINS_PATH,
    atomic_write_parquet,
    lookup_year_bins,
    scenario_mean_path,
    year_bin_mean_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def aggregate_scenario(
    *,
    storm_draw: int,
    scenario: str,
    storm_draw_dir: Path,
    storm_draw_table_path: str = STORM_DRAW_TABLE_PATH,
    time_bins_path: str = TIME_BINS_PATH,
) -> Path:
    out_path = scenario_mean_path(storm_draw_dir, storm_draw, scenario)
    if out_path.exists():
        logger.info("scenario-mean already present at %s — skipping.", out_path)
        return out_path

    year_bins = lookup_year_bins(
        storm_draw, scenario,
        storm_draw_table_path=storm_draw_table_path,
        time_bins_path=time_bins_path,
    )

    missing = [
        yb for yb in year_bins
        if not year_bin_mean_path(storm_draw_dir, storm_draw, scenario, yb).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"sd={storm_draw} {scenario}: year_bin-mean missing for {missing}. "
            "Upstream aggregate_year_bin task(s) reported success but did not "
            "produce the expected output — fail loudly."
        )

    frames = []
    for yb in year_bins:
        p = year_bin_mean_path(storm_draw_dir, storm_draw, scenario, yb)
        df = pd.read_parquet(p)
        if not df.empty:
            frames.append(df)
    logger.info("sd=%d %s: %d/%d year_bin means present, %d non-empty.",
                storm_draw, scenario, len(year_bins), len(year_bins), len(frames))

    if not frames:
        atomic_write_parquet(
            pd.DataFrame(columns=['location_id', 'year', 'exposed_mean', 'total_population']),
            out_path,
        )
        logger.warning("All year_bin means empty for %s/%s — wrote empty.", storm_draw, scenario)
        append_done(storm_draw_dir, TIER_AGG_SCENARIO,
                    storm_draw=storm_draw, scenario=scenario)
        return out_path

    combined = pd.concat(frames, axis=0, ignore_index=True)
    atomic_write_parquet(combined, out_path)
    append_done(storm_draw_dir, TIER_AGG_SCENARIO,
                storm_draw=storm_draw, scenario=scenario)
    return out_path


@click.command()
@click.option('--storm-draw',         type=int, required=True)
@click.option('--scenario',           type=str, required=True)
@click.option('--storm-draw-dir',     type=click.Path(path_type=Path), default=str(STORM_DRAW_DIR), show_default=True)
@click.option('--storm-draw-table',   type=str, default=STORM_DRAW_TABLE_PATH, show_default=True)
@click.option('--time-bins-path',     type=str, default=TIME_BINS_PATH, show_default=True)
def main(storm_draw, scenario, storm_draw_dir, storm_draw_table, time_bins_path):
    out_path = aggregate_scenario(
        storm_draw=storm_draw, scenario=scenario,
        storm_draw_dir=Path(storm_draw_dir),
        storm_draw_table_path=storm_draw_table,
        time_bins_path=time_bins_path,
    )
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
