"""Finalize aggregation: run agg_scenario → agg_storm_draw → postprocess in serial.

Single job that fans in after all predict_year_bin tasks complete.
Each step uses its own skip-check so partial runs resume cleanly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from idd_tc_mortality.predict.aggregate_scenario import aggregate_scenario
from idd_tc_mortality.predict.aggregate_storm_draw import aggregate_storm_draw
from idd_tc_mortality.predict.paths import (
    DRAWS_DIR,
    SCENARIOS,
    STORM_DRAW_DIR,
    STORM_DRAW_TABLE_PATH,
    TIME_BINS_PATH,
)
from idd_tc_mortality.predict.postprocess import (
    DEFAULT_HIERARCHY_PATH,
    DEFAULT_OBS_PATH,
    DEFAULT_POP_PATH,
    write_postprocess_outputs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option('--storm-draw-dir',    type=click.Path(path_type=Path), default=str(STORM_DRAW_DIR), show_default=True)
@click.option('--storm-draw-table',  type=str, default=STORM_DRAW_TABLE_PATH, show_default=True)
@click.option('--time-bins-path',    type=str, default=TIME_BINS_PATH, show_default=True)
@click.option('--storm-draws',       type=str, default='', help="Comma-separated; empty = all in table.")
@click.option('--scenarios',         type=str, default='', help="Comma-separated; empty = all SCENARIOS.")
@click.option('--hierarchy-path',    type=str, default=DEFAULT_HIERARCHY_PATH, show_default=True)
@click.option('--pop-path',          type=str, default=DEFAULT_POP_PATH, show_default=True)
@click.option('--obs-path',          type=str, default=DEFAULT_OBS_PATH, show_default=True)
@click.option('--skip-postprocess',  is_flag=True,
              help="Run agg_scenario + agg_storm_draw but skip postprocess.")
def main(
    storm_draw_dir, storm_draw_table, time_bins_path,
    storm_draws, scenarios,
    hierarchy_path, pop_path, obs_path,
    skip_postprocess,
):
    storm_draw_dir = Path(storm_draw_dir)

    sd_table  = pd.read_csv(storm_draw_table)
    sd_list   = (
        [int(x) for x in storm_draws.split(',')]
        if storm_draws
        else sorted(sd_table['storm_draw'].unique().tolist())
    )
    sc_list = scenarios.split(',') if scenarios else list(SCENARIOS)

    # Step 1: agg_scenario — one call per (sd, sc)
    logger.info("Step 1: aggregate_scenario for %d storm_draws × %d scenarios.",
                len(sd_list), len(sc_list))
    for sd in sd_list:
        for sc in sc_list:
            aggregate_scenario(
                storm_draw=sd, scenario=sc,
                storm_draw_dir=storm_draw_dir,
                storm_draw_table_path=storm_draw_table,
                time_bins_path=time_bins_path,
            )

    # Step 2: agg_storm_draw — one call per sd
    logger.info("Step 2: aggregate_storm_draw for %d storm_draws.", len(sd_list))
    for sd in sd_list:
        aggregate_storm_draw(
            storm_draw=sd,
            storm_draw_dir=storm_draw_dir,
        )

    # Step 3: postprocess
    if not skip_postprocess:
        logger.info("Step 3: postprocess.")
        write_postprocess_outputs(
            draw_base=storm_draw_dir,
            hierarchy_path=hierarchy_path,
            pop_path=pop_path,
            obs_path=obs_path,
        )
    else:
        logger.info("Step 3: skipped (--skip-postprocess).")


if __name__ == "__main__":
    main()
