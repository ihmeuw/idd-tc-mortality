"""Year-bin predict: run all basins for one (storm_draw, scenario, year_bin),
then aggregate basins into the year_bin mean.

Replaces the old two-task pattern of predict_basin × 7 + agg_year_bin as
separate jobmon tasks. Per-basin skip-checks are preserved: predict_basin
exits immediately if the basin mean already exists on disk, so partial runs
are cheap to resume.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from idd_tc_mortality.predict.aggregate_year_bin import aggregate_year_bin
from idd_tc_mortality.predict.paths import (
    BASIN_LEVELS,
    DRAWS_DIR,
    ISLAND_COV_PATH,
    NEW_SDI_PATH,
    OLD_SDI_PATH,
    STAGE4_DIR,
    STORM_DRAW_DIR,
    STORM_DRAW_TABLE_PATH,
)
from idd_tc_mortality.predict.predict_tc import predict_basin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option('--storm-draw',         type=int, required=True)
@click.option('--scenario',           type=str, required=True)
@click.option('--year-bin',           type=str, required=True)
@click.option('--basins',             type=str, default='',
              help="Comma-separated basin subset; empty = all 7 BASIN_LEVELS.")
@click.option('--stage4-dir',         type=click.Path(path_type=Path), default=str(STAGE4_DIR), show_default=True)
@click.option('--storm-draw-dir',     type=click.Path(path_type=Path), default=str(STORM_DRAW_DIR), show_default=True)
@click.option('--draws-dir',          type=click.Path(path_type=Path), default=str(DRAWS_DIR), show_default=True)
@click.option('--storm-draw-table',   type=str, default=STORM_DRAW_TABLE_PATH, show_default=True)
@click.option('--island-cov-path',    type=str, default=ISLAND_COV_PATH, show_default=True)
@click.option('--old-sdi-path',       type=str, default=OLD_SDI_PATH, show_default=True)
@click.option('--new-sdi-path',       type=str, default=NEW_SDI_PATH, show_default=True)
@click.option('--exposure-col',       type=str, default='person_storm_hours', show_default=True,
              help="Climada column to use as exposure. Default 'person_storm_hours'. "
                   "Pass 'total_population_exposed' for experimental runs.")
def main(
    storm_draw, scenario, year_bin, basins,
    stage4_dir, storm_draw_dir, draws_dir, storm_draw_table,
    island_cov_path, old_sdi_path, new_sdi_path,
    exposure_col,
):
    basin_list = basins.split(',') if basins else list(BASIN_LEVELS)
    storm_draw_dir = Path(storm_draw_dir)

    for basin in basin_list:
        predict_basin(
            storm_draw=storm_draw, scenario=scenario, year_bin=year_bin, basin=basin,
            stage4_dir=Path(stage4_dir), storm_draw_dir=storm_draw_dir,
            draws_dir=Path(draws_dir), storm_draw_table_path=storm_draw_table,
            island_cov_path=island_cov_path,
            old_sdi_path=old_sdi_path, new_sdi_path=new_sdi_path,
            exposure_col=exposure_col,
        )

    aggregate_year_bin(
        storm_draw=storm_draw, scenario=scenario, year_bin=year_bin,
        storm_draw_dir=storm_draw_dir,
    )


if __name__ == "__main__":
    main()
