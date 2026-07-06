"""
Consolidated-predict worker: one jobmon task = one (or a few) (storm_draw,
scenario) cells.

Refits the chosen model once, then for each cell pushdown-reads that storm
draw's climate-model slice filtered to the scenario, predicts the 16 toggles
(seed = storm_draw, per the consolidated seeding scheme), aggregates within the
draw (sum over storm_id & tc_risk_draw, ÷100), and atomic-writes a partial.

Driven by the cell manifest from `run-build-predict-cells`; consumes its task's
cells via `idd_tools.jobmon.inflate_cells`. Model/frame come from task args so
the same cell manifest serves every death-model × A0/A1-frame combination.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd

from idd_tools.jobmon import inflate_cells
from idd_tools.jobmon.atomic_io import atomic_write_parquet

from idd_tc_mortality.predict.consolidated import (
    bulk_sdi_table,
    predict_storm_draw,
    prep_frame,
)
from idd_tc_mortality.predict.paths import STORM_DRAW_TABLE_PATH
from idd_tc_mortality.refit_with_objects import refit_model_with_objects

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--cells-file", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--task-index", required=True, type=int)
@click.option("--focus-model", required=True, type=click.Path(exists=True, dir_okay=False),
              help="focus_model.json for the death model to predict.")
@click.option("--consolidated-path", required=True, type=str,
              help="A0 or A1 consolidated CLIMADA parquet to slice.")
@click.option("--out-dir", required=True, type=click.Path(path_type=Path))
@click.option("--data-path", required=True, type=str, help="input.parquet to refit on.")
@click.option("--folds-path", required=True, type=str)
@click.option("--storm-draw-table", default=STORM_DRAW_TABLE_PATH, show_default=True)
@click.option("--is-island-path", required=True, type=str)
@click.option("--old-sdi-path", required=True, type=str)
@click.option("--new-sdi-path", required=True, type=str)
@click.option("--exposure-col", default="person_storm_hours", show_default=True)
def main(cells_file, task_index, focus_model, consolidated_path, out_dir,
         data_path, folds_path, storm_draw_table, is_island_path,
         old_sdi_path, new_sdi_path, exposure_col):
    focus = json.loads(Path(focus_model).read_text())
    data = pd.read_parquet(data_path)
    folds = pd.read_parquet(folds_path)
    refit_out = refit_model_with_objects(focus, data, folds, n_seeds=1, n_folds=2)
    for stage in ("s1", "s2", "bulk", "tail"):
        if refit_out["is"][stage].get("failed"):
            raise SystemExit(f"IS stage {stage!r} failed; cannot predict.")

    tbl = pd.read_csv(storm_draw_table)
    mv_map = dict(zip(tbl["storm_draw"],
                      tbl["source_id"].astype(str) + "_" + tbl["variant_label"].astype(str)))
    is_island = pd.read_parquet(is_island_path)
    sdi = bulk_sdi_table(old_sdi_path, new_sdi_path)

    tasks = json.load(open(cells_file))["tasks"]
    if not 0 <= task_index < len(tasks):
        raise click.UsageError(f"--task-index {task_index} out of range ({len(tasks)} tasks).")
    cells = inflate_cells(tasks[task_index]["task_args"])
    logger.info("task %d: %d cell(s)", task_index, len(cells))

    out_partials = Path(out_dir) / "partials"
    basin_partials = Path(out_dir) / "basin_partials"
    out_partials.mkdir(parents=True, exist_ok=True)
    basin_partials.mkdir(parents=True, exist_ok=True)
    out_path = out_partials / f"cell_{task_index:05d}.parquet"
    basin_out_path = basin_partials / f"cell_{task_index:05d}.parquet"

    loc_frames, basin_frames = [], []
    for cell in cells:
        sd, sc = int(cell["storm_draw"]), str(cell["scenario"])
        mv = mv_map[sd]
        sl = pd.read_parquet(
            consolidated_path,
            filters=[("source_id_variant_label", "=", mv), ("experiment_id", "=", sc)],
        )
        if sl.empty:
            logger.info("  sd=%d %s (%s): no rows — skip", sd, sc, mv)
            continue
        sl = prep_frame(sl, is_island, sdi, exposure_col=exposure_col)
        if sl.empty:
            continue
        loc_agg, basin_agg = predict_storm_draw(refit_out, focus, data, sl, sd)
        loc_agg.insert(0, "storm_draw", sd)
        basin_agg.insert(0, "storm_draw", sd)
        loc_frames.append(loc_agg)
        basin_frames.append(basin_agg)
        logger.info("  sd=%d %s: %d (year,location) rows", sd, sc, len(loc_agg))
        del sl   # free this storm draw's slice before reading the next (keeps peak at one slice)

    loc_result = (pd.concat(loc_frames, ignore_index=True) if loc_frames
                  else pd.DataFrame(columns=["storm_draw", "experiment_id", "year", "location_id"]))
    basin_result = (pd.concat(basin_frames, ignore_index=True) if basin_frames
                    else pd.DataFrame(columns=["storm_draw", "experiment_id", "year", "basin"]))
    atomic_write_parquet(loc_result, out_path)
    atomic_write_parquet(basin_result, basin_out_path)
    logger.info("wrote %s (%d) + %s (%d)", out_path, len(loc_result), basin_out_path, len(basin_result))


if __name__ == "__main__":
    main()
