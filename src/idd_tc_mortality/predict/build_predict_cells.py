"""
Build the consolidated-predict cell manifest.

A cell is one (storm_draw, scenario) predict job. Tasks bundle whole storm
draws (all scenarios run serially), `--storm-draws-per-task` of them, so the
per-task refit/load overhead is amortised over many sub-second cells. Each cell
carries an `sd_group` (a block of N consecutive storm draws); the partition
fixes `sd_group`, so one task = N storm draws × all scenarios, run serially.

Model/frame-independent: the same manifest is reused across death models and
A0/A1 frames (the orchestrator passes those as task args).

    cells -> build_hierarchical_cellset -> rectangular_partition(fix=[sd_group])

Usage:
    run-build-predict-cells \\
        --output-path /…/04-predict/<vintage>/predict_cells.json \\
        --storm-draws-per-task 2
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import click
import pandas as pd

from idd_tools.jobmon import build_hierarchical_cellset, rectangular_partition

from idd_tc_mortality.predict.paths import STORM_DRAW_TABLE_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CELL_AXES = ["sd_group", "storm_draw", "scenario"]
FIX_AXES = ["sd_group"]   # one task per block of storm draws (all scenarios inside)
SCENARIOS = ["historical", "ssp126", "ssp245", "ssp585"]


def enumerate_cells(
    storm_draw_table_path: str, scenarios: list[str], storm_draws_per_task: int,
) -> list[dict]:
    """One cell per (storm_draw, scenario); sd_group blocks consecutive storm draws."""
    tbl = pd.read_csv(storm_draw_table_path)
    storm_draws = sorted(int(x) for x in tbl["storm_draw"].unique())
    cells = []
    for rank, sd in enumerate(storm_draws):
        grp = rank // storm_draws_per_task
        for sc in scenarios:
            cells.append({"sd_group": grp, "storm_draw": sd, "scenario": sc})
    return cells


def build_cells_manifest(
    output_path: str | Path,
    *,
    storm_draw_table_path: str = STORM_DRAW_TABLE_PATH,
    scenarios: list[str] | None = None,
    storm_draws_per_task: int = 2,
    workflow_name: str = "predict-consolidated-cells",
) -> dict:
    scenarios = scenarios or SCENARIOS
    cells = enumerate_cells(storm_draw_table_path, scenarios, storm_draws_per_task)
    if not cells:
        raise ValueError("No cells enumerated — check the storm_draw_table.")

    cellset = build_hierarchical_cellset(cells, axes=CELL_AXES)

    def _features(group_key: dict) -> dict:
        return {"sd_group": group_key["sd_group"]}

    task_manifest = rectangular_partition(
        cellset,
        fix=FIX_AXES,             # one task per sd_group = N storm draws × all scenarios
        workflow_name=workflow_name,
        task_template="predict_cell",
        max_per_task=None,        # whole group in one task
        features_fn=_features,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(task_manifest.model_dump_json(indent=2))

    cells_per_task = Counter(len(t.task_args.get("cells", [])) for t in task_manifest.tasks)
    logger.info("Wrote %d cells in %d tasks (%d storm draws/task × %d scenarios) to %s",
                len(cells), len(task_manifest.tasks), storm_draws_per_task, len(scenarios), out)
    logger.info("Cells-per-task distribution: %s", dict(cells_per_task))
    return {"n_cells": len(cells), "n_tasks": len(task_manifest.tasks)}


@click.command()
@click.option("--output-path", required=True, type=click.Path(dir_okay=False),
              help="Where to write the predict cell manifest JSON.")
@click.option("--storm-draw-table", default=STORM_DRAW_TABLE_PATH, show_default=True)
@click.option("--storm-draws-per-task", type=int, default=2, show_default=True,
              help="Storm draws (all scenarios each) bundled serially per task.")
@click.option("--workflow-name", default="predict-consolidated-cells", show_default=True)
def main(output_path, storm_draw_table, storm_draws_per_task, workflow_name):
    """Build the predict cell manifest (storm-draw-blocked, all scenarios per task)."""
    build_cells_manifest(
        output_path,
        storm_draw_table_path=storm_draw_table,
        storm_draws_per_task=storm_draws_per_task,
        workflow_name=workflow_name,
    )


if __name__ == "__main__":
    main()
