"""
Consolidated-predict orchestrator (idd_tools.jobmon).

Submits one task per (storm_draw, scenario) cell for ONE death model on ONE
frame (A0 or A1), via `submit_with_manifest`, then fans the partials into a
hierarchy-rolled summary.

Probe-first (idd_tools `with_probe`): default submits `--probe-n` tasks; inspect
sacct, set `--memory`/`--runtime`, resubmit with `--no-probe`. A0 and A1 frames
are separate invocations — probe EACH (A1's subnational cells are heavier).

Flow per (model, frame):
    run-build-predict-cells  ->  predict_cells.json   (once, model/frame-independent)
    run-predict-consolidated ... (default: probe-n tasks; inspect sacct)
    run-predict-consolidated ... --no-probe --memory <m> --runtime <r>
        -> 04-predict/<vintage>/<mid>_<frame>/partials/cell_*.parquet
        -> 04-predict/<vintage>/<mid>_<frame>/summary.parquet   (fan-in rollup)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd

from idd_tools.jobmon import (
    Task,
    TaskManifest,
    TaskTemplateSpec,
    filter_already_done,
    submit_with_manifest,
    with_probe,
)

from idd_tc_mortality.predict.consolidated import rollup_and_summarize, rollup_basin
from idd_tc_mortality.predict.paths import STORM_DRAW_TABLE_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SLURM_CLUSTER = "slurm"
_PROJECT = "proj_rapidresponse"
_QUEUE = "all.q"
_DEFAULT_MEM = "6G"     # generous default for the probe; tighten via --memory after sacct
_DEFAULT_RT = "30m"


def _rollup(out_dir: Path, hierarchy_path: str, storm_draw_table: str) -> Path:
    """Fan-in: concat the cell partials, roll up the hierarchy, summarize across draws."""
    partials = sorted((out_dir / "partials").glob("cell_*.parquet"))
    if not partials:
        raise FileNotFoundError(f"No partials under {out_dir / 'partials'}.")
    frames = [pd.read_parquet(p) for p in partials]
    per_draw = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    draw_ids = sorted(int(x) for x in pd.read_csv(storm_draw_table)["storm_draw"].unique())
    hierarchy_df = pd.read_parquet(hierarchy_path)
    summary = rollup_and_summarize(per_draw, hierarchy_df, draw_ids)
    out_path = out_dir / "summary.parquet"
    summary.to_parquet(out_path, index=False)
    logger.info("Rollup: %d partials -> %s (%d rows)", len(partials), out_path, len(summary))

    # Basin diagnostic summary (exposed + deaths per basin × scenario × year).
    bpartials = sorted((out_dir / "basin_partials").glob("cell_*.parquet"))
    if bpartials:
        bframes = [pd.read_parquet(p) for p in bpartials]
        per_draw_basin = pd.concat([f for f in bframes if not f.empty], ignore_index=True)
        basin_summary = rollup_basin(per_draw_basin)
        basin_summary.to_parquet(out_dir / "basin_summary.parquet", index=False)
        logger.info("Basin rollup: %d partials -> basin_summary.parquet (%d rows)",
                    len(bpartials), len(basin_summary))
    return out_path


@click.command()
@click.option("--cells-file", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--focus-model", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--consolidated-path", required=True, type=str, help="A0 or A1 frame.")
@click.option("--out-dir", required=True, type=click.Path(path_type=Path))
@click.option("--data-path", required=True, type=str)
@click.option("--folds-path", required=True, type=str)
@click.option("--hierarchy-path", required=True, type=str)
@click.option("--storm-draw-table", default=STORM_DRAW_TABLE_PATH, show_default=True)
@click.option("--is-island-path", required=True, type=str)
@click.option("--old-sdi-path", required=True, type=str)
@click.option("--new-sdi-path", required=True, type=str)
@click.option("--exposure-col", default="person_storm_hours", show_default=True)
@click.option("--workflow-name", default="predict-consolidated", show_default=True)
@click.option("--memory", default=_DEFAULT_MEM, show_default=True, help="Slurm memory per task.")
@click.option("--runtime", default=_DEFAULT_RT, show_default=True, help="Slurm runtime per task.")
@click.option("--max-attempts", type=int, default=None)
@with_probe()   # adds --probe-n / --no-probe -> probe_n, no_probe
def main(cells_file, focus_model, consolidated_path, out_dir, data_path, folds_path,
         hierarchy_path, storm_draw_table, is_island_path, old_sdi_path, new_sdi_path,
         exposure_col, workflow_name, memory, runtime, max_attempts, probe_n, no_probe):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_tasks = json.load(open(cells_file))["tasks"]
    probe_only = None if no_probe else probe_n

    resources = {"memory": memory, "runtime": runtime}

    task_args_common = {
        "cells_file": str(cells_file), "focus_model": str(focus_model),
        "consolidated_path": consolidated_path, "out_dir": str(out_dir),
        "data_path": data_path, "folds_path": folds_path,
        "storm_draw_table": storm_draw_table, "is_island_path": is_island_path,
        "old_sdi_path": old_sdi_path, "new_sdi_path": new_sdi_path,
        "exposure_col": exposure_col,
    }
    manifest = TaskManifest(
        workflow_name=workflow_name,
        tasks=[
            Task(
                index=i, task_id=f"task_{i:05d}", task_template="predict_cell",
                task_args={**task_args_common, "task_index": i},
                task_features=t.get("task_features", {}),
                depends_on=[],
            )
            for i, t in enumerate(cells_tasks)
        ],
    )
    if not probe_only:
        manifest = filter_already_done(
            manifest,
            lambda task: (
                (out_dir / "partials" / f"cell_{task.task_args['task_index']:05d}.parquet").exists()
                and (out_dir / "basin_partials" / f"cell_{task.task_args['task_index']:05d}.parquet").exists()
            ),
        )

    def _log_path(task: Task) -> tuple[str, str]:
        tag = f"task_{task.task_args['task_index']:05d}"
        d = out_dir / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"{tag}.out"), str(d / f"{tag}.err")

    templates = {
        "predict_cell": TaskTemplateSpec(
            command_template=(
                "run-predict-cell"
                " --cells-file {cells_file} --task-index {task_index}"
                " --focus-model {focus_model} --consolidated-path {consolidated_path}"
                " --out-dir {out_dir} --data-path {data_path} --folds-path {folds_path}"
                " --storm-draw-table {storm_draw_table} --is-island-path {is_island_path}"
                " --old-sdi-path {old_sdi_path} --new-sdi-path {new_sdi_path}"
                " --exposure-col {exposure_col}"
            ),
            node_args=["task_index"],
            task_args=["cells_file", "focus_model", "consolidated_path", "out_dir",
                       "data_path", "folds_path", "storm_draw_table", "is_island_path",
                       "old_sdi_path", "new_sdi_path", "exposure_col"],
            op_args=[],
        ),
    }

    result = submit_with_manifest(
        manifest, output_dir=out_dir, templates=templates, resources=resources,
        probe_only=probe_only,
        tool_name="idd-tc-mortality", cluster_name=_SLURM_CLUSTER,
        project=_PROJECT, queue=_QUEUE, log_path_fn=_log_path,
        **({} if max_attempts is None else {"max_attempts": max_attempts}),
    )

    if probe_only:
        logger.info("Probe (%d tasks) finished (%s). Inspect sacct (max RSS + Elapsed), "
                    "set --memory/--runtime, resubmit with --no-probe.", probe_only, result.status)
        return
    if result.status != "D":
        logger.error("Workflow %s status %r — not rolling up.", result.workflow_id, result.status)
        return
    logger.info("All cells done. Rolling up.")
    _rollup(out_dir, hierarchy_path, storm_draw_table)


if __name__ == "__main__":
    main()
