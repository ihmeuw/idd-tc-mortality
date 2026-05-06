"""
CLI entry point for parallel evaluate orchestration.

Splits the full evaluation grid into (s1_spec_id, threshold_quantile) groups —
90 groups for the preliminary run — and submits one Slurm job per group via
jobmon. After all workers complete, aggregates partial dh_results into a single
dh_results.parquet.

Usage:
    run-evaluate-orchestrate \\
        --specs-path    <path/to/manifest.json> \\
        --results-dir   <dir/with/.pkl files> \\
        --data-path     <path/to/input.parquet> \\
        --output-dir    <dir> \\
        --fold-assignments <path/to/fold_assignments.parquet>
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path

import click
import pandas as pd
import pyarrow.parquet as pq

from idd_tc_mortality.cache import component_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SLURM_CLUSTER = "slurm"
_SLURM_RESOURCES = {
    "cores":   1,
    "memory":  "4G",
    "runtime": "2h",
    "queue":   "all.q",
    "project": "proj_rapidresponse",
}


def _load_is_groups(specs_path: str) -> list[tuple[str, float]]:
    """Return all (s1_spec_id, threshold_quantile) pairs from the IS manifest."""
    with open(specs_path) as f:
        manifest: dict[str, dict] = json.load(f)

    s1_spec_ids: set[str] = set()
    thresholds: set[float] = set()

    for cid, spec in manifest.items():
        if spec.get("fold_tag", "is") != "is":
            continue
        if spec["component"] == "s1":
            s1_spec_ids.add(cid)
        q = spec.get("threshold_quantile")
        if q is not None:
            thresholds.add(q)

    groups = [(s1_id, q) for s1_id in sorted(s1_spec_ids) for q in sorted(thresholds)]
    return groups


def _aggregate_partials(output_path: Path) -> None:
    """Concatenate all partial dh_results into dh_results.parquet."""
    partials_dir = output_path / "partials"
    partial_files = sorted(partials_dir.glob("dh_*.parquet"))

    if not partial_files:
        logger.warning("No partial files found in %s — dh_results.parquet not written.", partials_dir)
        return

    dfs = [pd.read_parquet(p) for p in partial_files]
    combined = pd.concat(dfs, ignore_index=True)

    out_path = output_path / "dh_results.parquet"
    fd, tmp = tempfile.mkstemp(dir=output_path, suffix=".parquet.tmp")
    os.close(fd)
    try:
        combined.to_parquet(tmp, index=False)
        meta = pq.read_metadata(tmp)
        if meta.num_rows != len(combined):
            raise RuntimeError(
                f"Parquet row count mismatch: wrote {len(combined)}, "
                f"metadata reports {meta.num_rows}."
            )
        if out_path.exists():
            out_path.unlink()
        os.replace(tmp, out_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info(
        "Aggregated %d partial files → %d rows → %s",
        len(partial_files), len(combined), out_path,
    )


@click.command()
@click.option(
    "--specs-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to manifest.json written by run-fit-orchestrate.",
)
@click.option(
    "--results-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing fitted .pkl result files.",
)
@click.option(
    "--data-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to cleaned training data parquet file.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory where dh_results.parquet and partials/ are written.",
)
@click.option(
    "--fold-assignments",
    "fold_assignments_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to fold_assignments.parquet for OOS evaluation.",
)
@click.option(
    "--workflow-name",
    default="evaluate",
    show_default=True,
    help="Jobmon workflow name.",
)
def main(
    specs_path: str,
    results_dir: str,
    data_path: str,
    output_dir: str,
    fold_assignments_path: str | None,
    workflow_name: str,
) -> None:
    """Orchestrate parallel evaluation: submit one job per (S1, threshold) group."""
    from jobmon.client.api import Tool

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    groups = _load_is_groups(specs_path)
    logger.info("Found %d (s1_spec_id, threshold) groups to evaluate.", len(groups))

    tool = Tool("idd-tc-mortality")
    unique_name = f"{workflow_name}_{uuid.uuid4()}"
    wf = tool.create_workflow(
        workflow_args=unique_name,
        name=unique_name,
        description="evaluate grid run",
    )

    fa_str = str(fold_assignments_path) if fold_assignments_path else ""

    tt = tool.get_task_template(
        template_name="evaluate_worker",
        command_template=(
            "run-evaluate"
            " --specs-path {specs_path}"
            " --results-dir {results_dir}"
            " --data-path {data_path}"
            " --output-dir {output_dir}"
            " --fold-assignments {fold_assignments_path}"
            " --s1-spec-id {s1_spec_id}"
            " --threshold-quantile {threshold_quantile}"
        ),
        node_args=["s1_spec_id", "threshold_quantile"],
        task_args=["specs_path", "results_dir", "data_path", "output_dir", "fold_assignments_path"],
        op_args=[],
    )
    tt.update_default_compute_resources(_SLURM_CLUSTER, **_SLURM_RESOURCES)

    tasks = [
        tt.create_task(
            s1_spec_id=s1_id,
            threshold_quantile=str(q),
            specs_path=specs_path,
            results_dir=results_dir,
            data_path=data_path,
            output_dir=output_dir,
            fold_assignments_path=fa_str,
            cluster_name=_SLURM_CLUSTER,
        )
        for s1_id, q in groups
    ]

    wf.add_tasks(tasks)
    wf.bind()
    logger.info("Submitted %d jobmon tasks for workflow %r", len(tasks), workflow_name)
    wf.run()

    logger.info("All workers complete. Aggregating partial results.")
    _aggregate_partials(output_path)
