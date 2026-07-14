"""
CLI entry point for parallel evaluate orchestration.

Splits the full evaluation grid into bundles of BUNDLE_SIZE (s1_spec_id, s2_spec_id, threshold_quantile)
triples per job and submits via idd_tools.jobmon.submit_with_manifest. In task-file mode,
submits one job per task in the file.

No fit stage: the IS spec manifest + fold assignments are generated here from the
grid and the data, and every model component is RE-FIT in-memory by each worker
(fitting is ~0.1s; reading pickled fits from NFS at scale is what melted the
filesystem). One (s1, s2) cross per task runs its full bulk×tail cartesian.

Usage:
    run-evaluate-orchestrate \\
        --data-path  <path/to/input.parquet> \\
        --output-dir <dir>

Run probe first (default); confirm resources; then resubmit with --no-probe.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path

import click
import pandas as pd
import pyarrow.parquet as pq

from idd_tools.jobmon import (
    Task,
    TaskManifest,
    TaskTemplateSpec,
    calibrate_bundle_size,
    filter_already_done,
    submit_with_manifest,
    with_probe,
)

from idd_tc_mortality.cache import component_id
from idd_tc_mortality.cv import compute_fold_assignments
from idd_tc_mortality.grid.grid import enumerate_component_specs
from idd_tc_mortality.thresholds import compute_thresholds

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SLURM_CLUSTER = "slurm"
_SLURM_RESOURCES = {
    "cores":   1,
    "memory":  "1G",     # probe n=2: max 0.70 GiB, 60% of 1G
    "runtime": "11m",    # probe n=2: median 7 min, 87% of 8m; set 11m (1.57× observed)
    "queue":   "all.q",
    "project": "proj_rapidresponse",
}

BUNDLE_SIZE = 1  # one (s1, s2) pair per task: a single s1/s2 cross with its full
                 # bulk×tail cartesian, all re-fit in-memory (no NFS model reads).


def _load_is_groups(specs_path: str) -> list[tuple[str, str, float]]:
    """Return cov-matched (s1_spec_id, s2_spec_id, threshold_quantile) triples from the IS manifest.

    Only cov-matched pairs are included: each s2 is paired with the s1 whose
    covariate_combo matches. threshold_quantile comes from the s2 spec.
    """
    with open(specs_path) as f:
        manifest: dict[str, dict] = json.load(f)

    s1_specs: dict[str, dict] = {}  # cid → spec
    s2_specs: dict[str, dict] = {}  # cid → spec

    for cid, spec in manifest.items():
        if spec.get("fold_tag", "is") != "is":
            continue
        if spec["component"] == "s1":
            s1_specs[cid] = spec
        elif spec["component"] == "s2":
            s2_specs[cid] = spec

    def _ck(spec: dict) -> str:
        return json.dumps(spec["covariate_combo"], sort_keys=True)

    groups: list[tuple[str, str, float]] = []
    for s1_cid in sorted(s1_specs):
        s1_ck = _ck(s1_specs[s1_cid])
        for s2_cid in sorted(s2_specs):
            if _ck(s2_specs[s2_cid]) == s1_ck:
                q = float(s2_specs[s2_cid]["threshold_quantile"])
                groups.append((s1_cid, s2_cid, q))

    return groups


def _filter_to_survivors(specs: list[dict], survivors: dict) -> list[dict]:
    """Keep only specs matching the per-stage survivor allow-lists.

    ``survivors`` maps each stage ('s1','s2','bulk','tail') to
    ``{'families': [...], 'exposure_modes': [...]}``. A spec whose stage has a
    rule is kept iff its family AND exposure_mode are both listed; stages with
    no rule are kept in full, and non-stage keys (e.g. '_source') are ignored.
    Thresholds and covariate sets are not filtered — retained specs keep their
    full coverage.
    """
    kept: list[dict] = []
    for s in specs:
        rule = survivors.get(s["component"])
        if rule is None:
            kept.append(s)
        elif s.get("family") in rule["families"] and s.get("exposure_mode") in rule["exposure_modes"]:
            kept.append(s)
    return kept


def _parse_tier_map(s: str) -> dict[int, str]:
    """Parse '0:1G,1:1G,3:2G' into {0:'1G', 1:'1G', 3:'2G'} (|s2_cov| tier → resource)."""
    out: dict[int, str] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise click.UsageError(f"tier map entry {pair!r} must be 'tier:value' (e.g. '4:2G').")
        k, v = pair.split(":", 1)
        out[int(k.strip())] = v.strip()
    return out


def _build_manifest_and_folds(
    data_path: str,
    output_path: Path,
    n_seeds: int = 5,
    n_folds: int = 5,
    survivors: dict | None = None,
    refined_specs_path: str | None = None,
) -> tuple[str, str]:
    """Build the IS spec manifest + fold assignments, inject full-data threshold_rate,
    and write manifest.json + fold_assignments.parquet into output_path. No fitting,
    no .pkl — each evaluate worker re-fits every component (IS + OOS folds) in-memory.

    The IS spec list comes from one of two sources:
      - ``refined_specs_path`` set: load the explicit spec list verbatim (as produced
        by ``run-build-refined-specs-post2000``). Enumeration and the survivor filter
        are both skipped — this is the refined-grid path, whose per-family tail
        exposures cannot be expressed by ``enumerate(mode="refined")`` or ``--survivors``.
      - otherwise: enumerate the preliminary grid, optionally narrowed by ``survivors``.

    threshold_rate is computed once from the FULL data and baked into every
    s2/bulk/tail spec, so OOS fold-fits use the same threshold as the in-sample fit
    rather than recomputing the quantile from a fold subset. Returns
    (manifest_path, fold_assignments_path).
    """
    df = pd.read_parquet(data_path)

    if refined_specs_path is not None:
        with open(refined_specs_path) as f:
            specs = json.load(f)
        logger.info(
            "Loaded %d refined IS specs from %s (enumeration + survivor filter skipped).",
            len(specs), refined_specs_path,
        )
    else:
        specs = enumerate_component_specs(mode="preliminary")
        if survivors is not None:
            specs = _filter_to_survivors(specs, survivors)
            logger.info("Survivor filter applied: %d IS specs remain.", len(specs))
    death_rate = df["deaths"].values / df["exposed"].values
    threshold_map = compute_thresholds(death_rate)
    for spec in specs:
        if spec["threshold_quantile"] is not None:
            spec["threshold_rate"] = threshold_map[spec["threshold_quantile"]]

    manifest = {component_id(s): s for s in specs}
    manifest_path = output_path / "manifest.json"
    fd, tmp = tempfile.mkstemp(dir=output_path, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, manifest_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    logger.info("Wrote IS spec manifest (%d specs) to %s", len(manifest), manifest_path)

    fold_df = compute_fold_assignments(df, n_seeds=n_seeds, n_folds=n_folds)
    fold_path = output_path / "fold_assignments.parquet"
    fd, tmp = tempfile.mkstemp(dir=output_path, suffix=".parquet.tmp")
    os.close(fd)
    try:
        fold_df.to_parquet(tmp, index=True)
        os.replace(tmp, fold_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    logger.info("Wrote fold assignments (%d rows, %d seeds × %d folds) to %s",
                len(fold_df), n_seeds, n_folds, fold_path)

    return str(manifest_path), str(fold_path)


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
    default=None,
    type=click.Path(dir_okay=False),
    help="(Deprecated; ignored.) The manifest is now generated from the grid + data "
         "inside this command — there is no separate fit stage.",
)
@click.option(
    "--results-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="(Deprecated; ignored.) Models are re-fit in-memory; no .pkl files are read.",
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
@click.option(
    "--decouple-covs",
    is_flag=True,
    default=False,
    help="Forward to run-evaluate workers: if set, assemble DH configs from "
         "the full s1_cov × s2_cov × bulk_cov × tail_cov Cartesian product "
         "rather than the cov-matched subset. Ignored when --task-file is set "
         "(task mode is implicitly decoupled).",
)
@click.option(
    "--task-file",
    "task_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a task JSON file (produced by run-build-evaluate-tasks). "
         "When provided, one Slurm job is submitted per task in the file, "
         "and workers run as --task-file/--task-index. Takes precedence over "
         "the legacy (s1_spec_id, threshold) bundled partitioning.",
)
@click.option(
    "--skip-model-predictions",
    is_flag=True,
    default=False,
    help="Forward to run-evaluate workers. Skips the per-DH-config "
         "model_predictions/*.parquet writes (one per IS config + one per "
         "OOS fold per config). Required at refined scale where each worker "
         "would otherwise emit ~10M tiny NFS writes.",
)
@click.option(
    "--survivors",
    "survivors",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a survivors JSON (per-stage family + exposure_mode allow-lists "
         "from preliminary screening). When set, the enumerated grid is filtered "
         "to the screening winners before the manifest is built — a much smaller "
         "cov-matched run. Pair WITHOUT --skip-model-predictions to generate the "
         "per-config predictions the intermediate stage needs.",
)
@click.option(
    "--refined-specs",
    "refined_specs",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to an explicit refined IS spec JSON list (from "
         "run-build-refined-specs-post2000). When set, the manifest is built from "
         "this list verbatim instead of enumerating the preliminary grid, and "
         "--survivors is ignored. Required for refined runs, whose per-family tail "
         "exposures cannot be expressed by enumeration or the survivor allow-lists.",
)
@click.option(
    "--manifest-only",
    "manifest_only",
    is_flag=True,
    default=False,
    help="Build manifest.json + fold_assignments.parquet into --output-dir and exit "
         "without submitting. Use this to produce the manifest a downstream task "
         "builder (e.g. run-build-refined-cells) needs before submission.",
)
@click.option(
    "--cells-file",
    "cells_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a partitioned cell manifest (from run-build-refined-cells). Submits "
         "one job per task in cells mode (worker scores explicit DH-config cells). "
         "Uses the existing manifest.json + fold_assignments.parquet in --output-dir "
         "(run --manifest-only first); does NOT rebuild them. Resources are tiered by "
         "the task's s2_n_cov feature via --tier-memory / --tier-runtime.",
)
@click.option(
    "--probe-tiers",
    "probe_tiers",
    is_flag=True,
    default=False,
    help="Cells mode only: submit one representative task per distinct |s2_cov| tier "
         "(the 5 probes) with the given resources, then stop. Inspect sacct per task, "
         "set --tier-memory / --tier-runtime, and resubmit without this flag.",
)
@click.option(
    "--tier-memory",
    "tier_memory",
    default=None,
    help="Cells mode: per-|s2_cov| memory asks, e.g. '0:1G,1:1G,2:1G,3:2G,4:4G'. "
         "Tiers not listed fall back to --memory / the default.",
)
@click.option(
    "--tier-runtime",
    "tier_runtime",
    default=None,
    help="Cells mode: per-|s2_cov| runtime asks, e.g. '0:5m,1:5m,2:10m,3:20m,4:40m'. "
         "Tiers not listed fall back to --runtime / the default.",
)
@click.option(
    "--scope",
    is_flag=True,
    default=False,
    help="Forward to run-evaluate workers. Enables --scope (per-step "
         "tick-tock logging with elapsed time + RSS memory). Intended for "
         "small probe runs — emits millions of lines on large tasks.",
)
@click.option(
    "--memory",
    "memory",
    default=None,
    show_default=False,
    help=f"Override per-task Slurm memory ask (e.g. '2G', '500M'). "
         f"Default {_SLURM_RESOURCES['memory']} is calibrated for post-2000 data "
         f"at BUNDLE_SIZE={BUNDLE_SIZE}; adjust if data or bundle size changes.",
)
@click.option(
    "--runtime",
    "runtime",
    default=None,
    show_default=False,
    help=f"Override per-task Slurm runtime ask (e.g. '30m', '2h'). "
         f"Default {_SLURM_RESOURCES['runtime']} assumes BUNDLE_SIZE={BUNDLE_SIZE} "
         f"groups × ~100s/group; revalidate after probe if bundle size changes.",
)
@click.option(
    "--calibrate",
    "calibrate",
    is_flag=True,
    default=False,
    help="Run bundle-size calibration before the full workflow. Submits micro-tasks "
         "in parallel with generous resources (4G/1h), fits t(n) = a + b×n from sacct, "
         "and prints recommended BUNDLE_SIZE and Slurm resources. Does not submit the "
         "production workflow — update the constants in this file and rerun without "
         "--calibrate. Use --calibrate-sizes to override bundle sizes (default 1,2,4).",
)
@click.option(
    "--calibrate-sizes",
    "calibrate_sizes",
    default="1,2,4",
    show_default=True,
    help="Comma-separated bundle sizes for calibration tasks. Default '1,2,4' is "
         "appropriate for evaluate where one group ≈ 100s. For orchestrators where "
         "a single work unit is milliseconds (e.g. fit_component), use larger values "
         "such as '500,1000,2000'.",
)
@click.option(
    "--max-attempts",
    "max_attempts",
    type=int,
    default=None,
    help="Per-task jobmon retry budget. Pass 1 to disable retries entirely "
         "(no resource-escalating re-runs on failure) — useful for diagnostic "
         "probes where you want the true first-attempt outcome. Default: "
         "submit_with_manifest's default.",
)
@with_probe()
def main(
    specs_path: str,
    results_dir: str,
    data_path: str,
    output_dir: str,
    fold_assignments_path: str | None,
    workflow_name: str,
    decouple_covs: bool,
    task_file: str | None,
    skip_model_predictions: bool,
    survivors: str | None,
    refined_specs: str | None,
    manifest_only: bool,
    cells_file: str | None,
    probe_tiers: bool,
    tier_memory: str | None,
    tier_runtime: str | None,
    scope: bool,
    memory: str | None,
    runtime: str | None,
    calibrate: bool,
    calibrate_sizes: str,
    max_attempts: int | None,
    no_probe: bool,
    probe_n: int,
) -> None:
    """Orchestrate parallel evaluation: one job per bundle (legacy mode) or per task
    (task-file mode). Run probe first; confirm RSS and runtime; then resubmit with --no-probe."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logs_dir = output_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(task: Task) -> tuple[str, str]:
        return (
            str(logs_dir / f"{task.task_id}.out"),
            str(logs_dir / f"{task.task_id}.err"),
        )

    # ----- Cells mode (refined structure-C grid) -------------------------------
    # Submit the partitioned cell manifest from run-build-refined-cells. Reuses the
    # manifest.json + fold_assignments.parquet already in --output-dir (from an
    # earlier --manifest-only run); does NOT rebuild them. One job per cell-task,
    # with Slurm resources tiered by the task's |s2_cov| feature.
    if cells_file is not None:
        specs_path = str(output_path / "manifest.json")
        fold_assignments_path = str(output_path / "fold_assignments.parquet")
        if not Path(specs_path).exists():
            raise click.UsageError(
                f"--cells-file given but {specs_path} is missing. Run "
                "run-evaluate-orchestrate --refined-specs ... --manifest-only first."
            )
        with open(cells_file) as f:
            cells_tasks = json.load(f)["tasks"]

        mem_by_tier = _parse_tier_map(tier_memory) if tier_memory else {}
        rt_by_tier = _parse_tier_map(tier_runtime) if tier_runtime else {}
        default_mem = memory or _SLURM_RESOURCES["memory"]
        default_rt = runtime or _SLURM_RESOURCES["runtime"]

        def _resources_for(task: Task) -> dict:
            k = task.task_features.get("s2_n_cov")
            return {
                "memory": mem_by_tier.get(k, default_mem),
                "runtime": rt_by_tier.get(k, default_rt),
            }

        if probe_tiers:
            first_per_tier: dict[int, int] = {}
            for i, t in enumerate(cells_tasks):
                first_per_tier.setdefault(t["task_features"]["s2_n_cov"], i)
            submit_indices = [first_per_tier[k] for k in sorted(first_per_tier)]
            logger.info(
                "Probe-tiers: submitting %d tasks (one per |s2_cov| tier) {tier: task_index}=%s",
                len(submit_indices), {k: first_per_tier[k] for k in sorted(first_per_tier)},
            )
        else:
            submit_indices = list(range(len(cells_tasks)))

        skip_predictions_flag = " --skip-model-predictions" if skip_model_predictions else ""
        scope_flag = " --scope" if scope else ""

        manifest = TaskManifest(
            workflow_name=workflow_name,
            tasks=[
                Task(
                    index=j,
                    task_id=f"task_{i:05d}",
                    task_template="evaluate_cells",
                    task_args={
                        "specs_path": specs_path,
                        "data_path": data_path,
                        "output_dir": output_dir,
                        "fold_assignments_path": fold_assignments_path,
                        "cells_file": str(cells_file),
                        "task_index": i,
                    },
                    task_features={"s2_n_cov": cells_tasks[i]["task_features"]["s2_n_cov"]},
                    depends_on=[],
                )
                for j, i in enumerate(submit_indices)
            ],
        )
        if not probe_tiers:
            manifest = filter_already_done(
                manifest,
                lambda task: (
                    output_path / "partials"
                    / f"dh_task_{task.task_args['task_index']:05d}.parquet"
                ).exists(),
            )
        templates = {
            "evaluate_cells": TaskTemplateSpec(
                command_template=(
                    "run-evaluate"
                    " --specs-path {specs_path}"
                    " --data-path {data_path}"
                    " --output-dir {output_dir}"
                    " --fold-assignments {fold_assignments_path}"
                    " --cells-file {cells_file}"
                    " --task-index {task_index}"
                    + skip_predictions_flag
                    + scope_flag
                ),
                node_args=["task_index"],
                task_args=["specs_path", "data_path", "output_dir",
                           "fold_assignments_path", "cells_file"],
                op_args=[],
            ),
        }

        result = submit_with_manifest(
            manifest,
            output_dir=output_path,
            templates=templates,
            resources=_resources_for,
            tool_name="idd-tc-mortality",
            cluster_name=_SLURM_CLUSTER,
            project=_SLURM_RESOURCES["project"],
            queue=_SLURM_RESOURCES["queue"],
            log_path_fn=_log_path,
            **({} if max_attempts is None else {"max_attempts": max_attempts}),
        )

        if probe_tiers:
            logger.info(
                "Probe-tiers workflow %d finished (status %r). Inspect sacct per task "
                "(max RSS + runtime), then resubmit with --tier-memory / --tier-runtime "
                "and without --probe-tiers.",
                result.workflow_id, result.status,
            )
            return
        if result.status != "D":
            logger.error("Cells workflow %d finished with status %r — not aggregating.",
                         result.workflow_id, result.status)
            return
        logger.info("All cell tasks complete. Aggregating partial results.")
        _aggregate_partials(output_path)
        return

    # No fit stage: generate the IS spec manifest + fold assignments here, from the
    # grid and the data. Workers re-fit every component in-memory, so there are no
    # .pkl files. Any --specs-path / --results-dir / --fold-assignments inputs are ignored.
    # With --survivors, the grid is first filtered to the screening winners.
    if refined_specs and survivors:
        raise click.UsageError("--refined-specs and --survivors are mutually exclusive.")
    survivors_cfg = json.loads(Path(survivors).read_text()) if survivors else None
    specs_path, fold_assignments_path = _build_manifest_and_folds(
        data_path, output_path, survivors=survivors_cfg, refined_specs_path=refined_specs,
    )
    if manifest_only:
        logger.info(
            "--manifest-only: manifest + fold assignments written to %s; not submitting.",
            output_path,
        )
        return

    _resources = dict(_SLURM_RESOURCES)
    if memory is not None:
        _resources["memory"] = memory
    if runtime is not None:
        _resources["runtime"] = runtime
    logger.info("Per-task Slurm resources: memory=%s runtime=%s",
                _resources["memory"], _resources["runtime"])

    fa_str = str(fold_assignments_path) if fold_assignments_path else ""
    decouple_flag = " --decouple-covs" if decouple_covs else ""
    skip_predictions_flag = " --skip-model-predictions" if skip_model_predictions else ""
    scope_flag = " --scope" if scope else ""
    probe_only_n: int | None = None if no_probe else probe_n

    if calibrate:
        groups = _load_is_groups(specs_path)
        try:
            _requested_sizes = [int(s.strip()) for s in calibrate_sizes.split(",")]
        except ValueError:
            logger.error("--calibrate-sizes must be comma-separated integers, got %r.", calibrate_sizes)
            return
        calib_sizes = [n for n in _requested_sizes if n <= len(groups)]
        if len(calib_sizes) < 2:
            logger.error("Need at least 2 groups to calibrate; found %d.", len(groups))
            return
        calib_dir = output_path / "calibration"
        calib_dir.mkdir(parents=True, exist_ok=True)
        calib_bundles_dir = calib_dir / "bundles"
        calib_bundles_dir.mkdir(parents=True, exist_ok=True)
        calib_tasks = []
        for i, n in enumerate(calib_sizes):
            bp = calib_bundles_dir / f"bundle_n{n:02d}.json"
            bp.write_text(json.dumps([[s1, s2, q] for s1, s2, q in groups[:n]]))
            calib_tasks.append(Task(
                index=i,
                task_id=f"calib_n{n:02d}",
                task_template="evaluate_worker_bundle",
                task_args={
                    "specs_path": specs_path,
                    "results_dir": results_dir,
                    "data_path": data_path,
                    "output_dir": str(calib_dir),
                    "fold_assignments_path": fa_str,
                    "bundle_file": str(bp),
                    "bundle_index": i,
                },
                task_features={},
                depends_on=[],
            ))
        calib_manifest = TaskManifest(
            workflow_name=f"{workflow_name}-calibrate",
            tasks=calib_tasks,
        )
        calib_templates = {
            "evaluate_worker_bundle": TaskTemplateSpec(
                command_template=(
                    "run-evaluate"
                    " --specs-path {specs_path}"
                    " --results-dir {results_dir}"
                    " --data-path {data_path}"
                    " --output-dir {output_dir}"
                    " --fold-assignments {fold_assignments_path}"
                    " --bundle-file {bundle_file}"
                    " --bundle-index {bundle_index}"
                    + decouple_flag
                    + skip_predictions_flag
                    + scope_flag
                ),
                node_args=["bundle_index"],
                task_args=["specs_path", "results_dir", "data_path", "output_dir",
                           "fold_assignments_path", "bundle_file"],
                op_args=[],
            ),
        }
        calib_result = calibrate_bundle_size(
            calib_manifest,
            templates=calib_templates,
            sizes=calib_sizes,
            output_dir=calib_dir,
            tool_name="idd-tc-mortality",
            cluster_name=_SLURM_CLUSTER,
            project=_resources["project"],
            queue=_resources["queue"],
            log_method=logger.info,
        )
        logger.info(
            "\nUpdate run_evaluate_orchestrate.py with calibrated settings:\n"
            "  BUNDLE_SIZE = %d\n"
            "  _SLURM_RESOURCES['memory']  = '%s'\n"
            "  _SLURM_RESOURCES['runtime'] = '%s'",
            calib_result.recommended_bundle_size,
            calib_result.recommended_memory,
            calib_result.recommended_runtime,
        )
        return

    if task_file is not None:
        with open(task_file) as f:
            task_doc = json.load(f)
        n_tasks = len(task_doc["tasks"])
        logger.info("Task-file mode: %d tasks to evaluate.", n_tasks)

        manifest = TaskManifest(
            workflow_name=workflow_name,
            tasks=[
                Task(
                    index=i,
                    task_id=f"task_{i:05d}",
                    task_template="evaluate_worker_taskfile",
                    task_args={
                        "specs_path": specs_path,
                        "results_dir": results_dir,
                        "data_path": data_path,
                        "output_dir": output_dir,
                        "fold_assignments_path": fa_str,
                        "task_file": str(task_file),
                        "task_index": i,
                    },
                    task_features={},
                    depends_on=[],
                )
                for i in range(n_tasks)
            ],
        )
        manifest = filter_already_done(
            manifest,
            lambda task: (
                output_path / "partials" / f"dh_task_{task.task_args['task_index']:05d}.parquet"
            ).exists(),
        )
        templates = {
            "evaluate_worker_taskfile": TaskTemplateSpec(
                command_template=(
                    "run-evaluate"
                    " --specs-path {specs_path}"
                    " --results-dir {results_dir}"
                    " --data-path {data_path}"
                    " --output-dir {output_dir}"
                    " --fold-assignments {fold_assignments_path}"
                    " --task-file {task_file}"
                    " --task-index {task_index}"
                    + skip_predictions_flag
                    + scope_flag
                ),
                node_args=["task_index"],
                task_args=["specs_path", "results_dir", "data_path", "output_dir",
                           "fold_assignments_path", "task_file"],
                op_args=[],
            ),
        }
    else:
        groups = _load_is_groups(specs_path)
        n_groups = len(groups)
        n_bundles = math.ceil(n_groups / BUNDLE_SIZE)
        logger.info(
            "Legacy mode: %d (s1, s2, threshold) triples → %d bundles of ≤%d.",
            n_groups, n_bundles, BUNDLE_SIZE,
        )

        bundles_dir = output_path / "bundles"
        bundles_dir.mkdir(parents=True, exist_ok=True)
        bundle_paths: list[Path] = []
        for i in range(n_bundles):
            chunk = groups[i * BUNDLE_SIZE:(i + 1) * BUNDLE_SIZE]
            bp = bundles_dir / f"bundle_{i:05d}.json"
            bp.write_text(json.dumps([[s1, s2, q] for s1, s2, q in chunk]))
            bundle_paths.append(bp)

        manifest = TaskManifest(
            workflow_name=workflow_name,
            tasks=[
                Task(
                    index=i,
                    task_id=f"bundle_{i:05d}",
                    task_template="evaluate_worker_bundle",
                    task_args={
                        "specs_path": specs_path,
                        "results_dir": results_dir,
                        "data_path": data_path,
                        "output_dir": output_dir,
                        "fold_assignments_path": fa_str,
                        "bundle_file": str(bundle_paths[i]),
                        "bundle_index": i,
                    },
                    task_features={},
                    depends_on=[],
                )
                for i in range(n_bundles)
            ],
        )
        manifest = filter_already_done(
            manifest,
            lambda task: (
                output_path / "partials"
                / f"dh_bundle_{task.task_args['bundle_index']:05d}.parquet"
            ).exists(),
        )
        templates = {
            "evaluate_worker_bundle": TaskTemplateSpec(
                command_template=(
                    "run-evaluate"
                    " --specs-path {specs_path}"
                    " --results-dir {results_dir}"
                    " --data-path {data_path}"
                    " --output-dir {output_dir}"
                    " --fold-assignments {fold_assignments_path}"
                    " --bundle-file {bundle_file}"
                    " --bundle-index {bundle_index}"
                    + decouple_flag
                    + skip_predictions_flag
                    + scope_flag
                ),
                node_args=["bundle_index"],
                task_args=["specs_path", "results_dir", "data_path", "output_dir",
                           "fold_assignments_path", "bundle_file"],
                op_args=[],
            ),
        }

    result = submit_with_manifest(
        manifest,
        output_dir=output_path,
        templates=templates,
        resources={"memory": _resources["memory"], "runtime": _resources["runtime"]},
        tool_name="idd-tc-mortality",
        cluster_name=_SLURM_CLUSTER,
        project=_resources["project"],
        queue=_resources["queue"],
        probe_only=probe_only_n,
        log_path_fn=_log_path,
        **({} if max_attempts is None else {"max_attempts": max_attempts}),
    )

    if probe_only_n is not None:
        logger.info(
            "Probe complete: workflow %d, status %r. "
            "Verify RSS < %s and runtime < %s, then resubmit with --no-probe.",
            result.workflow_id, result.status,
            _resources["memory"], _resources["runtime"],
        )
        return

    if result.status != "D":
        logger.error(
            "Workflow %d finished with status %r — not aggregating.",
            result.workflow_id, result.status,
        )
        return

    logger.info("All workers complete. Aggregating partial results.")
    _aggregate_partials(output_path)
