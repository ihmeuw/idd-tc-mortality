"""
CLI entry point for orchestrating a full grid fit run.

Usage (preliminary):
    run-fit-orchestrate \\
        --mode preliminary \\
        --data-path <path/to/data.parquet> \\
        --output-dir <dir> \\
        [--local]

Usage (refined — IS specs pre-built via run-build-refined-specs):
    run-build-refined-specs --output-path <specs.json>
    run-fit-orchestrate \\
        --specs-file <specs.json> \\
        --data-path <path/to/data.parquet> \\
        --output-dir <dir> \\
        [--local]

When --specs-file is provided, enumerate_component_specs is bypassed:
the IS spec list is read from JSON exactly as produced by
run-build-refined-specs (or any compatible builder). OOS specs are then
expanded from the IS list as usual. This is the only supported path for
refined runs — the --mode refined branch of enumerate_component_specs
is not wired through this CLI.

Steps
-----
1.  Call enumerate_component_specs to build the full spec list.
2.  Write a manifest.json to output-dir mapping component_id → spec.
    Existing manifest is overwritten (each run is idempotent by design).
3a. --local: iterate over specs sequentially, call fit_one_component, save results.
3b. default: submit one Slurm job per spec via jobmon using the run-fit-component
    entry point.  The workflow blocks until all jobs complete or one fails.

Jobmon resources
----------------
Each component fit is single-threaded and lightweight (max RSS ~280 MB observed
2026-05-17; ask is 512M). Runtime is short; 5 min is generous for any single spec.
Queue and project are site defaults and can be adjusted here if needed.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import click
import pandas as pd

from idd_tc_mortality.cache import component_id, save_result
from idd_tc_mortality.cv import compute_fold_assignments
from idd_tc_mortality.grid.grid import enumerate_component_specs
from idd_tc_mortality.fit.fit_component import fit_one_component
from idd_tc_mortality.thresholds import compute_thresholds

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SLURM_CLUSTER = "slurm"
_SLURM_RESOURCES = {
    "cores": 1,
    "memory": "1G",     # 6570-spec bundles: 444 MB (2026-05-22), 513 MB (2026-06-08 post-2000); 1G headroom
    "runtime": "9m",    # 34 specs × 9s each = 306s; ~3× safety = 9m
    "queue": "all.q",
    "project": "proj_rapidresponse",
}

# Probe confirmed fit_component runs in ~9s per spec. Jobmon overhead at
# per-spec granularity destroyed a prior run (took a full day at 22k tasks).
# Bundle to target ≥5 min per task: ceil(300 / 9) = 34 specs per bundle.
BUNDLE_SIZE = 6570


def _make_oos_specs(is_specs: list[dict], n_seeds: int, n_folds: int) -> list[dict]:
    """Clone each IS spec for all OOS folds, updating fold_tag.

    Returns a flat list of specs with fold_tag 's{seed}_f{fold}' for each
    combination of seed in [0, n_seeds) and fold in [0, n_folds).
    """
    oos: list[dict] = []
    for spec in is_specs:
        for seed in range(n_seeds):
            for fold in range(n_folds):
                oos_spec = {**spec, "fold_tag": f"s{seed}_f{fold}"}
                oos.append(oos_spec)
    return oos


def _save_fold_assignments(
    df: pd.DataFrame, output_dir: Path, n_seeds: int, n_folds: int
) -> Path:
    """Compute and save basin-stratified fold assignments to output_dir.

    Returns path to the written parquet file.
    """
    fold_df = compute_fold_assignments(df, n_seeds=n_seeds, n_folds=n_folds)
    out_path = output_dir / "fold_assignments.parquet"
    fd, tmp = tempfile.mkstemp(dir=output_dir, suffix=".parquet.tmp")
    os.close(fd)
    try:
        fold_df.to_parquet(tmp, index=True)
        os.replace(tmp, out_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    logger.info("Wrote fold assignments (%d rows, %d seeds) to %s", len(fold_df), n_seeds, out_path)
    return out_path


def _write_manifest(specs: list[dict], output_dir: Path) -> Path:
    """Write specs to manifest.json in output_dir. Returns path to manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {component_id(s): s for s in specs}
    manifest_path = output_dir / "manifest.json"

    # Atomic write so a partially-written manifest is never visible.
    fd, tmp = tempfile.mkstemp(dir=output_dir, suffix=".json.tmp")
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

    logger.info("Wrote manifest with %d specs to %s", len(specs), manifest_path)
    return manifest_path


def _subset_df_for_spec(spec: dict, df: pd.DataFrame, fold_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return the training subset of df for a spec's fold_tag.

    IS specs (fold_tag='is') use the full df. OOS specs subset to rows where
    the seed column != held_out_fold.
    """
    fold_tag = spec.get("fold_tag", "is")
    if fold_tag == "is" or fold_df is None:
        return df
    seed_part, fold_part = fold_tag.split("_")
    seed_idx = int(seed_part[1:])
    held_out_fold = int(fold_part[1:])
    col = f"seed_{seed_idx}"
    train_mask = fold_df[col].values != held_out_fold
    return df.loc[df.index[train_mask]].copy()


def _run_local(
    specs: list[dict],
    df: pd.DataFrame,
    output_dir: Path,
    fold_df: pd.DataFrame | None = None,
) -> None:
    """Fit all specs sequentially in the current process."""
    n = len(specs)
    failures: list[tuple[str, dict, str]] = []
    for i, spec in enumerate(specs, 1):
        cid = component_id(spec)
        logger.info("[%d/%d] Fitting %s (%s/%s)", i, n, cid, spec["component"], spec.get("family"))
        df_train = _subset_df_for_spec(spec, df, fold_df)
        result = fit_one_component(spec, df_train)
        if not result.converged:
            failures.append((cid, spec, result.meta.get("fit_error", "no error message")))
        save_result(result, spec, output_dir, overwrite=True)

    n_failed = len(failures)
    n_converged = n - n_failed
    logger.info("Fit complete: %d/%d converged, %d non-converged.", n_converged, n, n_failed)
    if failures:
        logger.warning("Non-converged specs:")
        for cid, spec, err in failures:
            logger.warning(
                "  %s  %s/%s  fold_tag=%s  -> %s",
                cid, spec["component"], spec.get("family"),
                spec.get("fold_tag", "is"), err,
            )


# TODO: move _parse_runtime_seconds to idd_tools.jobmon once lessons-learned complete
def _parse_runtime_seconds(s: str) -> int:
    import re
    m = re.fullmatch(r"(\d+)(s|m|h)", s)
    if not m:
        raise ValueError(f"Unrecognised runtime format: {s!r}")
    return int(m.group(1)) * {"s": 1, "m": 60, "h": 3600}[m.group(2)]


def _parse_memory_gib(s: str) -> float:
    """Parse a Slurm memory string ('1G', '512M') to GiB."""
    import re
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([MG])", s)
    if not m:
        raise ValueError(f"Unrecognised memory format: {s!r}")
    val = float(m.group(1))
    return val / 1024.0 if m.group(2) == "M" else val


def _run_with_manifest(
    specs: list[dict],
    manifest_path: Path,
    data_path: str,
    output_dir: Path,
    workflow_name: str,
    fold_assignments_path: Path | None,
    queue: str,
) -> None:
    """Submit fit specs via idd-tools submit_with_manifest.

    Three-phase execution:
    1. Build a TaskManifest for all pending specs (filter_already_done skips
       any spec whose .pkl already exists — safe to rerun after partial failure).
    2. Probe with 2 tasks; verify RSS < 400 MB and runtime < 300 s before scale.
    3. Re-filter (probe outputs now exist) and submit the remaining tasks.
    """
    from idd_tools.jobmon import (
        Task,
        TaskManifest,
        TaskTemplateSpec,
        collect_workflow,
        filter_already_done,
        submit_with_manifest,
    )
    from idd_tc_mortality.cache import result_exists

    fa_str = str(fold_assignments_path) if fold_assignments_path else ""

    # Write bundle files: each is a JSON list of spec_ids for one Slurm task.
    bundle_dir = output_dir / ".bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    batches = [specs[i:i + BUNDLE_SIZE] for i in range(0, len(specs), BUNDLE_SIZE)]
    for i, batch in enumerate(batches):
        bundle_file = bundle_dir / f"bundle_{i:04d}.json"
        bundle_file.write_text(json.dumps([component_id(s) for s in batch]))
    logger.info(
        "Wrote %d bundle files (%d specs, bundle_size=%d) to %s",
        len(batches), len(specs), BUNDLE_SIZE, bundle_dir,
    )

    def _build_manifest() -> TaskManifest:
        return TaskManifest(
            workflow_name=workflow_name,
            tasks=[
                Task(
                    index=i,
                    task_id=f"fit_component-bundle-{i:04d}",
                    task_template="fit_component",
                    task_args={
                        "bundle_file": str(bundle_dir / f"bundle_{i:04d}.json"),
                        "manifest_path": str(manifest_path),
                        "data_path": str(data_path),
                        "output_dir": str(output_dir),
                        "fold_assignments_path": fa_str,
                    },
                    task_features={},
                    depends_on=[],
                )
                for i in range(len(batches))
            ],
        )

    def _done(task: Task) -> bool:
        # A bundle is done only when every spec in it has a result.
        # The worker handles partial bundles internally (per-spec skip).
        bf = Path(task.task_args["bundle_file"])
        spec_ids = json.loads(bf.read_text())
        return all(result_exists(sid, output_dir) for sid in spec_ids)

    fit_template = TaskTemplateSpec(
        command_template=(
            "run-fit-component"
            " --bundle-file {bundle_file}"
            " --manifest {manifest_path}"
            " --data-path {data_path}"
            " --output-dir {output_dir}"
            " --fold-assignments {fold_assignments_path}"
        ),
        node_args=["bundle_file"],
        task_args=["manifest_path", "data_path", "output_dir", "fold_assignments_path"],
        op_args=[],
    )
    submit_kwargs: dict = dict(
        output_dir=output_dir,
        templates={"fit_component": fit_template},
        resources={"memory": _SLURM_RESOURCES["memory"], "runtime": _SLURM_RESOURCES["runtime"]},
        tool_name="idd-tc-mortality",
        cluster_name=_SLURM_CLUSTER,
        project=_SLURM_RESOURCES["project"],
        queue=queue,
        log_method=logger.info,
    )

    full_manifest = _build_manifest()
    pending = filter_already_done(full_manifest, _done)
    if not pending.tasks:
        logger.info("All %d specs already done. Nothing to submit.", len(specs))
        return

    # --- Phase 1: 2-task probe ---
    logger.info("Submitting 2-task probe (memory=%s, runtime=%s).",
                _SLURM_RESOURCES["memory"], _SLURM_RESOURCES["runtime"])
    probe_result = submit_with_manifest(pending, probe_only=2, **submit_kwargs)
    if probe_result.status != "D":
        raise RuntimeError(
            f"Probe workflow {probe_result.workflow_id} completed with status "
            f"{probe_result.status!r} — fix the failure before submitting at scale."
        )
    probe_df = collect_workflow(probe_result.workflow_id)
    max_rss = probe_df["max_rss_gib"].max()
    max_rt = probe_df["elapsed_seconds"].max()
    if pd.isna(max_rss) or pd.isna(max_rt):
        raise RuntimeError("Probe sacct data unavailable — wait ~1 min and rerun.")
    mem_gib = _parse_memory_gib(_SLURM_RESOURCES["memory"])
    if max_rss >= 0.8 * mem_gib:
        raise RuntimeError(
            f"Probe RSS {max_rss:.3f} GiB ≥ 80% of memory ask "
            f"{_SLURM_RESOURCES['memory']} ({mem_gib:.2f} GiB) — "
            "increase _SLURM_RESOURCES['memory'] and rerun."
        )
    wall_limit_s = _parse_runtime_seconds(_SLURM_RESOURCES["runtime"])
    if max_rt >= 0.8 * wall_limit_s:
        raise RuntimeError(
            f"Probe runtime {max_rt}s ≥ 80% of wall limit {wall_limit_s}s — "
            "increase _SLURM_RESOURCES['runtime'] and rerun."
        )
    logger.info("Probe OK (max_rss=%.3f GiB, max_rt=%ds). Submitting full manifest.", max_rss, max_rt)

    # --- Phase 2: full run (re-filter; probe outputs now exist on disk) ---
    pending = filter_already_done(full_manifest, _done)
    if not pending.tasks:
        logger.info("All specs done after probe (nothing left to submit).")
        return
    result = submit_with_manifest(pending, **submit_kwargs)
    logger.info(
        "Workflow %d completed with status %r (%d tasks submitted).",
        result.workflow_id, result.status, result.n_tasks_submitted,
    )


@click.command()
@click.option(
    "--mode",
    required=False,
    default=None,
    type=click.Choice(["preliminary", "refined"]),
    help="Grid mode. 'preliminary' uses fixed defaults. Required unless "
         "--specs-file is provided.",
)
@click.option(
    "--specs-file",
    required=False,
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a JSON file of IS component spec dicts (as produced by "
         "run-build-refined-specs). If provided, enumerate_component_specs "
         "is skipped and these specs are used directly. Takes precedence "
         "over --mode and all enumeration override flags.",
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
    help="Directory where fitted results and the manifest are written.",
)
@click.option(
    "--thresholds",
    multiple=True,
    type=float,
    default=None,
    help="Threshold quantile levels (e.g. --thresholds 0.80 --thresholds 0.85). "
         "Required for --mode refined. Ignored in preliminary mode unless provided "
         "to override defaults.",
)
@click.option(
    "--covariate-combos",
    multiple=True,
    default=None,
    help="Covariate combo flag dicts as JSON strings "
         "(e.g. '{\"wind_speed\":true,\"sdi\":false,\"basin\":false,\"is_island\":false}'). "
         "Required for --mode refined.",
)
@click.option(
    "--bulk-families",
    multiple=True,
    default=None,
    help="Bulk distribution family names (e.g. --bulk-families gamma --bulk-families nb). "
         "Required for --mode refined.",
)
@click.option(
    "--tail-families",
    multiple=True,
    default=None,
    help="Tail distribution family names. Required for --mode refined.",
)
@click.option(
    "--local",
    is_flag=True,
    default=False,
    help="Run sequentially in the current process instead of submitting to Slurm.",
)
@click.option(
    "--skip-oos",
    is_flag=True,
    default=False,
    help="Fit IS components only; skip OOS fold specs. Useful for smoke-testing "
         "the pipeline locally before submitting the full IS+OOS array to the cluster.",
)
@click.option(
    "--workflow-name",
    default="fit",
    show_default=True,
    help="Jobmon workflow name. Change this to run multiple non-competing workflows.",
)
@click.option(
    "--n-seeds",
    default=5,
    show_default=True,
    type=int,
    help="Number of CV seeds (random shuffles of the stratified k-fold assignment).",
)
@click.option(
    "--n-folds",
    default=5,
    show_default=True,
    type=int,
    help="Number of CV folds per seed.",
)
@click.option(
    "--sample",
    default=None,
    type=int,
    help="Randomly sample this many IS specs per component type (s1, s2, bulk, tail). "
         "Useful for smoke-testing speed and resource requirements before a full run.",
)
@click.option(
    "--queue",
    default=_SLURM_RESOURCES["queue"],
    show_default=True,
    help="Slurm queue to submit jobs to (ignored under --local).",
)
def main(
    mode: str | None,
    specs_file: str | None,
    data_path: str,
    output_dir: str,
    thresholds: tuple[float, ...],
    covariate_combos: tuple[str, ...],
    bulk_families: tuple[str, ...],
    tail_families: tuple[str, ...],
    local: bool,
    skip_oos: bool,
    workflow_name: str,
    n_seeds: int,
    n_folds: int,
    sample: int | None,
    queue: str,
) -> None:
    """Orchestrate a full grid fit run: enumerate specs, write manifest, submit jobs."""
    output_path = Path(output_dir)

    if specs_file is not None:
        is_specs = json.loads(Path(specs_file).read_text())
        if sample is not None:
            raise click.UsageError(
                "--sample is incompatible with --specs-file. Trim the JSON file directly."
            )
        logger.info("Loaded %d IS component specs from %s.", len(is_specs), specs_file)
    else:
        if mode is None:
            raise click.UsageError("Either --mode or --specs-file must be provided.")
        # Parse covariate_combos from JSON strings (click passes them as strings).
        parsed_combos: list[dict] | None = None
        if covariate_combos:
            try:
                parsed_combos = [json.loads(c) for c in covariate_combos]
            except json.JSONDecodeError as exc:
                raise click.UsageError(
                    f"Could not parse --covariate-combos as JSON: {exc}"
                ) from exc

        is_specs = enumerate_component_specs(
            mode=mode,
            thresholds=list(thresholds) if thresholds else None,
            covariate_combos=parsed_combos,
            bulk_families=list(bulk_families) if bulk_families else None,
            tail_families=list(tail_families) if tail_families else None,
            sample=sample,
        )
        logger.info("Enumerated %d IS component specs (mode=%s).", len(is_specs), mode)

    # Load data to compute threshold_rate values and populate specs before writing
    # the manifest. This makes the manifest self-contained: workers read threshold_rate
    # directly from their spec rather than re-computing from data.
    df = pd.read_parquet(data_path)
    death_rate = df["deaths"].values / df["exposed"].values
    threshold_map = compute_thresholds(death_rate)
    for spec in is_specs:
        if spec["threshold_quantile"] is not None:
            spec["threshold_rate"] = threshold_map[spec["threshold_quantile"]]

    # Build OOS specs (same threshold_rate as IS — trained on subset but evaluated
    # against the same held-out threshold so comparisons are valid).
    if skip_oos:
        oos_specs: list[dict] = []
        logger.info("--skip-oos: fitting IS components only.")
    else:
        oos_specs = _make_oos_specs(is_specs, n_seeds=n_seeds, n_folds=n_folds)
    all_specs = is_specs + oos_specs
    logger.info(
        "Total specs: %d IS + %d OOS = %d.",
        len(is_specs), len(oos_specs), len(all_specs),
    )

    output_path.mkdir(parents=True, exist_ok=True)

    # Save fold assignments so workers can subset df for OOS fits.
    # Skipped when --skip-oos since workers won't need them.
    fold_assignments_path: Path | None = None
    if not skip_oos:
        fold_assignments_path = _save_fold_assignments(
            df, output_path, n_seeds=n_seeds, n_folds=n_folds
        )

    manifest_path = _write_manifest(all_specs, output_path)

    if local:
        logger.info("Running locally (sequential).")
        fold_df = pd.read_parquet(fold_assignments_path) if fold_assignments_path else None
        _run_local(all_specs, df, output_path, fold_df=fold_df)
        logger.info("All %d components complete.", len(all_specs))
    else:
        _run_with_manifest(
            all_specs, manifest_path, data_path, output_path,
            workflow_name, fold_assignments_path, queue=queue,
        )
