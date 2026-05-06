"""
CLI entry point for orchestrating a full grid fit run.

Usage (preliminary):
    run-fit-orchestrate \\
        --mode preliminary \\
        --data-path <path/to/data.parquet> \\
        --output-dir <dir> \\
        [--local]

Usage (refined — all four override parameters are required):
    run-fit-orchestrate \\
        --mode refined \\
        --data-path <path/to/data.parquet> \\
        --output-dir <dir> \\
        --thresholds 0.80 --thresholds 0.85 \\
        --bulk-families gamma --bulk-families nb \\
        --tail-families gpd --tail-families gamma \\
        --covariate-combos '{"wind_speed":true,"sdi":true,"basin":false,"is_island":false}' \\
        [--local]

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
Each component fit is single-threaded and light on memory (< 2 GB).
Runtime is dominated by GLM convergence; 30 min is generous for any single spec.
Queue and project are site defaults and can be adjusted here if needed.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
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
    "memory": "2G",
    "runtime": "30m",
    "queue": "all.q",
    "project": "proj_rapidresponse",
}


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
    for i, spec in enumerate(specs, 1):
        cid = component_id(spec)
        logger.info("[%d/%d] Fitting %s (%s/%s)", i, n, cid, spec["component"], spec.get("family"))
        df_train = _subset_df_for_spec(spec, df, fold_df)
        result = fit_one_component(spec, df_train)
        save_result(result, spec, output_dir, overwrite=True)


def _run_jobmon(
    specs: list[dict],
    manifest_path: Path,
    data_path: str,
    output_dir: Path,
    workflow_name: str,
    fold_assignments_path: Path | None,
) -> None:
    """Submit one Slurm job per spec via jobmon and wait for completion."""
    from jobmon.client.api import Tool  # deferred import — jobmon not needed for --local

    tool = Tool("idd-tc-mortality")
    unique_name = f"{workflow_name}_{uuid.uuid4()}"
    wf = tool.create_workflow(
        workflow_args=unique_name,
        name=unique_name,
        description="fit grid run",
    )

    # Entry-point command installed via pyproject.toml [tool.poetry.scripts].
    # --fold-assignments is always passed so IS and OOS specs share one template.
    tt = tool.get_task_template(
        template_name="fit_component",
        command_template=(
            "run-fit-component"
            " --spec-id {spec_id}"
            " --manifest {manifest_path}"
            " --data-path {data_path}"
            " --output-dir {output_dir}"
            " --fold-assignments {fold_assignments_path}"
        ),
        node_args=["spec_id"],
        task_args=["manifest_path", "data_path", "output_dir", "fold_assignments_path"],
        op_args=[],
    )
    tt.update_default_compute_resources(_SLURM_CLUSTER, **_SLURM_RESOURCES)

    fa_str = str(fold_assignments_path) if fold_assignments_path else ""
    tasks = [
        tt.create_task(
            spec_id=component_id(spec),
            manifest_path=str(manifest_path),
            data_path=str(data_path),
            output_dir=str(output_dir),
            fold_assignments_path=fa_str,
            cluster_name=_SLURM_CLUSTER,
        )
        for spec in specs
    ]

    wf.add_tasks(tasks)
    wf.bind()
    logger.info("Submitted %d jobmon tasks for workflow %r", len(tasks), workflow_name)
    wf.run()


@click.command()
@click.option(
    "--mode",
    required=True,
    type=click.Choice(["preliminary", "refined"]),
    help="Grid mode. 'preliminary' uses fixed defaults; "
         "'refined' requires --thresholds, --covariate-combos, "
         "--bulk-families, and --tail-families.",
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
def main(
    mode: str,
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
) -> None:
    """Orchestrate a full grid fit run: enumerate specs, write manifest, submit jobs."""
    output_path = Path(output_dir)

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
        _run_jobmon(
            all_specs, manifest_path, data_path, output_path,
            workflow_name, fold_assignments_path,
        )
