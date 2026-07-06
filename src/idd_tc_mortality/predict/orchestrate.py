"""Build the jobmon DAG for the predict-and-aggregate pipeline.

Two jobmon tiers:

    Tier 1  predict_year_bin     (sd, sc, yb)    predict all 7 basins + agg_year_bin inline
                │
                ▼  (all T1 tasks fan in to one global serial job)
    Tier 2  aggregate_finalize   (global)         agg_scenario → agg_storm_draw → postprocess

The task graph is enumerated entirely from canonical sources:

  - storm_draw_table.csv  -> (storm_draw -> (model, variant))
  - time_bins.csv         -> ((model, variant, scenario) -> [year_bin])
  - SCENARIOS, BASIN_LEVELS in paths.py

No filesystem stats on stage4_v2 or the storm_draw output tree are used to
gate task creation. The predict_basin task itself logs and exits cleanly if
the climada source folder happens to be absent.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from pathlib import Path

import click
import pandas as pd

from idd_tc_mortality.predict.done_manifest import (
    DEFAULT_SCAN_WORKERS,
    TIER_AGG_SCENARIO,
    TIER_AGG_STORM_DRAW,
    TIER_AGG_YEAR_BIN,
    manifest_path as _manifest_path,
    read_manifest,
    scan_and_write,
)
from idd_tc_mortality.predict.paths import (
    BASIN_LEVELS,
    DRAWS_DIR,
    ISLAND_COV_PATH,
    NEW_SDI_PATH,
    OLD_SDI_PATH,
    SCENARIOS,
    STAGE4_DIR,
    STORM_DRAW_DIR,
    STORM_DRAW_TABLE_PATH,
    TIME_BINS_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SLURM_CLUSTER = "slurm"

# Cached list of (model, variant) pairs that actually have stage4_v2 data.
# Used to filter the storm_draw_table to only the storm_draws we have inputs for.
_MODEL_VARIANTS_PATH = Path(__file__).resolve().parents[3] / 'notebooks' / 'model_variants.json'

_PREDICT_RESOURCES = {
    "cores":   1,
    "memory":  "1G",
    "runtime": "10m",
    "queue":   "all.q",
    "project": "proj_rapidresponse",
}
_FINALIZE_RESOURCES = {
    "cores":   1,
    "memory":  "1G",
    "runtime": "10m",
    "queue":   "all.q",
    "project": "proj_rapidresponse",
}


def _load_time_bins(time_bins_path: str) -> pd.DataFrame:
    df = pd.read_csv(time_bins_path)
    df = df[df['start_year'] >= 1970].copy()
    return df[['model', 'variant', 'scenario', 'bin_idx', 'start_year', 'end_year']].drop_duplicates()


def _setup_storm_draw_dirs(
    storm_draws: list[int],
    storm_draw_table: pd.DataFrame,
    storm_draw_dir: Path,
) -> None:
    for sd in storm_draws:
        row = storm_draw_table.loc[storm_draw_table['storm_draw'] == sd]
        if row.empty:
            raise ValueError(f"storm_draw={sd} not in storm_draw_table")
        model   = str(row['source_id'].values[0])
        variant = str(row['variant_label'].values[0])
        sd_dir  = storm_draw_dir / f'storm_draw_{sd}'
        sd_dir.mkdir(parents=True, exist_ok=True)
        info_path = sd_dir / 'info.json'
        if not info_path.exists():
            info_path.write_text(json.dumps({"model": model, "variant": variant}))


def _enumerate_year_bins(
    storm_draws: list[int],
    storm_draw_table: pd.DataFrame,
    time_bins_df: pd.DataFrame,
    scenarios: tuple[str, ...],
) -> dict[tuple[int, str], list[str]]:
    """Return (sd, sc) -> [year_bins] from time_bins_df. No filesystem scan."""
    year_bins_by_sd_sc: dict[tuple[int, str], list[str]] = defaultdict(list)
    for sd in storm_draws:
        row = storm_draw_table.loc[storm_draw_table['storm_draw'] == sd]
        if row.empty:
            continue
        model   = str(row['source_id'].values[0])
        variant = str(row['variant_label'].values[0])
        bins_for_mv = time_bins_df[
            (time_bins_df['model'] == model) & (time_bins_df['variant'] == variant)
        ]
        for sc in scenarios:
            seen: set[str] = set()
            for _, b in bins_for_mv[bins_for_mv['scenario'] == sc].iterrows():
                yb = f"{int(b['start_year'])}-{int(b['end_year'])}"
                if yb not in seen:
                    year_bins_by_sd_sc[(sd, sc)].append(yb)
                    seen.add(yb)
    return year_bins_by_sd_sc


@click.command()
@click.option('--storm-draws',          type=str, default='', help="Comma-separated storm_draw ids; empty = all in table.")
@click.option('--scenarios',            type=str, default='', help="Comma-separated scenarios; empty = all SCENARIOS.")
@click.option('--basins',               type=str, default='', help="Comma-separated basins; empty = all BASIN_LEVELS.")
@click.option('--storm-draw-dir',       type=click.Path(path_type=Path), default=str(STORM_DRAW_DIR), show_default=True)
@click.option('--stage4-dir',           type=click.Path(path_type=Path), default=str(STAGE4_DIR), show_default=True)
@click.option('--draws-dir',            type=click.Path(path_type=Path), default=str(DRAWS_DIR), show_default=True)
@click.option('--storm-draw-table',     type=str, default=STORM_DRAW_TABLE_PATH, show_default=True)
@click.option('--time-bins-path',       type=str, default=TIME_BINS_PATH, show_default=True)
@click.option('--island-cov-path',      type=str, default=ISLAND_COV_PATH, show_default=True)
@click.option('--old-sdi-path',         type=str, default=OLD_SDI_PATH, show_default=True)
@click.option('--new-sdi-path',         type=str, default=NEW_SDI_PATH, show_default=True)
@click.option('--model-variants-path',  type=click.Path(path_type=Path), default=str(_MODEL_VARIANTS_PATH), show_default=True,
              help="JSON file listing (model, variant) pairs with stage4 data. storm_draws outside this set are dropped.")
@click.option('--workflow-name',        type=str, default='predict-tc-mortality')
@click.option('--predict-memory',       type=str, default=_PREDICT_RESOURCES["memory"], show_default=True)
@click.option('--predict-runtime',      type=str, default=_PREDICT_RESOURCES["runtime"], show_default=True)
@click.option('--dry-run',              is_flag=True, help="Print task counts but don't submit.")
@click.option('--skip-manifest-scan',    is_flag=True,
              help="Skip step 0 (filesystem scan to rebuild the done-manifest). "
                   "Trust the existing manifest at <draw_base>/.done_manifest.jsonl. "
                   "Use after a clean run where the manifest is known fresh; saves "
                   "the NFS-walk cost.")
@click.option('--only-manifest-scan',    is_flag=True,
              help="Run only step 0: rebuild the done-manifest and exit. Equivalent "
                   "to `run-build-done-manifest`. No tasks are enumerated or submitted.")
@click.option('--manifest-workers',      type=int, default=DEFAULT_SCAN_WORKERS, show_default=True,
              help="Thread-pool size for the step-0 NFS walk.")
@click.option('--finalize-memory',        type=str, default=_FINALIZE_RESOURCES["memory"],  show_default=True,
              help="Memory ask for the single fan-in aggregate_finalize task.")
@click.option('--finalize-runtime',       type=str, default=_FINALIZE_RESOURCES["runtime"], show_default=True,
              help="Runtime ask for the single fan-in aggregate_finalize task.")
@click.option('--skip-postprocess',       is_flag=True,
              help="Pass --skip-postprocess to aggregate_finalize (runs agg steps but skips "
                   "write_postprocess_outputs). Use only if you intend to run "
                   "`run-predict-postprocess` manually afterwards.")
@click.option('--exposure-col',          type=str, default='person_storm_hours', show_default=True,
              help="Climada column to use as the exposure measure. Default is "
                   "'person_storm_hours'. Pass 'total_population_exposed' for "
                   "experimental population-exposed runs.")
def main(
    storm_draws, scenarios, basins,
    storm_draw_dir, stage4_dir, draws_dir,
    storm_draw_table, time_bins_path,
    island_cov_path, old_sdi_path, new_sdi_path,
    model_variants_path,
    workflow_name, predict_memory, predict_runtime,
    dry_run, skip_manifest_scan, only_manifest_scan, manifest_workers,
    finalize_memory, finalize_runtime, skip_postprocess,
    exposure_col,
):
    storm_draw_dir = Path(storm_draw_dir)
    stage4_dir     = Path(stage4_dir)
    draws_dir      = Path(draws_dir)

    # --- Step 0: done-manifest scan ---------------------------------------
    # The manifest at <draw_base>/.done_manifest.jsonl lists every terminal
    # output already on disk. We use it to prune enumeration so jobmon only
    # sees tasks whose output is missing.
    #
    # Modes:
    #   default                : scan filesystem, rewrite manifest, then proceed
    #   --skip-manifest-scan   : trust existing manifest, proceed
    #   --only-manifest-scan   : scan filesystem, rewrite manifest, exit
    if only_manifest_scan and skip_manifest_scan:
        raise click.UsageError("--only-manifest-scan and --skip-manifest-scan are mutually exclusive.")
    if not skip_manifest_scan:
        logger.info("Step 0: scanning %s for done outputs with %d workers ...",
                    storm_draw_dir, manifest_workers)
        t0 = time.monotonic()
        total, counts = scan_and_write(storm_draw_dir, workers=manifest_workers)
        for tier in (TIER_AGG_YEAR_BIN, TIER_AGG_SCENARIO, TIER_AGG_STORM_DRAW):
            logger.info("  %s: %d done", tier, counts.get(tier, 0))
        logger.info("Step 0: wrote %d records to %s in %.1fs.",
                    total, _manifest_path(storm_draw_dir), time.monotonic() - t0)
    if only_manifest_scan:
        logger.info("--only-manifest-scan: exiting after step 0.")
        return
    done = read_manifest(storm_draw_dir)
    logger.info(
        "Manifest: agg_year_bin=%d, agg_scenario=%d, agg_storm_draw=%d done.",
        len(done[TIER_AGG_YEAR_BIN]),
        len(done[TIER_AGG_SCENARIO]), len(done[TIER_AGG_STORM_DRAW]),
    )

    sd_table = pd.read_csv(storm_draw_table)

    # Filter to (model, variant) pairs that actually have stage4_v2 data.
    mv_data = json.loads(Path(model_variants_path).read_text())
    mv_set = {(d['model'], d['variant']) for d in mv_data['model_variants']}
    in_mv = sd_table.apply(lambda r: (r['source_id'], r['variant_label']) in mv_set, axis=1)
    n_dropped = (~in_mv).sum()
    if n_dropped:
        dropped = sd_table.loc[~in_mv, ['storm_draw', 'source_id', 'variant_label']]
        logger.info("Dropping %d storm_draws lacking stage4 data:", n_dropped)
        for _, r in dropped.iterrows():
            logger.info("  storm_draw=%d  (%s, %s)", r['storm_draw'], r['source_id'], r['variant_label'])
    sd_table = sd_table[in_mv].reset_index(drop=True)

    if storm_draws:
        sd_list = [int(x) for x in storm_draws.split(',')]
        # Honor the filter even when caller passes an explicit list.
        sd_list = [sd for sd in sd_list if sd in set(sd_table['storm_draw'])]
    else:
        sd_list = sorted(sd_table['storm_draw'].unique().tolist())

    sc_tuple    = tuple(scenarios.split(',')) if scenarios else SCENARIOS
    basin_tuple = tuple(basins.split(','))    if basins    else BASIN_LEVELS

    time_bins_df = _load_time_bins(time_bins_path)

    logger.info("Setup: %d storm_draw dirs.", len(sd_list))
    _setup_storm_draw_dirs(sd_list, sd_table, storm_draw_dir)

    year_bins_by_sd_sc = _enumerate_year_bins(sd_list, sd_table, time_bins_df, sc_tuple)

    # Full enumeration (pre-manifest pruning) — for the "would have been" counts in the log.
    # Tier 1 terminal output is the year_bin mean (written by aggregate_year_bin inline).
    all_year_bin_keys   = {(sd, sc, yb) for (sd, sc), ybs in year_bins_by_sd_sc.items() for yb in ybs}
    all_scenario_keys   = set(year_bins_by_sd_sc.keys())
    all_storm_draw_keys = {sd for sd, _ in all_scenario_keys}

    # Prune against the done-manifest: a task is created only if its terminal
    # output is missing. Downstream dependencies are added only on upstream
    # tasks that *were* created (done upstreams have their files on disk
    # already and need no dependency edge).
    year_bin_keys   = {k for k in all_year_bin_keys   if k not in done[TIER_AGG_YEAR_BIN]}
    scenario_keys   = {k for k in all_scenario_keys   if k not in done[TIER_AGG_SCENARIO]}
    storm_draw_keys = {k for k in all_storm_draw_keys if k not in done[TIER_AGG_STORM_DRAW]}

    n_yb = len(year_bin_keys)
    n_sc = len(scenario_keys)
    n_sd = len(storm_draw_keys)
    n_finalize = 1 if (n_yb + n_sc + n_sd) > 0 else 0
    total = n_yb + n_finalize

    logger.info("Task counts (after manifest pruning):")
    logger.info("  predict_year_bin  : %d / %d  enumerated / would-be (sd×sc×yb)",
                n_yb, len(all_year_bin_keys))
    logger.info("  aggregate_finalize: %d / 1   enumerated (pending: agg_scenario=%d, agg_storm_draw=%d)",
                n_finalize, n_sc, n_sd)
    logger.info("  ---------------------------------")
    logger.info("  total             : %d", total)

    if dry_run:
        logger.info("Dry run — exiting before jobmon submission.")
        return

    from jobmon.client.api import Tool

    predict_resources = dict(_PREDICT_RESOURCES)
    predict_resources["memory"]  = predict_memory
    predict_resources["runtime"] = predict_runtime
    finalize_resources = dict(_FINALIZE_RESOURCES)
    finalize_resources["memory"]  = finalize_memory
    finalize_resources["runtime"] = finalize_runtime

    logs_dir = storm_draw_dir / 'logs'
    for sub in ('predict_year_bin', 'finalize'):
        (logs_dir / sub).mkdir(parents=True, exist_ok=True)

    tool = Tool("idd-tc-mortality")
    unique_name = f"{workflow_name}_{uuid.uuid4()}"
    wf = tool.create_workflow(
        workflow_args=unique_name,
        name=unique_name,
        description="predict + aggregate TC mortality across storm_draws",
    )

    tt_predict_yb = tool.get_task_template(
        template_name="predict_year_bin",
        command_template=(
            "run-predict-year-bin"
            " --storm-draw {storm_draw}"
            " --scenario {scenario}"
            " --year-bin {year_bin}"
            " --basins {basins}"
            " --stage4-dir {stage4_dir}"
            " --storm-draw-dir {storm_draw_dir}"
            " --draws-dir {draws_dir}"
            " --storm-draw-table {storm_draw_table}"
            " --island-cov-path {island_cov_path}"
            " --old-sdi-path {old_sdi_path}"
            " --new-sdi-path {new_sdi_path}"
            " --exposure-col {exposure_col}"
        ),
        node_args=["storm_draw", "scenario", "year_bin"],
        task_args=["basins", "stage4_dir", "storm_draw_dir", "draws_dir", "storm_draw_table",
                   "island_cov_path", "old_sdi_path", "new_sdi_path",
                   "exposure_col"],
        op_args=[],
    )
    tt_finalize = tool.get_task_template(
        template_name="aggregate_finalize",
        command_template=(
            "run-aggregate-finalize"
            " --storm-draw-dir {storm_draw_dir}"
            " --storm-draw-table {storm_draw_table}"
            " --time-bins-path {time_bins_path}"
            " --storm-draws {storm_draws_str}"
            " --scenarios {scenarios_str}"
            " {skip_pp_flag}"
        ),
        node_args=[],
        task_args=["storm_draw_dir", "storm_draw_table", "time_bins_path",
                   "storm_draws_str", "scenarios_str", "skip_pp_flag"],
        op_args=[],
    )

    tt_predict_yb.update_default_compute_resources(_SLURM_CLUSTER, **predict_resources)
    tt_finalize.update_default_compute_resources(_SLURM_CLUSTER, **finalize_resources)

    sd_str       = str(storm_draw_dir)
    stage4_str   = str(stage4_dir)
    draws_str    = str(draws_dir)

    # Accumulate (child, parent) tuples across all tiers; wire them via
    # add_upstream AFTER wf.add_tasks() so jobmon sees a complete node set
    # before resolving any edges. Cleaner shape than per-loop add_upstream;
    # matches what idd-tools/jobmon submit_with_manifest emits.
    dependencies: list[tuple[object, object]] = []

    basins_str = ','.join(basin_tuple)

    # --- Tier 1: predict_year_bin --- (all basins + agg_year_bin inline)
    year_bin_tasks: dict[tuple[int, str, str], object] = {}
    for (sd, sc), ybs in year_bins_by_sd_sc.items():
        for yb in ybs:
            if (sd, sc, yb) not in year_bin_keys:
                continue
            tag = f"sd{sd}_{sc}_{yb}"
            task = tt_predict_yb.create_task(
                storm_draw=str(sd), scenario=sc, year_bin=yb,
                basins=basins_str,
                stage4_dir=stage4_str, storm_draw_dir=sd_str, draws_dir=draws_str,
                storm_draw_table=storm_draw_table,
                island_cov_path=island_cov_path,
                old_sdi_path=old_sdi_path, new_sdi_path=new_sdi_path,
                exposure_col=exposure_col,
                cluster_name=_SLURM_CLUSTER,
                compute_resources={
                    **predict_resources,
                    "stdout": str(logs_dir / 'predict_year_bin' / f"{tag}.out"),
                    "stderr": str(logs_dir / 'predict_year_bin' / f"{tag}.err"),
                },
            )
            year_bin_tasks[(sd, sc, yb)] = task

    # --- Tier 2: aggregate_finalize --- (single global serial job: agg_scenario → agg_storm_draw → postprocess)
    finalize_tasks: list[object] = []
    if n_finalize > 0:
        finalize_task = tt_finalize.create_task(
            storm_draw_dir=sd_str,
            storm_draw_table=storm_draw_table,
            time_bins_path=time_bins_path,
            storm_draws_str=','.join(str(s) for s in sd_list),
            scenarios_str=','.join(sc_tuple),
            skip_pp_flag="--skip-postprocess" if skip_postprocess else "",
            cluster_name=_SLURM_CLUSTER,
            compute_resources={
                **finalize_resources,
                "stdout": str(logs_dir / 'finalize' / "finalize.out"),
                "stderr": str(logs_dir / 'finalize' / "finalize.err"),
            },
        )
        for upstream in year_bin_tasks.values():
            dependencies.append((finalize_task, upstream))
        finalize_tasks.append(finalize_task)

    all_tasks = list(year_bin_tasks.values()) + finalize_tasks
    wf.add_tasks(all_tasks)

    # Wire all dependencies AFTER add_tasks (Script-1 pattern). jobmon sees the
    # full node set before any edge resolution; cleaner code shape and matches
    # what idd-tools/jobmon submit_with_manifest emits. Note: this is not
    # expected to materially reduce bind time vs the in-loop add_upstream
    # variant — the dominant cost is database I/O on edge inserts inside
    # jobmon's backend, which happens regardless of construction order.
    for child, parent in dependencies:
        child.add_upstream(parent)
    logger.info("Wired %d dependencies.", len(dependencies))

    logger.info("Binding workflow with %d tasks ...", len(all_tasks))
    wf.bind()
    logger.info("Submitting workflow %r ...", unique_name)
    status = wf.run()
    logger.info("Workflow finished with status: %s", status)


if __name__ == "__main__":
    main()
