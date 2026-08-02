"""
Build the refined-grid evaluate task manifest as a *cell set*, using idd_tools'
cells/partition machinery rather than a hand-rolled Cartesian task file.

Structure (post-2000 refined grid, 2026-06-12 decision):
    - s1_cov  : free  (any of the 16 covariate subsets)
    - s2_cov  : free  (independent of s1_cov)
    - bulk_cov ⊆ s2_cov
    - tail_cov ⊆ bulk_cov
i.e. the chain ``s2 ⊇ bulk ⊇ tail`` with s1 standing on its own. The two
hurdle stages (s1, s2) — the ones most likely to need rich, *different*
covariate sets — pick covariates independently; the rate stages (bulk, then
the data-starved tail) use progressively sparser subsets of s2's covariates.

Why cells, not a Cartesian task file
------------------------------------
A *cell* is one fully-specified DH config — its four component spec_ids. We
enumerate exactly the cells satisfying the nesting above (a custom subset of
the complete most-detailed cell space), hand them to
``build_hierarchical_cellset``, then ``rectangular_partition(fix=[s1_spec_id,
s2_spec_id])`` groups them into one task per (s1, s2) — which, because the
family/exposure axes are pinned, is exactly one task per (s1_cov, s2_cov,
threshold). The worker scores the explicit cells (no blind bulk×tail
Cartesian), so the ``tail ⊆ bulk`` nesting is honoured without pinning bulk
per task or filtering inside the worker.

The pinned family/exposure/covariate decisions are imported from
``build_refined_specs_post2000`` so the spec list and the cell enumeration
share a single source of truth. Thresholds are read from the manifest (the
set of s2 ``threshold_quantile`` values present), so the cells always match
whatever spec list the manifest was built from.

Per-task size scales as ``14 × 3^|s2_cov|`` (14, 42, 126, 378, 1,134 for
|s2_cov| = 0..4). The ``s2_n_cov`` task feature is attached so the submitter
can ask size-tiered Slurm resources (one probe per |s2_cov| tier).

Usage:
    run-build-refined-cells \\
        --manifest-path /…/02-evaluate/<date>_refined/manifest.json \\
        --output-path   /…/02-evaluate/<date>_refined/cells_manifest.json
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import click

from idd_tools.jobmon import (
    build_hierarchical_cellset,
    inflate_cells,
    pack_to_target,
    rectangular_partition,
)

from idd_tc_mortality.evaluate.build_half_coupled_multiconfig_tasks import (
    _load_spec_lookup,
    _spec_id_for,
    _tail_subsets_of as _cov_subsets,  # "all cov subsets of a given cov" — name is generic
)
from idd_tc_mortality.grid import build_refined_specs_post2000, build_refined_specs_tailvariant

# Which spec module's locked family/exposure/covariate decisions drive the
# enumeration. The manifest must have been built from the matching spec list,
# or every cell will miss its spec_id lookup and be skipped.
GRID_MODULES = {
    "post2000": build_refined_specs_post2000,
    "tailvariant": build_refined_specs_tailvariant,
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Cell schema: a cell is one DH config, identified by its four component
# spec_ids. Families/exposures/covs/threshold are all encoded inside the ids
# (the manifest resolved them), so the four ids fully specify the config.
CELL_AXES = ["s1_spec_id", "s2_spec_id", "bulk_spec_id", "tail_spec_id"]

# Partition key: grouping by (s1_spec_id, s2_spec_id) == grouping by
# (s1_cov, s2_cov, threshold), since s1/s2 family+exposure are pinned and the
# s2 spec_id encodes its threshold. One task per (s1, s2) group.
FIX_AXES = ["s1_spec_id", "s2_spec_id"]


def _cov_n_on(cov: dict[str, bool]) -> int:
    """Number of 'on' covariate axes in a covariate_combo dict."""
    return sum(1 for v in cov.values() if v)


def _cell_key(cell: dict) -> tuple[str, str, str, str]:
    return (
        cell["s1_spec_id"], cell["s2_spec_id"],
        cell["bulk_spec_id"], cell["tail_spec_id"],
    )


def _done_cells_from_partials(
    cells_manifest_path: str | Path, partials_dir: str | Path
) -> tuple[set[tuple[str, str, str, str]], int]:
    """Cells already computed under a PRIOR cells manifest.

    A partial ``dh_task_{i:05d}.parquet`` covers exactly the cells of task
    ``i`` in the manifest that produced it, so the (manifest, partials dir)
    pair recovers the done-cell set. Used by ``--exclude-done-*`` to re-pack
    only the remaining cells after a partition change, instead of deleting
    completed work. The kept partials must then be RENAMED to
    ``dh_keep_*.parquet``: the assembler globs ``dh_*.parquet`` (so they are
    still concatenated into dh_results), while done-detection matches only
    ``dh_task_{i:05d}.parquet`` under the NEW numbering (so they can no
    longer mark the wrong task as done).
    """
    with open(cells_manifest_path) as f:
        doc = json.load(f)
    done: set[tuple[str, str, str, str]] = set()
    n_partials = 0
    for i, task in enumerate(doc["tasks"]):
        if (Path(partials_dir) / f"dh_task_{i:05d}.parquet").exists():
            n_partials += 1
            for cell in inflate_cells(task["task_args"]):
                done.add(_cell_key(cell))
    return done, n_partials


def enumerate_refined_cells(
    manifest_path: str, grid_mod=build_refined_specs_post2000
) -> tuple[list[dict], dict[str, int], int]:
    """Enumerate the structure-C config cells from the refined manifest.

    Returns ``(cells, s2id_to_ncov, n_skipped)`` where ``cells`` is the list of
    cell dicts (each with the four spec_id axes), ``s2id_to_ncov`` maps each
    s2 spec_id to its |s2_cov| (for size-tiered resources), and ``n_skipped``
    counts cells dropped because a component spec was absent from the manifest
    (should be 0 for a manifest built from the matching spec list).
    """
    lookup = _load_spec_lookup(manifest_path)

    with open(manifest_path) as f:
        manifest: dict[str, dict] = json.load(f)
    thresholds = sorted(
        {
            float(spec["threshold_quantile"])
            for spec in manifest.values()
            if spec["component"] == "s2" and spec.get("threshold_quantile") is not None
        }
    )
    if not thresholds:
        raise ValueError(f"No s2 thresholds found in manifest {manifest_path}.")
    logger.info("Thresholds from manifest: %s", thresholds)

    cells: list[dict] = []
    s2id_to_ncov: dict[str, int] = {}
    n_skipped = 0

    for s1_cov in grid_mod.COV_COMBOS:
        for s1_fam, s1_exp in grid_mod.S1_FAMILY_MODES:
            s1_id = _spec_id_for(lookup, "s1", s1_fam, s1_exp, None, s1_cov)
            if s1_id is None:
                n_skipped += 1
                continue

            for s2_cov in grid_mod.COV_COMBOS:
                for q in thresholds:
                    for s2_fam, s2_exp in grid_mod.S2_FAMILY_MODES:
                        s2_id = _spec_id_for(lookup, "s2", s2_fam, s2_exp, q, s2_cov)
                        if s2_id is None:
                            n_skipped += 1
                            continue
                        s2id_to_ncov[s2_id] = _cov_n_on(s2_cov)

                        for bulk_cov in _cov_subsets(s2_cov):
                            for b_fam, b_exp in grid_mod.BULK_FAMILY_MODES:
                                bulk_id = _spec_id_for(lookup, "bulk", b_fam, b_exp, q, bulk_cov)
                                if bulk_id is None:
                                    n_skipped += 1
                                    continue

                                for tail_cov in _cov_subsets(bulk_cov):
                                    for t_fam, t_exp in grid_mod.TAIL_FAMILY_MODES:
                                        tail_id = _spec_id_for(
                                            lookup, "tail", t_fam, t_exp, q, tail_cov
                                        )
                                        if tail_id is None:
                                            n_skipped += 1
                                            continue
                                        cells.append(
                                            {
                                                "s1_spec_id":   s1_id,
                                                "s2_spec_id":   s2_id,
                                                "bulk_spec_id": bulk_id,
                                                "tail_spec_id": tail_id,
                                            }
                                        )

    return cells, s2id_to_ncov, n_skipped


def build_refined_cells_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    workflow_name: str = "evaluate-refined-cells",
    max_per_task: int | None = None,
    grid: str = "post2000",
    pack_target_s: float | None = None,
    pack_marginal_s: float = 0.49,
    pack_load_s: float = 60.0,
    exclude_done_manifest: str | Path | None = None,
    exclude_done_partials: str | Path | None = None,
) -> dict:
    """Build the partitioned cell manifest and write it to ``output_path`` as JSON.

    Returns the manifest as a plain dict (also written to disk).
    """
    if (exclude_done_manifest is None) != (exclude_done_partials is None):
        raise ValueError(
            "--exclude-done-manifest and --exclude-done-partials must be given together."
        )
    grid_mod = GRID_MODULES[grid]
    manifest_path = str(manifest_path)
    cells, s2id_to_ncov, n_skipped = enumerate_refined_cells(manifest_path, grid_mod)
    if exclude_done_manifest is not None:
        done, n_partials = _done_cells_from_partials(
            exclude_done_manifest, exclude_done_partials
        )
        before = len(cells)
        cells = [c for c in cells if _cell_key(c) not in done]
        logger.info(
            "Excluded %d already-computed cells (from %d partials under the prior "
            "manifest); %d cells remain. RENAME the kept partials to "
            "dh_keep_*.parquet before submitting with the new manifest.",
            before - len(cells), n_partials, len(cells),
        )
    if not cells:
        raise ValueError("Enumeration produced no cells — check the manifest contents.")

    cellset = build_hierarchical_cellset(cells, axes=CELL_AXES)

    # Cells per (s1, s2) group = (bulk modes × tail modes) × 3^|s2_cov|; the
    # leading factor is grid-dependent (post2000: 14, tailvariant: 64).
    chain_modes = len(grid_mod.BULK_FAMILY_MODES) * len(grid_mod.TAIL_FAMILY_MODES)

    def _features(group_key: dict) -> dict:
        # group_key carries the fix axes: {s1_spec_id, s2_spec_id}. |s2_cov|
        # determines the task's cell count, so tier on it.
        k = s2id_to_ncov.get(group_key["s2_spec_id"], -1)
        return {"s2_n_cov": k, "expected_n_cells": chain_modes * (3 ** k) if k >= 0 else -1}

    if pack_target_s is not None:
        # Pack whole groups (and mix small ones) into ~pack_target_s tasks.
        # Cost model: pack_load_s per distinct (s1, s2) group in a task +
        # pack_marginal_s per cell — both measured from the 2026-07-27 probes.
        task_manifest = pack_to_target(
            cellset,
            group_by=FIX_AXES,
            marginal_s=pack_marginal_s,
            load_s=pack_load_s,
            target_s=pack_target_s,
            workflow_name=workflow_name,
            task_template="evaluate_cells",
        ).manifest
    else:
        task_manifest = rectangular_partition(
            cellset,
            fix=FIX_AXES,
            workflow_name=workflow_name,
            task_template="evaluate_cells",
            max_per_task=max_per_task,
            features_fn=_features,
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(task_manifest.model_dump_json(indent=2))

    # Report the per-|s2_cov| task-size distribution (drives the probe-by-tier plan).
    size_by_tier: Counter = Counter()
    cells_by_tier: Counter = Counter()
    for task in task_manifest.tasks:
        k = task.task_features.get("s2_n_cov", -1)
        size_by_tier[k] += 1
        cells_by_tier[k] += len(task.task_args.get("cells", []))

    sizes = sorted(len(t.task_args.get("cells", [])) for t in task_manifest.tasks)
    logger.info(
        "Wrote %d cells in %d tasks to %s (skipped %d cells with a missing manifest spec). "
        "Cells/task min=%d median=%d max=%d.",
        len(cells), len(task_manifest.tasks), out, n_skipped,
        sizes[0], sizes[len(sizes) // 2], sizes[-1],
    )
    logger.info("Task-size distribution by |s2_cov|:")
    for k in sorted(size_by_tier):
        n_tasks = size_by_tier[k]
        n_cells = cells_by_tier[k]
        logger.info(
            "  |s2_cov|=%d : %d tasks × %d cells/task = %d cells",
            k, n_tasks, (n_cells // n_tasks if n_tasks else 0), n_cells,
        )

    return json.loads(task_manifest.model_dump_json())


@click.command()
@click.option(
    "--manifest-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the refined manifest.json (from run-evaluate-orchestrate --refined-specs "
         "--manifest-only).",
)
@click.option(
    "--output-path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Where to write the partitioned cell manifest JSON.",
)
@click.option(
    "--workflow-name",
    default="evaluate-refined-cells",
    show_default=True,
    help="Jobmon workflow name carried on the manifest.",
)
@click.option(
    "--max-per-task",
    type=int,
    default=None,
    help="Optional cap on cells per task. None (default) = one task per (s1, s2) "
         "group (the natural 512-task partition). Set this to split the fat "
         "|s2_cov|=4 groups if the probes show they exceed the resource ask.",
)
@click.option(
    "--grid",
    type=click.Choice(sorted(GRID_MODULES)),
    default="post2000",
    show_default=True,
    help="Which locked spec-module decisions drive the enumeration. Must match "
         "the spec list the manifest was built from.",
)
@click.option(
    "--pack-target-s",
    type=float,
    default=None,
    help="Pack groups into ~this-many-second tasks via idd_tools pack_to_target "
         "(mixing small groups; whole groups preserved where they fit). "
         "Overrides the one-task-per-(s1,s2) partition and --max-per-task.",
)
@click.option(
    "--pack-marginal-s",
    type=float,
    default=0.49,
    show_default=True,
    help="Packing cost model: seconds per cell (measured 2026-07-27 probes).",
)
@click.option(
    "--pack-load-s",
    type=float,
    default=60.0,
    show_default=True,
    help="Packing cost model: seconds per distinct (s1, s2) group in a task "
         "(startup + shared component fits).",
)
@click.option(
    "--exclude-done-manifest",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A PRIOR cells manifest: cells covered by its existing partials are "
         "excluded from the new build. Use with --exclude-done-partials.",
)
@click.option(
    "--exclude-done-partials",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Partials dir holding dh_task_*.parquet written under "
         "--exclude-done-manifest's numbering. Rename kept partials to "
         "dh_keep_*.parquet before submitting the new manifest.",
)
def main(
    manifest_path: str,
    output_path: str,
    workflow_name: str,
    max_per_task: int | None,
    grid: str,
    pack_target_s: float | None,
    pack_marginal_s: float,
    pack_load_s: float,
    exclude_done_manifest: str | None,
    exclude_done_partials: str | None,
) -> None:
    """Build the refined-grid cell manifest (structure C: s1 free, s2 ⊇ bulk ⊇ tail)."""
    build_refined_cells_manifest(
        manifest_path,
        output_path,
        workflow_name=workflow_name,
        max_per_task=max_per_task,
        grid=grid,
        pack_target_s=pack_target_s,
        pack_marginal_s=pack_marginal_s,
        pack_load_s=pack_load_s,
        exclude_done_manifest=exclude_done_manifest,
        exclude_done_partials=exclude_done_partials,
    )


if __name__ == "__main__":
    main()
