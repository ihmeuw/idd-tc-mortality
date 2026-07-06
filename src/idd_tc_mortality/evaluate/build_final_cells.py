"""
Build the FINAL-grid evaluate task manifest as a *cell set* (20260608 cycle).

The final grid is the explicit 256-config grid defined in ``build_final_specs``
(the 20260517 refined-final grid + the 2026-06-15 tail additions). Unlike
``build_refined_cells`` — which enumerates the *entire* structure-C space — this
enumerates exactly the hand-picked per-stage option sets as a full cartesian:

    s1(2 cov) × s2(2 cov) × bulk(2 exp × 4 cov) × tail(4 fam/exp × 2 cov) = 256

Every combination is nesting-valid (tail_cov ⊆ bulk_cov ⊆ s2_cov; s1 free), so
all 256 cells survive and ``n_skipped`` should be 0 against a manifest built
from the matching spec list.

cells → ``build_hierarchical_cellset`` → ``rectangular_partition(fix=[s1, s2])``:
one task per (s1, s2) = 2 × 2 = 4 tasks, each scoring 8 bulk × 8 tail = 64
explicit cells. The ``s2_n_cov`` task feature is attached so the submitter can
ask size-tiered Slurm resources (one probe per |s2_cov| tier).

The per-stage option sets are imported from ``build_final_specs`` so the spec
list and the cell enumeration share one source of truth.

Usage:
    run-build-final-cells \\
        --manifest-path /…/02-evaluate/20260608_final/manifest.json \\
        --output-path   /…/02-evaluate/20260608_final/cells_manifest.json
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import click

from idd_tools.jobmon import build_hierarchical_cellset, rectangular_partition

from idd_tc_mortality.evaluate.build_half_coupled_multiconfig_tasks import (
    _load_spec_lookup,
    _spec_id_for,
)
from idd_tc_mortality.grid.build_final_specs import (
    BULK_COVS,
    BULK_EXPOSURES,
    BULK_FAMILY,
    S1_COVS,
    S1_FAMILY_MODE,
    S2_COVS,
    S2_FAMILY_MODE,
    TAIL_COVS,
    TAIL_FAMILY_EXPOSURES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CELL_AXES = ["s1_spec_id", "s2_spec_id", "bulk_spec_id", "tail_spec_id"]
FIX_AXES = ["s1_spec_id", "s2_spec_id"]

# Per-(s1, s2) task cell count = |bulk options| × |tail options|.
_CELLS_PER_TASK = len(BULK_EXPOSURES) * len(BULK_COVS) * len(TAIL_FAMILY_EXPOSURES) * len(TAIL_COVS)


def _cov_n_on(cov: dict[str, bool]) -> int:
    """Number of 'on' covariate axes in a covariate_combo dict."""
    return sum(1 for v in cov.values() if v)


def enumerate_final_cells(manifest_path: str) -> tuple[list[dict], dict[str, int], int]:
    """Enumerate the explicit final-grid config cells from the manifest.

    Returns ``(cells, s2id_to_ncov, n_skipped)``. ``cells`` is the list of cell
    dicts (each with the four component spec_ids); ``s2id_to_ncov`` maps each s2
    spec_id to its |s2_cov| (for size-tiered resources); ``n_skipped`` counts
    cells dropped because a component spec was absent from the manifest (should
    be 0 for a manifest built from ``build_final_specs.build_specs``).
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

    for s1_cov in S1_COVS:
        s1_id = _spec_id_for(lookup, "s1", S1_FAMILY_MODE[0], S1_FAMILY_MODE[1], None, s1_cov)
        if s1_id is None:
            n_skipped += 1
            continue

        for s2_cov in S2_COVS:
            for q in thresholds:
                s2_id = _spec_id_for(lookup, "s2", S2_FAMILY_MODE[0], S2_FAMILY_MODE[1], q, s2_cov)
                if s2_id is None:
                    n_skipped += 1
                    continue
                s2id_to_ncov[s2_id] = _cov_n_on(s2_cov)

                for b_em in BULK_EXPOSURES:
                    for b_cov in BULK_COVS:
                        bulk_id = _spec_id_for(lookup, "bulk", BULK_FAMILY, b_em, q, b_cov)
                        if bulk_id is None:
                            n_skipped += 1
                            continue

                        for t_fam, t_em in TAIL_FAMILY_EXPOSURES:
                            for t_cov in TAIL_COVS:
                                tail_id = _spec_id_for(lookup, "tail", t_fam, t_em, q, t_cov)
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


def build_final_cells_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    workflow_name: str = "evaluate-final-cells",
    max_per_task: int | None = None,
) -> dict:
    """Build the partitioned cell manifest and write it to ``output_path`` as JSON.

    Returns the manifest as a plain dict (also written to disk).
    """
    manifest_path = str(manifest_path)
    cells, s2id_to_ncov, n_skipped = enumerate_final_cells(manifest_path)
    if not cells:
        raise ValueError("Enumeration produced no cells — check the manifest contents.")

    cellset = build_hierarchical_cellset(cells, axes=CELL_AXES)

    def _features(group_key: dict) -> dict:
        k = s2id_to_ncov.get(group_key["s2_spec_id"], -1)
        return {"s2_n_cov": k, "expected_n_cells": _CELLS_PER_TASK}

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

    size_by_tier: Counter = Counter()
    cells_by_tier: Counter = Counter()
    for task in task_manifest.tasks:
        k = task.task_features.get("s2_n_cov", -1)
        size_by_tier[k] += 1
        cells_by_tier[k] += len(task.task_args.get("cells", []))

    logger.info(
        "Wrote %d cells in %d tasks to %s (skipped %d cells with a missing manifest spec).",
        len(cells), len(task_manifest.tasks), out, n_skipped,
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
    help="Path to the final manifest.json (from run-evaluate-orchestrate "
         "--refined-specs final_is_specs.json --manifest-only).",
)
@click.option(
    "--output-path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Where to write the partitioned cell manifest JSON.",
)
@click.option(
    "--workflow-name",
    default="evaluate-final-cells",
    show_default=True,
    help="Jobmon workflow name carried on the manifest.",
)
@click.option(
    "--max-per-task",
    type=int,
    default=None,
    help="Optional cap on cells per task. None (default) = one task per (s1, s2) "
         "group (the natural 4-task partition).",
)
def main(manifest_path: str, output_path: str, workflow_name: str, max_per_task: int | None) -> None:
    """Build the final-grid cell manifest (explicit 256-config cartesian)."""
    build_final_cells_manifest(
        manifest_path,
        output_path,
        workflow_name=workflow_name,
        max_per_task=max_per_task,
    )


if __name__ == "__main__":
    main()
