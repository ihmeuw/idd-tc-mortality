"""
Per-tier cost probes for the FINAL grid, and their analyzer.

``build`` mode: draws a stratified size sweep over the (tail_family @
threshold) cost tiers via idd_tools ``build_size_sweep_manifest`` — two
family-pure probe tasks per tier at tier-appropriate sizes — and writes it as
a cells manifest that ``run-evaluate-orchestrate --cells-file`` submits like
any other. Probe tasks should run in a SEPARATE --output-dir (the builder
copies manifest.json + fold_assignments.parquet there) so their partials
never collide with a real run's task numbering.

``analyze`` mode: given the probe workflow id, joins per-task elapsed times
(idd_tools ``collect_workflow``) to the sweep's tier/size labels and fits the
two-point slope per tier: marginal seconds-per-cell + shared startup. Prints
the ``--pack-costs`` string for ``run-build-final-cells``.

Usage:
    python -m idd_tc_mortality.evaluate.build_final_cost_probe build \\
        --manifest-path <dir>/manifest.json --grid tailvariant-v2000 \\
        --probe-dir <dir>/probe_costs
    # ... submit via run-evaluate-orchestrate --cells-file, then:
    python -m idd_tc_mortality.evaluate.build_final_cost_probe analyze \\
        --probe-dir <dir>/probe_costs --workflow-id NNN
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import click

from idd_tools.jobmon import build_size_sweep_manifest, build_hierarchical_cellset

from idd_tc_mortality.evaluate.build_final_cells import (
    CELL_AXES,
    GRID_SOURCES,
    enumerate_final_cells,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default probe sizes per tier family: sized so each probe task runs ~5-15
# minutes at the per-cell costs estimated from the 2026-07-29..31 runs
# (non-gpd ~1.5-3 s/cell; gpd_cens@0.70 ~15 s; gpd_cens@0.95 ~40 s).
DEFAULT_FAMILY_SIZES: dict[str, list[int]] = {
    "gpd_cens@0.95": [8, 16],
    "gpd_cens@0.70": [16, 32],
    "gpd_trunc@0.85": [24, 48],
    "gpd_trunc@0.70": [24, 48],
    "default": [128, 256],
}


def tier_sizes(tiers: list[str]) -> dict[str, list[int]]:
    """Per-tier probe sizes: explicit entry if present, else the default."""
    return {t: DEFAULT_FAMILY_SIZES.get(t, DEFAULT_FAMILY_SIZES["default"])
            for t in tiers}


def build_probe(manifest_path: str, grid: str, probe_dir: str | Path,
                seed: int = 0) -> dict:
    """Write the probe cells manifest + copies of manifest/folds into probe_dir."""
    grid_mod = GRID_SOURCES[grid]
    cells, _, n_skipped = enumerate_final_cells(manifest_path, grid_mod)
    assert n_skipped == 0, f"{n_skipped} cells missing specs"
    cellset = build_hierarchical_cellset(cells, axes=CELL_AXES)

    tiers = sorted({c["tier"] for c in cells})
    sizes = tier_sizes(tiers)
    plan = build_size_sweep_manifest(
        cellset,
        group_by=["tier"],
        sizes=sizes,
        workflow_name="final-cost-probe",
        task_template="evaluate_cells",
        random=True,
        seed=seed,
    )

    probe_dir = Path(probe_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    out = probe_dir / "cells_manifest_probe.json"
    out.write_text(plan.manifest.model_dump_json(indent=2))
    # The orchestrator requires manifest.json + fold_assignments.parquet in
    # the output dir it submits from.
    src_dir = Path(manifest_path).parent
    shutil.copy2(manifest_path, probe_dir / "manifest.json")
    shutil.copy2(src_dir / "fold_assignments.parquet",
                 probe_dir / "fold_assignments.parquet")

    logger.info("Wrote %d probe tasks (%d tiers x 2 sizes) to %s",
                len(plan.manifest.tasks), len(tiers), out)
    for task, size in zip(plan.manifest.tasks, plan.sizes):
        tier = task.task_args.get("tier", task.task_features.get("tier", "?"))
        logger.info("  task %-3s tier=%-22s n_cells=%d",
                    task.task_args.get("task_index", task.index), tier, size)
    return json.loads(plan.manifest.model_dump_json())


def fit_tier_costs(probe_manifest: dict, task_elapsed: dict[int, float],
                   ) -> tuple[dict[str, float], float]:
    """Two-point fit per tier: (marginal s/cell per tier, median startup s).

    With sizes n1 < n2 and elapsed t1, t2 per tier:
        marginal = (t2 - t1) / (n2 - n1);  startup = t1 - marginal * n1.
    """
    from idd_tools.jobmon import inflate_cells
    import statistics as st

    by_tier: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for i, task in enumerate(probe_manifest["tasks"]):
        if i not in task_elapsed:
            continue
        cells = inflate_cells(task["task_args"])
        tier = cells[0]["tier"]
        by_tier[tier].append((len(cells), task_elapsed[i]))

    marginals: dict[str, float] = {}
    startups: list[float] = []
    for tier, pts in by_tier.items():
        pts.sort()
        if len(pts) >= 2 and pts[-1][0] > pts[0][0]:
            (n1, t1), (n2, t2) = pts[0], pts[-1]
            m = (t2 - t1) / (n2 - n1)
            marginals[tier] = max(m, 0.01)
            startups.append(t1 - m * n1)
        elif pts:
            n1, t1 = pts[0]
            marginals[tier] = max(t1 / n1, 0.01)  # startup-inflated upper bound
    return marginals, (st.median(startups) if startups else 60.0)


@click.group()
def main() -> None:
    """Final-grid per-tier cost probes."""


@main.command()
@click.option("--manifest-path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--grid", type=click.Choice(sorted(GRID_SOURCES)), required=True)
@click.option("--probe-dir", required=True, type=click.Path(file_okay=False))
@click.option("--seed", type=int, default=0, show_default=True)
def build(manifest_path: str, grid: str, probe_dir: str, seed: int) -> None:
    """Build the probe cells manifest (submit with run-evaluate-orchestrate)."""
    build_probe(manifest_path, grid, probe_dir, seed=seed)


@main.command()
@click.option("--probe-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--workflow-id", required=True, type=int)
def analyze(probe_dir: str, workflow_id: int) -> None:
    """Fit per-tier costs from the probe workflow; print --pack-costs."""
    import re

    from idd_tools.jobmon import collect_workflow

    doc = json.loads((Path(probe_dir) / "cells_manifest_probe.json").read_text())
    df = collect_workflow(workflow_id)
    task_elapsed: dict[int, float] = {}
    for _, r in df.iterrows():
        m = re.search(r"task_index-(\d+)", str(r["task_name"]))
        if m and r["state"] == "COMPLETED":
            task_elapsed[int(m.group(1))] = float(r["elapsed_seconds"])

    marginals, startup = fit_tier_costs(doc, task_elapsed)
    logger.info("Fitted startup (shared): %.0fs", startup)
    for tier in sorted(marginals, key=marginals.get, reverse=True):
        logger.info("  %-24s %.2f s/cell", tier, marginals[tier])
    costs = ",".join(f"{t}:{v:.2f}" for t, v in sorted(marginals.items()))
    print(f"\n--pack-costs '{costs}' --pack-load-s {startup:.0f}")


if __name__ == "__main__":
    main()
