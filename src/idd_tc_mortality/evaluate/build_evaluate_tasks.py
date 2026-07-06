"""
CLI entry point for building an evaluate-tasks JSON file.

A task file partitions the evaluate workload into N independent Slurm tasks.
Each task is a JSON object describing:
    threshold_quantile     — single threshold quantile (float)
    s1_spec_ids            — list of S1 IS component_ids the task covers
    s2_spec_ids            — list of S2 IS component_ids the task covers
    bulk_spec_ids          — list of bulk IS component_ids the task covers
    tail_spec_ids          — list of tail IS component_ids the task covers

The worker (``run-evaluate --task-file ... --task-index N``) iterates the
Cartesian product of these per-stage lists at the task's threshold. Decoupling
is implicit — per-stage spec_id lists may span multiple covariate combos.

Granularity control
-------------------
``--n-tasks`` selects how many tasks the JSON file contains. The default
(``one_per_s1_threshold``) produces one task per (S1 spec × threshold) pair,
matching the existing orchestrator partitioning. Finer-grained options split
the tail or bulk spec lists across additional tasks so individual workers
own less work each.

Usage:
    run-build-evaluate-tasks \\
        --manifest-path /mnt/.../01-preliminary/<date>/manifest.json \\
        --output-path   /mnt/.../02-evaluate/<date>/tasks.json
        [--n-tasks-mode one_per_s1_threshold]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest reading
# ---------------------------------------------------------------------------

def _load_is_specs_by_component(manifest_path: str) -> dict[str, dict[str, dict]]:
    """Return {component: {component_id: spec}} restricted to IS specs."""
    with open(manifest_path) as f:
        manifest: dict[str, dict] = json.load(f)

    by_component: dict[str, dict[str, dict]] = {
        "s1": {}, "s2": {}, "bulk": {}, "tail": {},
    }
    for cid, spec in manifest.items():
        if spec.get("fold_tag", "is") != "is":
            continue
        component = spec.get("component")
        if component in by_component:
            by_component[component][cid] = spec
    return by_component


# ---------------------------------------------------------------------------
# Task-list builders
# ---------------------------------------------------------------------------

def _build_one_per_s1_threshold(
    by_component: dict[str, dict[str, dict]],
) -> list[dict]:
    """One task per (S1 spec × threshold). Each task carries all S2 / bulk /
    tail spec_ids that exist at its threshold — the worker runs the full
    cov-decoupled Cartesian product within each task."""
    s1s   = by_component["s1"]
    s2s   = by_component["s2"]
    bulks = by_component["bulk"]
    tails = by_component["tail"]

    thresholds = sorted({
        s.get("threshold_quantile") for s in (
            list(s2s.values()) + list(bulks.values()) + list(tails.values())
        )
        if s.get("threshold_quantile") is not None
    })

    tasks: list[dict] = []
    task_index = 0
    for s1_cid in sorted(s1s.keys()):
        for q in thresholds:
            s2_ids   = sorted(cid for cid, spec in s2s.items()   if spec.get("threshold_quantile") == q)
            bulk_ids = sorted(cid for cid, spec in bulks.items() if spec.get("threshold_quantile") == q)
            tail_ids = sorted(cid for cid, spec in tails.items() if spec.get("threshold_quantile") == q)
            tasks.append({
                "task_index":         task_index,
                "threshold_quantile": float(q),
                "s1_spec_ids":        [s1_cid],
                "s2_spec_ids":        s2_ids,
                "bulk_spec_ids":      bulk_ids,
                "tail_spec_ids":      tail_ids,
            })
            task_index += 1
    return tasks


def _build_single_config(
    by_component: dict[str, dict[str, dict]],
    n_configs: int = 1,
) -> list[dict]:
    """Single task containing N DH configurations (default 1) — one of each
    stage spec at the lowest available threshold, with the first N tail
    specs at that threshold. Use for benchmarking — N DH configs in one
    worker × (1 IS + 25 OOS) folds each — when paired with `--scope`.

    Tail is the dimension expanded (vs. bulk / s2) because the tail axis
    has the most specs (16 × 6 × 4 = 384 at a single threshold) and is
    typically the slowest stage to fit/predict — useful to probe."""
    s1s   = by_component["s1"]
    s2s   = by_component["s2"]
    bulks = by_component["bulk"]
    tails = by_component["tail"]

    if not (s1s and s2s and bulks and tails):
        raise ValueError("manifest is missing one or more component types")

    thresholds = sorted({
        s.get("threshold_quantile") for s in s2s.values()
        if s.get("threshold_quantile") is not None
    })
    if not thresholds:
        raise ValueError("no thresholds found in s2 specs")
    q = thresholds[0]

    s1_cid   = sorted(s1s.keys())[0]
    s2_cid   = sorted(cid for cid, spec in s2s.items()   if spec.get("threshold_quantile") == q)[0]
    bulk_cid = sorted(cid for cid, spec in bulks.items() if spec.get("threshold_quantile") == q)[0]

    available_tail_cids = sorted(
        cid for cid, spec in tails.items()
        if spec.get("threshold_quantile") == q
    )
    if n_configs < 1:
        raise ValueError(f"--n-configs must be >= 1; got {n_configs}")
    n_take = min(n_configs, len(available_tail_cids))
    if n_take < n_configs:
        logger.warning(
            "Requested n_configs=%d but only %d tail specs available at "
            "threshold=%s; capping to %d.",
            n_configs, len(available_tail_cids), q, n_take,
        )
    tail_cids = available_tail_cids[:n_take]

    return [{
        "task_index":         0,
        "threshold_quantile": float(q),
        "s1_spec_ids":        [s1_cid],
        "s2_spec_ids":        [s2_cid],
        "bulk_spec_ids":      [bulk_cid],
        "tail_spec_ids":      tail_cids,
    }]


def _cov_key(spec: dict) -> str:
    """Canonical string key for a spec's covariate_combo, stable across
    runs. Used to filter to cov-matched specs for coupled partitions."""
    return json.dumps(spec.get("covariate_combo", {}), sort_keys=True)


def _build_one_per_s1_threshold_coupled(
    by_component: dict[str, dict[str, dict]],
) -> list[dict]:
    """One task per (S1 spec × threshold), COUPLED — each task contains
    only S2 / bulk / tail spec_ids whose covariate_combo matches the S1
    spec's covariate_combo at that threshold. The worker iterates the
    cartesian over the supplied spec_ids, which is now coupled-equivalent.

    Preliminary uses coupled evaluation by design; this partition gives
    coupled semantics under --task-file mode (which forces decoupled
    worker iteration but produces coupled output when the per-stage
    spec_id lists are already cov-filtered)."""
    s1s   = by_component["s1"]
    s2s   = by_component["s2"]
    bulks = by_component["bulk"]
    tails = by_component["tail"]

    thresholds = sorted({
        s.get("threshold_quantile") for s in (
            list(s2s.values()) + list(bulks.values()) + list(tails.values())
        )
        if s.get("threshold_quantile") is not None
    })

    tasks: list[dict] = []
    task_index = 0
    for s1_cid in sorted(s1s.keys()):
        s1_cov = _cov_key(s1s[s1_cid])
        for q in thresholds:
            s2_ids = sorted(
                cid for cid, spec in s2s.items()
                if spec.get("threshold_quantile") == q and _cov_key(spec) == s1_cov
            )
            bulk_ids = sorted(
                cid for cid, spec in bulks.items()
                if spec.get("threshold_quantile") == q and _cov_key(spec) == s1_cov
            )
            tail_ids = sorted(
                cid for cid, spec in tails.items()
                if spec.get("threshold_quantile") == q and _cov_key(spec) == s1_cov
            )
            if not (s2_ids and bulk_ids and tail_ids):
                continue
            tasks.append({
                "task_index":         task_index,
                "threshold_quantile": float(q),
                "s1_spec_ids":        [s1_cid],
                "s2_spec_ids":        s2_ids,
                "bulk_spec_ids":      bulk_ids,
                "tail_spec_ids":      tail_ids,
            })
            task_index += 1
    return tasks


def _build_one_per_s1_s2_threshold_coupled(
    by_component: dict[str, dict[str, dict]],
) -> list[dict]:
    """One task per (S1 spec × S2 spec), COUPLED — each task contains only
    bulk / tail spec_ids whose covariate_combo matches the S1 / S2 pair's
    shared covariate_combo at that threshold. S2 must share S1's cov_key
    or the pair is skipped.

    The finer-grained coupled analog of one_per_s1_threshold_coupled. Use
    when you want many small coupled tasks for cluster parallelism."""
    s1s   = by_component["s1"]
    s2s   = by_component["s2"]
    bulks = by_component["bulk"]
    tails = by_component["tail"]

    tasks: list[dict] = []
    task_index = 0
    for s1_cid in sorted(s1s.keys()):
        s1_cov = _cov_key(s1s[s1_cid])
        for s2_cid in sorted(s2s.keys()):
            s2_spec = s2s[s2_cid]
            q = s2_spec.get("threshold_quantile")
            if q is None:
                continue
            if _cov_key(s2_spec) != s1_cov:
                continue
            bulk_ids = sorted(
                cid for cid, spec in bulks.items()
                if spec.get("threshold_quantile") == q and _cov_key(spec) == s1_cov
            )
            tail_ids = sorted(
                cid for cid, spec in tails.items()
                if spec.get("threshold_quantile") == q and _cov_key(spec) == s1_cov
            )
            if not (bulk_ids and tail_ids):
                continue
            tasks.append({
                "task_index":         task_index,
                "threshold_quantile": float(q),
                "s1_spec_ids":        [s1_cid],
                "s2_spec_ids":        [s2_cid],
                "bulk_spec_ids":      bulk_ids,
                "tail_spec_ids":      tail_ids,
            })
            task_index += 1
    return tasks


def _build_one_per_s1_s2_threshold(
    by_component: dict[str, dict[str, dict]],
) -> list[dict]:
    """One task per (S1 spec × S2 spec). Threshold is carried by S2.
    Each task enumerates all bulk × tail specs at that S2's threshold.

    For the preliminary grid (15 S1 IS specs × 72 S2 IS specs across all
    thresholds) this yields 15 × 72 = 1080 tasks of ~54 bulk × ~78 tail =
    ~4,200 DH configs each — 12× fewer configs per task than the legacy
    one_per_s1_threshold partition (~50K configs/task)."""
    s1s   = by_component["s1"]
    s2s   = by_component["s2"]
    bulks = by_component["bulk"]
    tails = by_component["tail"]

    tasks: list[dict] = []
    task_index = 0
    for s1_cid in sorted(s1s.keys()):
        for s2_cid in sorted(s2s.keys()):
            q = s2s[s2_cid].get("threshold_quantile")
            if q is None:
                continue
            bulk_ids_q = sorted(cid for cid, spec in bulks.items() if spec.get("threshold_quantile") == q)
            tail_ids_q = sorted(cid for cid, spec in tails.items() if spec.get("threshold_quantile") == q)
            tasks.append({
                "task_index":         task_index,
                "threshold_quantile": float(q),
                "s1_spec_ids":        [s1_cid],
                "s2_spec_ids":        [s2_cid],
                "bulk_spec_ids":      bulk_ids_q,
                "tail_spec_ids":      tail_ids_q,
            })
            task_index += 1
    return tasks


def _build_one_per_bulk_spec_threshold(
    by_component: dict[str, dict[str, dict]],
) -> list[dict]:
    """One task per (S1 spec × S2 cov × bulk spec × threshold). Each task
    enumerates all tail specs at that threshold. For the refined grid this
    yields 16 × 2 × 16 × 64 = 32,768 tasks of 384 DH configs each — the
    granularity that produced ~5 min wall time per task in the probe."""
    s1s   = by_component["s1"]
    s2s   = by_component["s2"]
    bulks = by_component["bulk"]
    tails = by_component["tail"]

    thresholds = sorted({
        s.get("threshold_quantile") for s in s2s.values()
        if s.get("threshold_quantile") is not None
    })

    tasks: list[dict] = []
    task_index = 0
    for s1_cid in sorted(s1s.keys()):
        for q in thresholds:
            s2_ids_q   = sorted(cid for cid, spec in s2s.items()   if spec.get("threshold_quantile") == q)
            bulk_ids_q = sorted(cid for cid, spec in bulks.items() if spec.get("threshold_quantile") == q)
            tail_ids_q = sorted(cid for cid, spec in tails.items() if spec.get("threshold_quantile") == q)
            for s2_cid in s2_ids_q:
                for bulk_cid in bulk_ids_q:
                    tasks.append({
                        "task_index":         task_index,
                        "threshold_quantile": float(q),
                        "s1_spec_ids":        [s1_cid],
                        "s2_spec_ids":        [s2_cid],
                        "bulk_spec_ids":      [bulk_cid],
                        "tail_spec_ids":      tail_ids_q,
                    })
                    task_index += 1
    return tasks


def _build_one_per_tail_fam_exp_threshold(
    by_component: dict[str, dict[str, dict]],
) -> list[dict]:
    """One task per (S1 spec × threshold × tail family × tail exposure_mode).
    Each task enumerates all S2 covs × bulk specs × tail covs at the fixed
    (family, exposure). For the refined grid this yields 16 × 2 × 6 × 4 =
    768 tasks of ~16,384 DH configs each — designed to minimize per-task
    NFS read fan-out (only 97 unique IS pickles per task) and worst-case
    cross-task pickle demand (~384 tasks share any one pickle, vs 6 K or
    16 K under finer partitions)."""
    s1s   = by_component["s1"]
    s2s   = by_component["s2"]
    bulks = by_component["bulk"]
    tails = by_component["tail"]

    thresholds = sorted({
        s.get("threshold_quantile") for s in s2s.values()
        if s.get("threshold_quantile") is not None
    })

    tasks: list[dict] = []
    task_index = 0
    for s1_cid in sorted(s1s.keys()):
        for q in thresholds:
            s2_ids_q   = sorted(cid for cid, spec in s2s.items()   if spec.get("threshold_quantile") == q)
            bulk_ids_q = sorted(cid for cid, spec in bulks.items() if spec.get("threshold_quantile") == q)

            # Group tail specs at this threshold by (family, exposure_mode).
            tail_groups: dict[tuple[str, str], list[str]] = {}
            for cid, spec in tails.items():
                if spec.get("threshold_quantile") != q:
                    continue
                key = (spec.get("family"), spec.get("exposure_mode"))
                tail_groups.setdefault(key, []).append(cid)

            for key in sorted(tail_groups.keys()):
                tail_ids = sorted(tail_groups[key])
                tasks.append({
                    "task_index":         task_index,
                    "threshold_quantile": float(q),
                    "s1_spec_ids":        [s1_cid],
                    "s2_spec_ids":        s2_ids_q,
                    "bulk_spec_ids":      bulk_ids_q,
                    "tail_spec_ids":      tail_ids,
                })
                task_index += 1
    return tasks


_MODES = {
    "one_per_s1_threshold":                  _build_one_per_s1_threshold,
    "one_per_s1_threshold_coupled":          _build_one_per_s1_threshold_coupled,
    "one_per_s1_s2_threshold":               _build_one_per_s1_s2_threshold,
    "one_per_s1_s2_threshold_coupled":       _build_one_per_s1_s2_threshold_coupled,
    "one_per_bulk_spec_threshold":           _build_one_per_bulk_spec_threshold,
    "one_per_tail_fam_exp_threshold":        _build_one_per_tail_fam_exp_threshold,
    "single_config":                         _build_single_config,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--manifest-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to manifest.json from run-fit-orchestrate / run-build-refined-specs.",
)
@click.option(
    "--output-path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Where to write the task JSON file.",
)
@click.option(
    "--mode",
    default="one_per_s1_threshold",
    show_default=True,
    type=click.Choice(sorted(_MODES.keys())),
    help="Partitioning strategy. 'one_per_s1_threshold' produces one task per "
         "(S1 spec × threshold). 'single_config' produces one task with "
         "--n-configs DH configs (default 1) for benchmarking.",
)
@click.option(
    "--n-configs",
    default=1,
    show_default=True,
    type=int,
    help="Number of DH configs in the single task (only used with "
         "--mode single_config). Tail spec is the varied dimension. Caps at "
         "the number of tail specs available at the chosen threshold (384 "
         "for the refined grid).",
)
def main(manifest_path: str, output_path: str, mode: str, n_configs: int) -> None:
    """Build an evaluate-task JSON file from a manifest."""
    by_component = _load_is_specs_by_component(manifest_path)
    counts = {c: len(d) for c, d in by_component.items()}
    logger.info(
        "Manifest IS specs: s1=%d, s2=%d, bulk=%d, tail=%d",
        counts["s1"], counts["s2"], counts["bulk"], counts["tail"],
    )

    if mode == "single_config":
        tasks = _build_single_config(by_component, n_configs=n_configs)
    else:
        tasks = _MODES[mode](by_component)

    doc = {
        "format_version": 1,
        "mode":           mode,
        "manifest_path":  str(manifest_path),
        "tasks":          tasks,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))

    # Per-task summary so the user can sanity-check the granularity.
    logger.info("Wrote %d tasks to %s (mode=%s)", len(tasks), out, mode)
    if tasks:
        first = tasks[0]
        ncfg = (len(first["s1_spec_ids"]) * len(first["s2_spec_ids"])
                * len(first["bulk_spec_ids"]) * len(first["tail_spec_ids"]))
        logger.info(
            "First task: threshold=%.2f, "
            "|s1|=%d, |s2|=%d, |bulk|=%d, |tail|=%d  (%d DH configs)",
            first["threshold_quantile"],
            len(first["s1_spec_ids"]),
            len(first["s2_spec_ids"]),
            len(first["bulk_spec_ids"]),
            len(first["tail_spec_ids"]),
            ncfg,
        )


if __name__ == "__main__":
    main()
