"""
Build a half-coupled covariate-chain task file in MULTI-CONFIG mode.

Difference from `build_half_coupled_tasks.py`:
  - `build_half_coupled_tasks.py`  emits ONE task per DH config (single-config).
    For the 4-axis post-2000 refined grid that's 22,500 → 17,500 tasks.
    Per-task overhead (~20-30s Python startup + manifest load + parquet
    load) makes that wasteful at this scale.
  - This module emits one task per (s1_cov × s2_cov × bulk_cov × threshold)
    tuple. Within each task the worker iterates the Cartesian over:
        bulk_exposures (configurable, e.g. {free, free+weight})
        × tail_specs   (all tail_cov ⊆ bulk_cov × tail whitelist)
    Half-coupled validity is preserved because tail_covs are enumerated
    as subsets of THIS task's pinned bulk_cov. Bulk_exposures sharing the
    same bulk_cov are listed in the same task — the Cartesian over them
    × tails is fully half-coupled-valid.

For the post-2000 refined grid:
  - (s1, s2, bulk) tuples with s1 ⊇ s2 ⊇ bulk: 4^4 = 256
  - × len(thresholds) tasks total
  - Per task: 14 × 2^|bulk_cov| DH configs (max 224 when bulk=all-on)
  - Per-task wall: max ~90s of work + ~30s overhead = ~2 min
  - Total evaluate wall on ~200 concurrent slots: ~3-5 min

Half-coupling rule (same as build_half_coupled_tasks.py):
    s1_cov ⊇ s2_cov ⊇ bulk_cov ⊇ tail_cov
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


COV_AXES: tuple[str, ...] = ("wind_speed", "sdi", "basin", "is_island")


# ---------------------------------------------------------------------------
# Manifest lookup (shared shape with build_half_coupled_tasks)
# ---------------------------------------------------------------------------

def _load_spec_lookup(manifest_path: str) -> dict[tuple, str]:
    """Build a lookup keyed by
        (component, family, exposure_mode, threshold_quantile, cov_json)
    → component_id. IS specs only (fold_tag='is')."""
    with open(manifest_path) as f:
        manifest: dict[str, dict] = json.load(f)

    lookup: dict[tuple, str] = {}
    for cid, spec in manifest.items():
        if spec.get("fold_tag", "is") != "is":
            continue
        key = (
            spec["component"],
            spec.get("family"),
            spec.get("exposure_mode"),
            spec.get("threshold_quantile"),
            json.dumps(spec["covariate_combo"], sort_keys=True),
        )
        lookup[key] = cid
    return lookup


def _spec_id_for(
    lookup: dict[tuple, str],
    stage: str,
    family: str,
    exposure_mode: str,
    threshold_quantile: float | None,
    cov: dict[str, bool],
) -> str | None:
    q = None if stage == "s1" else float(threshold_quantile)
    key = (stage, family, exposure_mode, q, json.dumps(cov, sort_keys=True))
    return lookup.get(key)


# ---------------------------------------------------------------------------
# Cov-chain enumeration helpers
# ---------------------------------------------------------------------------

def _all_cov_subsets() -> list[frozenset[str]]:
    return [
        frozenset(combo)
        for r in range(len(COV_AXES) + 1)
        for combo in itertools.combinations(COV_AXES, r)
    ]


def _subset_to_cov(s: frozenset[str]) -> dict[str, bool]:
    return {axis: (axis in s) for axis in COV_AXES}


def enumerate_s1_s2_bulk_triples() -> list[tuple[dict, dict, dict]]:
    """All (s1_cov, s2_cov, bulk_cov) triples with s1 ⊇ s2 ⊇ bulk.

    For the 4-axis space: 4^4 = 256 triples (each axis independently lands
    in one of 4 buckets: never-on / off-at-s1→s2 / off-at-s2→bulk /
    on-through-bulk).
    """
    subsets = _all_cov_subsets()
    out: list[tuple[dict, dict, dict]] = []
    for c1 in subsets:
        for c2 in subsets:
            if not c2.issubset(c1):
                continue
            for c3 in subsets:
                if not c3.issubset(c2):
                    continue
                out.append((_subset_to_cov(c1), _subset_to_cov(c2), _subset_to_cov(c3)))
    return out


def _tail_subsets_of(bulk_cov: dict[str, bool]) -> list[dict[str, bool]]:
    """All tail_cov subsets of a given bulk_cov."""
    on_axes = frozenset(axis for axis, on in bulk_cov.items() if on)
    return [
        _subset_to_cov(frozenset(combo))
        for r in range(len(on_axes) + 1)
        for combo in itertools.combinations(on_axes, r)
    ]


# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------

def build_multiconfig_task_file(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    thresholds: list[float],
    s1_family: str,
    s1_exposure_mode: str,
    s2_family: str,
    s2_exposure_mode: str,
    bulk_family: str,
    bulk_exposure_modes: list[str],
    tail_fam_exp_whitelist: list[tuple[str, str]],
) -> dict:
    """Build the multi-config half-coupled task file. One task per
    (s1_cov, s2_cov, bulk_cov, threshold). Each task uses Cartesian over
    bulk_exposure_modes × (tail_cov subsets × tail_fam_exp_whitelist).
    """
    lookup = _load_spec_lookup(str(manifest_path))
    triples = enumerate_s1_s2_bulk_triples()
    logger.info("Enumerated %d (s1, s2, bulk) triples.", len(triples))

    tasks: list[dict] = []
    skipped_tasks = 0
    for threshold in thresholds:
        for (s1_cov, s2_cov, bulk_cov) in triples:
            s1_sid = _spec_id_for(lookup, "s1", s1_family, s1_exposure_mode, None, s1_cov)
            s2_sid = _spec_id_for(lookup, "s2", s2_family, s2_exposure_mode, threshold, s2_cov)
            if s1_sid is None or s2_sid is None:
                skipped_tasks += 1
                continue

            # All bulk exposure_modes at this bulk_cov / threshold.
            bulk_sids: list[str] = []
            for be in bulk_exposure_modes:
                bsid = _spec_id_for(lookup, "bulk", bulk_family, be, threshold, bulk_cov)
                if bsid is not None:
                    bulk_sids.append(bsid)
            if not bulk_sids:
                skipped_tasks += 1
                continue

            # All tail specs: (tail_cov ⊆ bulk_cov) × tail whitelist.
            tail_sids: list[str] = []
            for tail_cov in _tail_subsets_of(bulk_cov):
                for (tfam, texp) in tail_fam_exp_whitelist:
                    tsid = _spec_id_for(lookup, "tail", tfam, texp, threshold, tail_cov)
                    if tsid is not None:
                        tail_sids.append(tsid)
            if not tail_sids:
                skipped_tasks += 1
                continue

            tasks.append({
                "task_index":         len(tasks),
                "threshold_quantile": float(threshold),
                "s1_spec_ids":        [s1_sid],
                "s2_spec_ids":        [s2_sid],
                "bulk_spec_ids":      sorted(bulk_sids),
                "tail_spec_ids":      sorted(tail_sids),
                "meta": {
                    "s1_cov":   s1_cov,
                    "s2_cov":   s2_cov,
                    "bulk_cov": bulk_cov,
                    "n_dh_configs": len(bulk_sids) * len(tail_sids),
                },
            })

    doc = {
        "format_version": 1,
        "mode":           "half_coupled_multiconfig",
        "manifest_path":  str(manifest_path),
        "pins": {
            "thresholds":          [float(t) for t in thresholds],
            "s1_family":           s1_family,
            "s1_exposure_mode":    s1_exposure_mode,
            "s2_family":           s2_family,
            "s2_exposure_mode":    s2_exposure_mode,
            "bulk_family":         bulk_family,
            "bulk_exposure_modes": list(bulk_exposure_modes),
            "tail_whitelist":      [list(p) for p in tail_fam_exp_whitelist],
        },
        "tasks": tasks,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))

    total_configs = sum(t["meta"]["n_dh_configs"] for t in tasks)
    logger.info(
        "Wrote %d tasks (covering %d DH configs) to %s. Skipped %d (s1, s2, bulk × threshold) "
        "tuples due to missing manifest entries.",
        len(tasks), total_configs, out, skipped_tasks,
    )
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_tail_whitelist(strs: tuple[str, ...]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in strs:
        if ":" not in s:
            raise click.UsageError(
                f"--tail-whitelist value {s!r} must be 'family:exposure_mode'."
            )
        fam, exp = s.split(":", 1)
        out.append((fam.strip(), exp.strip()))
    return out


@click.command()
@click.option(
    "--manifest-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the IS manifest.json from run-fit-orchestrate.",
)
@click.option(
    "--output-path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Where to write the task JSON.",
)
@click.option("--thresholds",          multiple=True, required=True, type=float,
              help="Threshold quantiles to include. Pass multiple times for "
                   "multiple thresholds, e.g. --thresholds 0.70 --thresholds 0.85.")
@click.option("--s1-family",           required=True, type=str)
@click.option("--s1-exposure-mode",    required=True, type=str)
@click.option("--s2-family",           required=True, type=str)
@click.option("--s2-exposure-mode",    required=True, type=str)
@click.option("--bulk-family",         required=True, type=str)
@click.option(
    "--bulk-exposure-modes",
    multiple=True,
    required=True,
    type=str,
    help="Bulk exposure_modes to include. Pass multiple times, "
         "e.g. --bulk-exposure-modes free --bulk-exposure-modes free+weight.",
)
@click.option(
    "--tail-whitelist",
    multiple=True,
    required=True,
    help="Whitelisted tail (family:exposure_mode), as in build_half_coupled_tasks.",
)
def main(
    manifest_path: str,
    output_path: str,
    thresholds: tuple[float, ...],
    s1_family: str,
    s1_exposure_mode: str,
    s2_family: str,
    s2_exposure_mode: str,
    bulk_family: str,
    bulk_exposure_modes: tuple[str, ...],
    tail_whitelist: tuple[str, ...],
) -> None:
    """Build a multi-config half-coupled task file (one task per
    s1_cov × s2_cov × bulk_cov × threshold; Cartesian within over
    bulk_exposure_modes × tail subsets × tail whitelist)."""
    parsed_tail = _parse_tail_whitelist(tail_whitelist)
    build_multiconfig_task_file(
        manifest_path=manifest_path,
        output_path=output_path,
        thresholds=list(thresholds),
        s1_family=s1_family,
        s1_exposure_mode=s1_exposure_mode,
        s2_family=s2_family,
        s2_exposure_mode=s2_exposure_mode,
        bulk_family=bulk_family,
        bulk_exposure_modes=list(bulk_exposure_modes),
        tail_fam_exp_whitelist=parsed_tail,
    )


if __name__ == "__main__":
    main()
