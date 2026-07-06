"""
Build a task file containing the half-coupled covariate-chain grid as a
collection of single-DH-config tasks.

Half-coupling rule
------------------
Downstream stage covariates must be a *subset* of upstream stage covariates:

    s1_cov ⊇ s2_cov ⊇ bulk_cov ⊇ tail_cov

A covariate can be dropped at any stage transition, but once dropped it can
never reappear in a downstream stage. For the 4-axis covariate space (the
``{wind_speed, sdi, basin, is_island}`` set), the number of valid chains is

    Σ_{k=0..4} C(4, k) × 4^k  =  5^4  =  625

across all S1 cov choices. Multiplying by the per-tail (family, exposure)
whitelist gives the total DH configurations.

Why single-config tasks
-----------------------
The existing task format takes per-stage ``*_spec_ids`` lists and the
worker iterates their Cartesian product. Half-coupled chains are *not* a
Cartesian product — they're a tree of subset constraints. The simplest
fit with the existing worker is to emit one task per valid
(s1, s2, bulk, tail) tuple, each with single-element spec_id lists; the
worker's Cartesian over 1×1×1×1 is trivially 1 DH config. No worker
changes required.

Total per-task overhead is meaningful (Python startup, manifest load,
parquet load — ~20–30 s per task) so this is best for grids of a few
thousand chains at most. Beyond that, the worker would need an explicit-
tuple list mode and that's a separate code change.
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
# Manifest lookup (shared shape with build_neighbor_tasks)
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
    family: str | None,
    exposure_mode: str | None,
    threshold_quantile: float | None,
    cov: dict[str, bool],
) -> str | None:
    q = None if stage == "s1" else float(threshold_quantile)
    key = (stage, family, exposure_mode, q, json.dumps(cov, sort_keys=True))
    return lookup.get(key)


# ---------------------------------------------------------------------------
# Half-coupled chain enumeration
# ---------------------------------------------------------------------------

def _all_cov_subsets() -> list[frozenset[str]]:
    """All 2^len(COV_AXES) subsets of the covariate axes as frozensets."""
    return [
        frozenset(combo)
        for r in range(len(COV_AXES) + 1)
        for combo in itertools.combinations(COV_AXES, r)
    ]


def _subset_to_cov(s: frozenset[str]) -> dict[str, bool]:
    """frozenset → {axis: True/False} dict in the manifest's format."""
    return {axis: (axis in s) for axis in COV_AXES}


def enumerate_half_coupled_chains() -> list[tuple[dict, dict, dict, dict]]:
    """All (s1_cov, s2_cov, bulk_cov, tail_cov) chains with c1 ⊇ c2 ⊇ c3 ⊇ c4.

    For the 4-axis space this is exactly 5^4 = 625 chains.
    """
    subsets = _all_cov_subsets()
    chains: list[tuple[dict, dict, dict, dict]] = []
    for c1 in subsets:
        for c2 in subsets:
            if not c2.issubset(c1):
                continue
            for c3 in subsets:
                if not c3.issubset(c2):
                    continue
                for c4 in subsets:
                    if not c4.issubset(c3):
                        continue
                    chains.append((
                        _subset_to_cov(c1),
                        _subset_to_cov(c2),
                        _subset_to_cov(c3),
                        _subset_to_cov(c4),
                    ))
    return chains


# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------

def build_half_coupled_task_file(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    threshold: float,
    s1_family: str,
    s1_exposure_mode: str,
    s2_family: str,
    s2_exposure_mode: str,
    bulk_family: str,
    bulk_exposure_mode: str,
    tail_fam_exp_whitelist: list[tuple[str, str]],
) -> dict:
    """For each valid half-coupled cov chain × tail (family, exposure_mode)
    in the whitelist, emit one task with single-element spec_id lists.
    """
    lookup = _load_spec_lookup(str(manifest_path))
    chains = enumerate_half_coupled_chains()
    logger.info("Enumerated %d half-coupled cov chains.", len(chains))

    tasks: list[dict] = []
    skipped_missing = 0
    for (s1_cov, s2_cov, bulk_cov, tail_cov) in chains:
        s1_sid   = _spec_id_for(lookup, "s1",   s1_family,   s1_exposure_mode,   None,      s1_cov)
        s2_sid   = _spec_id_for(lookup, "s2",   s2_family,   s2_exposure_mode,   threshold, s2_cov)
        bulk_sid = _spec_id_for(lookup, "bulk", bulk_family, bulk_exposure_mode, threshold, bulk_cov)

        if any(x is None for x in (s1_sid, s2_sid, bulk_sid)):
            # All 3 of these are identical across the tail whitelist; one
            # missing means every chain×tail in this iteration is unrunnable.
            skipped_missing += len(tail_fam_exp_whitelist)
            continue

        for tail_family, tail_exposure_mode in tail_fam_exp_whitelist:
            tail_sid = _spec_id_for(
                lookup, "tail", tail_family, tail_exposure_mode, threshold, tail_cov,
            )
            if tail_sid is None:
                skipped_missing += 1
                continue

            tasks.append({
                "task_index":         len(tasks),
                "threshold_quantile": float(threshold),
                "s1_spec_ids":        [s1_sid],
                "s2_spec_ids":        [s2_sid],
                "bulk_spec_ids":      [bulk_sid],
                "tail_spec_ids":      [tail_sid],
                "meta": {
                    "s1_cov":            s1_cov,
                    "s2_cov":            s2_cov,
                    "bulk_cov":          bulk_cov,
                    "tail_cov":          tail_cov,
                    "tail_family":       tail_family,
                    "tail_exposure_mode": tail_exposure_mode,
                },
            })

    doc = {
        "format_version": 1,
        "mode":           "half_coupled",
        "manifest_path":  str(manifest_path),
        "pins": {
            "threshold":           float(threshold),
            "s1_family":           s1_family,
            "s1_exposure_mode":    s1_exposure_mode,
            "s2_family":           s2_family,
            "s2_exposure_mode":    s2_exposure_mode,
            "bulk_family":         bulk_family,
            "bulk_exposure_mode":  bulk_exposure_mode,
            "tail_whitelist":      [list(p) for p in tail_fam_exp_whitelist],
        },
        "tasks": tasks,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))

    logger.info(
        "Wrote %d tasks to %s  (skipped %d (chain × tail) combos missing from manifest)",
        len(tasks), out, skipped_missing,
    )
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_tail_whitelist(strs: tuple[str, ...]) -> list[tuple[str, str]]:
    """Parse 'family:exposure_mode' strings into (family, exposure_mode) tuples."""
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
@click.option("--threshold",          required=True, type=float)
@click.option("--s1-family",          required=True, type=str)
@click.option("--s1-exposure-mode",   required=True, type=str)
@click.option("--s2-family",          required=True, type=str)
@click.option("--s2-exposure-mode",   required=True, type=str)
@click.option("--bulk-family",        required=True, type=str)
@click.option("--bulk-exposure-mode", required=True, type=str)
@click.option(
    "--tail-whitelist",
    multiple=True,
    required=True,
    help="Whitelisted tail (family:exposure_mode). Pass multiple times for "
         "multiple combinations, e.g. "
         "--tail-whitelist gamma:free+weight --tail-whitelist weibull:excluded",
)
def main(
    manifest_path: str,
    output_path: str,
    threshold: float,
    s1_family: str,
    s1_exposure_mode: str,
    s2_family: str,
    s2_exposure_mode: str,
    bulk_family: str,
    bulk_exposure_mode: str,
    tail_whitelist: tuple[str, ...],
) -> None:
    """Build a half-coupled covariate-chain task file (one task per DH config)."""
    parsed = _parse_tail_whitelist(tail_whitelist)
    build_half_coupled_task_file(
        manifest_path=manifest_path,
        output_path=output_path,
        threshold=threshold,
        s1_family=s1_family,
        s1_exposure_mode=s1_exposure_mode,
        s2_family=s2_family,
        s2_exposure_mode=s2_exposure_mode,
        bulk_family=bulk_family,
        bulk_exposure_mode=bulk_exposure_mode,
        tail_fam_exp_whitelist=parsed,
    )


if __name__ == "__main__":
    main()
