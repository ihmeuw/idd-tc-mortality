"""
Build a task file containing the single-cov-token *neighbours* of a small set
of candidate winner configurations.

Why this exists
---------------
The decoupled refined-evaluate grid has 12.5M DH configurations. Most of
them aren't analytically interesting. The original motivation for breaking
the `s1_cov == s2_cov == bulk_cov == tail_cov` coupling was to support
per-stage cov perturbation analysis of a chosen winner config — e.g.,
"is there a config identical to my TOPSIS winner except that `wind_speed`
is dropped from S2 only?" That's 16 DH configurations per winner (4 stage
covs × 4 axis flips), not 12.5M.

This builder takes a list of candidate winners (one row per winner with the
13 CONFIG_COLS values) and emits a task file with one task per
(winner × perturbation-stage) pair, containing the 4 single-axis-flipped
component_ids for that stage. Each task evaluates 4 sibling DH configs of
the winner via the existing run_evaluate worker.

Total DH configs in a 10-winner task file: 10 × 4 × 4 = 160. At ~0.66 s
per config that's ~100 s of compute in `--local` mode, or fast jobmon if
desired.

Usage
-----
Build the manifest-spec lookup from your fit-cache manifest; build the
neighbour task file from a winners CSV / parquet / passed-in DataFrame:

    run-build-neighbor-tasks \\
        --manifest-path /mnt/.../01-preliminary/<date>/manifest.json \\
        --winners-path  /tmp/winners.parquet \\
        --output-path   /mnt/.../02-evaluate/<date>/tasks_neighbours.json

The winners file must contain (at minimum) the 13 CONFIG_COLS:
threshold_quantile, s1_family, s1_exposure_mode, s1_cov,
s2_family, s2_exposure_mode, s2_cov, bulk_family, bulk_exposure_mode,
bulk_cov, tail_family, tail_exposure_mode, tail_cov.

Run the resulting task file via the standard orchestrator:

    run-evaluate-orchestrate --task-file <path> ... (everything else as usual)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


COV_AXES    = ["wind_speed", "sdi", "basin", "is_island"]
STAGE_NAMES = ("s1", "s2", "bulk", "tail")


# ---------------------------------------------------------------------------
# Manifest lookup
# ---------------------------------------------------------------------------

def _load_spec_lookup(manifest_path: str) -> dict[tuple, str]:
    """Build a lookup table for IS specs only:
        (component, family, exposure_mode, threshold_quantile, cov_json, fold_tag='is')
        -> component_id

    cov_json is the spec's covariate_combo serialised with sort_keys=True so
    that lookups normalised the same way always match.
    """
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
    """Look up the IS component_id for a specific (stage, fam, exp, q, cov).
    Returns None if no such spec was fit (caller decides whether to skip)."""
    q = None if stage == "s1" else float(threshold_quantile)
    key = (stage, family, exposure_mode, q, json.dumps(cov, sort_keys=True))
    return lookup.get(key)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_neighbor_task_file(
    winners_df: pd.DataFrame,
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Build the task JSON. Returns the parsed task document for inspection."""
    lookup = _load_spec_lookup(str(manifest_path))

    tasks: list[dict] = []
    skipped_missing = 0

    for w_idx, w in winners_df.iterrows():
        threshold = float(w["threshold_quantile"])

        # Parse the winner's per-stage covs (json strings → dicts).
        winner_covs: dict[str, dict[str, bool]] = {}
        for s in STAGE_NAMES:
            c = w[f"{s}_cov"]
            winner_covs[s] = json.loads(c) if isinstance(c, str) else dict(c)

        # Find the IS component_id of each of the four winner stage specs.
        winner_spec_ids: dict[str, str] = {}
        winner_resolution_failed = False
        for s in STAGE_NAMES:
            sid = _spec_id_for(
                lookup, s, w[f"{s}_family"], w[f"{s}_exposure_mode"],
                threshold, winner_covs[s],
            )
            if sid is None:
                logger.warning(
                    "Winner #%s: stage=%s spec not found in manifest "
                    "(family=%s, exp=%s, threshold=%s, cov=%s) — skipping winner.",
                    w_idx, s, w[f"{s}_family"], w[f"{s}_exposure_mode"],
                    threshold, winner_covs[s],
                )
                winner_resolution_failed = True
                break
            winner_spec_ids[s] = sid
        if winner_resolution_failed:
            continue

        # For each stage, emit one task containing the 4 single-axis-flipped
        # cov perturbations as that stage's spec_id list. The other three
        # stages stay pinned at the winner's spec_id.
        for perturb_stage in STAGE_NAMES:
            perturbed_sids: list[str] = []
            for axis in COV_AXES:
                new_cov = dict(winner_covs[perturb_stage])
                new_cov[axis] = not new_cov[axis]
                sid = _spec_id_for(
                    lookup, perturb_stage,
                    w[f"{perturb_stage}_family"],
                    w[f"{perturb_stage}_exposure_mode"],
                    threshold, new_cov,
                )
                if sid is None:
                    skipped_missing += 1
                    continue
                perturbed_sids.append(sid)

            if not perturbed_sids:
                continue

            task_spec: dict = {
                "task_index":         len(tasks),
                "threshold_quantile": threshold,
                "s1_spec_ids":        [winner_spec_ids["s1"]],
                "s2_spec_ids":        [winner_spec_ids["s2"]],
                "bulk_spec_ids":      [winner_spec_ids["bulk"]],
                "tail_spec_ids":      [winner_spec_ids["tail"]],
                "meta": {
                    "winner_idx":    int(w_idx) if isinstance(w_idx, (int, str)) else None,
                    "perturb_stage": perturb_stage,
                    "n_perturbed":   len(perturbed_sids),
                },
            }
            # Replace the perturbed stage's spec_id list with the 4 flips.
            task_spec[f"{perturb_stage}_spec_ids"] = perturbed_sids
            tasks.append(task_spec)

    doc = {
        "format_version": 1,
        "mode":           "neighbor_perturbations",
        "manifest_path":  str(manifest_path),
        "tasks":          tasks,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))

    total_configs = sum(t["meta"]["n_perturbed"] for t in tasks)
    logger.info(
        "Wrote %d perturbation tasks (%d DH configs total) to %s",
        len(tasks), total_configs, out,
    )
    if skipped_missing:
        logger.warning(
            "Skipped %d individual axis flips: corresponding spec wasn't in "
            "the fit manifest. Probably means a (cov × family × exposure) "
            "combination never made it into the refined grid build.",
            skipped_missing,
        )
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--manifest-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the IS manifest.json from run-fit-orchestrate.",
)
@click.option(
    "--winners-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a parquet or CSV containing the candidate-winner rows. "
         "Must include the 13 CONFIG_COLS.",
)
@click.option(
    "--output-path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Where to write the neighbour task JSON.",
)
def main(manifest_path: str, winners_path: str, output_path: str) -> None:
    """Build a task file containing the single-cov-token neighbours of every
    winner row in --winners-path."""
    winners_path_p = Path(winners_path)
    if winners_path_p.suffix == ".parquet":
        winners_df = pd.read_parquet(winners_path_p)
    elif winners_path_p.suffix in (".csv", ".tsv"):
        sep = "\t" if winners_path_p.suffix == ".tsv" else ","
        winners_df = pd.read_csv(winners_path_p, sep=sep)
    else:
        raise click.UsageError(f"Unsupported --winners-path extension: {winners_path_p.suffix}")

    build_neighbor_task_file(winners_df, manifest_path, output_path)


if __name__ == "__main__":
    main()
