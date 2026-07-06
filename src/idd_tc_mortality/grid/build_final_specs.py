"""
CLI entry point for building the FINAL-grid IS spec list (20260608 cycle).

The final grid is an explicit, hand-picked grid — the 20260517 refined-final
96-config grid (see ``02-evaluate/refined-final-20260517/tasks.json``) PLUS two
additions locked in on 2026-06-15:

  - Tail family/exposure: add ``(gpd, weight)``  (was {log_logistic/free+weight,
    log_logistic/weight, weibull/free}).
  - Tail covariate option: add ``sdi`` — tail cov now ∈ {(none), sdi}
    (was (none) only).

Combinations (full cartesian of the per-stage option sets below):
    s1   : 2 cov sets                                            =  2
    s2   : 2 cov sets                                            =  2
    bulk : 2 exposure modes × 4 cov sets                         =  8
    tail : 4 (family, exposure) pairs × 2 cov sets               =  8
    total DH configs = 2 × 2 × 8 × 8                              = 256
threshold pinned at 0.70.

Every combination is nesting-valid (tail_cov ⊆ bulk_cov ⊆ s2_cov; s1 free):
each bulk cov set contains ``sdi`` so the new tail ``sdi`` option always nests,
and all four bulk sets are subsets of both s2 sets. So all 256 survive.

This file is the auditable record for the final grid — to change it, edit the
constants below in a single commit. The companion ``build_final_cells.py``
imports these constants so the spec list and the cell enumeration share one
source of truth.

Usage:
    run-build-final-specs \\
        --output-path /mnt/team/idd/pub/idd_tc_mortality/01-refined/<date>/final_is_specs.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Covariate helper + per-stage option sets (the auditable grid definition)
# ---------------------------------------------------------------------------

COV_AXES = ["wind_speed", "sdi", "basin", "is_island"]


def _cov(*on: str) -> dict[str, bool]:
    """Covariate-combo dict with the named axes True, the rest False."""
    bad = set(on) - set(COV_AXES)
    if bad:
        raise ValueError(f"Unknown covariate axes: {sorted(bad)}")
    return {axis: (axis in on) for axis in COV_AXES}


THRESHOLDS: list[float] = [0.70]

# S1 / S2 — logit/free, the two covariate sets from the 20260517 final.
S1_FAMILY_MODE: tuple[str, str] = ("logit", "free")
S2_FAMILY_MODE: tuple[str, str] = ("logit", "free")
S1_COVS: list[dict[str, bool]] = [
    _cov("wind_speed", "sdi", "basin"),
    _cov("wind_speed", "sdi", "basin", "is_island"),
]
S2_COVS: list[dict[str, bool]] = [
    _cov("wind_speed", "sdi", "basin"),
    _cov("wind_speed", "sdi", "basin", "is_island"),
]

# Bulk — scaled_logit, both exposure modes, the four sdi-anchored cov sets.
BULK_FAMILY: str = "scaled_logit"
BULK_EXPOSURES: list[str] = ["free", "free+weight"]
BULK_COVS: list[dict[str, bool]] = [
    _cov("sdi"),
    _cov("sdi", "basin"),
    _cov("wind_speed", "sdi"),
    _cov("wind_speed", "sdi", "basin"),
]

# Tail — the three 20260517 pairs PLUS (gpd, weight); cov ∈ {(none), sdi}.
TAIL_FAMILY_EXPOSURES: list[tuple[str, str]] = [
    ("log_logistic", "free+weight"),
    ("log_logistic", "weight"),
    ("weibull", "free"),
    ("gpd", "weight"),          # 2026-06-15 addition
]
TAIL_COVS: list[dict[str, bool]] = [
    _cov(),                     # intercept-only (the 20260517 tail)
    _cov("sdi"),                # 2026-06-15 addition
]


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------

def build_specs(thresholds: list[float] | None = None) -> list[dict]:
    """Return the flat list of IS component spec dicts for the final grid.

    Same shape as ``enumerate_component_specs`` / ``build_refined_specs_post2000``
    output: one dict per distinct (component, family, exposure_mode, threshold,
    covariate_combo). These are the components that the 256 DH configs assemble
    from — 2 s1 + 2 s2 + 8 bulk + 8 tail = 20 distinct IS specs.
    """
    thresholds = thresholds if thresholds is not None else THRESHOLDS
    specs: list[dict] = []

    for cov in S1_COVS:
        specs.append({
            "component":          "s1",
            "covariate_combo":    cov,
            "threshold_quantile": None,
            "threshold_rate":     None,
            "family":             S1_FAMILY_MODE[0],
            "exposure_mode":      S1_FAMILY_MODE[1],
            "fold_tag":           "is",
        })

    for q in thresholds:
        for cov in S2_COVS:
            specs.append({
                "component":          "s2",
                "covariate_combo":    cov,
                "threshold_quantile": q,
                "threshold_rate":     None,
                "family":             S2_FAMILY_MODE[0],
                "exposure_mode":      S2_FAMILY_MODE[1],
                "fold_tag":           "is",
            })
        for em in BULK_EXPOSURES:
            for cov in BULK_COVS:
                specs.append({
                    "component":          "bulk",
                    "covariate_combo":    cov,
                    "threshold_quantile": q,
                    "threshold_rate":     None,
                    "family":             BULK_FAMILY,
                    "exposure_mode":      em,
                    "fold_tag":           "is",
                })
        for family, em in TAIL_FAMILY_EXPOSURES:
            for cov in TAIL_COVS:
                specs.append({
                    "component":          "tail",
                    "covariate_combo":    cov,
                    "threshold_quantile": q,
                    "threshold_rate":     None,
                    "family":             family,
                    "exposure_mode":      em,
                    "fold_tag":           "is",
                })

    return specs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--output-path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Where to write the final-grid IS spec JSON file.",
)
@click.option(
    "--thresholds",
    multiple=True,
    type=float,
    default=tuple(THRESHOLDS),
    show_default=True,
    help="Threshold quantile levels. Defaults to the final-grid decision (0.70).",
)
def main(output_path: str, thresholds: tuple[float, ...]) -> None:
    """Build the final-grid IS spec list and write it to JSON."""
    specs = build_specs(list(thresholds))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(specs, indent=2))

    counts: dict[str, int] = {}
    for s in specs:
        counts[s["component"]] = counts.get(s["component"], 0) + 1
    logger.info(
        "Wrote %d IS specs to %s (per-component: %s)",
        len(specs), out, counts,
    )


if __name__ == "__main__":
    main()
