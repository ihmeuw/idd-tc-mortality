"""
CLI entry point for building the post-2000 refined-grid IS spec list.

Fork of build_refined_specs.py with the screening decisions from the
2026-05-17 post-2000 evaluate cycle locked in. The all-years version
(`build_refined_specs.py`) is preserved for reproducing the
`preliminary_decisions.qmd` decisions; this file is the auditable record
for `preliminary_decisions_post2000.qmd`.

Differences from the all-years constants:
  - Bulk family narrowed to `scaled_logit` only (was {beta, scaled_logit}).
  - Tail families narrowed to {gpd, log_logistic, weibull} (was 6 families
    including lognormal / truncated_normal / gamma).
  - Tail exposure modes per family (per-family dict, not a Cartesian):
      gpd:          [free, weight, free+weight]
      log_logistic: [free, weight, free+weight]
      weibull:      [free]
    The `excluded` exposure mode is dropped from all tail families on this
    cycle — top-25 calibrated survivors had no `excluded` entries.
  - Default thresholds (0.70, 0.85), reflecting the post-2000 threshold
    decision: 0.90 was the only threshold that couldn't sustain bulk-of-
    storms calibration after dropping the top-25 mega-events; 0.70 was
    the cleanest IS-OOS generalizer; 0.85 was the best non-0.70 threshold.

Each constant traces back to a locked-in callout in
`reports/preliminary_decisions_post2000.qmd`. Editing this file is the
auditable record of changing the post-2000 refined grid. To change the
grid, edit the constants below in a single commit.

Usage:
    run-build-refined-specs-post2000 \\
        --output-path /mnt/team/idd/pub/idd_tc_mortality/01-refined/<date>/refined_is_specs.json \\
        [--thresholds 0.70 --thresholds 0.85]
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import click
import numpy as np

from idd_tc_mortality.constants import QUANTILE_LEVELS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def snap_to_quantile_levels(thresholds: list[float]) -> list[float]:
    """Snap requested thresholds to the canonical QUANTILE_LEVELS floats.

    The pipeline keys thresholds by np.linspace values, where e.g. 0.80 is
    stored as 0.7999999999999999. A CLI-parsed exact 0.8 would KeyError in
    the orchestrator's threshold_map and mismatch dh_results joins, so every
    requested threshold must resolve to its canonical representation.
    """
    snapped: list[float] = []
    for q in thresholds:
        j = int(np.argmin(np.abs(QUANTILE_LEVELS - q)))
        if abs(float(QUANTILE_LEVELS[j]) - q) > 1e-6:
            raise ValueError(
                f"threshold {q} is not one of QUANTILE_LEVELS "
                f"{[round(float(x), 4) for x in QUANTILE_LEVELS]}"
            )
        snapped.append(float(QUANTILE_LEVELS[j]))
    return snapped


# ---------------------------------------------------------------------------
# Screening decisions (locked-in by reports/preliminary_decisions_post2000.qmd)
# ---------------------------------------------------------------------------

# All 2^4 = 16 subsets of the four covariate axes.
COV_AXES = ["wind_speed", "sdi", "basin", "is_island"]
COV_COMBOS: list[dict[str, bool]] = [
    {axis: bool(bit) for axis, bit in zip(COV_AXES, bits)}
    for bits in itertools.product([False, True], repeat=len(COV_AXES))
]

# S1 — Decision: family=logit, exposure_mode=free.
S1_FAMILY_MODES: list[tuple[str, str]] = [("logit", "free")]

# S2 — Decision: family=logit, exposure_mode=free.
S2_FAMILY_MODES: list[tuple[str, str]] = [("logit", "free")]

# Bulk — POST-2000 Decision: family=scaled_logit only,
# exposure_mode in {free, free+weight}. (All-years kept beta too;
# post-2000 data showed scaled_logit dominates beta by 14-38% across
# thresholds — see post2000 qmd Bulk section.)
BULK_FAMILY_MODES: list[tuple[str, str]] = list(itertools.product(
    ["scaled_logit"],
    ["free", "free+weight"],
))

# Tail — POST-2000 Decision: families {gpd, log_logistic, weibull} only,
# with per-family exposure modes. (All-years kept 6 families × 4 exposures
# including `excluded`; post-2000 top-25 survivors had only these 3
# families and weibull only ever appeared with `free`.)
TAIL_FAMILY_EXPOSURES: dict[str, list[str]] = {
    "gpd":          ["free", "weight", "free+weight"],
    "log_logistic": ["free", "weight", "free+weight"],
    "weibull":      ["free"],
}
TAIL_FAMILY_MODES: list[tuple[str, str]] = [
    (family, exposure)
    for family, exposures in TAIL_FAMILY_EXPOSURES.items()
    for exposure in exposures
]


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------

def build_specs(thresholds: list[float]) -> list[dict]:
    """Return the flat list of IS component spec dicts for the post-2000
    refined grid. Same shape as `enumerate_component_specs` output."""
    specs: list[dict] = []
    for combo in COV_COMBOS:
        for family, em in S1_FAMILY_MODES:
            specs.append({
                "component":          "s1",
                "covariate_combo":    combo,
                "threshold_quantile": None,
                "threshold_rate":     None,
                "family":             family,
                "exposure_mode":      em,
                "fold_tag":           "is",
            })
        for q in thresholds:
            for family, em in S2_FAMILY_MODES:
                specs.append({
                    "component":          "s2",
                    "covariate_combo":    combo,
                    "threshold_quantile": q,
                    "threshold_rate":     None,
                    "family":             family,
                    "exposure_mode":      em,
                    "fold_tag":           "is",
                })
            for family, em in BULK_FAMILY_MODES:
                specs.append({
                    "component":          "bulk",
                    "covariate_combo":    combo,
                    "threshold_quantile": q,
                    "threshold_rate":     None,
                    "family":             family,
                    "exposure_mode":      em,
                    "fold_tag":           "is",
                })
            for family, em in TAIL_FAMILY_MODES:
                specs.append({
                    "component":          "tail",
                    "covariate_combo":    combo,
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
    help="Where to write the post-2000 refined IS spec JSON file.",
)
@click.option(
    "--thresholds",
    multiple=True,
    type=float,
    default=(0.70, 0.85),
    show_default=True,
    help="Threshold quantile levels to include. Defaults to the post-2000 "
         "qmd Threshold-section decision (0.70, 0.85).",
)
def main(output_path: str, thresholds: tuple[float, ...]) -> None:
    """Build the post-2000 refined-grid IS spec list and write it to JSON."""
    specs = build_specs(snap_to_quantile_levels(list(thresholds)))
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
