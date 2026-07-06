"""
CLI entry point for building the refined-grid IS spec list.

Run this AFTER the preliminary screening has chosen which families,
exposure modes, covariate combinations, and thresholds to keep. The
output is a JSON file of IS component spec dicts that
`run-fit-orchestrate --specs-file <path>` consumes directly, bypassing
enumerate_component_specs.

Usage:
    run-build-refined-specs \\
        --output-path /mnt/team/idd/pub/idd_tc_mortality/01-refined/<date>/refined_is_specs.json \\
        [--thresholds 0.70 --thresholds 0.75]

The screening decisions (kept families, kept exposure modes, kept
covariate combinations) are hard-coded as constants in this module.
They are intentionally not CLI flags: each constant traces back to a
locked-in callout in `reports/preliminary_decisions.qmd`, and editing
this file is the auditable record of changing the refined grid. To
change the grid, edit the constants below in a single commit.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Screening decisions (locked-in by reports/preliminary_decisions.qmd)
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

# Bulk — Decision: family in {beta, scaled_logit}, exposure_mode in {free, free+weight}.
BULK_FAMILY_MODES: list[tuple[str, str]] = list(itertools.product(
    ["beta", "scaled_logit"],
    ["free", "free+weight"],
))

# Tail — Decision: family in {gpd, log_logistic, lognormal, truncated_normal,
# gamma, weibull} (drop nb, poisson). Exposure mode left unrestricted at all
# four rate exposure modes per the qmd's decision summary.
TAIL_FAMILY_MODES: list[tuple[str, str]] = list(itertools.product(
    ["gpd", "log_logistic", "lognormal", "truncated_normal", "gamma", "weibull"],
    ["free", "weight", "free+weight", "excluded"],
))


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------

def build_specs(thresholds: list[float]) -> list[dict]:
    """Return the flat list of IS component spec dicts for the refined grid.

    The shape of each spec matches what `enumerate_component_specs` produces
    so that downstream code (orchestrator, run-fit-component) is unaware of
    whether the manifest came from enumerate or from this builder.
    """
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
    help="Where to write the refined IS spec JSON file.",
)
@click.option(
    "--thresholds",
    multiple=True,
    type=float,
    default=(0.70, 0.75),
    show_default=True,
    help="Threshold quantile levels to include. Defaults to the qmd's "
         "Threshold-section decision (0.70, 0.75).",
)
def main(output_path: str, thresholds: tuple[float, ...]) -> None:
    """Build the refined-grid IS spec list and write it to JSON."""
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
