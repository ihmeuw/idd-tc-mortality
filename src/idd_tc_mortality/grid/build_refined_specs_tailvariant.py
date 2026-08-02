"""
CLI entry point for building the honest-mean tail-variant REFINED-grid IS spec list.

Fork of build_refined_specs_post2000.py with the 20260722 tail-variant
intermediate-cull decisions locked in (2026-07-27, SAME decisions for both
vintages v1985/v2000):

  - Thresholds: {0.70, 0.85, 0.95} — drop 0.75, 0.80, 0.90.
  - Tail: ALL 8 honest-mean families kept (gamma, lognormal, truncated_normal,
    weibull_mean, gpd_cens, log_logistic_cens, gpd_trunc, log_logistic_trunc),
    full Cartesian with all four rate exposure modes including `excluded` —
    the intermediate cull did not narrow the tail dimension.
  - Bulk: no narrowing — scaled_logit x {free, free+weight} (the preliminary
    decision carried forward).
  - S1/S2: logit/free carried forward.
  - Covariates: the standard refined-stage move from the 3 coarse coupled
    preliminary sets to all 2^4 = 16 per-component combos.

Each constant traces back to the intermediate-cull decision logged in
`.claude/DECISIONS.md`. Editing this file is the auditable record of changing
the tail-variant refined grid. To change the grid, edit the constants below
in a single commit.

Usage:
    run-build-refined-specs-tailvariant \\
        --output-path <OUTPUT_ROOT>/01-refined/tailvariants/refined_is_specs.json
"""

from __future__ import annotations

import itertools
import json
import logging
from collections import Counter
from pathlib import Path

import click

from idd_tc_mortality.grid.build_refined_specs_post2000 import snap_to_quantile_levels

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Screening decisions (locked-in by the 20260722 intermediate cull)
# ---------------------------------------------------------------------------

# All 2^4 = 16 subsets of the four covariate axes.
COV_AXES = ["wind_speed", "sdi", "basin", "is_island"]
COV_COMBOS: list[dict[str, bool]] = [
    {axis: bool(bit) for axis, bit in zip(COV_AXES, bits)}
    for bits in itertools.product([False, True], repeat=len(COV_AXES))
]

# S1 — Decision: family=logit, exposure_mode=free (carried from preliminary).
S1_FAMILY_MODES: list[tuple[str, str]] = [("logit", "free")]

# S2 — Decision: family=logit, exposure_mode=free (carried from preliminary).
S2_FAMILY_MODES: list[tuple[str, str]] = [("logit", "free")]

# Bulk — Decision: no intermediate narrowing; the preliminary decision stands:
# family=scaled_logit only, exposure_mode in {free, free+weight}.
BULK_FAMILY_MODES: list[tuple[str, str]] = list(itertools.product(
    ["scaled_logit"],
    ["free", "free+weight"],
))

# Tail — Decision: the intermediate cull kept ALL 8 honest-mean families and
# did not restrict exposure modes, so the full Cartesian survives (contrast
# with the 20260517 post-2000 cycle's per-family exposure dict).
HONEST_TAIL_FAMILIES: list[str] = [
    "gamma", "lognormal", "truncated_normal",   # already report a mean
    "weibull_mean",                              # D
    "gpd_cens", "log_logistic_cens",             # C
    "gpd_trunc", "log_logistic_trunc",           # B
]
TAIL_EXPOSURE_MODES: list[str] = ["free", "weight", "free+weight", "excluded"]
TAIL_FAMILY_MODES: list[tuple[str, str]] = list(itertools.product(
    HONEST_TAIL_FAMILIES,
    TAIL_EXPOSURE_MODES,
))

# Threshold decision: {0.70, 0.85, 0.95}. Encoded as the CLI default below so
# the flag remains overridable for probes; values are snapped to the canonical
# QUANTILE_LEVELS floats before use.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.70, 0.85, 0.95)


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------

def build_specs(thresholds: list[float]) -> list[dict]:
    """Return the flat list of IS component spec dicts for the tail-variant
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
    help="Where to write the tail-variant refined IS spec JSON file.",
)
@click.option(
    "--thresholds",
    multiple=True,
    type=float,
    default=DEFAULT_THRESHOLDS,
    show_default=True,
    help="Threshold quantile levels to include. Defaults to the 20260722 "
         "intermediate-cull decision (0.70, 0.85, 0.95).",
)
def main(output_path: str, thresholds: tuple[float, ...]) -> None:
    """Build the tail-variant refined-grid IS spec list and write it to JSON."""
    specs = build_specs(snap_to_quantile_levels(list(thresholds)))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(specs, indent=2))

    by_comp = Counter(s["component"] for s in specs)
    tail_fams = Counter(s["family"] for s in specs if s["component"] == "tail")
    logger.info("Wrote %d IS specs to %s", len(specs), out)
    logger.info("  per-component: %s", dict(by_comp))
    logger.info("  tail families: %s", dict(tail_fams))


if __name__ == "__main__":
    main()
