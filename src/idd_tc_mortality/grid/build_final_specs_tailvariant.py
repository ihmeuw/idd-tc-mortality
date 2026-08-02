"""
CLI entry point for the honest-mean tail-variant FINAL-grid IS spec lists.

Unlike prior cycles, the two vintages reached DIFFERENT refined decisions
(2026-07-28, `notebooks/20260722/dh_refined_diagnostics_{v1985,v2000}.ipynb`,
two-stage TOPSIS cull at CULL_PCT=50), so this module carries one explicit
grid per vintage. Each grid's option sets are the union of the merit groups'
top-10 covariate patterns with the 20260714 baseline sets, exactly as printed
by each notebook's final cell.

    v2000: thresholds {0.70, 0.95}; families {gamma, log_logistic_trunc,
           gpd_cens, log_logistic_cens, lognormal};
           8 s1 x 5 s2 x (2 x 11) bulk x (10 x 9) tail = 158,400 configs.
    v1985: thresholds {0.70, 0.85}; families {gamma, log_logistic_trunc,
           gpd_trunc, log_logistic_cens, lognormal};
           6 s1 x 3 s2 x (2 x 9) bulk x (7 x 12) tail = 54,432 configs.

Editing this file is the auditable record of changing either final grid.
The companion ``build_final_cells --grid tailvariant-<vintage>`` imports these
constants so the spec list and the cell enumeration share one source of truth.

Usage:
    run-build-final-specs-tailvariant --vintage v2000 \\
        --output-path <OUTPUT_ROOT>/01-refined/tailvariants/final_is_specs_v2000.json
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import click

from idd_tc_mortality.grid.build_refined_specs_post2000 import snap_to_quantile_levels

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COV_AXES = ["wind_speed", "sdi", "basin", "is_island"]


def _covs(*patterns: str) -> list[dict[str, bool]]:
    """Covariate-combo dicts from '+'-joined pattern strings ('' = intercept)."""
    out = []
    for p in patterns:
        on = set(p.split("+")) if p else set()
        bad = on - set(COV_AXES)
        if bad:
            raise ValueError(f"Unknown covariate axes in {p!r}: {sorted(bad)}")
        out.append({axis: (axis in on) for axis in COV_AXES})
    return out


# ---------------------------------------------------------------------------
# Per-vintage grids (the auditable decisions — from the refined notebooks)
# ---------------------------------------------------------------------------

GRIDS: dict[str, SimpleNamespace] = {
    "v2000": SimpleNamespace(
        THRESHOLDS=[0.70, 0.95],
        S1_FAMILY_MODE=("logit", "free"),
        S2_FAMILY_MODE=("logit", "free"),
        S1_COVS=_covs(
            "basin+is_island+sdi", "basin+is_island+sdi+wind_speed",
            "basin+sdi", "basin+sdi+wind_speed", "is_island+sdi",
            "is_island+sdi+wind_speed", "sdi", "sdi+wind_speed",
        ),
        S2_COVS=_covs(
            "", "basin+is_island+sdi+wind_speed", "basin+sdi+wind_speed",
            "is_island+sdi+wind_speed", "sdi+wind_speed",
        ),
        BULK_FAMILY="scaled_logit",
        BULK_EXPOSURES=["free", "free+weight"],
        BULK_COVS=_covs(
            "", "basin+is_island+sdi+wind_speed", "basin+is_island+wind_speed",
            "basin+sdi", "basin+sdi+wind_speed", "basin+wind_speed",
            "is_island+sdi+wind_speed", "is_island+wind_speed", "sdi",
            "sdi+wind_speed", "wind_speed",
        ),
        TAIL_FAMILY_EXPOSURES=[
            ("gamma", "excluded"), ("gamma", "free+weight"),
            ("gpd_cens", "excluded"), ("gpd_cens", "free"),
            ("log_logistic_cens", "excluded"), ("log_logistic_cens", "free+weight"),
            ("log_logistic_trunc", "excluded"), ("log_logistic_trunc", "free+weight"),
            ("lognormal", "excluded"), ("lognormal", "free+weight"),
        ],
        TAIL_COVS=_covs(
            "", "basin+is_island+sdi+wind_speed", "basin+sdi",
            "basin+sdi+wind_speed", "is_island+sdi", "is_island+sdi+wind_speed",
            "sdi", "sdi+wind_speed", "wind_speed",
        ),
    ),
    "v1985": SimpleNamespace(
        THRESHOLDS=[0.70, 0.85],
        S1_FAMILY_MODE=("logit", "free"),
        S2_FAMILY_MODE=("logit", "free"),
        S1_COVS=_covs(
            "basin", "basin+is_island", "basin+is_island+sdi",
            "basin+is_island+sdi+wind_speed", "basin+sdi", "basin+sdi+wind_speed",
        ),
        S2_COVS=_covs(
            "basin+is_island+sdi+wind_speed", "basin+sdi+wind_speed", "sdi+wind_speed",
        ),
        BULK_FAMILY="scaled_logit",
        BULK_EXPOSURES=["free", "free+weight"],
        BULK_COVS=_covs(
            "basin+is_island+sdi", "basin+is_island+sdi+wind_speed",
            "basin+is_island+wind_speed", "basin+sdi", "basin+sdi+wind_speed",
            "is_island+sdi", "is_island+sdi+wind_speed", "sdi", "sdi+wind_speed",
        ),
        TAIL_FAMILY_EXPOSURES=[
            ("gamma", "free+weight"), ("gamma", "weight"),
            ("gpd_trunc", "free+weight"),
            ("log_logistic_cens", "free+weight"),
            ("log_logistic_trunc", "free+weight"),
            ("lognormal", "free+weight"), ("lognormal", "weight"),
        ],
        TAIL_COVS=_covs(
            "", "basin+is_island+sdi+wind_speed", "basin+is_island+wind_speed",
            "basin+sdi", "basin+sdi+wind_speed", "basin+wind_speed",
            "is_island+sdi", "is_island+sdi+wind_speed", "is_island+wind_speed",
            "sdi", "sdi+wind_speed", "wind_speed",
        ),
    ),
}


def n_configs(vintage: str) -> int:
    """The DH-config count implied by a vintage's grid (cross of option sets)."""
    g = GRIDS[vintage]
    return (len(g.THRESHOLDS) * len(g.S1_COVS) * len(g.S2_COVS)
            * len(g.BULK_EXPOSURES) * len(g.BULK_COVS)
            * len(g.TAIL_FAMILY_EXPOSURES) * len(g.TAIL_COVS))


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------

def build_specs(vintage: str) -> list[dict]:
    """Return the flat IS component spec list for a vintage's final grid.

    Same shape as ``build_final_specs.build_specs`` output: one dict per
    distinct (component, family, exposure_mode, threshold, covariate_combo).
    """
    g = GRIDS[vintage]
    thresholds = snap_to_quantile_levels(list(g.THRESHOLDS))
    specs: list[dict] = []

    for cov in g.S1_COVS:
        specs.append({
            "component":          "s1",
            "covariate_combo":    cov,
            "threshold_quantile": None,
            "threshold_rate":     None,
            "family":             g.S1_FAMILY_MODE[0],
            "exposure_mode":      g.S1_FAMILY_MODE[1],
            "fold_tag":           "is",
        })
    for q in thresholds:
        for cov in g.S2_COVS:
            specs.append({
                "component":          "s2",
                "covariate_combo":    cov,
                "threshold_quantile": q,
                "threshold_rate":     None,
                "family":             g.S2_FAMILY_MODE[0],
                "exposure_mode":      g.S2_FAMILY_MODE[1],
                "fold_tag":           "is",
            })
        for em in g.BULK_EXPOSURES:
            for cov in g.BULK_COVS:
                specs.append({
                    "component":          "bulk",
                    "covariate_combo":    cov,
                    "threshold_quantile": q,
                    "threshold_rate":     None,
                    "family":             g.BULK_FAMILY,
                    "exposure_mode":      em,
                    "fold_tag":           "is",
                })
        for fam, em in g.TAIL_FAMILY_EXPOSURES:
            for cov in g.TAIL_COVS:
                specs.append({
                    "component":          "tail",
                    "covariate_combo":    cov,
                    "threshold_quantile": q,
                    "threshold_rate":     None,
                    "family":             fam,
                    "exposure_mode":      em,
                    "fold_tag":           "is",
                })
    return specs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--vintage",
    type=click.Choice(sorted(GRIDS)),
    required=True,
    help="Which vintage's final grid to build (the grids differ this cycle).",
)
@click.option(
    "--output-path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Where to write the final IS spec JSON file.",
)
def main(vintage: str, output_path: str) -> None:
    """Build a vintage's tail-variant final-grid IS spec list."""
    specs = build_specs(vintage)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(specs, indent=2))

    by_comp = Counter(s["component"] for s in specs)
    logger.info(
        "Wrote %d IS specs for %s to %s (per-component: %s); implied DH grid: %s configs",
        len(specs), vintage, out, dict(by_comp), f"{n_configs(vintage):,}",
    )


if __name__ == "__main__":
    main()
