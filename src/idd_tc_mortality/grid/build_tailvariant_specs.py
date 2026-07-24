"""Build the honest-mean tail-variant PRELIMINARY spec list.

Same general preliminary settings as the standard run (3 coarse coupled covariate sets, 6
thresholds 0.70-0.95, s1/s2/bulk unchanged) but with an HONEST-MEAN tail family set: the
median-reporting families (gpd, weibull, log_logistic) are dropped and replaced by their
finite-mean variants, so assembled E[rate] is a true expectation.

Tail families
-------------
Standard-enumerated (cov x exposure apply, like any rate tail):
    gamma, lognormal, truncated_normal   -- already report a mean
    weibull_mean                         -- D  (Weibull mean, replaces median Weibull)
    gpd_cens, log_logistic_cens          -- C  (censored mean)
    gpd_trunc, log_logistic_trunc        -- B  (renormalized-truncated mean)
Count (offset): nb, poisson.
Covariate-free (A), appended specially: gpd_shadow, log_logistic_shadow. A shadow fit is
    intercept-only and ignores log_exposed-in-X, so it is enumerated intercept-only with only
    weighted/unweighted exposure -- NOT the full cov x exposure grid, which would emit ~120
    redundant duplicate fits.

The median-vs-mean contrast figure is recovered POST-HOC on the winning config (same fit
object, base-median predict vs variant-mean predict) -- it is deliberately NOT a grid arm here.

Run:
    python -m idd_tc_mortality.grid.build_tailvariant_specs \\
        --output-path /mnt/team/idd/pub/idd_tc_mortality/01-preliminary/<vintage>_tailvariants/is_specs.json
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import click

from idd_tc_mortality.constants import QUANTILE_LEVELS
from idd_tc_mortality.grid.grid import enumerate_component_specs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Standard-enumerated honest tail rate families (cov x exposure apply) + count families.
HONEST_TAIL_STANDARD: list[str] = [
    "gamma", "lognormal", "truncated_normal",   # already report a mean
    "weibull_mean",                              # D
    "gpd_cens", "log_logistic_cens",             # C
    "gpd_trunc", "log_logistic_trunc",           # B
]
TAIL_COUNT: list[str] = ["nb", "poisson"]

# Covariate-free shadow (A) families: intercept-only, weighted vs unweighted only.
SHADOW_FAMILIES: list[str] = ["gpd_shadow", "log_logistic_shadow"]
SHADOW_EXPOSURES: list[str] = ["weight", "excluded"]  # log_exposed-in-X is moot for A
_INTERCEPT_ONLY: dict[str, bool] = {"wind_speed": False, "sdi": False, "basin": False, "is_island": False}


def build_specs() -> list[dict]:
    """Return the flat honest-mean preliminary component-spec list.

    s1/s2/bulk + the standard honest tails + count tails come from the shared preliminary
    enumerator (only the tail family list is overridden); the covariate-free A tails are
    appended with their reduced (intercept-only x threshold x {weight, unweighted}) grid.
    """
    specs = enumerate_component_specs(
        mode="preliminary",
        tail_families=HONEST_TAIL_STANDARD + TAIL_COUNT,
    )
    for fam in SHADOW_FAMILIES:
        for q in (float(x) for x in QUANTILE_LEVELS):
            for em in SHADOW_EXPOSURES:
                specs.append({
                    "component":          "tail",
                    "covariate_combo":    dict(_INTERCEPT_ONLY),
                    "threshold_quantile": q,
                    "threshold_rate":     None,
                    "family":             fam,
                    "exposure_mode":      em,
                    "fold_tag":           "is",
                })
    return specs


@click.command()
@click.option(
    "--output-path", required=True, type=click.Path(dir_okay=False),
    help="Where to write the honest-mean preliminary IS spec JSON.",
)
def main(output_path: str) -> None:
    """Build the honest-mean preliminary spec list and write it to JSON."""
    specs = build_specs()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(specs, indent=2))

    by_comp = Counter(s["component"] for s in specs)
    tail_fams = Counter(s["family"] for s in specs if s["component"] == "tail")
    logger.info("Wrote %d specs to %s", len(specs), out)
    logger.info("  per-component: %s", dict(by_comp))
    logger.info("  tail families: %s", dict(tail_fams))


if __name__ == "__main__":
    main()
