"""Generate the 5-version comparison set of draw-level direct-deaths files.

Given one A1/A0 blend (``--pred-root``/``--mid``/``--cell``), writes five
date-prefixed draw-level files (parquet + netCDF) for side-by-side comparison in
``notebooks/.../sr_version_comparison.ipynb``:

    <prefix>_unadjusted_direct_deaths
    <prefix>_adjusted_mean_direct_deaths
    <prefix>_adjusted_median_direct_deaths
    <prefix>_adjusted_mean_sr31_guard_direct_deaths
    <prefix>_adjusted_median_sr31_guard_direct_deaths

The four adjusted variants are the 2 (rake statistic: mean/median) x 2 (SR-31
guard: off/on) matrix; the fifth file is the un-raked blend.

The unadjusted blend is read once — from the blend dir's existing
``draw_level_<cell>_ssp_unadjusted.parquet`` if present, else rebuilt from the
a0/a1 partials — and every adjusted variant is derived from it, so the expensive
blend read happens a single time. Each variant is written and dropped before the
next is built, so at most one adjusted frame is held alongside the base. All
computation reuses ``finalize_deliverable``'s library functions; this script only
orchestrates the matrix.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import click
import pandas as pd

from idd_tc_mortality.predict.finalize_deliverable import (
    DEFAULT_CELL,
    DEFAULT_OBS_YEARS,
    DEFAULT_REF_SCENARIOS,
    DEFAULT_SCENARIOS,
    apply_super_region_adjustment,
    blend_export,
    load_fhs_population,
    load_or_build_blend_summary,
    merge_population_and_rate,
    population_index,
    super_region_ratios,
    write_draw_level,
)
from idd_tc_mortality.predict.paths import (
    DIRECT_RISK_DIR,
    FHS_FUTURE_POP_PATH,
    FHS_PAST_POP_PATH,
    STORM_DRAW_TABLE_PATH,
)
from idd_tc_mortality.predict.postprocess import (
    DEFAULT_HIERARCHY_PATH,
    DEFAULT_OBS_PATH,
    build_ancestor_map,
    build_observed_deaths,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The deaths columns each adjusted variant is derived from (population + death_rate
# are re-attached per variant after scaling).
CORE_COLS = ["storm_draw", "ssp_scenario", "year_id", "location_id", "deaths"]


def _csv(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def load_unadjusted_core(blend_dir, a0_dir, a1_dir, hierarchy_df, cell,
                         scenarios) -> pd.DataFrame:
    """Unadjusted draw-level deaths (``CORE_COLS``).

    Reads the blend dir's existing ``draw_level_<cell>_ssp_unadjusted.parquet``
    if present (its ``deaths`` column IS ``blend_export`` output — the file was
    written by a prior finalize run), else rebuilds from the a0/a1 partials.
    """
    existing = Path(blend_dir) / f"draw_level_{cell}_ssp_unadjusted.parquet"
    if existing.exists():
        logger.info("reading existing unadjusted blend %s", existing)
        return pd.read_parquet(existing, columns=CORE_COLS)
    logger.info("unadjusted blend absent — rebuilding from a0/a1 partials")
    return blend_export(a0_dir, a1_dir, hierarchy_df, cell, scenarios)[CORE_COLS]


@click.command()
@click.option("--pred-root", required=True, type=click.Path(path_type=Path),
              help="Predict output root for the vintage, e.g. 04-predict/<vintage>.")
@click.option("--mid", required=True, help="Death-model id (the <mid> in <mid>_{a0,a1,blend}).")
@click.option("--prefix", required=True,
              help="Filename date prefix, e.g. 2026_07_13.")
@click.option("--a0-dir", type=click.Path(path_type=Path), default=None,
              help="A0 partials dir. Default <pred-root>/<mid>_a0.")
@click.option("--a1-dir", type=click.Path(path_type=Path), default=None,
              help="A1 partials dir. Default <pred-root>/<mid>_a1.")
@click.option("--blend-dir", type=click.Path(path_type=Path), default=None,
              help="Blend dir with summary.parquet + the unadjusted draw-level file. "
                   "Default <pred-root>/<mid>_blend.")
@click.option("--out-dir", type=click.Path(path_type=Path), default=DIRECT_RISK_DIR,
              show_default=True, help="Where the 10 files (5 versions x parquet+nc) go.")
@click.option("--summary-path", type=click.Path(path_type=Path), default=None,
              help="Blend summary for the rake ratios. Default <blend-dir>/summary.parquet; "
                   "built from the a0/a1 partials if absent.")
@click.option("--storm-draw-table", default=STORM_DRAW_TABLE_PATH, show_default=True,
              help="storm_draw_table.csv — draw ids used only when building a missing summary.")
@click.option("--cell", default=DEFAULT_CELL, show_default=True, help="Toggle-cell column name.")
@click.option("--scenarios", default=",".join(DEFAULT_SCENARIOS), show_default=True,
              help="Comma-separated SSP scenarios to keep (only used on a rebuild).")
@click.option("--hierarchy-path", default=DEFAULT_HIERARCHY_PATH, show_default=True)
@click.option("--obs-path", default=DEFAULT_OBS_PATH, show_default=True)
@click.option("--past-pop-path", default=FHS_PAST_POP_PATH, show_default=True)
@click.option("--future-pop-path", default=FHS_FUTURE_POP_PATH, show_default=True)
@click.option("--ref-scenarios", default=",".join(DEFAULT_REF_SCENARIOS), show_default=True,
              help="Comma-separated scenarios for the rake predicted baseline.")
@click.option("--obs-year-start", default=DEFAULT_OBS_YEARS[0], show_default=True, type=int)
@click.option("--obs-year-end", default=DEFAULT_OBS_YEARS[1], show_default=True, type=int)
def main(pred_root, mid, prefix, a0_dir, a1_dir, blend_dir, out_dir, summary_path,
         storm_draw_table, cell, scenarios, hierarchy_path, obs_path, past_pop_path,
         future_pop_path, ref_scenarios, obs_year_start, obs_year_end):
    a0_dir = a0_dir or pred_root / f"{mid}_a0"
    a1_dir = a1_dir or pred_root / f"{mid}_a1"
    blend_dir = blend_dir or pred_root / f"{mid}_blend"
    summary_path = summary_path or blend_dir / "summary.parquet"
    out_dir = Path(out_dir)
    scenarios = _csv(scenarios)
    ref_scenarios = _csv(ref_scenarios)
    obs_years = (obs_year_start, obs_year_end)

    hierarchy_df = pd.read_parquet(hierarchy_path)

    logger.info("loading unadjusted blend + adjustment inputs")
    base = load_unadjusted_core(blend_dir, a0_dir, a1_dir, hierarchy_df, cell, scenarios)
    logger.info("    %d rows | storm_draws=%d locations=%d years %d-%d",
                len(base), base["storm_draw"].nunique(), base["location_id"].nunique(),
                base["year_id"].min(), base["year_id"].max())
    summary = load_or_build_blend_summary(summary_path, a0_dir, a1_dir, hierarchy_df,
                                          storm_draw_table)
    obs = build_observed_deaths(str(obs_path), build_ancestor_map(hierarchy_df))
    pop = load_fhs_population(past_pop_path, future_pop_path,
                              int(base["year_id"].min()), int(base["year_id"].max()))
    pop2 = population_index(pop)

    def emit(name: str, deaths: pd.DataFrame) -> None:
        d = merge_population_and_rate(deaths, pop)
        base_path = out_dir / f"{prefix}_{name}_direct_deaths"
        for p in write_draw_level(d, base_path, pop2):
            logger.info("wrote %s", p)

    emit("unadjusted", base)
    for stat, guard in itertools.product(("mean", "median"), (False, True)):
        name = f"adjusted_{stat}" + ("_sr31_guard" if guard else "")
        ratios = super_region_ratios(summary, obs, hierarchy_df, cell, ref_scenarios,
                                     obs_years, sr31_guard=guard, statistic=stat)
        logger.info("(%s) ratios: %s", name,
                    {int(k): round(v, 4) for k, v in ratios.dropna().items()})
        adj = apply_super_region_adjustment(base, ratios, hierarchy_df)
        emit(name, adj)
        del adj


if __name__ == "__main__":
    main()
