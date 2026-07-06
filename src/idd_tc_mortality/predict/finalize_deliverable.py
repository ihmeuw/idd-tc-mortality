"""Finalize the direct-deaths deliverable for one death model's A1/A0 blend.

Chains the three post-processing steps that produced the shipped 2026-06-24 M1
deliverable (``/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/direct_deaths.*``):

    (a) blend export             -> draw-level A1/A0-blended deaths for one toggle
                                    cell, columns renamed to deliverable names
    (b) super-region median adj  -> scale each location by its super-region's
                                    observed/predicted median ratio; reroll Global
    (c) FHS-population rate merge -> attach FHS population + death_rate (per 100k)

and writes ``draw_level_<cell>_ssp_{unadjusted,adjusted}.{parquet,nc}``.

This is the promotion to committed code of three ad-hoc heredocs run in-session;
the computational logic is kept equivalent to them (verified byte-for-byte against
the shipped file — see ``run-finalize-deliverable`` verification in DECISIONS.md).

Omitted intermediate (not a silent change): the original chain merged a
``build_population`` (country-and-up only) population + death_rate onto the frame
between (a) and (b); step (c) then DROPPED and fully overwrote both columns with
the FHS population. That intermediate never affected the shipped bytes, so it is
omitted here as dead computation. The final population/death_rate come entirely
from the FHS past+future population, which also covers admin-1 (level 4).

SR-31 caveat / OPEN DECISION: super-region 31 (Central Europe / Eastern Europe /
Central Asia) has observed median deaths = 0, so its median ratio is 0 and its
deaths + rates are hard-zeroed. The shipped file used that hard zero. The older
20260515 notebook instead replaced a 0 ratio with 0.1 x the smallest non-zero
ratio; that guard is exposed as ``--sr31-guard/--no-sr31-guard``, defaulting to
``--no-sr31-guard`` (what shipped). See ``.claude/DECISIONS.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

from idd_tc_mortality.predict.consolidated import draw_level_blend, rollup_blend
from idd_tc_mortality.predict.paths import (
    FHS_FUTURE_POP_PATH,
    FHS_PAST_POP_PATH,
    STORM_DRAW_TABLE_PATH,
    atomic_write_parquet,
)
from idd_tc_mortality.predict.postprocess import (
    DEFAULT_HIERARCHY_PATH,
    DEFAULT_OBS_PATH,
    build_ancestor_map,
    build_observed_deaths,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GROUP_COLS = ["storm_draw", "ssp_scenario", "year_id"]
GLOBAL_LOCATION_ID = 1
DEFAULT_CELL = "deaths_c1_s1_o0_b1"
DEFAULT_SCENARIOS = ("ssp126", "ssp245", "ssp585")
DEFAULT_REF_SCENARIOS = ("historical", "ssp245")
DEFAULT_OBS_YEARS = (2000, 2023)
DEFAULT_SPLIT_YEAR = 2024


# ---------------------------------------------------------------------------
# (a) blend export
# ---------------------------------------------------------------------------

def blend_export(a0_dir, a1_dir, hierarchy_df: pd.DataFrame, cell: str,
                 scenarios) -> pd.DataFrame:
    """Draw-level A1/A0 blend for one cell, columns renamed to deliverable names.

    Wraps ``draw_level_blend`` (which returns storm_draw/experiment_id/year/
    location_id/deaths for all FHS levels) and renames the two columns the
    deliverable uses different names for.
    """
    df = draw_level_blend(Path(a0_dir), Path(a1_dir), hierarchy_df, cell,
                          scenarios=list(scenarios))
    return df.rename(columns={"experiment_id": "ssp_scenario", "year": "year_id"})


# ---------------------------------------------------------------------------
# (b) super-region median adjustment
# ---------------------------------------------------------------------------

def load_or_build_blend_summary(summary_path, a0_dir, a1_dir, hierarchy_df,
                                storm_draw_table) -> pd.DataFrame:
    """Read the blend summary, or build it from the a0/a1 partials if absent.

    Step (b)'s predicted-median baseline needs the summarized (mean across storm
    draws) A1/A0-blend at super-region level. That summary was previously produced
    by a manual ``rollup_blend`` step; building it on demand here keeps
    ``run-finalize-deliverable`` self-sufficient for a rerun (no paste-a-snippet gap).
    """
    summary_path = Path(summary_path)
    if summary_path.exists():
        return pd.read_parquet(summary_path)
    logger.info("blend summary %s absent — building from a0/a1 partials", summary_path)
    draw_ids = sorted(int(x) for x in pd.read_csv(storm_draw_table)["storm_draw"].unique())
    summary, _ = rollup_blend(Path(a0_dir), Path(a1_dir), hierarchy_df, draw_ids)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(summary, summary_path)
    logger.info("    wrote %s (%d rows)", summary_path, len(summary))
    return summary


def super_region_median_ratios(summary: pd.DataFrame, obs_by_loc: pd.DataFrame,
                               hierarchy_df: pd.DataFrame, cell: str,
                               ref_scenarios, obs_years, sr31_guard: bool = False
                               ) -> pd.Series:
    """obs_median / pred_median per super-region (level 1) for one cell.

    obs_median  = median observed super-region deaths over ``obs_years``.
    pred_median = median of the blend summary ``mean`` over ``ref_scenarios`` x
                  ``obs_years``.

    A super-region with 0 observed median gets ratio 0 (its deaths hard-zeroed)
    unless ``sr31_guard`` replaces a 0 ratio with 0.1 x the smallest non-zero
    ratio (the old 20260515-notebook behavior).
    """
    y0, y1 = obs_years
    lvl = hierarchy_df[["location_id", "level", "super_region_id"]]
    s = summary.merge(lvl, on="location_id", how="left")
    obs = obs_by_loc.merge(hierarchy_df[["location_id", "level"]], on="location_id",
                           how="left")
    obs_med = (obs[(obs["level"] == 1) & (obs["year"].between(y0, y1))]
               .groupby("location_id")["deaths"].median())
    pred_med = (s[(s["level"] == 1) & (s["cell"] == cell)
                  & (s["experiment_id"].isin(list(ref_scenarios)))
                  & (s["year"].between(y0, y1))]
                .groupby("location_id")["mean"].median())
    ratio = (obs_med / pred_med).replace([np.inf, -np.inf], np.nan)
    if sr31_guard:
        nonzero_min = ratio[ratio > 0].min()
        ratio = ratio.mask(ratio == 0, 0.1 * nonzero_min)
    return ratio


def apply_super_region_adjustment(draws: pd.DataFrame, ratios: pd.Series,
                                  hierarchy_df: pd.DataFrame,
                                  group_cols=GROUP_COLS,
                                  global_id: int = GLOBAL_LOCATION_ID) -> pd.DataFrame:
    """Scale each location's deaths by ITS super-region's ratio; reroll Global.

    Every location is scaled by the ratio of the super-region it belongs to (via
    ``super_region_id``); locations whose super-region has no ratio are left
    unscaled (x1). ``global_id`` is then recomputed as the sum of the adjusted
    super-regions so the hierarchy stays internally consistent.
    """
    loc2sr = hierarchy_df.set_index("location_id")["super_region_id"]
    sr_ids = list(ratios.index)
    adj = draws.copy()
    adj["deaths"] = adj["deaths"] * adj["location_id"].map(loc2sr).map(ratios).fillna(1.0)
    g = (adj[adj["location_id"].isin(sr_ids)]
         .groupby(list(group_cols), as_index=False)["deaths"].sum())
    adj = adj.merge(g.rename(columns={"deaths": "_g"}), on=list(group_cols), how="left")
    adj.loc[adj["location_id"] == global_id, "deaths"] = \
        adj.loc[adj["location_id"] == global_id, "_g"]
    return adj.drop(columns="_g")


# ---------------------------------------------------------------------------
# (c) FHS-population rate merge
# ---------------------------------------------------------------------------

def _load_population(path, ymin: int, ymax: int) -> pd.DataFrame:
    """FHS population as (location_id, year_id, population) for [ymin, ymax].

    Handles both the past (draws/statistic) and future (summary 'draws' var with a
    'statistic' dim) file layouts, selecting all-age both-sex (sex_id=3,
    age_group_id=22), the single scenario, and the 'mean' statistic; any leftover
    non-(location, year) dim is averaged out.
    """
    ds = xr.open_dataset(path)
    var = ("draws" if "draws" in ds.data_vars
           else "population" if "population" in ds.data_vars
           else list(ds.data_vars)[0])
    da = ds[var]
    sel = {}
    if "statistic" in da.dims or "statistic" in da.coords:
        sel["statistic"] = "mean"
    if "scenario" in da.dims:
        sel["scenario"] = int(ds["scenario"].values[0])
    if "sex_id" in da.dims:
        sel["sex_id"] = 3
    if "age_group_id" in da.dims:
        sel["age_group_id"] = 22
    da = da.sel(**sel)
    extra = [d for d in da.dims if d not in ("location_id", "year_id")]
    if extra:
        da = da.mean(extra)
    out = da.to_dataframe("population").reset_index()[["location_id", "year_id", "population"]]
    return out[out["year_id"].between(ymin, ymax)]


def load_fhs_population(past_pop_path, future_pop_path, ymin: int, ymax: int,
                        split_year: int = DEFAULT_SPLIT_YEAR) -> pd.DataFrame:
    """Past population for years < ``split_year`` spliced with future for >=, over
    [ymin, ymax]. Returns (location_id, year_id, population) with integer ids."""
    past = _load_population(past_pop_path, ymin, split_year - 1)
    future = _load_population(future_pop_path, split_year, ymax)
    pop = pd.concat([past, future], ignore_index=True)
    return pop.dropna(subset=["population"]).astype({"location_id": "int64",
                                                     "year_id": "int64"})


def merge_population_and_rate(draws: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Attach ``population`` and ``death_rate`` (per 100,000); overwrite any existing."""
    d = draws.drop(columns=["population", "death_rate"], errors="ignore")
    d = d.merge(pop, on=["location_id", "year_id"], how="left")
    d["death_rate"] = d["deaths"] / d["population"] * 1e5
    return d


def _write_nc(df: pd.DataFrame, path: str, pop2: pd.Series) -> None:
    """netCDF: deaths 4-D (0-filled), population 2-D, death_rate = deaths/pop*1e5."""
    ds = df.set_index(["storm_draw", "ssp_scenario", "year_id", "location_id"])[["deaths"]].to_xarray()
    ds["deaths"] = ds["deaths"].fillna(0.0)
    ds["population"] = pop2.to_xarray()
    ds["death_rate"] = ds["deaths"] / ds["population"] * 1e5
    ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})


def write_deliverable(unadjusted: pd.DataFrame, adjusted: pd.DataFrame,
                      pop: pd.DataFrame, out_dir, cell: str) -> list[Path]:
    """Write the four ``draw_level_<cell>_ssp_{unadjusted,adjusted}.{parquet,nc}`` files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pop2 = (pop.drop_duplicates(["location_id", "year_id"])
            .set_index(["year_id", "location_id"])["population"])
    written: list[Path] = []
    for tag, df in (("unadjusted", unadjusted), ("adjusted", adjusted)):
        base = out_dir / f"draw_level_{cell}_ssp_{tag}"
        atomic_write_parquet(df, base.with_suffix(".parquet"))
        _write_nc(df, str(base.with_suffix(".nc")), pop2)
        written += [base.with_suffix(".parquet"), base.with_suffix(".nc")]
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _csv(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


@click.command()
@click.option("--pred-root", required=True, type=click.Path(path_type=Path),
              help="Predict output root for the vintage, e.g. 04-predict/<vintage>.")
@click.option("--mid", required=True, help="Death-model id (the <mid> in <mid>_{a0,a1,blend}).")
@click.option("--a0-dir", type=click.Path(path_type=Path), default=None,
              help="A0 partials dir. Default <pred-root>/<mid>_a0.")
@click.option("--a1-dir", type=click.Path(path_type=Path), default=None,
              help="A1 partials dir. Default <pred-root>/<mid>_a1.")
@click.option("--blend-dir", type=click.Path(path_type=Path), default=None,
              help="Blend dir holding summary.parquet. Default <pred-root>/<mid>_blend.")
@click.option("--out-dir", type=click.Path(path_type=Path), default=None,
              help="Where the 4 files are written. Default = blend dir.")
@click.option("--summary-path", type=click.Path(path_type=Path), default=None,
              help="Blend summary for the median ratios. Default <blend-dir>/summary.parquet; "
                   "built from the a0/a1 partials if absent.")
@click.option("--storm-draw-table", default=STORM_DRAW_TABLE_PATH, show_default=True,
              help="storm_draw_table.csv — draw ids used only when building a missing blend summary.")
@click.option("--cell", default=DEFAULT_CELL, show_default=True, help="Toggle-cell column name.")
@click.option("--scenarios", default=",".join(DEFAULT_SCENARIOS), show_default=True,
              help="Comma-separated SSP scenarios to keep.")
@click.option("--hierarchy-path", default=DEFAULT_HIERARCHY_PATH, show_default=True)
@click.option("--obs-path", default=DEFAULT_OBS_PATH, show_default=True)
@click.option("--past-pop-path", default=FHS_PAST_POP_PATH, show_default=True)
@click.option("--future-pop-path", default=FHS_FUTURE_POP_PATH, show_default=True)
@click.option("--ref-scenarios", default=",".join(DEFAULT_REF_SCENARIOS), show_default=True,
              help="Comma-separated scenarios for the median-adjust predicted baseline.")
@click.option("--obs-year-start", default=DEFAULT_OBS_YEARS[0], show_default=True, type=int)
@click.option("--obs-year-end", default=DEFAULT_OBS_YEARS[1], show_default=True, type=int)
@click.option("--split-year", default=DEFAULT_SPLIT_YEAR, show_default=True, type=int,
              help="First future year: past pop for years < this, future pop for >=.")
@click.option("--sr31-guard/--no-sr31-guard", default=False, show_default=True,
              help="Replace a 0 super-region ratio with 0.1 x smallest non-zero ratio. "
                   "Default (--no-sr31-guard) = what shipped 2026-06-24.")
def main(pred_root, mid, a0_dir, a1_dir, blend_dir, out_dir, summary_path, storm_draw_table,
         cell, scenarios, hierarchy_path, obs_path, past_pop_path, future_pop_path,
         ref_scenarios, obs_year_start, obs_year_end, split_year, sr31_guard):
    a0_dir = a0_dir or pred_root / f"{mid}_a0"
    a1_dir = a1_dir or pred_root / f"{mid}_a1"
    blend_dir = blend_dir or pred_root / f"{mid}_blend"
    out_dir = out_dir or blend_dir
    summary_path = summary_path or blend_dir / "summary.parquet"
    scenarios = _csv(scenarios)
    ref_scenarios = _csv(ref_scenarios)
    obs_years = (obs_year_start, obs_year_end)

    hierarchy_df = pd.read_parquet(hierarchy_path)

    logger.info("(a) blend export: cell=%s scenarios=%s", cell, scenarios)
    unadj = blend_export(a0_dir, a1_dir, hierarchy_df, cell, scenarios)
    logger.info("    %d rows | storm_draws=%d locations=%d years %d-%d",
                len(unadj), unadj["storm_draw"].nunique(), unadj["location_id"].nunique(),
                unadj["year_id"].min(), unadj["year_id"].max())

    logger.info("(b) super-region median adjustment (guard=%s)", sr31_guard)
    summary = load_or_build_blend_summary(summary_path, a0_dir, a1_dir, hierarchy_df,
                                          storm_draw_table)
    amap = build_ancestor_map(hierarchy_df)
    obs = build_observed_deaths(str(obs_path), amap)
    ratios = super_region_median_ratios(summary, obs, hierarchy_df, cell,
                                        ref_scenarios, obs_years, sr31_guard=sr31_guard)
    logger.info("    ratios: %s", {int(k): round(v, 4) for k, v in ratios.dropna().items()})
    adj = apply_super_region_adjustment(unadj, ratios, hierarchy_df)

    logger.info("(c) FHS-population rate merge")
    pop = load_fhs_population(past_pop_path, future_pop_path,
                              int(unadj["year_id"].min()), int(unadj["year_id"].max()),
                              split_year)
    logger.info("    %d (loc,year) rows; %d locations; years %d-%d",
                len(pop), pop["location_id"].nunique(), pop["year_id"].min(),
                pop["year_id"].max())
    unadj = merge_population_and_rate(unadj, pop)
    adj = merge_population_and_rate(adj, pop)
    logger.info("    location pop coverage: %.0f%%",
                100 * adj.drop_duplicates("location_id")["population"].notna().mean())

    for p in write_deliverable(unadj, adj, pop, out_dir, cell):
        logger.info("wrote %s", p)


if __name__ == "__main__":
    main()
