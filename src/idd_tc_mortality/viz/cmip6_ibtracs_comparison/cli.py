"""``run-cmip6-ibtracs-comparison``: aggregate → predict → stats → export (+ sanity)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd

from idd_tc_mortality.viz.cmip6_ibtracs_comparison.constants import (
    MORTALITY_YEARS,
    N_DRAWS,
    RAKE_MODEL_JSON,
    RUNS,
    STORM_YEARS,
    WORK_DIR,
    load_hierarchy,
    outcome_counts,
)

_WORK = click.option("--work-dir", type=click.Path(path_type=Path), default=WORK_DIR, show_default=True)


@click.group()
def main():
    """CMIP6 (tc-risk / CLIMADA) vs IBTrACS comparability report."""


@main.command()
@click.option("--part", type=click.Choice(["tracks", "climada", "ibtracs", "observed"]), required=True)
@_WORK
def aggregate(part, work_dir):
    """Storm counts by category and geography (model and observed) or observed deaths."""
    from idd_tc_mortality.viz.cmip6_ibtracs_comparison import mortality, storms

    if part == "observed":
        click.echo(f"[aggregate:observed] {mortality.run_observed(work_dir)}")
    else:
        storms.run_aggregate(part, work_dir, load_hierarchy(), STORM_YEARS)


@main.command()
@click.option("--mv", type=click.Choice(list(RUNS)), default=None, help="One CMIP6 run.")
@click.option("--ibtracs", is_flag=True, help="Predict on the IBTrACS storm record instead.")
@click.option("--n-fit-draws", type=int, default=None, help="Probe: use only the first N fit draws.")
@click.option("--rake-json", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=RAKE_MODEL_JSON,
              show_default=True, help="model.json holding the super-region rake ratios.")
@_WORK
def predict(mv, ibtracs, n_fit_draws, rake_json, work_dir):
    """Shipped mortality model on one CMIP6 run or on IBTrACS storms (1985-2023)."""
    from idd_tc_mortality.viz.cmip6_ibtracs_comparison import mortality

    if ibtracs == (mv is not None):
        raise click.UsageError("give exactly one of --mv / --ibtracs")
    t0 = time.time()
    path = (mortality.run_predict_ibtracs(work_dir, MORTALITY_YEARS, n_fit_draws, rake_json) if ibtracs
            else mortality.run_predict_mv(mv, work_dir, MORTALITY_YEARS, n_fit_draws, rake_json))
    click.echo(f"[predict] wrote {path} in {time.time() - t0:.0f}s")


@main.command()
@click.option("--storm-only", is_flag=True, help="Skip the mortality cells (storm half only).")
@_WORK
def stats(storm_only, work_dir):
    """Rank-based match measures for every cell; writes stats/ parquet tables."""
    from idd_tc_mortality.viz.cmip6_ibtracs_comparison.stats import run_stats

    run_stats(work_dir, storm_only=storm_only)


@main.command()
@click.option("--out-dir", type=click.Path(path_type=Path), default=None,
              help="Artifact bundle directory [default: <work-dir>/artifact].")
@_WORK
def export(out_dir, work_dir):
    """JSON data files, overview, simplified geometry and the page."""
    from idd_tc_mortality.viz.cmip6_ibtracs_comparison.export import run_export

    run_export(work_dir, out_dir or Path(work_dir) / "artifact")


@main.command()
@_WORK
def sanity(work_dir):
    """Compare global Cat3+/4+/5 annual means with the 20260924 scripts' methods.

    Recomputes those scripts' quantities on the same inputs (CLIMADA: storms keyed
    WITHOUT basin, as that script did; tc-risk: all tracks; IBTrACS: usa_wind only,
    keyed by season) next to this pipeline's counts, historical years 1970-2014.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as pads

    from idd_tc_mortality.viz.cmip6_ibtracs_comparison.constants import (
        IBTRACS_NC,
        KT_TO_MS,
        MODEL_FRAMES,
        TRACKS_PARQUET,
    )

    work_dir = Path(work_dir)
    cats = {"cat3plus": 50.0, "cat4plus": 58.0, "cat5": 70.0}
    y0, y1 = 1970, 2014
    out = {"years": [y0, y1], "climada": {}, "tracks": {}, "ibtracs_land": {}, "ibtracs_all": {}}

    def member_mean(counts: pd.DataFrame) -> dict:
        """Mean annual count per member (mean over draws, then over years and members)."""
        c = outcome_counts(counts[(counts["level"] == "global") & counts["year"].between(y0, y1)])
        full = pd.MultiIndex.from_product([list(RUNS), range(N_DRAWS), range(y0, y1 + 1)], names=["run", "draw", "year"])
        res = {}
        for k in cats:
            s = c.set_index(["run", "draw", "year"])[k].reindex(full, fill_value=0)
            res[k] = float(s.groupby(["run", "year"]).mean().mean())
        return res

    out["climada"]["pipeline"] = member_mean(pd.read_parquet(work_dir / "counts_model_land.parquet"))
    out["tracks"]["pipeline"] = member_mean(pd.read_parquet(work_dir / "counts_model_all.parquet"))

    tb = pads.dataset(str(MODEL_FRAMES["a0"])).to_table(
        columns=["source_id_variant_label", "tc_risk_draw", "year", "storm_id", "max_wind_speed"],
        filter=(pc.field("experiment_id") == "historical") & (pc.field("year") >= y0) & (pc.field("year") <= y1))
    g = (tb.group_by(["source_id_variant_label", "tc_risk_draw", "year", "storm_id"])
           .aggregate([("max_wind_speed", "max")]).to_pandas())
    full = pd.MultiIndex.from_product([list(RUNS), range(N_DRAWS), range(y0, y1 + 1)])
    out["climada"]["script_20260924_no_basin_key"] = {
        k: float(g[g["max_wind_speed_max"] >= thr].groupby(["source_id_variant_label", "tc_risk_draw", "year"]).size()
                 .reindex(full, fill_value=0).groupby(level=[0, 2]).mean().mean()) for k, thr in cats.items()}

    tr = pd.read_parquet(TRACKS_PARQUET, columns=["model_variant", "scenario", "draw", "year", "max_vmax"])
    tr = tr[(tr["scenario"] == "historical") & tr["year"].between(y0, y1)]
    out["tracks"]["script_20260924"] = {
        k: float(tr[tr["max_vmax"] >= thr].groupby(["model_variant", "draw", "year"]).size()
                 .reindex(full, fill_value=0).groupby(level=[0, 2]).mean().mean()) for k, thr in cats.items()}

    obs_land = outcome_counts(pd.read_parquet(work_dir / "counts_obs_land.parquet"))
    ol = obs_land[(obs_land["level"] == "global")].set_index("year").reindex(range(y0, y1 + 1), fill_value=0)
    out["ibtracs_land"]["pipeline"] = {k: float(ol[k].mean()) for k in cats}
    obs_all = outcome_counts(pd.read_parquet(work_dir / "counts_obs_all.parquet"))
    oa = obs_all[(obs_all["level"] == "global")].set_index("year").reindex(range(y0, y1 + 1), fill_value=0)
    out["ibtracs_all"]["pipeline_sid_year_with_wmo_fallback"] = {k: float(oa[k].mean()) for k in cats}
    import warnings

    import xarray as xr
    ds = xr.open_dataset(IBTRACS_NC, decode_times=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mx = np.nanmax(ds["usa_wind"].values.astype("float32"), axis=1) * KT_TO_MS
    season = ds["season"].values.astype(int)
    keep = (season >= y0) & (season <= y1)
    out["ibtracs_all"]["script_20260924_season_usa_only"] = {
        k: float(pd.Series(season[keep & (mx >= thr)]).value_counts().reindex(range(y0, y1 + 1), fill_value=0).mean())
        for k, thr in cats.items()}
    (work_dir / "sanity.json").write_text(json.dumps(out, indent=1))
    click.echo(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
