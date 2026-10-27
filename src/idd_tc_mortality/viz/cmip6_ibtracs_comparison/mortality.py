"""Shipped mortality model (gpd_cens e7a78b19) predicted on CMIP6 and on IBTrACS storms.

Mirrors :mod:`idd_tc_mortality.predict.blend_historical` (same refit, draw models, A1/A0
blend, shipped super-region rake ratios) with two differences that matter for a
hazard-vs-model comparison:

* every run is predicted with the SAME 100 fit draws (the ship pairs fit seed =
  storm_draw, which gives MRI r1 a single draw); deaths are averaged over the fit draws,
  so the level matches the ship's draw-mean basis and run-to-run differences are hazard;
* the tc-risk draw dimension is kept (not averaged), so the 800 run × draw annual
  values are available for the "observed percentile" measure.

IBTrACS storms are predicted with the same 100 fit draws, kept per fit draw (that band
is mortality-model uncertainty alone). Basin is a model covariate; IBTrACS storms whose
genesis is in no tc-risk box are recoded to ``'NA'`` for prediction, as the validate step
did for the training data.
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pads

from idd_tc_mortality.predict import blend_historical as bh
from idd_tc_mortality.predict.consolidated import prep_frame
from idd_tc_mortality.predict.postprocess import build_ancestor_map
from idd_tc_mortality.viz.cmip6_ibtracs_comparison.constants import (
    IBTRACS_LAND,
    MODEL_FRAMES,
    MORTALITY_YEARS,
    NO_BASIN,
    OBS_DEATHS,
    RAKE_MODEL_JSON,
)
from idd_tc_mortality.viz.cmip6_ibtracs_comparison.io import write_parquet_checked
from idd_tc_mortality.viz.cmip6_ibtracs_comparison.storms import read_ibtracs_land, splice_filter

KEYS = ["draw", "basin", "location_id", "year"]
DEATHS_DTYPES = {"draw": "int16", "basin": "string", "location_id": "int64", "year": "int16",
                 "deaths_unadj": "float64", "deaths_adj": "float64"}


# --- pure helpers -----------------------------------------------------------------------
def predict_deaths(sl: pd.DataFrame, models: dict, keys: list[str], average: bool = True) -> pd.DataFrame:
    """Expected deaths per ``keys`` under each fit-draw model.

    ``average=True`` returns the mean over fit draws (columns ``keys + ['deaths']``);
    ``False`` keeps one row per fit draw (``['fit_draw'] + keys + ['deaths']``).
    """
    grp = sl.groupby(keys, sort=True, observed=True)
    codes = grp.ngroup().to_numpy()
    base = grp.size().reset_index()[keys]
    n = len(base)
    acc = np.zeros(n)
    parts = []
    for seed, dm in models.items():
        pred = dm.predict(sl, outcome_draw=False, expected_bernoulli=True, seed=seed)
        d = np.bincount(codes, weights=np.asarray(pred["deaths"], dtype=float), minlength=n)
        if average:
            acc += d
        else:
            parts.append(base.assign(fit_draw=seed, deaths=d))
    if average:
        return base.assign(deaths=acc / len(models))
    out = pd.concat(parts, ignore_index=True)
    return out[["fit_draw"] + keys + ["deaths"]]


def blend_rollup(parts: dict[str, pd.DataFrame], hier: pd.DataFrame, ratios: pd.Series,
                 keys: list[str] = KEYS) -> pd.DataFrame:
    """A1/A0 blend → rake at most-detailed level → sum into every ancestor.

    ``parts`` maps frame key (``'a0'`` / ``'a1'``) to a frame with ``keys + ['deaths']``
    where ``location_id`` is the most-detailed unit. Scaling most-detailed rows by their
    super-region's ratio and rolling up is identical to the ship's rake (regions and
    super-regions scale by the same ratio; Global becomes the sum of raked super-regions).
    Returns ``keys + ['deaths_unadj', 'deaths_adj']`` with ``location_id`` = every ancestor.
    """
    md = pd.concat([df.assign(frame=k) for k, df in parts.items()], ignore_index=True)
    blend, parents = bh.blend_a1_a0(md, hier)
    sr = hier.set_index("location_id")["super_region_id"]
    blend["deaths_adj"] = blend["deaths"] * blend["location_id"].map(sr).map(ratios).fillna(1.0)
    amap = build_ancestor_map(hier)
    other = [k for k in keys if k != "location_id"]
    out = (blend.merge(amap, on="location_id")
                .groupby(other + ["ancestor"], as_index=False)[["deaths", "deaths_adj"]].sum()
                .rename(columns={"ancestor": "location_id", "deaths": "deaths_unadj"}))
    click.echo(f"[blend] parents dropped {parents}; most-detailed units {blend['location_id'].nunique()}")
    return out[keys + ["deaths_unadj", "deaths_adj"]]


def observed_deaths(hier: pd.DataFrame, path: Path = OBS_DEATHS) -> pd.DataFrame:
    """Observed deaths per (basin, location_id, year) summed into every ancestor."""
    d = pd.read_parquet(path, columns=["basin", "location_id", "year", "deaths"])
    g = d.groupby(["basin", "location_id", "year"], as_index=False)["deaths"].sum()
    amap = build_ancestor_map(hier)
    out = (g.merge(amap, on="location_id")
            .groupby(["basin", "ancestor", "year"], as_index=False)["deaths"].sum()
            .rename(columns={"ancestor": "location_id"}))
    return out.astype({"basin": "string", "location_id": "int64", "year": "int16", "deaths": "float64"})


# --- frames -----------------------------------------------------------------------------
def model_frame(mv: str, frame_key: str, years: tuple[int, int]) -> pd.DataFrame:
    """One member's storm × unit rows over the spliced historical / ssp245 years."""
    flt = (pc.field("source_id_variant_label") == mv) & splice_filter("experiment_id", "year", years)
    tb = pads.dataset(str(MODEL_FRAMES[frame_key])).to_table(filter=flt)
    df = tb.to_pandas().rename(columns={"tc_risk_draw": "draw"})
    if df.empty:
        raise RuntimeError(f"no rows for {mv} in {MODEL_FRAMES[frame_key]} over {years}")
    return df


def ibtracs_frame(frame_key: str, years: tuple[int, int], storms: pd.DataFrame) -> pd.DataFrame:
    """IBTrACS CLIMADA rows with the tc-risk basin covariate (no-box storms → 'NA')."""
    df = read_ibtracs_land(IBTRACS_LAND[frame_key], years)
    basin = storms.set_index("storm_id")["tc_basin"]
    df["basin"] = df["storm_id"].map(basin)
    if df["basin"].isna().any():
        raise RuntimeError("IBTrACS CLIMADA storms absent from the storm table")
    df.loc[df["basin"] == NO_BASIN, "basin"] = "NA"
    df["draw"] = 0
    return df


def prepare(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """prep_frame (island, SDI, exposed>0) + the exposed_population rename the ship applies."""
    sl = prep_frame(df.copy(), ctx["is_island"], ctx["sdi"])
    if "total_population_exposed" in sl.columns and "exposed_population" not in sl.columns:
        sl = sl.rename(columns={"total_population_exposed": "exposed_population"})
    if sl.empty:
        raise RuntimeError("prep_frame dropped every row")
    return sl


def build_models(ctx: dict, seeds: list[int]) -> dict:
    """Refit the focus model once and build one DrawModel per fit seed (as the ship does)."""
    from idd_tc_mortality.refit_with_objects import refit_model_with_objects
    from idd_tc_mortality.uncertainty import build_draw_models

    t0 = time.time()
    refit = refit_model_with_objects(focus_model=ctx["focus"], data=ctx["data"],
                                     fold_assignments=ctx["folds"], n_seeds=1, n_folds=2)
    bad = [s for s in ("s1", "s2", "bulk", "tail") if refit["is"][s].get("failed")]
    if bad:
        raise RuntimeError(f"refit failed stages {bad}")
    models = {d: build_draw_models(refit, ctx["focus"], ctx["data"], n_draws=1,
                                   draw_coefs=True, draw_scale=True, seed=d)[0] for d in seeds}
    click.echo(f"[models] refit + {len(models)} draw models in {time.time() - t0:.0f}s")
    return models


def fit_seeds(ctx: dict) -> list[int]:
    """The ship's 100 fit seeds (storm_draw ids), used for every run here."""
    return sorted({d for ds in ctx["mv_draws"].values() for d in ds})


# --- stage entry points -----------------------------------------------------------------
def run_predict_mv(mv: str, work_dir: Path, years: tuple[int, int] = MORTALITY_YEARS,
                   n_fit_draws: int | None = None, rake_json: Path = RAKE_MODEL_JSON) -> Path:
    ctx = bh._ctx()
    ratios = bh.shipped_ratios(Path(rake_json))   # raises if the staged / shipped model.json is absent
    seeds = fit_seeds(ctx)[:n_fit_draws] if n_fit_draws else fit_seeds(ctx)
    models = build_models(ctx, seeds)
    hier = pd.read_parquet(bh.HIER_PATH)
    parts = {}
    for fk in ("a0", "a1"):
        t0 = time.time()
        sl = prepare(model_frame(mv, fk, years), ctx)
        parts[fk] = predict_deaths(sl, models, KEYS, average=True)
        click.echo(f"[predict {mv} {fk}] {len(sl):,} rows × {len(models)} fit draws → "
                   f"{len(parts[fk]):,} rows, {time.time() - t0:.0f}s")
    out = blend_rollup(parts, hier, ratios)
    suffix = f"_probe{len(seeds)}" if n_fit_draws else ""
    path = Path(work_dir) / "mortality" / f"model_{mv}{suffix}.parquet"
    return write_parquet_checked(out, path, DEATHS_DTYPES)


def run_predict_ibtracs(work_dir: Path, years: tuple[int, int] = MORTALITY_YEARS,
                        n_fit_draws: int | None = None, rake_json: Path = RAKE_MODEL_JSON) -> Path:
    ctx = bh._ctx()
    ratios = bh.shipped_ratios(Path(rake_json))
    seeds = fit_seeds(ctx)[:n_fit_draws] if n_fit_draws else fit_seeds(ctx)
    models = build_models(ctx, seeds)
    hier = pd.read_parquet(bh.HIER_PATH)
    storms = pd.read_parquet(Path(work_dir) / "ibtracs_storms.parquet", columns=["storm_id", "tc_basin"])
    parts = {}
    for fk in ("a0", "a1"):
        t0 = time.time()
        sl = prepare(ibtracs_frame(fk, years, storms), ctx)
        p = predict_deaths(sl, models, ["basin", "location_id", "year"], average=False)
        parts[fk] = p.rename(columns={"fit_draw": "draw"})
        click.echo(f"[predict ibtracs {fk}] {len(sl):,} rows × {len(models)} fit draws → "
                   f"{len(parts[fk]):,} rows, {time.time() - t0:.0f}s")
    out = blend_rollup(parts, hier, ratios)
    suffix = f"_probe{len(seeds)}" if n_fit_draws else ""
    path = Path(work_dir) / "mortality" / f"ibtracs_pred{suffix}.parquet"
    return write_parquet_checked(out, path, DEATHS_DTYPES)


def run_observed(work_dir: Path) -> Path:
    hier = pd.read_parquet(bh.HIER_PATH)
    return write_parquet_checked(observed_deaths(hier), Path(work_dir) / "mortality" / "observed.parquet")
