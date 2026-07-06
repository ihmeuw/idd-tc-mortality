"""
Consolidated-frame prediction pipeline (new format).

Replaces the file-based predict_tc + aggregate_* chain. Inputs are the single
consolidated CLIMADA parquets (one row per tc_risk_draw × storm × year ×
location, with source_id_variant_label / experiment_id / basin columns). For a
chosen model we:

  per storm draw N (1..100):
    - slice the frame to N's climate model (source_id_variant_label)
    - seed = N: build the (c,s) coefficient/scale draw  (build_draw_models n_draws=1)
    - seed = N: predict the (o,b) outcomes               (DrawModel.predict)
    - sum over (storm_id, tc_risk_draw), / N_TC_DRAWS     -> expected deaths
      per (experiment_id, year, location) for draw N, 16 toggle columns
  then:
    - roll the most-detailed locations up the GBD hierarchy (sum)
    - summarize across the 100 storm draws -> mean / 2.5 / 97.5 per cell

Seeding (per storm draw N): coefficients and outcome/Bernoulli flips are both
seeded by N via numpy SeedSequence, so results are perfectly reproducible and
each storm draw (hence each model/variant) gets independent coin flips. The
re-seed before predict makes the run checkpoint-resumable.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from idd_tc_mortality.predict.postprocess import build_ancestor_map
from idd_tc_mortality.uncertainty import build_draw_models

N_TC_DRAWS = 100   # fixed divisor: sum over (storm, tc_risk) then / 100
TOGGLES = [(c, s, o, b) for c in (0, 1) for s in (0, 1) for o in (0, 1) for b in (0, 1)]
DEATHS_COLS = [f"deaths_c{c}_s{s}_o{o}_b{b}" for (c, s, o, b) in TOGGLES]


# ---------------------------------------------------------------------------
# Covariate prep
# ---------------------------------------------------------------------------

def bulk_sdi_table(old_sdi_path: str, new_sdi_path: str) -> pd.DataFrame:
    """(location_id, year, sdi) for 1970-2100, mean over the draw dim.

    Past file is 1980-2023, future 2024-2100 -> concat native ranges (no gap,
    no overlap); backfill 1970-1979 by duplicating 1980.
    """
    past = (xr.open_dataset(old_sdi_path)["draws"].mean("draw")
            .to_dataframe("sdi").reset_index().rename(columns={"year_id": "year"}))
    fut = xr.open_dataset(new_sdi_path)["draws"].mean("draw")
    if "scenario" in fut.dims:
        fut = fut.squeeze("scenario", drop=True)
    fut = fut.to_dataframe("sdi").reset_index().rename(columns={"year_id": "year"})
    cols = ["location_id", "year", "sdi"]
    base = pd.concat([past[cols], fut[cols]], ignore_index=True)
    y1980 = base[base["year"] == 1980][["location_id", "sdi"]]
    backfill = [y1980.assign(year=y) for y in range(1970, 1980)]
    return pd.concat([base] + backfill, ignore_index=True)[cols]


def prep_frame(
    df: pd.DataFrame,
    is_island: pd.DataFrame,
    sdi: pd.DataFrame,
    exposure_col: str = "person_storm_hours",
) -> pd.DataFrame:
    """Merge is_island + sdi, rename to model names, drop unpredictable rows."""
    df = df.merge(is_island, on="location_id", how="left")
    df = df.rename(columns={"max_wind_speed": "wind_speed", exposure_col: "exposed"})
    df = df.merge(sdi, on=["location_id", "year"], how="left")
    df = df[(df["exposed"] > 0) & (df["total_population"] > 0)].copy()
    miss = df["sdi"].isna() | df["is_island"].isna()
    if miss.any():
        df = df[~miss].copy()
    return df


# ---------------------------------------------------------------------------
# Per-storm-draw predict + within-draw aggregation
# ---------------------------------------------------------------------------

def predict_storm_draw(refit_out, focus, data, slice_df, storm_draw):
    """16-toggle predict for one storm draw; sum over (storm_id, tc_risk_draw),
    / N_TC_DRAWS. Returns two aggregates of the same predictions:

      loc_agg:   expected deaths per (experiment_id, year, location_id)   [FHS rollup]
      basin_agg: expected deaths + exposed per (experiment_id, year, basin) [basin diagnostic]
    """
    slice_df = slice_df.sort_values(
        ["storm_id", "tc_risk_draw", "location_id", "year"]
    ).reset_index(drop=True)

    work = slice_df[["experiment_id", "year", "location_id", "basin", "exposed"]].copy()
    for c in (0, 1):
        for s in (0, 1):
            dm = build_draw_models(
                refit_out, focus, data,
                n_draws=1, draw_coefs=bool(c), draw_scale=bool(s), seed=storm_draw,
            )[0]
            for o in (False, True):
                for b in (False, True):
                    pred = dm.predict(slice_df, outcome_draw=o, expected_bernoulli=b, seed=storm_draw)
                    work[f"deaths_c{c}_s{int(s)}_o{int(o)}_b{int(b)}"] = pred["deaths"].values

    loc_agg = work.groupby(["experiment_id", "year", "location_id"], as_index=False)[DEATHS_COLS].sum()
    loc_agg[DEATHS_COLS] = loc_agg[DEATHS_COLS] / N_TC_DRAWS

    basin_agg = work.groupby(["experiment_id", "year", "basin"], as_index=False)[
        ["exposed"] + DEATHS_COLS
    ].sum()
    basin_agg[["exposed"] + DEATHS_COLS] = basin_agg[["exposed"] + DEATHS_COLS] / N_TC_DRAWS
    basin_agg = basin_agg.rename(columns={"exposed": "exposed_mean"})
    return loc_agg, basin_agg


# ---------------------------------------------------------------------------
# Hierarchy rollup + across-draw summary
# ---------------------------------------------------------------------------

def rollup_and_summarize(
    per_draw: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    draw_ids: list[int],
) -> pd.DataFrame:
    """per_draw: (storm_draw, experiment_id, year, location_id, <16 deaths>),
    most-detailed locations. Returns (experiment_id, year, location_id, cell,
    mean, lower, upper) with location rolled up the hierarchy and summary taken
    across the supplied draw_ids (missing draw cells filled with 0)."""
    ancestor_map = build_ancestor_map(hierarchy_df)
    valid = set(ancestor_map["location_id"].unique())
    missing = set(per_draw["location_id"].unique()) - valid
    if missing:
        raise ValueError(f"location_ids absent from hierarchy: {sorted(missing)[:10]}")

    # sum most-detailed -> every ancestor
    rolled = (
        per_draw.merge(ancestor_map, on="location_id")
        .drop(columns="location_id")
        .rename(columns={"ancestor": "location_id"})
        .groupby(["storm_draw", "experiment_id", "year", "location_id"], as_index=False)[DEATHS_COLS]
        .sum()
    )

    # full (experiment, year, location) grid x draws so absent draws count as 0
    keys = rolled[["experiment_id", "year", "location_id"]].drop_duplicates()
    grid = keys.merge(pd.DataFrame({"storm_draw": draw_ids}), how="cross")
    full = grid.merge(
        rolled, on=["storm_draw", "experiment_id", "year", "location_id"], how="left"
    )
    full[DEATHS_COLS] = full[DEATHS_COLS].fillna(0.0)

    out_frames = []
    gcols = ["experiment_id", "year", "location_id"]
    for cell in DEATHS_COLS:
        g = full.groupby(gcols)[cell]
        summary = pd.DataFrame({
            "mean":  g.mean(),
            "lower": g.quantile(0.025),
            "upper": g.quantile(0.975),
        }).reset_index()
        summary["cell"] = cell
        out_frames.append(summary)
    return pd.concat(out_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

SIP_BASINS = {"SI": "SIP", "SP": "SIP", "AU": "SIP"}


def rollup_basin(per_draw_basin: pd.DataFrame) -> pd.DataFrame:
    """per_draw_basin: (storm_draw, experiment_id, year, basin, exposed_mean, <16 deaths>).
    Combine SI/SP/AU into SIP, sum within basin per draw, then mean across storm draws.
    Returns (experiment_id, year, basin, exposed_mean, <16 deaths>)."""
    df = per_draw_basin.copy()
    df["basin"] = df["basin"].replace(SIP_BASINS)
    cols = ["exposed_mean"] + DEATHS_COLS
    per_draw = df.groupby(["storm_draw", "experiment_id", "year", "basin"], as_index=False)[cols].sum()
    return per_draw.groupby(["experiment_id", "year", "basin"], as_index=False)[cols].mean()


def rollup_blend(a0_dir, a1_dir, hierarchy_df: pd.DataFrame, draw_ids: list[int]):
    """Blend the two frames into one all-FHS-levels summary.

    Most-detailed selected set = A1's admin-1 units (the subnationalized countries)
    + A0's country rows for every country that was NOT subnationalized. The A0
    rows for the subnationalized parents are dropped (their country totals come
    from summing the A1 children up the hierarchy). Returns (summary, dropped).
    """
    def _load(d):
        ps = sorted((Path(d) / "partials").glob("cell_*.parquet"))
        frames = [pd.read_parquet(p) for p in ps]
        return pd.concat([f for f in frames if not f.empty], ignore_index=True)

    a0 = _load(a0_dir)
    a1 = _load(a1_dir)
    # Subnationalized A0 countries = the level-3 parents of the A1 (level-4) units.
    sub = hierarchy_df[hierarchy_df["location_id"].isin(a1["location_id"].unique())]
    subnationalized = sorted(int(x) for x in sub["path_to_top_parent"].str.split(",").str[3].unique())
    a0_keep = a0[~a0["location_id"].isin(subnationalized)]
    blend = pd.concat([a1, a0_keep], ignore_index=True)
    return rollup_and_summarize(blend, hierarchy_df, draw_ids), subnationalized


def draw_level_blend(a0_dir, a1_dir, hierarchy_df: pd.DataFrame, cell: str,
                     scenarios: list[str] | None = None) -> pd.DataFrame:
    """Draw-level (per storm_draw) FHS-rolled deaths for ONE toggle cell, blended.

    Same blend selection as ``rollup_blend`` (A1 admin-1 for the subnationalized
    countries + A0 for the rest), but rolled up the hierarchy *keeping*
    storm_draw and NOT collapsing to mean/lower/upper. Returns
    ``(storm_draw, experiment_id, year, location_id, deaths)`` for all FHS levels.
    """
    keep = ["storm_draw", "experiment_id", "year", "location_id", cell]

    def _load(d):
        ps = sorted((Path(d) / "partials").glob("cell_*.parquet"))
        frames = [pd.read_parquet(p, columns=keep) for p in ps]
        return pd.concat([f for f in frames if not f.empty], ignore_index=True)

    a0 = _load(a0_dir)
    a1 = _load(a1_dir)
    sub = hierarchy_df[hierarchy_df["location_id"].isin(a1["location_id"].unique())]
    subnationalized = {int(x) for x in sub["path_to_top_parent"].str.split(",").str[3].unique()}
    blend = pd.concat([a1, a0[~a0["location_id"].isin(subnationalized)]], ignore_index=True)
    if scenarios is not None:
        blend = blend[blend["experiment_id"].isin(scenarios)]

    amap = build_ancestor_map(hierarchy_df)
    rolled = (
        blend.merge(amap, on="location_id")
        .drop(columns="location_id")
        .rename(columns={"ancestor": "location_id"})
        .groupby(["storm_draw", "experiment_id", "year", "location_id"], as_index=False)[cell]
        .sum()
        .rename(columns={cell: "deaths"})
    )
    return rolled.astype({"storm_draw": "int32", "year": "int16", "location_id": "int32"})


def run_a0(
    *,
    refit_out,
    focus,
    data,
    consolidated_path: str,
    storm_draw_table_path: str,
    is_island_path: str,
    old_sdi_path: str,
    new_sdi_path: str,
    hierarchy_path: str,
    exposure_col: str = "person_storm_hours",
    max_storm_draws: int | None = None,
) -> pd.DataFrame:
    """Full predict + A0 rollup for one model. Returns the summarized frame."""
    sd_tbl = pd.read_csv(storm_draw_table_path)
    sd_tbl["mv"] = sd_tbl["source_id"].astype(str) + "_" + sd_tbl["variant_label"].astype(str)
    if max_storm_draws is not None:
        sd_tbl = sd_tbl[sd_tbl["storm_draw"] <= max_storm_draws].copy()
    draw_ids = sorted(sd_tbl["storm_draw"].tolist())

    is_island = pd.read_parquet(is_island_path)
    sdi = bulk_sdi_table(old_sdi_path, new_sdi_path)
    hierarchy_df = pd.read_parquet(hierarchy_path)

    per_draw_frames: list[pd.DataFrame] = []
    for mv, grp in sd_tbl.groupby("mv"):
        sds = grp["storm_draw"].tolist()
        t0 = time.time()
        sl = pd.read_parquet(
            consolidated_path,
            filters=[("source_id_variant_label", "=", mv)],
        )
        if sl.empty:
            print(f"  [skip] {mv}: no rows in frame")
            continue
        sl = prep_frame(sl, is_island, sdi, exposure_col=exposure_col)
        print(f"  [{mv}] {len(sl):,} prepped rows, storm_draws={sds} ({time.time()-t0:.1f}s load+prep)")
        for sd in sds:
            tsd = time.time()
            agg, _basin_agg = predict_storm_draw(refit_out, focus, data, sl, storm_draw=sd)
            agg.insert(0, "storm_draw", sd)
            per_draw_frames.append(agg)
            print(f"     storm_draw={sd}: {len(agg):,} rows ({time.time()-tsd:.1f}s)")

    per_draw = pd.concat(per_draw_frames, ignore_index=True)
    print(f"[rollup] {len(per_draw):,} per-draw rows across {len(draw_ids)} draws")
    return rollup_and_summarize(per_draw, hierarchy_df, draw_ids)
