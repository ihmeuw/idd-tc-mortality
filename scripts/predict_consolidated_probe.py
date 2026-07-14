"""
predict_consolidated_probe.py

PROBE for the new consolidated-frame prediction path. Runs ONE storm draw of
ONE model through predict + within-draw aggregation, so we can eyeball the
mechanics (seeding, coefficient draw, 16 toggles, sum->/100) before scaling to
100 storm draws + hierarchy rollup.

Seeding scheme (per storm draw N):
  - build_draw_models(n_draws=1, seed=N)  -> coefficient/scale draw seeded by N
  - DrawModel.predict(seed=N)             -> o/b flips re-seeded by N
Both deterministic => reproducible; re-seed at predict => checkpoint-resumable.

Run (project env):
    python scripts/predict_consolidated_probe.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import xarray as xr

from idd_tc_mortality.refit_with_objects import refit_model_with_objects
from idd_tc_mortality.uncertainty import build_draw_models

ROOT = Path("/mnt/team/idd/pub/idd_tc_mortality")
A0_PATH = "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4b_v2/_consolidated/storm_exposure_all.parquet"
IS_ISLAND_PATH = ROOT / "00-data" / "current" / "is_island.parquet"
OLD_SDI = "/mnt/share/forecasting/data/16/past/sdi/past_sdi_s130v66/sdi.nc"
NEW_SDI = "/mnt/share/forecasting/data/32/future/sdi/future_sdi_s130v66/sdi.nc"
STORM_DRAW_TABLE = "/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv"

DATA_PATH = ROOT / "00-data" / "current" / "input.parquet"
FOLDS_PATH = ROOT / "02-evaluate" / "20260608_final" / "fold_assignments.parquet"
FOCUS_PATH = ROOT / "03-draws" / "20260608" / "6004b730b8d8f96cb6d8443378b04f09" / "focus_model.json"

N_TC_DRAWS = 100  # fixed divisor: sum over (storm, tc_risk) then / 100
EXPOSURE_COL = "person_storm_hours"


def bulk_sdi_table() -> pd.DataFrame:
    """(location_id, year, sdi) for 1970-2100. Past file is 1980-2023, future
    2024-2100 -> just concat their native ranges (no overlap, no gap); backfill
    1970-1979 by duplicating 1980. SDI is the mean over the draw dim."""
    past = (xr.open_dataset(OLD_SDI)["draws"].mean("draw")
            .to_dataframe("sdi").reset_index().rename(columns={"year_id": "year"}))
    fut = xr.open_dataset(NEW_SDI)["draws"].mean("draw")
    if "scenario" in fut.dims:
        fut = fut.squeeze("scenario", drop=True)
    fut = fut.to_dataframe("sdi").reset_index().rename(columns={"year_id": "year"})
    cols = ["location_id", "year", "sdi"]
    base = pd.concat([past[cols], fut[cols]], ignore_index=True)
    y1980 = base[base["year"] == 1980][["location_id", "sdi"]]
    backfill = [y1980.assign(year=y) for y in range(1970, 1980)]
    return pd.concat([base] + backfill, ignore_index=True)[cols]


def prep(df: pd.DataFrame, is_island: pd.DataFrame, sdi: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(is_island, on="location_id", how="left")
    df = df.rename(columns={"max_wind_speed": "wind_speed", EXPOSURE_COL: "exposed"})
    df = df.merge(sdi, on=["location_id", "year"], how="left")
    n0 = len(df)
    df = df[(df["exposed"] > 0) & (df["total_population"] > 0)].copy()
    miss = df["sdi"].isna() | df["is_island"].isna()
    if miss.any():
        df = df[~miss].copy()
    print(f"   prep: {n0:,} -> {len(df):,} rows after exposure/cov filters")
    return df


def predict_one_storm_draw(refit_out, focus, data, slice_df, storm_draw) -> pd.DataFrame:
    """16-toggle predict for one storm draw, then sum over (storm_id, tc_risk_draw)
    and divide by N_TC_DRAWS -> expected deaths per (experiment_id, year, location)."""
    slice_df = slice_df.sort_values(
        ["storm_id", "tc_risk_draw", "location_id", "year"]
    ).reset_index(drop=True)

    out = slice_df[["experiment_id", "year", "location_id"]].copy()
    for c in (0, 1):
        for s in (0, 1):
            dm = build_draw_models(
                refit_out, focus, data,
                n_draws=1, draw_coefs=bool(c), draw_scale=bool(s), seed=storm_draw,
            )[0]
            for o in (False, True):
                for b in (False, True):
                    pred = dm.predict(slice_df, outcome_draw=o, expected_bernoulli=b, seed=storm_draw)
                    out[f"deaths_c{c}_s{s}_o{int(o)}_b{int(b)}"] = pred["deaths"].values

    death_cols = [c for c in out.columns if c.startswith("deaths_")]
    agg = out.groupby(["experiment_id", "year", "location_id"], as_index=False)[death_cols].sum()
    agg[death_cols] = agg[death_cols] / N_TC_DRAWS
    return agg


def main():
    t0 = time.time()
    focus = json.loads(FOCUS_PATH.read_text())
    data = pd.read_parquet(DATA_PATH)
    folds = pd.read_parquet(FOLDS_PATH)
    print(f"[refit] model 1 on {len(data):,} rows ...")
    refit_out = refit_model_with_objects(focus, data, folds, n_seeds=1, n_folds=2)
    print(f"        done {time.time()-t0:.1f}s")

    sd_tbl = pd.read_csv(STORM_DRAW_TABLE)
    row = sd_tbl[sd_tbl["storm_draw"] == 1].iloc[0]
    mv = f"{row['source_id']}_{row['variant_label']}"
    print(f"[slice] storm_draw=1 -> model/variant {mv!r}, experiment=historical")

    a0 = pd.read_parquet(A0_PATH)
    sl = a0[(a0["source_id_variant_label"] == mv) & (a0["experiment_id"] == "historical")].copy()
    print(f"        slice rows: {len(sl):,}  "
          f"(storms={sl['storm_id'].nunique()}, tc_risk_draws={sl['tc_risk_draw'].nunique()}, "
          f"years {sl['year'].min()}-{sl['year'].max()}, locs={sl['location_id'].nunique()})")

    print("[prep] merging is_island + sdi ...")
    sdi = bulk_sdi_table()
    isl = pd.read_parquet(IS_ISLAND_PATH)
    sl = prep(sl, isl, sdi)

    print("[predict] storm_draw=1, 16 toggles ...")
    tp = time.time()
    agg = predict_one_storm_draw(refit_out, focus, data, sl, storm_draw=1)
    print(f"          done {time.time()-tp:.1f}s -> {len(agg):,} (experiment,year,location) rows")

    mean_cell = "deaths_c0_s0_o0_b1"  # everything-at-mean cell
    print(f"\n=== aggregated deaths, mean cell {mean_cell} ===")
    by_year = agg.groupby("year")[mean_cell].sum()
    print("total expected deaths/year (first 10):")
    print(by_year.head(10).to_string())
    print(f"\ngrand total {mean_cell}: {agg[mean_cell].sum():,.1f}")
    print("\nper-toggle grand totals:")
    for col in sorted(c for c in agg.columns if c.startswith("deaths_")):
        print(f"   {col}: {agg[col].sum():,.1f}")
    print(f"\n[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
