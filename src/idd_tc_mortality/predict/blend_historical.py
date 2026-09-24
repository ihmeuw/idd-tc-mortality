"""Historical-scenario predictions on the SHIPPED A1/A0 blend, for plotting.

The shipped deliverable (``direct_risk/2026_08_20_v2000_*``, built by
``04-predict/20260806_shipped/_ship_gpd_cens_blend.py``) covers the ssp scenarios
2015-2100 only; historical was predicted just for the rake window and not written.
This module reproduces that script's prediction, blend and roll-up for the
historical scenario over any year window, then applies the SHIPPED rake ratios
(read from ``2026_08_20_v2000_model.json``) so the result is on the same footing as
the deliverable. Output: draw-level parquet(s) in the figures working slot.

Same cell as the ship: deaths_c1_s1_o0_b1, fit-draw seed = storm_draw, paired draws.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import numpy as np
import pandas as pd

from idd_tc_mortality.predict.paths import atomic_write_parquet

ROOT = Path("/mnt/team/idd/pub/idd_tc_mortality")
DR = Path("/mnt/team/rapidresponse/pub/tropical-storms/direct_risk")
FRAMES = {
    "a0": "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4b_v2/_consolidated/storm_exposure_all.parquet",
    "a1": "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4b_v2_admin1/_consolidated/storm_exposure_all.parquet",
}
STORM_DRAW_TABLE = "/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv"
IS_ISLAND_PATH = ROOT / "00-data" / "current" / "is_island.parquet"
OLD_SDI = "/mnt/share/forecasting/data/32/past/sdi/past_sdi_s130v89/sdi.nc"
SSP245_SDI = "/mnt/share/forecasting/data/32/future/sdi/future_sdi_s130v89/sdi.nc"
HIER_PATH = "/mnt/team/idd/pub/forecast-mbp/01-raw_data/gbd/fhs_2023_modeling_hierarchy.parquet"
DATA_PATH = ROOT / "00-data" / "20260722_v2000" / "input.parquet"
FOLDS_PATH = ROOT / "02-evaluate" / "20260722_v2000_final" / "fold_assignments.parquet"
MID = "e7a78b19e3468d0beb8cb2161dd6d154"
FOCUS_PATH = ROOT / "03-draws" / "20260722_v2000" / MID / "focus_model.json"
SHIP_MODEL_JSON = DR / "2026_08_20_v2000_model.json"
OUT_DIR = ROOT / "05-figures" / "working" / f"shipped_{MID[:8]}"

DEATHS_DTYPES = {"storm_draw": "int64", "ssp_scenario": "string", "year_id": "int64",
                 "location_id": "int64", "deaths": "float64"}


# --- pure helpers (tested) ----------------------------------------------------------
def subnationalized_parents(hier: pd.DataFrame, a1_location_ids) -> list[int]:
    """Country ids (4th element of path_to_top_parent) of the admin-1 units in the A1 frame."""
    sub = hier[hier["location_id"].isin(set(a1_location_ids))]
    return sorted(int(x) for x in sub["path_to_top_parent"].str.split(",").str[3].unique())


def blend_a1_a0(md: pd.DataFrame, hier: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """A1 admin-1 rows + A0 rows for countries that are NOT subnationalized (ship rule).

    ``md`` carries a ``frame`` column in {'a0', 'a1'}. Returns the blended
    most-detailed frame (``frame`` dropped) and the dropped parent ids.
    """
    a1 = md[md["frame"] == "a1"].drop(columns="frame")
    a0 = md[md["frame"] == "a0"].drop(columns="frame")
    parents = subnationalized_parents(hier, a1["location_id"].unique())
    blend = pd.concat([a1, a0[~a0["location_id"].isin(parents)]], ignore_index=True)
    return blend, parents


def rollup(blend: pd.DataFrame, hier: pd.DataFrame) -> pd.DataFrame:
    """Sum every most-detailed row into each of its ancestors (incl. itself)."""
    from idd_tc_mortality.predict.postprocess import build_ancestor_map

    amap = build_ancestor_map(hier)
    out = (blend.merge(amap, on="location_id")
           .groupby(["storm_draw", "ssp_scenario", "year_id", "ancestor"], as_index=False)["deaths"].sum()
           .rename(columns={"ancestor": "location_id"}))
    return out.astype({"storm_draw": "int64", "year_id": "int64", "location_id": "int64"})


def shipped_ratios(model_json: Path = SHIP_MODEL_JSON) -> pd.Series:
    """Super-region rake ratios recorded by the ship script, indexed by int super_region_id."""
    meta = json.loads(Path(model_json).read_text())
    return pd.Series({int(k): float(v) for k, v in meta["ratios"].items()}, name="ratio")


# --- prediction (mirrors _ship_gpd_cens_blend.run_mv) ---------------------------------
_CTX: dict | None = None


def _ctx() -> dict:
    """Per-process context: training data, folds, covariate tables, focus model, draw table."""
    global _CTX
    if _CTX is None:
        from idd_tc_mortality.predict.consolidated import bulk_sdi_table

        tbl = pd.read_csv(STORM_DRAW_TABLE)
        tbl["mv"] = tbl["source_id"].astype(str) + "_" + tbl["variant_label"].astype(str)
        _CTX = {
            "data": pd.read_parquet(DATA_PATH),
            "folds": pd.read_parquet(FOLDS_PATH),
            "is_island": pd.read_parquet(IS_ISLAND_PATH),
            # historical rows use the ssp245 SDI table, as the ship script did
            "sdi": bulk_sdi_table(OLD_SDI, SSP245_SDI),
            "focus": json.loads(FOCUS_PATH.read_text()),
            "mv_draws": tbl.groupby("mv")["storm_draw"].apply(list).to_dict(),
        }
    return _CTX


def predict_variant(mv: str, years: tuple[int, int], scenario: str = "historical") -> pd.DataFrame:
    """Most-detailed (location, year) deaths per paired draw for one climate variant, both frames."""
    from idd_tc_mortality.predict.consolidated import N_TC_DRAWS, prep_frame
    from idd_tc_mortality.refit_with_objects import refit_model_with_objects
    from idd_tc_mortality.uncertainty import build_draw_models

    c = _ctx()
    t0 = time.time()
    refit = refit_model_with_objects(focus_model=c["focus"], data=c["data"],
                                     fold_assignments=c["folds"], n_seeds=1, n_folds=2)
    bad = [s for s in ("s1", "s2", "bulk", "tail") if refit["is"][s].get("failed")]
    if bad:
        raise RuntimeError(f"refit failed stages {bad}")
    dms = {d: build_draw_models(refit, c["focus"], c["data"], n_draws=1,
                                draw_coefs=True, draw_scale=True, seed=d)[0]
           for d in c["mv_draws"][mv]}
    parts = []
    for fkey, fpath in FRAMES.items():
        sl = pd.read_parquet(fpath, filters=[("source_id_variant_label", "=", mv),
                                             ("experiment_id", "=", scenario)])
        sl = sl[sl["year"].between(*years)]
        if sl.empty:
            continue
        sl = prep_frame(sl.copy(), c["is_island"], c["sdi"])
        if sl.empty:
            continue
        if "total_population_exposed" in sl.columns and "exposed_population" not in sl.columns:
            sl = sl.rename(columns={"total_population_exposed": "exposed_population"})
        for d, dm in dms.items():
            pred = dm.predict(sl, outcome_draw=False, expected_bernoulli=True, seed=d)
            g = (pd.DataFrame({"location_id": sl["location_id"].values,
                               "year_id": sl["year"].values,
                               "deaths": pred["deaths"].values})
                 .groupby(["location_id", "year_id"], as_index=False)["deaths"].sum())
            g["deaths"] /= N_TC_DRAWS
            g["ssp_scenario"] = scenario
            g["storm_draw"] = d
            g["frame"] = fkey
            parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    click.echo(f"[mv] {mv}: {len(c['mv_draws'][mv])} draws, {len(out):,} rows, {time.time() - t0:.0f}s")
    return out


def _write(df: pd.DataFrame, path: Path) -> None:
    cols = list(DEATHS_DTYPES)
    atomic_write_parquet(df[cols].astype(DEATHS_DTYPES), path)


@click.command()
@click.option("--years", nargs=2, type=int, default=(1985, 2014), show_default=True, help="Historical year window.")
@click.option("--workers", default=4, show_default=True)
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_DIR, show_default=True)
def main(years, workers, out_dir):
    """Predict the historical scenario on the shipped A1/A0 blend and write draw-level parquets."""
    from idd_tc_mortality.predict.finalize_deliverable import apply_super_region_adjustment

    hier = pd.read_parquet(HIER_PATH)
    mvs = sorted(_ctx()["mv_draws"])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        md = pd.concat(list(pool.map(predict_variant, mvs, [tuple(years)] * len(mvs))), ignore_index=True)
    blend, parents = blend_a1_a0(md, hier)
    click.echo(f"[blend] parents dropped {parents}; most-detailed locations {blend['location_id'].nunique()}")
    unadj = rollup(blend, hier)
    ratios = shipped_ratios()
    adj = apply_super_region_adjustment(unadj, ratios, hier)
    out_dir = Path(out_dir)
    _write(unadj, out_dir / "historical_blend_unadjusted.parquet")
    _write(adj, out_dir / "historical_blend_adjusted_mean.parquet")
    meta = {"mid": MID, "scenario": "historical", "years": list(years), "frames": FRAMES,
            "blend_parents_dropped": parents, "rake_ratios_from": str(SHIP_MODEL_JSON),
            "ratios": {str(k): float(v) for k, v in ratios.items()},
            "rows": len(unadj), "locations": int(unadj["location_id"].nunique())}
    (out_dir / "historical_blend_meta.json").write_text(json.dumps(meta, indent=1))
    g = adj[adj.location_id == 1].groupby("year_id")["deaths"].mean()
    click.echo(f"[check] adjusted global historical mean: {years[0]}={g.get(years[0], np.nan):.0f} "
               f"2000={g.get(2000, np.nan):.0f} {years[1]}={g.get(years[1], np.nan):.0f}")
    click.echo(f"[wrote] {out_dir}")


if __name__ == "__main__":
    main()
