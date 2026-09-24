"""All-storm global TC counts by Saffir-Simpson category from the TC-risk tracks.

Companion to storm_count_by_category.py (land-affecting storms, CLIMADA max
wind). Here every synthetic track in the TC-risk output counts, whether or not
it reaches land: per track, max of vmax_trks (m/s) over its lifetime.

Stage ``extract`` (Slurm, multiprocessing): read every
tracks_{basin}_{model}_{scenario}_{variant}_{start}_{end}[_e{k}].nc under
tc_risk/output/cmip6 for the 7 most-detailed basins (GL excluded) and the 100
draws CLIMADA uses (unsuffixed file = draw 0, _e{k} = draw k+1, k <= 98), and
write one row per track to track_max_vmax.parquet (atomic write).

Stage ``plot``: count tracks per (member, scenario, draw, year) at each
threshold, zero-fill, mean over draws per member, then the same panels as the
CLIMADA figure. Observed dots: all IBTrACS storms, per-storm max of usa_wind
(1-min sustained, kt -> m/s), 1970 onward. Thresholds: Cat 3 >= 50,
Cat 4 >= 58, Cat 5 >= 70 m/s (idd_climate_models.constants).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TC_RISK = Path("/mnt/team/rapidresponse/pub/tropical-storms/tc_risk/output/cmip6")
OUT_DIR = Path("/mnt/team/rapidresponse/pub/tropical-storms/figures/tc_risk")
TRACKS_PARQUET = OUT_DIR / "track_max_vmax.parquet"
IBTRACS_NC = Path.home() / "climada/data/IBTrACS.ALL.v04r01.nc"
BASINS = ("AU", "EP", "NA", "NI", "SI", "SP", "WP")
N_DRAWS = 100
KT_TO_MS = 0.514444
_DRAW_RE = re.compile(r"_e(\d+)\.nc$")

SCHEMA = pa.schema([
    ("model_variant", pa.string()), ("scenario", pa.string()), ("basin", pa.string()),
    ("time_bin", pa.string()), ("draw", pa.int16()), ("track", pa.int32()),
    ("year", pa.int16()), ("max_vmax", pa.float32()),
])


def list_track_files() -> list[str]:
    files = []
    for basin in BASINS:
        files += glob.glob(str(TC_RISK / "*" / "*" / "*" / "*" / basin / "tracks_*.nc"))
    keep = []
    for f in files:
        m = _DRAW_RE.search(f)
        draw = int(m.group(1)) + 1 if m else 0
        if draw < N_DRAWS:
            keep.append(f)
    return sorted(keep)


def _read_one(path: str):
    with netCDF4.Dataset(path) as d:
        v = np.ma.filled(d["vmax_trks"][:].astype("float32"), np.nan)
        y = np.asarray(d["tc_years"][:], dtype="int16")
    if v.size == 0:
        return path, y[:0], np.empty(0, dtype="float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mx = np.nanmax(v, axis=1).astype("float32")
    return path, y, mx


def _meta(path: str):
    p = Path(path)
    basin, time_bin, scenario, variant, model = p.parent.name, p.parents[1].name, p.parents[2].name, p.parents[3].name, p.parents[4].name
    m = _DRAW_RE.search(path)
    return f"{model}_{variant}", scenario, basin, time_bin, (int(m.group(1)) + 1 if m else 0)


def extract(procs: int) -> None:
    files = list_track_files()
    print(f"{len(files):,} track files; probe: 44 ms/file single, 5.6 ms/file wall at 16 procs "
          f"-> ~{len(files) * 0.0056 / 60:.0f} min at 16 procs", flush=True)
    cols = {k: [] for k in SCHEMA.names}
    t0 = time.time()
    with Pool(procs) as pool:
        for i, (path, years, mx) in enumerate(pool.imap_unordered(_read_one, files, chunksize=25), 1):
            n = len(years)
            if n:
                mv, sc, ba, tb, dr = _meta(path)
                cols["model_variant"].append(np.full(n, mv, dtype=object))
                cols["scenario"].append(np.full(n, sc, dtype=object))
                cols["basin"].append(np.full(n, ba, dtype=object))
                cols["time_bin"].append(np.full(n, tb, dtype=object))
                cols["draw"].append(np.full(n, dr, dtype="int16"))
                cols["track"].append(np.arange(n, dtype="int32"))
                cols["year"].append(years)
                cols["max_vmax"].append(mx)
            if i % 20000 == 0:
                print(f"  {i:,}/{len(files):,} files, {time.time() - t0:.0f}s", flush=True)
    table = pa.Table.from_arrays(
        [pa.array(np.concatenate(cols[k]), type=SCHEMA.field(k).type) for k in SCHEMA.names], schema=SCHEMA)
    tmp = TRACKS_PARQUET.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    md = pq.read_metadata(tmp)
    if md.num_rows != table.num_rows or pq.read_schema(tmp).names != SCHEMA.names:
        raise RuntimeError(f"parquet validation failed: {md.num_rows} rows vs {table.num_rows}")
    os.replace(tmp, TRACKS_PARQUET)
    print(f"wrote {TRACKS_PARQUET}: {table.num_rows:,} tracks from {len(files):,} files in {time.time() - t0:.0f}s")


def ibtracs_all_storm_counts(categories: dict[str, float], year0: int = 1970) -> pd.DataFrame:
    import xarray as xr
    ds = xr.open_dataset(IBTRACS_NC, decode_times=False)
    wind = ds["usa_wind"].values.astype("float32") * KT_TO_MS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mx = np.nanmax(wind, axis=1)
    season = ds["season"].values.astype(int)
    keep = season >= year0
    years = np.arange(year0, season[keep].max() + 1)
    out = []
    for cat, thr in categories.items():
        sel = keep & (mx >= thr)
        cnt = pd.Series(season[sel]).value_counts().reindex(years, fill_value=0)
        out.append(pd.DataFrame({"year": years, "count": cnt.values, "category": cat}))
    print(f"  IBTrACS: {keep.sum()} storms {year0}-{years[-1]}, usa_wind available for "
          f"{np.isfinite(mx[keep]).sum()}", flush=True)
    return pd.concat(out, ignore_index=True)


def member_counts(categories: dict[str, float], year_ranges: dict[str, tuple[int, int]]) -> pd.DataFrame:
    df = pd.read_parquet(TRACKS_PARQUET, columns=["model_variant", "scenario", "draw", "year", "max_vmax"])
    out = []
    for scen, (y0, y1) in year_ranges.items():
        s = df[(df["scenario"] == scen) & (df["year"] >= y0) & (df["year"] <= y1)]
        members = s[["model_variant", "draw"]].drop_duplicates()
        full = pd.MultiIndex.from_frame(members.merge(pd.DataFrame({"year": np.arange(y0, y1 + 1)}), how="cross"))
        for cat, thr in categories.items():
            cnt = (s[s["max_vmax"] >= thr].groupby(["model_variant", "draw", "year"]).size()
                   .reindex(full, fill_value=0).rename("count").reset_index())
            mem = cnt.groupby(["model_variant", "year"])["count"].mean().reset_index()
            mem = mem.rename(columns={"model_variant": "source_id_variant_label"})
            mem["experiment"], mem["category"] = scen, cat
            out.append(mem)
        print(f"  {scen}: {len(s):,} tracks, {members['model_variant'].nunique()} members x "
              f"{members['draw'].nunique()} draws", flush=True)
    return pd.concat(out, ignore_index=True)


def plot_stage() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import storm_count_by_category as climada_fig  # shared panels, thresholds, year ranges
    model_df = member_counts(climada_fig.CATEGORIES, climada_fig.YEAR_RANGES)
    obs_df = ibtracs_all_storm_counts(climada_fig.CATEGORIES)
    fig = climada_fig.plot(model_df, obs_df)
    for ax, cat in zip(fig.axes, climada_fig.CATEGORIES):
        ax.set_title(f"{cat} (TC-risk track max vmax >= {climada_fig.CATEGORIES[cat]:.0f} m/s)", fontsize=13, fontweight="bold")
    fig.suptitle("All synthetic TCs (TC-risk tracks) by scenario and Saffir-Simpson category\n"
                 "(members: 8 model/variant realizations, each the mean over 100 draws; band = 10th-90th "
                 "percentile across members; dots: all IBTrACS storms, usa_wind)", fontsize=13)
    png = OUT_DIR / "storm_count_by_category.png"
    fig.savefig(png, dpi=150)
    print(f"Saved {png}")
    print("\nMean annual count per member (model, all years):\n",
          model_df.groupby(["category", "experiment"])["count"].mean().unstack("experiment").round(2).to_string())
    print("\nIBTrACS mean annual count 1970-2014:\n",
          obs_df[obs_df["year"] <= 2014].groupby("category")["count"].mean().round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["extract", "plot"], required=True)
    ap.add_argument("--procs", type=int, default=32, help="worker processes for extract")
    a = ap.parse_args()
    extract(a.procs) if a.stage == "extract" else plot_stage()
