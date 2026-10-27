"""Storm counts by Saffir-Simpson category and geography — model (CMIP6) and observed (IBTrACS).

Two storm sources, both on the same 18 m/s tropical-storm floor:

* ``all``  — every track. Model: tc-risk lifetime ``max_vmax`` per track
  (``track_max_vmax.parquet``). Observed: IBTrACS per-storm lifetime max of ``usa_wind``
  (1-min, kt → m/s); where ``usa_wind`` is absent, ``wmo_wind`` / 0.88 (10-min → 1-min).
  Levels: global and basin.
* ``land`` — land-affecting storms: CLIMADA ``max_wind_speed`` over exposed admin0 units
  (model consolidated frame; IBTrACS CLIMADA storm files). Levels: country (that unit's
  max wind), region / super-region / global / basin (each storm once, at its worst
  category over the units in the aggregate).

Model storms are keyed by (run, draw, year, basin, storm_id): ``storm_id`` in the CLIMADA
frame is unique only within a basin. IBTrACS storms are assigned a tc-risk basin by their
genesis point (``constants.assign_tcrisk_basin``) and keyed by the genesis calendar year
(the sid prefix, which is also the year the CLIMADA IBTrACS files use).

Outputs are sparse (zero rows omitted); ``stats`` densifies over the full
run × draw × year × geography grid.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pads

from idd_tc_mortality.viz.cmip6_ibtracs_comparison.constants import (
    CAT_COLS,
    GLOBAL_ID,
    IBTRACS_LAND,
    IBTRACS_NC,
    KT_TO_MS,
    MODEL_FRAMES,
    N_DRAWS,
    NO_BASIN,
    RUNS,
    SPLICE,
    STORM_YEARS,
    TEN_MIN_TO_ONE_MIN,
    TRACKS_PARQUET,
    assign_tcrisk_basin,
    categorize,
    wrap_lon,
)
from idd_tc_mortality.viz.cmip6_ibtracs_comparison.io import write_parquet_checked

COUNT_KEYS = ["year", "level", "geo"]
COUNT_DTYPES = {"year": "int16", "level": "string", "geo": "string", **{c: "int16" for c in CAT_COLS}}
MODEL_COUNT_DTYPES = {"run": "string", "draw": "int16", **COUNT_DTYPES}


# --- pure helpers -----------------------------------------------------------------------
def splice_filter(scenario_field: str, year_field: str, years: tuple[int, int]):
    """pyarrow filter: historical years ≤ 2014, ssp245 after, clipped to ``years``."""
    y0, y1 = years
    parts = []
    for scen, (a, b) in SPLICE.items():
        lo, hi = max(a, y0), min(b, y1)
        if lo <= hi:
            parts.append((pc.field(scenario_field) == scen)
                         & (pc.field(year_field) >= lo) & (pc.field(year_field) <= hi))
    if not parts:
        raise ValueError(f"no splice years inside {years}")
    flt = parts[0]
    for p in parts[1:]:
        flt = flt | p
    return flt


def counts_by_category(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Rows per unique ``keys`` with the number of storms in each category code 1..6.

    ``df`` carries a ``cat`` column from :func:`categorize`; code 0 (below 18 m/s) is
    dropped. Key combinations with no storm at or above the floor are absent.
    """
    sub = df[df["cat"] > 0]
    if sub.empty:
        return pd.DataFrame(columns=list(keys) + CAT_COLS)
    tab = sub.groupby(list(keys) + ["cat"], observed=True).size().unstack("cat", fill_value=0)
    tab = tab.reindex(columns=range(1, 7), fill_value=0)
    tab.columns = CAT_COLS
    return tab.reset_index()


def storm_counts_by_level(rows: pd.DataFrame, hier: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per (keys, year, level, geo) category counts from storm × admin0 rows.

    ``rows`` needs ``year, basin, storm_id, location_id, max_wind_speed`` plus ``keys``.
    A storm is one (keys, year, basin, storm_id). Country counts use the unit's own max
    wind; region / super-region / global / basin aggregates count each storm once at its
    worst category over the units inside the aggregate. Storms with basin ``NO_BASIN``
    count everywhere except at the basin level.
    """
    loc = hier.set_index("location_id")
    level = rows["location_id"].map(loc["level"])
    if level.isna().any() or (level != 3).any():
        bad = rows.loc[level.isna() | (level != 3), "location_id"].unique()[:10]
        raise ValueError(f"expected admin0 (level 3) location ids only; offenders {bad}")
    rows = rows.assign(region=rows["location_id"].map(loc["region_id"]).astype(int),
                       super_region=rows["location_id"].map(loc["super_region_id"]).astype(int))
    storm_key = list(keys) + ["year", "basin", "storm_id"]
    specs = [("country", "location_id"), ("region", "region"), ("super_region", "super_region"),
             ("basin", "basin"), ("global", None)]
    out = []
    for lvl, unit in specs:
        gkeys = storm_key + ([unit] if unit and unit not in storm_key else [])
        g = rows.groupby(gkeys, observed=True)["max_wind_speed"].max().reset_index()
        g["cat"] = categorize(g["max_wind_speed"])
        ckeys = list(keys) + ["year"] + ([unit] if unit else [])
        c = counts_by_category(g, ckeys)
        c["level"] = lvl
        if unit == "basin":
            c["geo"] = c["basin"].astype(str)
            c = c[c["geo"] != NO_BASIN].drop(columns="basin")
        elif unit:
            c["geo"] = c[unit].astype(int).astype(str)
            c = c.drop(columns=unit)
        else:
            c["geo"] = str(GLOBAL_ID)
        out.append(c)
    res = pd.concat(out, ignore_index=True)[list(keys) + COUNT_KEYS + CAT_COLS]
    for c in CAT_COLS:
        res[c] = res[c].astype("int16")
    return res


def check_member_grid(counts: pd.DataFrame) -> None:
    """Every run present with all 100 draws, else raise."""
    runs = set(counts["run"].astype(str).unique())
    if runs != set(RUNS):
        raise RuntimeError(f"runs {sorted(runs)} != expected {sorted(RUNS)}")
    per_run = counts.groupby("run", observed=True)["draw"].nunique()
    if (per_run != N_DRAWS).any():
        raise RuntimeError(f"draws per run != {N_DRAWS}: {per_run.to_dict()}")


# --- model side -------------------------------------------------------------------------
def track_counts(years: tuple[int, int] = STORM_YEARS, path: Path = TRACKS_PARQUET) -> pd.DataFrame:
    """All-track counts per (run, draw, year) at basin and global level (sparse)."""
    t0 = time.time()
    tb = pads.dataset(str(path)).to_table(
        columns=["model_variant", "scenario", "basin", "draw", "year", "max_vmax"],
        filter=splice_filter("scenario", "year", years))
    df = tb.to_pandas(categories=["model_variant", "scenario", "basin"]).rename(columns={"model_variant": "run"})
    df["cat"] = categorize(df["max_vmax"])
    b = counts_by_category(df, ["run", "draw", "year", "basin"])
    b["level"] = "basin"
    b["geo"] = b["basin"].astype(str)
    g = counts_by_category(df, ["run", "draw", "year"])
    g["level"] = "global"
    g["geo"] = str(GLOBAL_ID)
    out = pd.concat([b.drop(columns="basin"), g], ignore_index=True)[["run", "draw"] + COUNT_KEYS + CAT_COLS]
    out["run"] = out["run"].astype(str)
    check_member_grid(out)
    click.echo(f"[tracks] {len(df):,} tracks {years[0]}-{years[1]} → {len(out):,} count rows, {time.time() - t0:.0f}s")
    return out


def model_land_counts(hier: pd.DataFrame, years: tuple[int, int] = STORM_YEARS,
                      path: Path = MODEL_FRAMES["a0"]) -> pd.DataFrame:
    """Land-affecting counts per (run, draw, year, level, geo) from the CLIMADA admin0 frame."""
    t0 = time.time()
    tb = pads.dataset(str(path)).to_table(
        columns=["source_id_variant_label", "tc_risk_draw", "year", "storm_id", "location_id",
                 "max_wind_speed", "basin"],
        filter=splice_filter("experiment_id", "year", years))
    df = (tb.to_pandas(categories=["source_id_variant_label", "basin"])
            .rename(columns={"source_id_variant_label": "run", "tc_risk_draw": "draw"}))
    click.echo(f"[climada] {len(df):,} storm×admin0 rows read, {time.time() - t0:.0f}s")
    out = storm_counts_by_level(df, hier, keys=["run", "draw"])
    out["run"] = out["run"].astype(str)
    check_member_grid(out)
    click.echo(f"[climada] {len(out):,} count rows, {time.time() - t0:.0f}s")
    return out


# --- observed side ----------------------------------------------------------------------
def _lifetime_max_ms(ds, name: str) -> np.ndarray:
    w = ds[name].values.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(w, axis=1) * KT_TO_MS


def ibtracs_storm_table(nc_path: Path = IBTRACS_NC) -> pd.DataFrame:
    """One row per IBTrACS storm: year, genesis point, tc-risk basin, lifetime max winds.

    ``wind_1min`` is ``usa_wind`` where present, else ``wmo_wind / 0.88``; ``wind_source``
    records which. ``year`` is the genesis calendar year (sid prefix).
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path, decode_times=False)
    sid = ds["sid"].values.astype(str)
    lat = ds["lat"].values.astype(float)
    lon = ds["lon"].values.astype(float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    has = ok.any(axis=1)
    first = ok.argmax(axis=1)
    idx = np.arange(len(sid))
    glat = np.where(has, lat[idx, first], np.nan)
    glon = np.where(has, lon[idx, first], np.nan)
    usa = _lifetime_max_ms(ds, "usa_wind")
    wmo = _lifetime_max_ms(ds, "wmo_wind")
    wind = np.where(np.isfinite(usa), usa, wmo * TEN_MIN_TO_ONE_MIN)
    source = np.where(np.isfinite(usa), "usa", np.where(np.isfinite(wmo), "wmo_10min_adj", "none"))
    basin = ds["basin"].values.astype(str)
    df = pd.DataFrame({
        "storm_id": sid,
        "year": np.array([int(s[:4]) for s in sid], dtype=np.int16),
        "season": ds["season"].values.astype(np.int16),
        "track_type": ds["track_type"].values.astype(str),
        "ib_basin": basin[idx, first],
        "tc_basin": assign_tcrisk_basin(glon, glat),
        "genesis_lon": wrap_lon(glon),
        "genesis_lat": glat,
        "usa_max": usa, "wmo_max": wmo, "wind_1min": wind, "wind_source": source,
    })
    return df


def ibtracs_wind_coverage(storms: pd.DataFrame, years: tuple[int, int] = STORM_YEARS) -> pd.DataFrame:
    """Storms per (IBTrACS basin, decade) by intensity source — for the methods tab."""
    s = storms[~storms["track_type"].str.startswith("spur") & storms["year"].between(*years)]
    s = s.assign(decade=(s["year"] // 10 * 10).astype(int))
    tab = (s.groupby(["ib_basin", "decade", "wind_source"]).size().unstack("wind_source", fill_value=0)
             .reindex(columns=["usa", "wmo_10min_adj", "none"], fill_value=0).reset_index())
    tab["n"] = tab[["usa", "wmo_10min_adj", "none"]].sum(axis=1)
    return tab


def ibtracs_all_counts(storms: pd.DataFrame, years: tuple[int, int] = STORM_YEARS) -> pd.DataFrame:
    """Observed all-storm counts per (year, level, geo); spur tracks excluded."""
    s = storms[~storms["track_type"].str.startswith("spur") & storms["year"].between(*years)].copy()
    s["cat"] = categorize(s["wind_1min"])
    b = counts_by_category(s, ["year", "tc_basin"])
    b["level"] = "basin"
    b["geo"] = b["tc_basin"].astype(str)
    b = b[b["geo"] != NO_BASIN].drop(columns="tc_basin")
    g = counts_by_category(s, ["year"])
    g["level"] = "global"
    g["geo"] = str(GLOBAL_ID)
    out = pd.concat([b, g], ignore_index=True)[COUNT_KEYS + CAT_COLS]
    click.echo(f"[ibtracs all] {len(s):,} storms {years[0]}-{years[1]}, "
               f"{(s['cat'] > 0).sum():,} at ≥18 m/s, {(s['wind_source'] == 'none').sum():,} without any wind")
    return out


def read_ibtracs_land(root: Path, years: tuple[int, int]) -> pd.DataFrame:
    files = sorted(Path(root).glob("*/*/storm_paf/storm_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no IBTrACS CLIMADA storm files under {root}")
    tb = pads.dataset([str(f) for f in files]).to_table(
        columns=["storm_id", "year", "location_id", "max_wind_speed", "person_storm_hours",
                 "total_population", "total_population_exposed"])
    df = tb.to_pandas()
    df["year"] = df["year"].astype(int)
    return df[df["year"].between(*years)].reset_index(drop=True)


def ibtracs_land_counts(storms: pd.DataFrame, hier: pd.DataFrame, years: tuple[int, int] = STORM_YEARS,
                        root: Path = IBTRACS_LAND["a0"]) -> pd.DataFrame:
    """Observed land-affecting counts per (year, level, geo) from the CLIMADA IBTrACS files."""
    df = read_ibtracs_land(root, years)
    basin = storms.set_index("storm_id")["tc_basin"]
    df["basin"] = df["storm_id"].map(basin)
    if df["basin"].isna().any():
        missing = df.loc[df["basin"].isna(), "storm_id"].unique()[:10]
        raise RuntimeError(f"CLIMADA IBTrACS storms absent from the netCDF: {missing}")
    out = storm_counts_by_level(df, hier, keys=[])
    n_storms = df.groupby(["year", "basin", "storm_id"]).ngroups
    click.echo(f"[ibtracs land] {len(df):,} storm×admin0 rows, {n_storms:,} storms, "
               f"{(df['basin'] == NO_BASIN).sum():,} rows outside every tc-risk box")
    return out


# --- stage entry points -----------------------------------------------------------------
def run_aggregate(part: str, work_dir: Path, hier: pd.DataFrame, years: tuple[int, int] = STORM_YEARS) -> None:
    work_dir = Path(work_dir)
    if part == "tracks":
        write_parquet_checked(track_counts(years), work_dir / "counts_model_all.parquet", MODEL_COUNT_DTYPES)
    elif part == "climada":
        write_parquet_checked(model_land_counts(hier, years), work_dir / "counts_model_land.parquet",
                              MODEL_COUNT_DTYPES)
    elif part == "ibtracs":
        storms = ibtracs_storm_table()
        write_parquet_checked(storms, work_dir / "ibtracs_storms.parquet")
        write_parquet_checked(ibtracs_wind_coverage(storms, years), work_dir / "ibtracs_wind_coverage.parquet")
        write_parquet_checked(ibtracs_all_counts(storms, years), work_dir / "counts_obs_all.parquet", COUNT_DTYPES)
        write_parquet_checked(ibtracs_land_counts(storms, hier, years), work_dir / "counts_obs_land.parquet",
                              COUNT_DTYPES)
    else:
        raise ValueError(part)
    click.echo(f"[aggregate:{part}] wrote to {work_dir}")
