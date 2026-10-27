"""Export the stats tables as static JSON (one file per domain / source / level / outcome),
per-start-year overview summaries, simplified GBD geometry and a metadata file for the
artifact page.

Per-geo statistics are stored as arrays aligned with a file-level ``stat_keys`` list (one
key per pair | series | stat | window | detrend | start year) so the keys are written once per
file, not once per place.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd

from idd_tc_mortality.viz.cmip6_ibtracs_comparison.constants import (
    ADMIN0_SIMPLIFIED,
    BASIN_LABELS,
    CLIMADA_VERSION,
    DEFAULT_START,
    GCMS,
    LEVEL_LABELS,
    MORTALITY_STARTS,
    MORTALITY_WINDOWS,
    MORTALITY_YEARS,
    N_SURROGATES,
    OUTCOME_LABELS,
    RAKE_MODEL_JSON,
    RUNS,
    SOURCE_LABELS,
    STORM_STARTS,
    STORM_WINDOWS,
    STORM_YEARS,
    TCRISK_BOXES,
    WINDOW_LABELS,
    load_hierarchy,
)
from idd_tc_mortality.viz.cmip6_ibtracs_comparison.io import write_json
from idd_tc_mortality.viz.cmip6_ibtracs_comparison.stats import (
    CEIL_COLS,
    METRIC_COLS,
    MORTALITY_TAGS,
    QUANTILES,
    RAW_BAND_START,
)

PAGE = Path(__file__).with_name("page.html")
ROUND = 4


# --- pure shaping -----------------------------------------------------------------------
def _num(v, nd: int = ROUND):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, nd)


def _nums(a, nd: int = ROUND) -> list:
    return [_num(x, nd) for x in a]


def stat_key(pair: str, series: str, stat: str, window: str, detrended: bool, start_year: int) -> str:
    return f"{pair}|{series}|{stat}|{window}|{'d' if detrended else 'r'}|{int(start_year)}"


def band_key(band: str, window: str, detrended: bool, start_year: int = RAW_BAND_START) -> str:
    return f"{band}|{window}|{'d' if detrended else 'r'}|{int(start_year)}"


def ceil_key(window: str, detrended: bool, start_year: int, stat: str) -> str:
    return f"{window}|{'d' if detrended else 'r'}|{int(start_year)}|{stat}"


def _stat_keys(stats: pd.DataFrame) -> list[str]:
    return [stat_key(r.pair, r.series, r.stat, r.window, bool(r.detrended), r.start_year) for r in stats.itertuples(index=False)]


def shape_cell_group(stats: pd.DataFrame, band: pd.DataFrame, series: pd.DataFrame, ceiling: pd.DataFrame,
                     names: dict[str, str], years: np.ndarray) -> dict:
    """The per-geo payload of one (domain, source, level, outcome) file."""
    geos: dict[str, dict] = {}
    new = lambda geo: {"name": names.get(geo, geo), "series": {}}
    for geo, s in series.groupby("geo", sort=True):
        g = geos.setdefault(geo, new(geo))
        for (sid, stat), v in s.groupby(["series", "stat"]):
            vals = _nums(v.sort_values("year")["value"].to_numpy())
            if sid == "obs":
                g["obs"] = vals
            else:
                g["series"].setdefault(sid, {})[stat] = vals
    for geo, b in band.groupby("geo"):
        g = geos.setdefault(geo, new(geo))
        g["band"] = {}
        for (bk, window, det, sy), v in b.groupby(["band", "window", "detrended", "start_year"]):
            v = v.sort_values("start")
            g["band"][band_key(bk, window, bool(det), int(sy))] = {
                "start": v["start"].astype(int).tolist(), "end": v["end"].astype(int).tolist(),
                "q": [_nums(v[f"q{q:02d}"].to_numpy()) for q in QUANTILES],
                "obs": _nums(v["obs"].to_numpy()), "pct": _nums(v["obs_pct"].to_numpy(), 3),
            }
    keys = sorted(set(_stat_keys(stats))) if len(stats) else []
    pos = {k: i for i, k in enumerate(keys)}
    st = stats.assign(skey=_stat_keys(stats)) if len(stats) else stats
    for geo, rows in st.groupby("geo"):
        g = geos.setdefault(geo, new(geo))
        arr: list = [None] * len(keys)
        for r in rows.itertuples(index=False):
            arr[pos[r.skey]] = [_num(getattr(r, c)) for c in METRIC_COLS]
        g["stats"] = arr
    for geo, rows in ceiling.groupby("geo"):
        g = geos.setdefault(geo, new(geo))
        g["ceiling"] = {ceil_key(r.window, bool(r.detrended), r.start_year, r.stat): [_num(getattr(r, c)) for c in CEIL_COLS]
                        for r in rows.itertuples(index=False)}
    return {"years": [int(y) for y in years], "metrics": METRIC_COLS, "stat_keys": keys, "ceil_cols": CEIL_COLS, "geos": geos}


OVERVIEW_COLS = ["domain", "source", "level", "outcome", "pair", "series", "stat", "window", "detrended", "start_year",
                 "n_geos", "spearman", "kendall", "hit_rate", "updown", "ratio", "pct_mean", "sur_pct", "n_sig", "ceil_med"]


def overview_rows(stats: pd.DataFrame, ceiling: pd.DataFrame | None = None) -> list[list]:
    """Median of each measure across the kept geos per (domain, source, level, outcome, pair,
    series, stat, window, detrended, start_year), the geo count, the number of geos whose ρ
    falls outside the chance band (two-sided p < 0.05) and the median run-vs-run ceiling."""
    keys = ["domain", "source", "level", "outcome", "pair", "series", "stat", "window", "detrended", "start_year"]
    med = stats.groupby(keys, observed=True)[["spearman", "kendall", "hit_rate", "updown", "ratio", "pct_mean", "sur_pct"]].median()
    n = stats.groupby(keys, observed=True)["geo"].nunique().rename("n_geos")
    sig = stats.assign(_sig=(stats["sur_p"] < 0.05).astype(int)).groupby(keys, observed=True)["_sig"].sum().rename("n_sig")
    tab = med.join(n).join(sig).reset_index()
    if ceiling is not None and len(ceiling):
        ckeys = ["domain", "source", "level", "outcome", "window", "detrended", "start_year", "stat"]
        cm = ceiling.groupby(ckeys, observed=True)["ceil_med"].median().reset_index()
        tab = tab.merge(cm, on=ckeys, how="left")
    else:
        tab["ceil_med"] = np.nan
    rows = []
    for r in tab.itertuples(index=False):
        rows.append([r.domain, r.source, r.level, r.outcome, r.pair, r.series, r.stat, r.window,
                     "d" if r.detrended else "r", int(r.start_year), int(r.n_geos), _num(r.spearman), _num(r.kendall),
                     _num(r.hit_rate), _num(r.updown), _num(r.ratio), _num(r.pct_mean), _num(r.sur_pct), int(r.n_sig),
                     _num(r.ceil_med)])
    return rows


def spearman_index(stats: pd.DataFrame) -> dict:
    """{"keys": [stat keys], "geos": {geo: {outcome: [spearman per key]}}} — the overview's
    per-place lookup, stored as arrays in one fixed key order to keep the file small."""
    st = stats.assign(key=_stat_keys(stats))
    keys = sorted(st["key"].unique())
    pos = {k: i for i, k in enumerate(keys)}
    out: dict = {}
    for (geo, outcome), sub in st.groupby(["geo", "outcome"]):
        arr = [None] * len(keys)
        for k, v in zip(sub["key"], sub["spearman"]):
            arr[pos[k]] = _num(v)
        out.setdefault(geo, {})[outcome] = arr
    return {"keys": keys, "geos": out}


def box_geojson() -> dict:
    """tc-risk basin boxes as a GeoJSON FeatureCollection."""
    feats = []
    for b in TCRISK_BOXES:
        name = b.name.replace("tc-risk ", "")
        ring = [[b.lon0, b.lat0], [b.lon1, b.lat0], [b.lon1, b.lat1], [b.lon0, b.lat1], [b.lon0, b.lat0]]
        feats.append({"type": "Feature", "properties": {"id": name, "name": BASIN_LABELS.get(name, name)},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "features": feats}


# --- geometry ---------------------------------------------------------------------------
def build_geometry(hier: pd.DataFrame, out_dir: Path, tolerance: float = 0.08, grid: float = 0.01) -> dict[str, int]:
    """Simplified country / region / super-region polygons from the LBD admin0 layer."""
    import geopandas as gpd
    import shapely

    g = gpd.read_file(ADMIN0_SIMPLIFIED)[["loc_id", "geometry"]].rename(columns={"loc_id": "location_id"})
    g["location_id"] = g["location_id"].astype(int)
    h = hier.set_index("location_id")
    g = g[g["location_id"].isin(h.index[h["level"] == 3])].copy()
    g["region_id"] = g["location_id"].map(h["region_id"])
    g["super_region_id"] = g["location_id"].map(h["super_region_id"])
    g["name"] = g["location_id"].map(h["location_name"])
    sizes = {}
    layers = {
        "country": g[["location_id", "name", "geometry"]].rename(columns={"location_id": "id"}),
        "region": g.dissolve(by="region_id")[["geometry"]].reset_index().rename(columns={"region_id": "id"}),
        "super_region": g.dissolve(by="super_region_id")[["geometry"]].reset_index().rename(columns={"super_region_id": "id"}),
    }
    for key, layer in layers.items():
        layer = layer.copy()
        layer["id"] = layer["id"].astype(int)
        if "name" not in layer:
            layer["name"] = layer["id"].map(h["location_name"])
        geom = layer.geometry.simplify(tolerance, preserve_topology=True)
        geom = shapely.set_precision(geom.values, grid)
        layer = gpd.GeoDataFrame({"id": layer["id"].astype(str), "name": layer["name"]}, geometry=geom, crs=g.crs)
        layer = layer[~layer.geometry.is_empty]
        path = out_dir / "geo" / f"{key}.json"
        write_json(json.loads(layer.to_json(drop_id=True)), path)
        sizes[key] = path.stat().st_size
    path = out_dir / "geo" / "basin.json"
    write_json(box_geojson(), path)
    sizes["basin"] = path.stat().st_size
    return sizes


# --- stage ------------------------------------------------------------------------------
def _file_rel(domain: str, source: str, level: str, outcome: str) -> str:
    return f"data/{domain}/{source}/{level}/{outcome}.json"


def run_export(work_dir: Path, out_dir: Path) -> None:
    t0 = time.time()
    work_dir, out_dir = Path(work_dir), Path(out_dir)
    sdir = work_dir / "stats"
    stats = pd.read_parquet(sdir / "stats.parquet")
    band = pd.read_parquet(sdir / "band.parquet")
    series = pd.read_parquet(sdir / "series.parquet")
    ceiling = pd.read_parquet(sdir / "ceiling.parquet")
    geos = pd.read_parquet(sdir / "geos.parquet")
    hier = load_hierarchy()
    names = dict(zip(geos["geo"], geos["name"]))
    files = {}
    group_keys = ["domain", "source", "level", "outcome"]
    s_groups = {k: v for k, v in stats.groupby(group_keys)}
    b_groups = {k: v for k, v in band.groupby(group_keys)}
    c_groups = {k: v for k, v in ceiling.groupby(group_keys)}
    empty = stats.iloc[0:0]
    for key, se in series.groupby(group_keys):
        domain, _source, _level, _outcome = key
        years = np.arange(*STORM_YEARS) if domain == "storm" else np.arange(*MORTALITY_YEARS)
        years = np.append(years, years[-1] + 1)
        payload = shape_cell_group(s_groups.get(key, empty), b_groups.get(key, band.iloc[0:0]), se,
                                   c_groups.get(key, ceiling.iloc[0:0]), names, years)
        rel = _file_rel(*key)
        path = write_json(payload, out_dir / rel)
        files[rel] = path.stat().st_size
    for key, st in stats.groupby(["domain", "source", "level"]):
        rel = f"data/{key[0]}/{key[1]}/{key[2]}/spearman.json"
        path = write_json(spearman_index(st), out_dir / rel)
        files[rel] = path.stat().st_size
    click.echo(f"[export] {len(files)} data files, {sum(files.values()) / 1e6:.1f} MB, {time.time() - t0:.0f}s")

    ov = overview_rows(stats, ceiling)
    si = OVERVIEW_COLS.index("start_year")
    for start in sorted({r[si] for r in ov}):
        rel = f"data/overview/{start}.json"
        path = write_json({"columns": OVERVIEW_COLS, "rows": [r for r in ov if r[si] == start]}, out_dir / rel)
        files[rel] = path.stat().st_size
    sizes = build_geometry(hier, out_dir)
    click.echo(f"[export] geometry bytes {sizes}")

    extra = {}
    for name in ("sanity.json",):
        p = work_dir / name
        if p.exists():
            extra[name.split(".")[0]] = json.loads(p.read_text())
    cov = pd.read_parquet(work_dir / "ibtracs_wind_coverage.parquet")
    dropped = geos[~geos["kept"]].sort_values(["domain", "start_year", "level", "n_obs_land"])
    rake = {}
    if RAKE_MODEL_JSON.exists():
        meta_json = json.loads(RAKE_MODEL_JSON.read_text())
        rake = {"path": str(RAKE_MODEL_JSON.name), "frames": meta_json.get("climada_frames"),
                "ratios": meta_json.get("ratios"), "supersedes": meta_json.get("supersedes")}
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "climada_version": CLIMADA_VERSION,
        "storm_years": list(STORM_YEARS), "mortality_years": list(MORTALITY_YEARS),
        "storm_starts": list(STORM_STARTS), "mortality_starts": list(MORTALITY_STARTS), "default_start": DEFAULT_START,
        "n_surrogates": N_SURROGATES, "raw_band_start": RAW_BAND_START,
        "levels": list(LEVEL_LABELS.items()), "outcomes": list(OUTCOME_LABELS.items()),
        "sources": list(SOURCE_LABELS.items()), "mortality_tags": list(MORTALITY_TAGS),
        "runs": list(RUNS), "gcms": list(GCMS),
        "storm_windows": list(STORM_WINDOWS), "mortality_windows": list(MORTALITY_WINDOWS),
        "window_labels": WINDOW_LABELS, "metrics": METRIC_COLS, "ceil_cols": CEIL_COLS, "quantiles": list(QUANTILES),
        "geos": geos.to_dict(orient="records"),
        "dropped": dropped.to_dict(orient="records"),
        "basin_boxes": [{"basin": b.name.replace("tc-risk ", ""), "lon0": b.lon0, "lon1": b.lon1,
                         "lat0": b.lat0, "lat1": b.lat1} for b in TCRISK_BOXES],
        "ibtracs_wind_coverage": cov.to_dict(orient="records"),
        "rake": rake,
        "files": files, "geo_bytes": sizes, **extra,
    }
    write_json(meta, out_dir / "data" / "meta.json")
    if PAGE.exists():
        shutil.copy(PAGE, out_dir / "index.html")
    click.echo(f"[export] wrote {out_dir} in {time.time() - t0:.0f}s")
