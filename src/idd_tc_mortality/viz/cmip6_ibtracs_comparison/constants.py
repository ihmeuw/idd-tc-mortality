"""Shared definitions for the CMIP6 (tc-risk / CLIMADA) vs IBTrACS comparability report.

Period, intensity thresholds, outcome definitions, member → GCM mapping, geography levels
and the tc-risk basin assignment rule (genesis point inside the tc-risk basin box, with
longitudes wrapped to -180..180 — the rule that reproduces the basin labels of the
observed-deaths training file). The design decisions themselves are recorded in
``.claude/cmip6_ibtracs_comparison_prompt.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from idd_tc_mortality.predict.blend_historical import DR, HIER_PATH, ROOT, SHIP_MODEL_JSON
from idd_tc_mortality.viz.basin_map import ADMIN0_SIMPLIFIED, TCRISK_BOXES

__all__ = ["ADMIN0_SIMPLIFIED", "HIER_PATH", "SHIP_MODEL_JSON", "TCRISK_BOXES"]

TS = DR.parent  # the rapidresponse tropical-storms share, via the committed direct_risk constant
TRACKS_PARQUET = TS / "figures" / "tc_risk" / "track_max_vmax.parquet"
# CLIMADA stage4b_v3 (admin0 consolidated 2026-10-26, admin1 2026-10-18): drops the spurious
# inland units of v2 and adds West African exposure; same members, scenarios, years, draws
CLIMADA_VERSION = "stage4b_v3"
MODEL_FRAMES = {
    "a0": TS / "climada/output/stage4b_v3_admin0/_consolidated/storm_exposure_all.parquet",
    "a1": TS / "climada/output/stage4b_v3_admin1/_consolidated/storm_exposure_all.parquet",
}
IBTRACS_LAND = {
    "a0": TS / "climada/output/ibtracs_stage4b_pafs_admin0/ibtracs/official/historical",
    "a1": TS / "climada/output/ibtracs_stage4b_pafs_admin1/ibtracs/official/historical",
}
IBTRACS_NC = Path.home() / "climada" / "data" / "IBTrACS.ALL.v04r01.nc"
OBS_DEATHS = ROOT / "00-data" / "20260722_v1985" / "input.parquet"
# super-region rake ratios: the v3-frames recreation of the shipped deliverable (staged on the
# node until Bobby copies it to rapidresponse direct_risk); `predict --rake-json` overrides
RAKE_MODEL_JSON = ROOT / "04-predict" / "20261026_shipped_v3" / "direct_risk_staging" / "2026_10_26_v2000_model.json"
WORK_DIR = ROOT / "05-figures" / "working" / "cmip6_ibtracs_comparison"

# --- period and splice -----------------------------------------------------------------
STORM_YEARS = (1970, 2025)
MORTALITY_YEARS = (1985, 2023)
# model years come from `historical` up to 2014 and from `ssp245` after it
SPLICE = {"historical": (1970, 2014), "ssp245": (2015, 2025)}
# start years: every statistic restarts at X (detrend, windows, terciles, the ≥10-storm filter,
# chance bands, ceilings, effective n); the end stays at the period end
STORM_STARTS = (1970, 1980, 1985, 1990, 2000)
MORTALITY_STARTS = (1985, 1990, 2000)
DEFAULT_START = 1985

# --- intensity --------------------------------------------------------------------------
KT_TO_MS = 0.514444
# IBTrACS wmo_wind outside US-agency basins is a 10-min mean; the usual conversion to the
# 1-min convention of usa_wind / tc-risk is to divide by 0.88 (New Delhi reports 3-min).
TEN_MIN_TO_ONE_MIN = 1.0 / 0.88
# code = number of edges <= wind: 0 below tropical-storm strength (dropped), 1 TS,
# 2 Cat1, 3 Cat2, 4 Cat3, 5 Cat4, 6 Cat5
CATEGORY_EDGES = np.array([18.0, 33.0, 43.0, 50.0, 58.0, 70.0])
CAT_NAMES = {1: "TS", 2: "Cat1", 3: "Cat2", 4: "Cat3", 5: "Cat4", 6: "Cat5"}
CAT_COLS = [f"n{k}" for k in range(1, 7)]
OUTCOMES: dict[str, tuple[int, ...]] = {
    "total": (1, 2, 3, 4, 5, 6),
    "cat1": (2,), "cat2": (3,), "cat3": (4,), "cat4": (5,), "cat5": (6,),
    "cat3plus": (4, 5, 6), "cat4plus": (5, 6),
}
OUTCOME_LABELS = {
    "total": "Total storms (≥18 m/s)", "cat1": "Cat 1", "cat2": "Cat 2", "cat3": "Cat 3",
    "cat4": "Cat 4", "cat5": "Cat 5", "cat3plus": "Cat 3+", "cat4plus": "Cat 4+",
}
# storm sources: all tracks (tc-risk lifetime max; IBTrACS track max) vs land-hitting
# (CLIMADA max wind over exposed admin units)
SOURCES = ("all", "land")
SOURCE_LABELS = {"all": "All storms", "land": "Land-hitting storms"}

# --- members ----------------------------------------------------------------------------
RUNS = (
    "CMCC-ESM2_r1i1p1f1", "EC-Earth3_r1i1p1f1", "MPI-ESM1-2-HR_r1i1p1f1",
    "MRI-ESM2-0_r1i1p1f1", "MRI-ESM2-0_r2i1p1f1", "MRI-ESM2-0_r3i1p1f1",
    "MRI-ESM2-0_r4i1p1f1", "MRI-ESM2-0_r5i1p1f1",
)
N_DRAWS = 100


def gcm_of(run: str) -> str:
    """``'MRI-ESM2-0_r3i1p1f1'`` → ``'MRI-ESM2-0'``."""
    return run.split("_", 1)[0]


GCMS = tuple(sorted({gcm_of(r) for r in RUNS}))

# --- geography --------------------------------------------------------------------------
LEVELS = ("global", "basin", "super_region", "region", "country")
LEVEL_LABELS = {"global": "Global", "basin": "Basin", "super_region": "GBD super-region",
                "region": "GBD region", "country": "Country"}
GLOBAL_ID = 1
BASINS = ("AU", "EP", "NA", "NI", "SI", "SP", "WP")
BASIN_LABELS = {"AU": "Australia (tc-risk AU)", "EP": "East Pacific", "NA": "North Atlantic",
                "NI": "North Indian", "SI": "South Indian (tc-risk SI)",
                "SP": "South Pacific (tc-risk SP, 180–250°E)", "WP": "West Pacific"}
NO_BASIN = "XX"  # IBTrACS storms whose genesis is in no tc-risk box (South Atlantic etc.)
MIN_OBS_LAND_STORMS = 10

# --- windows ----------------------------------------------------------------------------
# centred moving averages only; non-overlapping blocks were dropped (too few points to rank)
STORM_WINDOWS = ("ma1", "ma5", "ma10")
MORTALITY_WINDOWS = ("ma5", "ma10", "ma15")
WINDOW_LABELS = {"ma1": "1-yr", "ma5": "5-yr moving", "ma10": "10-yr moving", "ma15": "15-yr moving"}

# --- chance band and ceiling -----------------------------------------------------------
# phase-randomized surrogates of the observed annual series (same power spectrum, random
# phases), smoothed like the real series and correlated with every model series
N_SURROGATES = 1000
SURROGATE_SEED = 20261026


# --- helpers ----------------------------------------------------------------------------
def wrap_lon(lon) -> np.ndarray:
    """Wrap longitudes to [-180, 180). IBTrACS keeps tracks continuous across the dateline."""
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def assign_tcrisk_basin(lon, lat) -> np.ndarray:
    """tc-risk basin whose box contains (lon, lat); ``NO_BASIN`` outside every box.

    Boxes are half-open (``lon0 <= lon < lon1``, ``lat0 <= lat < lat1``) so a point on a
    shared edge lands in exactly one basin. NaN coordinates map to ``NO_BASIN``.
    """
    lon = wrap_lon(lon)
    lat = np.asarray(lat, dtype=float)
    out = np.full(lon.shape, NO_BASIN, dtype=object)
    for b in TCRISK_BOXES:
        name = b.name.replace("tc-risk ", "")
        hit = (lon >= b.lon0) & (lon < b.lon1) & (lat >= b.lat0) & (lat < b.lat1) & (out == NO_BASIN)
        out[hit] = name
    return out.astype(str)


def categorize(wind) -> np.ndarray:
    """Category code 0..6 for max wind in m/s (0 = below 18 m/s or missing)."""
    w = np.asarray(wind, dtype=float)
    code = np.searchsorted(CATEGORY_EDGES, w, side="right").astype(np.int8)
    code[~np.isfinite(w)] = 0
    return code


def outcome_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """Add one column per outcome (sums of the ``n1..n6`` category columns)."""
    out = counts.copy()
    for name, codes in OUTCOMES.items():
        out[name] = sum(out[f"n{k}"] for k in codes)
    return out


def load_hierarchy(path: Path = HIER_PATH) -> pd.DataFrame:
    """GBD hierarchy with integer region / super-region ids."""
    h = pd.read_parquet(path)
    h = h[["location_id", "location_name", "level", "parent_id", "path_to_top_parent",
           "region_id", "super_region_id"]].copy()
    for c in ("region_id", "super_region_id"):
        h[c] = h[c].fillna(-1).astype(int)
    return h


def geo_label(level: str, geo: str, hier: pd.DataFrame) -> str:
    if level == "global":
        return "Global"
    if level == "basin":
        return BASIN_LABELS.get(geo, geo)
    names = hier.set_index("location_id")["location_name"]
    return str(names.get(int(geo), geo))
