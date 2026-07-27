"""Frame transforms for the driver-of-change sensitivities.

Each transform operates on the prepped per-storm-draw prediction frame (the
output of ``consolidated.prep_frame`` — one row per storm/tc_risk_draw/location,
carrying at least ``location_id``, ``year``, ``sdi``, ``exposed``,
``total_population``) and returns a modified copy. Rows before ``anchor_year``
are never touched; at ``anchor_year`` the transforms are no-ops by construction.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

ANCHOR_YEAR = 2023
SENSITIVITY_MODES = ("none", "sdi_const", "pop_const", "both")

# FHS all-age both-sex selectors (match paths.py FHS population convention).
_ALL_AGE = 22
_BOTH_SEX = 3


# ---------------------------------------------------------------------------
# Population ratio (from FHS all-age both-sex population)
# ---------------------------------------------------------------------------

def _load_fhs_pop(path: str, var: str) -> pd.DataFrame:
    """(location_id, year, pop) all-age both-sex from an FHS population nc."""
    da = xr.open_dataset(path)[var]
    if "statistic" in da.dims:
        da = da.sel(statistic="mean")
    if "scenario" in da.dims:
        da = da.squeeze("scenario", drop=True) if da.sizes["scenario"] == 1 else da.isel(scenario=0)
    if "sex_id" in da.dims:
        da = da.sel(sex_id=_BOTH_SEX)
    if "age_group_id" in da.dims:
        da = da.sel(age_group_id=_ALL_AGE)
    df = da.to_dataframe(name="pop").reset_index()
    df = df.rename(columns={"year_id": "year"})
    return df[["location_id", "year", "pop"]]


def population_ratio(
    past_path: str,
    future_path: str,
    anchor_year: int = ANCHOR_YEAR,
) -> dict[tuple[int, int], float]:
    """``{(location_id, year): pop(anchor) / pop(year)}`` for years >= anchor.

    Population is the FHS all-age both-sex series: past file for years <= anchor,
    future summary for years > anchor (the deliverable's past/future splice). Both
    numerator and denominator come from this single series, so the ratio is exactly
    1 at ``anchor_year`` and is internally consistent with the ``total_population``
    baked into ``person_storm_hours`` (verified equal to FHS s130v41).
    """
    past = _load_fhs_pop(past_path, "population")
    fut = _load_fhs_pop(future_path, "draws")
    pop = pd.concat([past[past["year"] <= anchor_year],
                     fut[fut["year"] > anchor_year]], ignore_index=True)

    anchor = pop[pop["year"] == anchor_year].set_index("location_id")["pop"]
    pop = pop[pop["year"] >= anchor_year].copy()
    pop["anchor_pop"] = pop["location_id"].map(anchor)
    pop = pop[pop["pop"] > 0]
    pop["ratio"] = pop["anchor_pop"] / pop["pop"]

    n_missing_anchor = int(pop["anchor_pop"].isna().sum())
    if n_missing_anchor:
        logger.warning("population_ratio: %d (loc,year) rows have no %d anchor population",
                       n_missing_anchor, anchor_year)
    pop = pop.dropna(subset=["ratio"])
    return {(int(l), int(y)): float(r)
            for l, y, r in zip(pop["location_id"], pop["year"], pop["ratio"])}


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

def freeze_sdi(frame: pd.DataFrame, sdi_table: pd.DataFrame,
               anchor_year: int = ANCHOR_YEAR) -> pd.DataFrame:
    """Hold SDI at its ``anchor_year`` value for all rows with ``year >= anchor_year``.

    The anchor value per location comes from ``sdi_table`` (the complete
    location x year table from ``bulk_sdi_table``), not the storm-gated frame.
    """
    anchor = (sdi_table[sdi_table["year"] == anchor_year]
              .drop_duplicates("location_id").set_index("location_id")["sdi"])
    out = frame.copy()
    mask = out["year"] >= anchor_year
    frozen = out.loc[mask, "location_id"].map(anchor)
    n_missing = int(frozen.isna().sum())
    if n_missing:
        logger.warning("freeze_sdi: %d rows (year>=%d) have no anchor SDI; left unchanged",
                       n_missing, anchor_year)
    out.loc[mask, "sdi"] = frozen.where(frozen.notna(), out.loc[mask, "sdi"]).values
    return out


def scale_exposure(frame: pd.DataFrame, pop_ratio: dict[tuple[int, int], float],
                   anchor_year: int = ANCHOR_YEAR) -> pd.DataFrame:
    """Scale ``exposed`` (person_storm_hours) by ``pop_ratio[(loc, year)]`` for
    rows with ``year >= anchor_year`` — holding population at its anchor level."""
    out = frame.copy()
    mask = out["year"] >= anchor_year
    keys = zip(out.loc[mask, "location_id"].astype(int), out.loc[mask, "year"].astype(int))
    ratios = np.array([pop_ratio.get(k, np.nan) for k in keys], dtype=float)
    n_missing = int(np.isnan(ratios).sum())
    if n_missing:
        logger.warning("scale_exposure: %d rows (year>=%d) have no population ratio; left unchanged",
                       n_missing, anchor_year)
    ratios = np.where(np.isnan(ratios), 1.0, ratios)
    out.loc[mask, "exposed"] = out.loc[mask, "exposed"].to_numpy(dtype=float) * ratios
    return out


def apply_sensitivity(
    frame: pd.DataFrame,
    mode: str,
    *,
    sdi_table: pd.DataFrame | None = None,
    pop_ratio: dict[tuple[int, int], float] | None = None,
    anchor_year: int = ANCHOR_YEAR,
) -> pd.DataFrame:
    """Dispatch the frame transform(s) for ``mode`` (one of ``SENSITIVITY_MODES``)."""
    if mode not in SENSITIVITY_MODES:
        raise ValueError(f"mode {mode!r} not in {SENSITIVITY_MODES}")
    if mode == "none":
        return frame
    out = frame
    if mode in ("sdi_const", "both"):
        if sdi_table is None:
            raise ValueError(f"mode {mode!r} needs sdi_table")
        out = freeze_sdi(out, sdi_table, anchor_year)
    if mode in ("pop_const", "both"):
        if pop_ratio is None:
            raise ValueError(f"mode {mode!r} needs pop_ratio")
        out = scale_exposure(out, pop_ratio, anchor_year)
    return out
