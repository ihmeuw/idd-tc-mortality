"""Three-panel storm figure: relative risk, population density, attributable person-years.

Ports the derivation in ``notebooks/temp/2026_09_23_katrina_plots_specific.ipynb``
onto stored inputs and the idd-figures map stack:

* hazard: the stage-1 per-storm maximum-wind grid (0.1 deg, NaN below 17 m/s);
* relative risk: intensity interpolated through the RR-by-windspeed sample
  curves (respiratory or CVD), clamped at the curve's top, zero below its bottom;
* PAF per pixel: ``(RR - 1) / RR * impact_days / 365`` — the notebook's fixed
  impact-duration assumption, exposed as ``--impact-days``;
* population: the 0.01 deg WGS84 population raster read as a window over the
  storm extent (water is NoData, empty land is 0); RR is nearest-resampled onto
  that grid, as rra-flooding does for flood fraction;
* attributable person-years per pixel: ``PAF * population``.

Two steps, so a notebook can restyle without recomputing: :func:`prepare_arrays`
does the data work and returns a :class:`PanelData`; :func:`build_figure` lays
it out. Colour and resolution follow rra-flooding's Hubei figure: Blues for the
hazard panel, viridis (log) for population, YlOrRd (log) for the attributable
panel, on the 0.01 deg grid. The layout is an idd-figures ``panel_grid`` spec
(maps row + colorbar row) so the map-to-bar gap is its own knob; map cells keep
the explicit aspect and box guard of ``map_facet``. Raster reading needs
``rasterio``, imported lazily so the pure functions stay importable without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

STAGE1_ROOT = Path(
    "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/ibtracs_stage1/"
    "ibtracs/official/historical"
)
RR_SAMPLES_DIR = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/relative_risk_samples")
RR_FILES = {"resp": "rd_rr_samples.csv", "cvd": "cvd_rr_samples.csv"}
OUTCOME_LABEL = {"resp": "respiratory", "cvd": "cardiovascular"}
POP_DIR = Path("/mnt/team/rapidresponse/pub/population-model/results/current/wgs84_0p01")
ADMIN1_SHP = Path("/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29/lbd_standard_admin_1.shp")
OUT_DIR = Path("/mnt/team/idd/pub/idd_tc_mortality/05-figures/working")

KNOTS_TO_MS = 0.514444
EARTH_RADIUS_KM = 6371.0088
DAYS_PER_YEAR = 365.0
PANEL_NAMES = ("rr", "density", "person_years")
CBAR_LABELS = ("Relative risk", "Population per km²", "Attributable person-years")


# --- inputs -------------------------------------------------------------------
def load_intensity(storm_id: str, basin: str, root: Path = STAGE1_ROOT) -> xr.DataArray:
    """Stage-1 per-storm maximum wind (m/s) on its 0.1 deg grid; NaN below threshold."""
    path = root / storm_id[:4] / basin / "intensity" / f"{storm_id}.nc"
    with xr.open_dataset(path) as ds:
        return ds["intensity"].load()


def load_rr_curve(outcome: str, root: Path = RR_SAMPLES_DIR) -> pd.DataFrame:
    """RR-by-windspeed sample table: ``windspeed`` in knots plus mean/min/max/sample_* columns."""
    return pd.read_csv(root / RR_FILES[outcome])


def read_population_window(path: Path, extent: list[float]):
    """Window of the population raster covering ``extent`` (lon_min, lon_max, lat_min, lat_max).

    Returns ``(counts, lat_centers, lon_centers, edges)`` where ``edges`` is the
    imshow extent of the window in the same order as ``extent``. NoData -> NaN.
    """
    import rasterio
    from rasterio.windows import from_bounds

    lon_min, lon_max, lat_min, lat_max = extent
    with rasterio.open(path) as src:
        win = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
        win = win.round_offsets().round_lengths()
        counts = src.read(1, window=win, masked=True).filled(np.nan).astype("float64")
        tf = src.window_transform(win)
    res_x, res_y = tf.a, -tf.e
    n_lat, n_lon = counts.shape
    lon = tf.c + res_x * (np.arange(n_lon) + 0.5)
    lat = tf.f - res_y * (np.arange(n_lat) + 0.5)
    edges = [tf.c, tf.c + res_x * n_lon, tf.f - res_y * n_lat, tf.f]
    return counts, lat, lon, edges


def load_admin1(extent: list[float], name: str | None = None, path: Path = ADMIN1_SHP):
    """Admin-1 polygons intersecting ``extent``; with ``name``, exactly that unit."""
    import geopandas as gpd

    lon_min, lon_max, lat_min, lat_max = extent
    gdf = gpd.read_file(path, bbox=(lon_min, lat_min, lon_max, lat_max))
    if name is None:
        return gdf
    unit = gdf[gdf["ADM1_NAME"] == name]
    if len(unit) != 1:
        msg = f"expected one admin-1 unit named {name!r} in extent, found {len(unit)}"
        raise ValueError(msg)
    return unit


# --- derivations ----------------------------------------------------------------
def rr_from_intensity(
    intensity_ms: np.ndarray, windspeed_knots: np.ndarray, rr: np.ndarray
) -> np.ndarray:
    """Relative risk per pixel from max wind speed via the sample curve.

    Below the curve's lowest windspeed -> 0; above its highest -> the RR at the
    highest; linear interpolation between. NaN intensity stays NaN. (The
    notebook's ``interpolate_rr_from_windspeed``, without its unused threshold.)
    """
    ws_ms = np.asarray(windspeed_knots, dtype=float) * KNOTS_TO_MS
    order = np.argsort(ws_ms)
    ws_ms, rr = ws_ms[order], np.asarray(rr, dtype=float)[order]
    x = np.asarray(intensity_ms, dtype=float)
    out = np.full(x.shape, np.nan)
    finite = np.isfinite(x)
    out[finite] = np.interp(x[finite], ws_ms, rr, left=0.0, right=rr[-1])
    return out


def paf_from_rr(rr: np.ndarray, impact_days: float) -> np.ndarray:
    """Per-pixel PAF ``(RR - 1) / RR * impact_days / 365``; NaN where RR is not > 0."""
    rr = np.asarray(rr, dtype=float)
    out = np.full(rr.shape, np.nan)
    ok = np.isfinite(rr) & (rr > 0)
    out[ok] = (rr[ok] - 1.0) / rr[ok] * (impact_days / DAYS_PER_YEAR)
    return out


def pixel_area_km2(lat_centers: np.ndarray, res_deg: float) -> np.ndarray:
    """Area (km^2) of a ``res_deg`` square pixel centred at each latitude."""
    d = math.radians(res_deg) * EARTH_RADIUS_KM
    return d * d * np.cos(np.radians(np.asarray(lat_centers, dtype=float)))


def _nearest_index(src: np.ndarray, dst: np.ndarray):
    order = np.argsort(src)
    s = src[order]
    step = float(np.median(np.diff(s))) if s.size > 1 else math.inf
    pos = np.searchsorted(s, dst)
    left = np.clip(pos - 1, 0, s.size - 1)
    right = np.clip(pos, 0, s.size - 1)
    pick = np.where(np.abs(s[right] - dst) < np.abs(s[left] - dst), right, left)
    valid = np.abs(s[pick] - dst) <= step / 2 + 1e-9
    return order[pick], valid


def nearest_resample(
    src: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour resample of a (lat, lon) grid onto target centres; outside -> NaN."""
    ilat, vlat = _nearest_index(np.asarray(src_lat, float), np.asarray(dst_lat, float))
    ilon, vlon = _nearest_index(np.asarray(src_lon, float), np.asarray(dst_lon, float))
    out = np.asarray(src, dtype=float)[np.ix_(ilat, ilon)].copy()
    out[~vlat, :] = np.nan
    out[:, ~vlon] = np.nan
    return out


def storm_extent(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, pad_deg: float):
    """Bounding box of pixels with ``values > 0``, padded: [lon_min, lon_max, lat_min, lat_max]."""
    rows, cols = np.where(np.isfinite(values) & (values > 0))
    if rows.size == 0:
        msg = "no affected pixels (values > 0)"
        raise ValueError(msg)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return [
        float(lon[cols].min() - pad_deg),
        float(lon[cols].max() + pad_deg),
        float(lat[rows].min() - pad_deg),
        float(lat[rows].max() + pad_deg),
    ]


def admin_extent(gdf, frac: float = 0.05):
    """Admin unit bbox padded by ``frac`` of its width/height (the Hubei figure's 5%)."""
    minx, miny, maxx, maxy = gdf.geometry.total_bounds
    dx, dy = frac * (maxx - minx), frac * (maxy - miny)
    return [float(minx - dx), float(maxx + dx), float(miny - dy), float(maxy + dy)]


def mask_outside(arr: np.ndarray, geometry, edges: list[float]) -> np.ndarray:
    """NaN everywhere outside ``geometry`` (all-touched, as the Hubei figure masks)."""
    from affine import Affine
    from rasterio.features import geometry_mask

    n_lat, n_lon = arr.shape
    lon_min, lon_max, lat_min, lat_max = edges
    tf = Affine((lon_max - lon_min) / n_lon, 0, lon_min, 0, -(lat_max - lat_min) / n_lat, lat_max)
    inside = geometry_mask([geometry], transform=tf, invert=True, out_shape=arr.shape, all_touched=True)
    out = np.asarray(arr, dtype=float).copy()
    out[~inside] = np.nan
    return out


def positive_or_nan(arr: np.ndarray) -> np.ndarray:
    """Copy with non-finite and non-positive values set to NaN (display masking)."""
    out = np.asarray(arr, dtype=float).copy()
    out[~(np.isfinite(out) & (out > 0))] = np.nan
    return out


# --- data step --------------------------------------------------------------------
@dataclass
class PanelData:
    """Everything :func:`build_figure` needs; produced by :func:`prepare_arrays`."""

    rr: np.ndarray
    density: np.ndarray
    person_years: np.ndarray
    raster_extent: list[float]  # imshow extent of the arrays
    extent: list[float]  # map view extent
    titles: tuple[str, str, str]
    totals: dict[str, float] = field(default_factory=dict)
    land_gdf: object = None  # admin-1 polygons in extent: grey backdrop + thin outlines
    unit_gdf: object = None  # the one admin-1 unit when restricted (red outline)


def prepare_arrays(
    *,
    storm_id: str,
    basin: str = "NA",
    outcome: str = "resp",
    rr_sample: str = "mean",
    impact_days: float = 20.0,
    pop_year: int | None = None,
    adm1_name: str | None = None,
    pad_deg: float = 0.5,
) -> PanelData:
    """Load inputs and derive the three 0.01 deg arrays, extents, totals and titles."""
    pop_year = pop_year or int(storm_id[:4])
    intensity = load_intensity(storm_id, basin)
    lat_i, lon_i = intensity["lat"].values, intensity["lon"].values
    curve = load_rr_curve(outcome)
    rr_coarse = positive_or_nan(
        rr_from_intensity(intensity.values, curve["windspeed"].values, curve[rr_sample].values)
    )

    unit = None
    if adm1_name is not None:
        unit = load_admin1(storm_extent(rr_coarse, lat_i, lon_i, pad_deg), adm1_name)
        extent = admin_extent(unit)
    else:
        extent = storm_extent(rr_coarse, lat_i, lon_i, pad_deg)

    counts, lat_p, lon_p, edges = read_population_window(POP_DIR / f"{pop_year}q1.tif", extent)
    rr = nearest_resample(rr_coarse, lat_i, lon_i, lat_p, lon_p)
    counts = positive_or_nan(counts)
    person_years = paf_from_rr(rr, impact_days) * counts
    res_deg = float(abs(lon_p[1] - lon_p[0]))
    density = counts / pixel_area_km2(lat_p, res_deg)[:, None]

    if unit is not None:
        geom = unit.geometry.iloc[0]
        rr, counts, density, person_years = (
            mask_outside(a, geom, edges) for a in (rr, counts, density, person_years)
        )
    person_years = positive_or_nan(person_years)

    totals = {
        "population": float(np.nansum(counts)),
        "person_years": float(np.nansum(person_years)),
        "affected_population": float(np.nansum(np.where(np.isfinite(rr), counts, 0.0))),
    }
    where = f"in {adm1_name}" if adm1_name else "in extent"
    label = OUTCOME_LABEL[outcome]
    name = str(intensity.attrs.get("storm_name", storm_id)).title()
    start = pd.to_datetime(intensity.attrs.get("start_date")).strftime("%B %d, %Y")
    titles = (
        f"Relative risk, {label}\n{name} ({start})",
        f"Population density ({pop_year})\n({totals['population']:,.0f} people {where})",
        (
            f"Attributable person-years, {label}\n"
            f"({totals['person_years']:,.0f} person-years, {impact_days:g} impact days)"
        ),
    )
    return PanelData(
        rr=rr,
        density=density,
        person_years=person_years,
        raster_extent=edges,
        extent=extent,
        titles=titles,
        totals=totals,
        land_gdf=load_admin1(extent),
        unit_gdf=unit,
    )


# --- figure ---------------------------------------------------------------------
def log_norm(values: np.ndarray):
    """LogNorm from max(0.1, min) to the 99th percentile of positive values (Hubei rule)."""
    from matplotlib.colors import LogNorm

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return LogNorm(0.1, 1.0)
    vmin = max(0.1, float(v.min()))
    vmax = float(np.percentile(v, 99))
    if vmax <= vmin:
        vmax = float(v.max())
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin, vmax)


def decade_ticks(norm) -> list[float] | None:
    """Powers of ten inside [vmin, vmax]; None when fewer than two (let matplotlib pick)."""
    k0 = math.ceil(math.log10(norm.vmin) - 1e-9)
    k1 = math.floor(math.log10(norm.vmax) + 1e-9)
    ticks = [10.0**k for k in range(k0, k1 + 1)]
    return ticks if len(ticks) >= 2 else None


def figure_geometry(
    fig_width: float,
    aspect: float,
    *,
    ncols: int = 3,
    wspace: float = 0.04,
    side_margins: tuple[float, float] = (0.02, 0.98),
    title_h: float = 1.0,
    gap_h: float = 0.25,
    cbar_h: float = 0.9,
    bottom_pad_h: float = 0.1,
) -> dict:
    """Outer-grid geometry for a maps row over a colorbar row, all in inches.

    The map row's height follows the panel width and the extent aspect
    (``idd_figures.lib.layouts.geometry``); ``gap_h`` is the map-to-bar gap,
    charged as gridspec ``hspace`` against the mean row height; ``title_h`` is
    the top allowance for the panel titles. Nothing is left to auto-layout.
    """
    from idd_figures.lib.layouts.geometry import gap_factor, map_row_height

    margins_lr = {"left": side_margins[0], "right": side_margins[1]}
    map_h = map_row_height(fig_width, margins=margins_lr, ncols=ncols, aspect=aspect, wspace=wspace)
    heights = [map_h, cbar_h]
    hspace = gap_h / (sum(heights) / len(heights))
    content = sum(heights) * gap_factor(len(heights), hspace)  # rows + the one gap
    fig_h = content + title_h + bottom_pad_h
    margins = {**margins_lr, "top": 1.0 - title_h / fig_h, "bottom": bottom_pad_h / fig_h}
    return {
        "fig_height": fig_h,
        "map_height": map_h,
        "height_ratios": heights,
        "hspace": hspace,
        "margins": margins,
    }


def _check_map_box(fig, ax, aspect: float, tol: float = 0.01) -> None:
    """Raise if the map cell's allocated box does not match the extent aspect (no letterboxing)."""
    fw, fh = fig.get_size_inches()
    pos = ax.get_position(original=True)
    box = (pos.height * fh) / (pos.width * fw)
    if abs(box - aspect) > tol * aspect:
        msg = f"map cell box aspect {box:.4f} != extent aspect {aspect:.4f}"
        raise ValueError(msg)


def _legend_paint(ax, _data=None, **kw):
    from idd_figures.lib.painters.legend import bin_legend_panel

    return bin_legend_panel(ax, **kw)


def _linear_norm(arr: np.ndarray):
    """Linear norm over the finite range of ``arr``."""
    from matplotlib.colors import Normalize

    return Normalize(vmin=float(np.nanmin(arr)), vmax=float(np.nanmax(arr)))


@dataclass
class PanelSpec:
    """One map panel and its colorbar, for :func:`build_panel_row`."""

    array: np.ndarray
    cmap: str  # matplotlib colormap name
    title: str
    cbar_label: str
    name: str  # axes name; the bar is ``cbar:<name>``
    norm: object = None  # matplotlib norm; None -> linear over the finite range
    ticks: list | None = None  # pinned colorbar tick positions (data units)


def build_panel_row(
    specs: list[PanelSpec],
    *,
    raster_extent: list[float],
    extent: list[float],
    land_gdf=None,
    boundary_gdf=None,
    boundary_style: dict | None = None,
    fig_width: float = 18.0,
    wspace: float = 0.04,
    title_h: float = 1.0,
    gap_h: float = 0.25,
    cbar_h: float = 0.9,
    bottom_pad_h: float = 0.1,
    title_fontsize: float = 18,
    cbar_fontsize: float = 14,
    cbar_label_fontsize: float = 16,
    water_color: str = "#A9CCE8",
    water_alpha: float = 1.0,
    land_color: str = "lightgray",
):
    """A row of map panels over a row of colorbars (one per panel); return the Figure.

    All panels share ``raster_extent`` (imshow extent of the arrays) and the map
    view ``extent``. Water is the axes background (``water_color``); ``land_color``
    fills ``land_gdf`` under the rasters so land without data reads grey while
    water stays water; NaN raster pixels are transparent. ``boundary_gdf`` is
    outlined with ``boundary_style``. Geometry comes from :func:`figure_geometry`;
    map cells get an explicit aspect and the letterbox guard.
    """
    import cartopy.crs as ccrs
    from idd_figures.lib.layouts.geometry import extent_aspect
    from idd_figures.lib.layouts.grids import cell, grid, paint, panel_grid
    from idd_figures.lib.painters.maps import map_cell_painter
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable

    n = len(specs)
    aspect = extent_aspect(extent)
    geo = figure_geometry(
        fig_width, aspect, ncols=n, wspace=wspace, title_h=title_h, gap_h=gap_h,
        cbar_h=cbar_h, bottom_pad_h=bottom_pad_h,
    )
    b_style = boundary_style or {"color": "black", "linewidth": 0.5}
    map_cells, bar_cells = [], []
    for j, s in enumerate(specs):
        cmap = colormaps[s.cmap]
        norm = s.norm if s.norm is not None else _linear_norm(s.array)
        map_cells.append(
            cell(
                (0, j),
                paint(
                    map_cell_painter,
                    None,
                    extent=tuple(extent),
                    raster=s.array,
                    raster_extent=raster_extent,
                    cmap=cmap,
                    norm=norm,
                    base_admin_gdf=land_gdf,
                    boundary_gdf=boundary_gdf,
                    ocean="fill",
                    ocean_style={"color": water_color, "alpha": water_alpha},
                    backdrop_style={"color": land_color},
                    boundary_style=b_style,
                ),
                projection=ccrs.PlateCarree(),
                title=s.title,
                title_kwargs={"fontsize": title_fontsize},
                name=s.name,
            )
        )
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        bar_cells.append(
            cell(
                (0, j),
                paint(
                    _legend_paint,
                    None,
                    mappable=sm,
                    use_colorbar=True,
                    cbar_label=s.cbar_label,
                    ticks=s.ticks,
                    fontsize=cbar_fontsize,
                    cbar_label_fontsize=cbar_label_fontsize,
                    bin_bottom=0.55,
                    bin_top=0.95,
                ),
                name=f"cbar:{s.name}",
            )
        )
    spec = grid(
        (2, 1),
        [
            cell((0, 0), grid((1, n), map_cells, wspace=wspace)),
            cell((1, 0), grid((1, n), bar_cells, wspace=wspace)),
        ],
        height_ratios=geo["height_ratios"],
        margins=geo["margins"],
        hspace=geo["hspace"],
    )
    fig = panel_grid(spec, figsize=(fig_width, geo["fig_height"]))
    for s in specs:
        ax = fig.axes_by_name[s.name]
        ax.set_aspect(1.0, adjustable="box")
        _check_map_box(fig, ax, aspect)
    return fig


def build_figure(
    data: PanelData,
    *,
    cbar_labels: tuple[str, str, str] = CBAR_LABELS,
    rr_cmap: str = "Blues",
    density_cmap: str = "viridis",
    person_years_cmap: str = "YlOrRd",
    boundary_style: dict | None = None,
    unit_style: dict | None = None,
    **style,
):
    """RR / density / person-years over three colorbars (the Hubei analogue).

    Linear Blues for RR, log viridis for density, log YlOrRd for person-years,
    decade ticks on the log bars. ``boundary_style`` styles the admin-1
    outlines, ``unit_style`` the restricted unit's outline. Remaining keywords
    go to :func:`build_panel_row` (sizes, gap, fonts, water/land colours).
    """
    dens_norm, py_norm = log_norm(data.density), log_norm(data.person_years)
    specs = [
        PanelSpec(data.rr, rr_cmap, data.titles[0], cbar_labels[0], PANEL_NAMES[0]),
        PanelSpec(
            data.density, density_cmap, data.titles[1], cbar_labels[1], PANEL_NAMES[1],
            norm=dens_norm, ticks=decade_ticks(dens_norm),
        ),
        PanelSpec(
            data.person_years, person_years_cmap, data.titles[2], cbar_labels[2], PANEL_NAMES[2],
            norm=py_norm, ticks=decade_ticks(py_norm),
        ),
    ]
    if data.unit_gdf is not None:
        boundary, b_style = data.unit_gdf, unit_style or {"color": "red", "linewidth": 1.5}
    else:
        boundary, b_style = data.land_gdf, boundary_style or {"color": "black", "linewidth": 0.5}
    return build_panel_row(
        specs, raster_extent=data.raster_extent, extent=data.extent, land_gdf=data.land_gdf,
        boundary_gdf=boundary, boundary_style=b_style, **style,
    )


# --- two-panel exposure figure: max intensity + exposure hours ------------------------
EXPOSURE_NAMES = ("intensity", "exposure_hours")
EXPOSURE_CBAR_LABELS = ("Wind speed (m/s)", "Exposure (hours)")


def load_exposure_hours(storm_id: str, basin: str, root: Path = STAGE1_ROOT) -> xr.DataArray:
    """Stage-1 per-storm hours with wind above threshold, on the intensity grid."""
    path = root / storm_id[:4] / basin / "exposure_hours" / f"{storm_id}.nc"
    with xr.open_dataset(path) as ds:
        return ds["exposure_hours"].load()


def grid_edges(lat: np.ndarray, lon: np.ndarray) -> list[float]:
    """imshow extent (pixel edges) of a regular grid given its centre coordinates."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    dlat = abs(float(np.median(np.diff(lat)))) if lat.size > 1 else 0.0
    dlon = abs(float(np.median(np.diff(lon)))) if lon.size > 1 else 0.0
    return [
        float(lon.min() - dlon / 2),
        float(lon.max() + dlon / 2),
        float(lat.min() - dlat / 2),
        float(lat.max() + dlat / 2),
    ]


@dataclass
class ExposureData:
    """Inputs for :func:`build_exposure_figure`; produced by :func:`prepare_exposure_arrays`."""

    intensity: np.ndarray  # lat descending, lon ascending (imshow origin upper)
    exposure_hours: np.ndarray
    raster_extent: list[float]
    extent: list[float]
    titles: tuple[str, str]
    land_gdf: object = None


def prepare_exposure_arrays(*, storm_id: str, basin: str = "NA", pad_deg: float = 0.5) -> ExposureData:
    """Load the stage-1 max-intensity and exposure-hours grids for one storm (native 0.1 deg)."""
    intensity = load_intensity(storm_id, basin)
    hours = load_exposure_hours(storm_id, basin)
    if intensity.shape != hours.shape:
        msg = f"intensity {intensity.shape} and exposure_hours {hours.shape} grids differ"
        raise ValueError(msg)
    lat, lon = intensity["lat"].values, intensity["lon"].values
    inten, hrs = intensity.values, hours.values
    if lat[0] < lat[-1]:  # imshow origin="upper" wants north first
        lat, inten, hrs = lat[::-1], inten[::-1], hrs[::-1]
    inten, hrs = positive_or_nan(inten), positive_or_nan(hrs)
    extent = storm_extent(inten, lat, lon, pad_deg)
    thr = float(hours.attrs.get("wind_threshold_m_s", 17.0))
    name = str(intensity.attrs.get("storm_name", storm_id)).title()
    start = pd.to_datetime(intensity.attrs.get("start_date")).strftime("%B %d, %Y")
    titles = (
        f"Max intensity, windspeed > {thr:g} m/s\n{name} ({start})",
        f"Exposure hours, windspeed > {thr:g} m/s\n{name} ({start})",
    )
    return ExposureData(
        intensity=inten,
        exposure_hours=hrs,
        raster_extent=grid_edges(lat, lon),
        extent=extent,
        titles=titles,
        land_gdf=load_admin1(extent),
    )


def build_exposure_figure(
    data: ExposureData,
    *,
    intensity_cmap: str = "viridis",
    hours_cmap: str = "magma",
    cbar_labels: tuple[str, str] = EXPOSURE_CBAR_LABELS,
    fig_width: float = 13.0,
    **style,
):
    """Max intensity (viridis) and exposure hours (magma) over two colorbars.

    Linear colour scales over each array's range. Remaining keywords go to
    :func:`build_panel_row` (sizes, gap, fonts, water/land colours, boundary_style).
    """
    specs = [
        PanelSpec(data.intensity, intensity_cmap, data.titles[0], cbar_labels[0], EXPOSURE_NAMES[0]),
        PanelSpec(data.exposure_hours, hours_cmap, data.titles[1], cbar_labels[1], EXPOSURE_NAMES[1]),
    ]
    return build_panel_row(
        specs, raster_extent=data.raster_extent, extent=data.extent, land_gdf=data.land_gdf,
        boundary_gdf=data.land_gdf, fig_width=fig_width, **style,
    )


@click.command()
@click.option("--storm-id", default="2005236N23285", show_default=True, help="IBTrACS id (stage-1 file stem).")
@click.option("--basin", default="NA", show_default=True, help="Stage-1 basin folder.")
@click.option("--outcome", type=click.Choice(sorted(RR_FILES)), default="resp", show_default=True)
@click.option("--rr-sample", default="mean", show_default=True, help="RR table column: mean, min, max, sample_001, ...")
@click.option("--impact-days", default=20.0, show_default=True, help="Impact duration assigned to every affected pixel.")
@click.option("--pop-year", type=int, default=None, help="Population raster year [default: storm year].")
@click.option("--adm1-name", default=None, help="Restrict panels and totals to one admin-1 unit (ADM1_NAME).")
@click.option("--pad-deg", default=0.5, show_default=True, help="Padding around the affected bbox (no admin unit).")
@click.option("--fig-width", default=18.0, show_default=True, help="Figure width in inches.")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Output .png [default: OUT_DIR/<storm>_<outcome>_three_panel.png].")
@click.option("--dpi", default=200, show_default=True)
def main(storm_id, basin, outcome, rr_sample, impact_days, pop_year, adm1_name, pad_deg, fig_width, out, dpi):
    """Three-panel storm map: relative risk, population density, attributable person-years."""
    from idd_figures.lib.io import save_figure

    out = out or OUT_DIR / f"{storm_id}_{outcome}_three_panel.png"
    data = prepare_arrays(
        storm_id=storm_id, basin=basin, outcome=outcome, rr_sample=rr_sample,
        impact_days=impact_days, pop_year=pop_year, adm1_name=adm1_name, pad_deg=pad_deg,
    )
    fig = build_figure(data, fig_width=fig_width)
    written = save_figure(fig, out, dpi=dpi)
    for k, v in data.totals.items():
        click.echo(f"{k}: {v:,.1f}")
    click.echo(f"wrote {written[0]}")


if __name__ == "__main__":
    main()
