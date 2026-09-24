"""Basin-definition map: IBTrACS v4 (solid) vs tc-risk (dashed) boxes, all basins.

Port of ``plot_basin_comparison`` (``notebooks/20260515/predict_scenario_plots.ipynb``)
onto the idd-figures map stack. Layout is an idd-figures ``panel_grid`` of three
rows — title band / Pacific-centred PlateCarree map / legend row — with explicit
geometry (row heights in inches; the map row's height follows the extent aspect)
and the map cell's box guard. The basemap is offline: water is the axes
background and land is the simplified LBD admin-0 layer with white borders, so
no Natural Earth download is needed.

Basin boxes are drawn with cartopy's ``add_geometries`` (dateline-safe). The
original's zero-width tc-risk SP rectangle (``lon0 == lon1 == 180``) is dropped:
it is a degenerate ring and the reason the notebook cell raised
``GEOSException: Points of LinearRing do not form a closed linestring`` on draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click

from idd_tc_mortality.viz.storm_three_panel import _check_map_box

ADMIN0_SIMPLIFIED = Path(
    "/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29/lbd_standard_admin_0_simplified.shp"
)
OUT_DIR = Path("/mnt/team/idd/pub/idd_tc_mortality/05-figures/working/basin_definitions")
EXTENT = (-180.0, 180.0, -60.0, 80.0)
TITLE = "IBTrACS v4 (solid) vs tc-risk (dashed) basin definitions — all basins"
SPLIT_LON = 135.0


@dataclass(frozen=True)
class BasinBox:
    name: str
    lon0: float
    lon1: float
    lat0: float
    lat1: float
    color: str

    def is_degenerate(self) -> bool:
        return not (self.lon1 > self.lon0 and self.lat1 > self.lat0)


# lons in -180..180; wrapping basins are two boxes
IBTRACS_BOXES = [
    BasinBox("IBTrACS NA", -100, 30, 0, 60, "#185FA5"),
    BasinBox("IBTrACS EP", -180, -100, 0, 60, "#3B6D11"),
    BasinBox("IBTrACS WP", 100, 180, 0, 60, "#854F0B"),
    BasinBox("IBTrACS NI", 30, 100, 0, 50, "#993C1D"),
    BasinBox("IBTrACS SI", 10, 135, -45, 0, "#0C447C"),
    BasinBox("IBTrACS SP", 135, 180, -45, 0, "#0F6E56"),
    BasinBox("IBTrACS SP (east)", -180, -70, -45, 0, "#0F6E56"),
    BasinBox("IBTrACS SA", -70, 10, -45, 0, "#534AB7"),
]
TCRISK_BOXES = [
    BasinBox("tc-risk NA", -100, 0, 0, 60, "#185FA5"),
    BasinBox("tc-risk EP", -180, -100, 0, 60, "#3B6D11"),
    BasinBox("tc-risk WP", 100, 180, 0, 60, "#854F0B"),
    BasinBox("tc-risk NI", 30, 100, 0, 50, "#993C1D"),
    BasinBox("tc-risk SI", 20, 100, -45, 0, "#D85A30"),
    BasinBox("tc-risk AU", 100, 180, -45, 0, "#BA7517"),
    BasinBox("tc-risk SP", -180, -110, -45, 0, "#7F77DD"),  # 180E-250E
]
# (lon, lat, text, colour)
NH_LABELS = [(-40, 30, "NA", "#185FA5"), (-145, 30, "EP", "#3B6D11"),
             (140, 30, "WP", "#854F0B"), (65, 25, "NI", "#993C1D")]
SH_LABELS = [(72, -25, "SI\nIBTrACS", "#0C447C"), (57, -38, "SI\ntc-risk", "#D85A30"),
             (140, -25, "SP\nIBTrACS", "#0F6E56"), (140, -38, "AU\ntc-risk", "#BA7517"),
             (-150, -25, "SP\nIBTrACS", "#0F6E56"), (-145, -38, "SP\ntc-risk", "#7F77DD"),
             (-30, -25, "SA\nIBTrACS", "#534AB7")]
BASIN_LEGEND = [("#185FA5", "NA"), ("#3B6D11", "EP"), ("#854F0B", "WP"), ("#993C1D", "NI"),
                ("#0C447C", "SI (IBTrACS)"), ("#D85A30", "SI (tc-risk)"), ("#0F6E56", "SP (IBTrACS)"),
                ("#BA7517", "AU (tc-risk)"), ("#7F77DD", "SP (tc-risk)"), ("#534AB7", "SA (IBTrACS only)")]


# --- painters (draw on a GeoAxes, return it) ------------------------------------------
def boxes_painter(ax, _data=None, *, boxes, solid=True, fill_alpha=0.12, linewidth=2.0, zorder=3):
    """Lon/lat rectangles: filled + solid outline (IBTrACS) or dashed outline only (tc-risk)."""
    import cartopy.crs as ccrs
    from shapely.geometry import box as shapely_box

    pc = ccrs.PlateCarree()
    for b in boxes:
        if b.is_degenerate():
            raise ValueError(f"degenerate basin box {b.name!r}: {b}")
        geom = [shapely_box(b.lon0, b.lat0, b.lon1, b.lat1)]
        if solid:
            ax.add_geometries(geom, crs=pc, facecolor=b.color, alpha=fill_alpha, edgecolor="none", zorder=zorder)
        ax.add_geometries(geom, crs=pc, facecolor="none", edgecolor=b.color, linewidth=linewidth,
                          linestyle="-" if solid else "--", zorder=zorder + 1)
    return ax


def labels_painter(ax, _data=None, *, labels, fontsize=9, zorder=6):
    """Bold coloured text at (lon, lat)."""
    import cartopy.crs as ccrs

    for lon, lat, text, color in labels:
        ax.text(lon, lat, text, transform=ccrs.PlateCarree(), ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color, zorder=zorder, linespacing=1.4)
    return ax


def split_line_painter(ax, _data=None, *, lon=SPLIT_LON, lat0=-45.0, lat1=0.0, color="#888780",
                       fontsize=7, zorder=5):
    """Dotted meridian marking the IBTrACS SI/SP split, labelled below."""
    import cartopy.crs as ccrs

    pc = ccrs.PlateCarree()
    ax.plot([lon, lon], [lat0, lat1], transform=pc, color=color, lw=1.5, ls=":", zorder=zorder, alpha=0.9)
    ax.text(lon, lat0 - 2, f"{lon:g}°E", transform=pc, fontsize=fontsize, color=color, ha="center", zorder=zorder + 1)
    return ax


def basin_map_painter(ax, _data=None, *, land_gdf=None, extent=EXTENT, water_color="#dbeaf4",
                      land_color="#d8d5cc", border_color="white", label_fontsizes=(10, 8)):
    """ONE complete cell: offline basemap + both box sets + split line + labels."""
    from idd_figures.lib.painters.maps import map_cell_painter

    map_cell_painter(
        ax, None, extent=tuple(extent), base_admin_gdf=land_gdf, ocean="fill",
        ocean_style={"color": water_color, "alpha": 1.0},
        backdrop_style={"color": land_color, "edgecolor": border_color, "linewidth": 0.3},
    )
    boxes_painter(ax, boxes=IBTRACS_BOXES, solid=True, zorder=3)
    boxes_painter(ax, boxes=TCRISK_BOXES, solid=False, linewidth=1.8, zorder=4)
    split_line_painter(ax)
    labels_painter(ax, labels=NH_LABELS, fontsize=label_fontsizes[0])
    labels_painter(ax, labels=SH_LABELS, fontsize=label_fontsizes[1])
    return ax


def legend_handles():
    """Proxy artists: line-style keys, then one swatch per basin colour."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="#888", alpha=0.3, edgecolor="#444", lw=1.5, label="IBTrACS basins (solid)"),
        Patch(facecolor="none", edgecolor="#444", linestyle="--", lw=1.5, label="tc-risk basins (dashed)"),
        Line2D([0], [0], color="#888780", lw=1.5, ls=":", label=f"{SPLIT_LON:g}°E  IBTrACS SI/SP split"),
    ]
    handles += [Patch(facecolor=c, alpha=0.6, label=lab) for c, lab in BASIN_LEGEND]
    return handles


# --- layout -----------------------------------------------------------------------------
def build_basin_map(land_gdf, *, fig_width=16.0, central_longitude=180.0, title=TITLE, title_h=0.5,
                    title_gap_h=0.35, legend_gap_h=0.35, legend_h=0.9, legend_ncol=7, title_fontsize=13,
                    legend_fontsize=9, side_margin=0.02, gridlines=True, **painter_kw):
    """Title band / gap / Pacific-centred map / gap / legend row as one explicit panel_grid.

    All heights are inches. The two gaps are their own spacer rows (blank cells), so
    the title-to-map and map-to-legend distances are independent knobs; the map row's
    height follows the extent aspect and the figure height is derived from the rest.
    ``**painter_kw`` goes to :func:`basin_map_painter` (water/land/border colours,
    label font sizes). Returns the Figure; named axes: title, map, legend.
    """
    import cartopy.crs as ccrs
    from idd_figures.lib.layouts.geometry import extent_aspect, panel_width
    from idd_figures.lib.layouts.grids import blank, cell, grid, label, legend, paint, panel_grid

    margins = {"left": side_margin, "right": 1.0 - side_margin, "top": 0.995, "bottom": 0.005}
    aspect = extent_aspect(EXTENT)
    map_h = panel_width(fig_width, margins=margins, ncols=1) * aspect
    heights = [title_h, max(title_gap_h, 1e-6), map_h, max(legend_gap_h, 1e-6), legend_h]
    fig_h = sum(heights) / (margins["top"] - margins["bottom"])  # hspace=0: rows realise these inches exactly

    handles = legend_handles()
    spec = grid(
        (5, 1),
        [
            cell((0, 0), label(title, fontsize=title_fontsize), name="title"),
            cell((1, 0), blank()),
            cell((2, 0), paint(basin_map_painter, None, land_gdf=land_gdf, **painter_kw),
                 projection=ccrs.PlateCarree(central_longitude=central_longitude), name="map"),
            cell((3, 0), blank()),
            cell((4, 0), legend((handles, [h.get_label() for h in handles]), ncol=legend_ncol,
                                fontsize=legend_fontsize, frameon=False), name="legend"),
        ],
        height_ratios=heights, margins=margins, hspace=0.0,
    )
    fig = panel_grid(spec, figsize=(fig_width, fig_h))
    ax = fig.axes_by_name["map"]
    ax.set_aspect(1.0, adjustable="box")
    _check_map_box(fig, ax, aspect)
    if gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
    return fig


@click.command()
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_DIR, show_default=True)
@click.option("--fig-width", default=16.0, show_default=True)
@click.option("--dpi", default=150, show_default=True)
@click.option("--land", type=click.Path(path_type=Path), default=ADMIN0_SIMPLIFIED, show_default=True,
              help="Land polygons (any geopandas-readable file).")
def main(out_dir, fig_width, dpi, land):
    """IBTrACS vs tc-risk basin-definition map (PNG + PDF) on the idd-figures map stack."""
    import geopandas as gpd
    import matplotlib
    from idd_figures.lib.io import save_figure

    matplotlib.use("Agg")
    fig = build_basin_map(gpd.read_file(land), fig_width=fig_width)
    for p in save_figure(fig, Path(out_dir) / "basin_comparison_global.png", formats=["png", "pdf"], dpi=dpi):
        click.echo(f"wrote {p}")


if __name__ == "__main__":
    main()
