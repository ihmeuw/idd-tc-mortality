"""Unit tests for the three-panel storm figure derivations and assembly.

Pure functions are tested exactly against the notebook's formulas; raster I/O
helpers that need rasterio are skipped when it is not installed.
"""

import math

import matplotlib
import numpy as np
import pytest
from click.testing import CliRunner

from idd_tc_mortality.viz import storm_three_panel as stp

matplotlib.use("Agg")


# --- rr_from_intensity ------------------------------------------------------------
def test_rr_below_curve_is_zero_above_is_clamped_between_is_linear():
    knots = np.array([10.0, 20.0, 30.0])
    rr = np.array([1.1, 1.2, 1.4])
    ms = knots * stp.KNOTS_TO_MS
    x = np.array([ms[0] - 1.0, ms[2] + 5.0, (ms[0] + ms[1]) / 2, np.nan])
    out = stp.rr_from_intensity(x, knots, rr)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(1.4)
    assert out[2] == pytest.approx(1.15)
    assert np.isnan(out[3])


def test_rr_handles_unsorted_curve():
    knots = np.array([30.0, 10.0, 20.0])
    rr = np.array([1.4, 1.1, 1.2])
    x = np.array([20.0 * stp.KNOTS_TO_MS])
    assert stp.rr_from_intensity(x, knots, rr)[0] == pytest.approx(1.2)


# --- paf_from_rr -----------------------------------------------------------------
def test_paf_formula_and_masking():
    rr = np.array([2.0, 1.0, 0.0, np.nan, -1.0])
    out = stp.paf_from_rr(rr, impact_days=365.0)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(0.0)
    assert np.isnan(out[2]) and np.isnan(out[3]) and np.isnan(out[4])


def test_paf_scales_with_impact_days():
    out = stp.paf_from_rr(np.array([2.0]), impact_days=20.0)
    assert out[0] == pytest.approx(0.5 * 20.0 / 365.0)


# --- pixel_area_km2 --------------------------------------------------------------
def test_pixel_area_equator_and_cosine_falloff():
    a = stp.pixel_area_km2(np.array([0.0, 60.0]), res_deg=0.01)
    side = math.radians(0.01) * stp.EARTH_RADIUS_KM
    assert a[0] == pytest.approx(side * side)
    assert a[1] == pytest.approx(a[0] * 0.5)


# --- nearest_resample ------------------------------------------------------------
def test_nearest_resample_maps_centres_and_nans_outside():
    src = np.array([[1.0, 2.0], [3.0, 4.0]])
    src_lat = np.array([1.5, 0.5])  # descending like the stage-1 grids
    src_lon = np.array([0.5, 1.5])
    dst_lat = np.array([1.75, 1.25, 0.75, 0.25, -0.75])  # last is outside
    dst_lon = np.array([0.25, 0.75, 1.25, 1.75, 2.75])  # last is outside
    out = stp.nearest_resample(src, src_lat, src_lon, dst_lat, dst_lon)
    assert out.shape == (5, 5)
    np.testing.assert_array_equal(out[0, :4], [1.0, 1.0, 2.0, 2.0])
    np.testing.assert_array_equal(out[3, :4], [3.0, 3.0, 4.0, 4.0])
    assert np.isnan(out[4, :]).all()
    assert np.isnan(out[:, 4]).all()


# --- extents ---------------------------------------------------------------------
def test_storm_extent_pads_bbox_of_positive_pixels():
    vals = np.full((3, 3), np.nan)
    vals[1, 1] = 5.0
    vals[2, 2] = 0.0  # not > 0, ignored
    lat = np.array([2.5, 1.5, 0.5])
    lon = np.array([0.5, 1.5, 2.5])
    assert stp.storm_extent(vals, lat, lon, pad_deg=0.5) == [1.0, 2.0, 1.0, 2.0]


def test_storm_extent_raises_without_affected_pixels():
    with pytest.raises(ValueError, match="no affected"):
        stp.storm_extent(np.zeros((2, 2)), np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.1)


def test_admin_extent_pads_five_percent():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 10.0, 20.0)], crs="EPSG:4326")
    assert stp.admin_extent(gdf) == pytest.approx([-0.5, 10.5, -1.0, 21.0])


# --- masking helpers -------------------------------------------------------------
def test_positive_or_nan():
    out = stp.positive_or_nan(np.array([1.0, 0.0, -2.0, np.nan, np.inf]))
    assert out[0] == 1.0
    assert np.isnan(out[1:]).all()


def test_mask_outside_geometry():
    pytest.importorskip("rasterio")
    from shapely.geometry import box

    arr = np.ones((4, 4))
    edges = [0.0, 4.0, 0.0, 4.0]
    out = stp.mask_outside(arr, box(0.0, 0.0, 2.0, 2.0), edges)  # lower-left quadrant
    assert np.isfinite(out[2:, :2]).all()
    assert np.isnan(out[:2, 2:]).all()


# --- colour scaling ----------------------------------------------------------------
def test_log_norm_follows_hubei_rule():
    v = np.concatenate([np.array([np.nan, 0.0, -1.0]), np.logspace(-2, 3, 1000)])
    n = stp.log_norm(v)
    assert n.vmin == pytest.approx(0.1)
    assert n.vmax == pytest.approx(np.percentile(np.logspace(-2, 3, 1000), 99))


def test_log_norm_empty_and_degenerate():
    assert (stp.log_norm(np.array([np.nan])).vmin, stp.log_norm(np.array([np.nan])).vmax) == (0.1, 1.0)
    n = stp.log_norm(np.array([5.0, 5.0]))
    assert n.vmin == 5.0 and n.vmax == 50.0


def test_decade_ticks():
    from matplotlib.colors import LogNorm

    assert stp.decade_ticks(LogNorm(0.5, 2000.0)) == [1.0, 10.0, 100.0, 1000.0]
    assert stp.decade_ticks(LogNorm(2.0, 8.0)) is None


# --- figure assembly ---------------------------------------------------------------
def test_figure_geometry_realises_declared_inches():
    geo = stp.figure_geometry(
        18.0, 1.25, wspace=0.04, title_h=1.0, gap_h=0.25, cbar_h=0.9, bottom_pad_h=0.1
    )
    usable = 18.0 * 0.96
    panel_w = usable / (3 + 0.04 * 2)
    assert geo["map_height"] == pytest.approx(panel_w * 1.25)
    assert geo["fig_height"] == pytest.approx(geo["map_height"] + 0.9 + 0.25 + 1.0 + 0.1)
    assert geo["margins"]["top"] == pytest.approx(1 - 1.0 / geo["fig_height"])
    assert geo["margins"]["bottom"] == pytest.approx(0.1 / geo["fig_height"])
    # the gap is hspace charged against the mean row height
    assert geo["hspace"] * (geo["map_height"] + 0.9) / 2 == pytest.approx(0.25)


def _synthetic_panel_data():
    rng = np.random.default_rng(0)
    rr = np.full((50, 40), np.nan)
    rr[10:40, 5:35] = rng.uniform(1.09, 1.15, size=(30, 30))
    density = rng.lognormal(3.0, 1.5, size=(50, 40))
    py = np.where(np.isfinite(rr), density * 0.01, np.nan)
    extent = [0.0, 4.0, 0.0, 5.0]
    return stp.PanelData(
        rr=rr, density=density, person_years=py, raster_extent=extent, extent=extent,
        titles=("a\nb", "c\nd", "e\nf"),
    )


def test_build_figure_has_three_maps_three_bars_and_declared_geometry():
    import matplotlib.pyplot as plt

    data = _synthetic_panel_data()
    fig = stp.build_figure(data, fig_width=18.0, gap_h=0.25, title_h=1.0, cbar_h=0.9)
    names = set(fig.axes_by_name)
    assert {"rr", "density", "person_years", "cbar:rr", "cbar:density", "cbar:person_years"} <= names
    geo = stp.figure_geometry(18.0, 1.25)
    w, h = fig.get_size_inches()
    assert (w, h) == pytest.approx((18.0, geo["fig_height"]))
    # map box aspect equals the extent aspect (the guard would have raised otherwise)
    pos = fig.axes_by_name["rr"].get_position(original=True)
    assert (pos.height * h) / (pos.width * w) == pytest.approx(1.25, rel=0.01)
    # the map-to-bar gap is the declared inches
    map_bottom = fig.axes_by_name["rr"].get_position(original=True).y0 * h
    bar_top = fig.axes_by_name["cbar:rr"].get_position(original=True).y1 * h
    assert map_bottom - bar_top == pytest.approx(0.25, abs=0.01)
    plt.close(fig)


def test_build_figure_style_knobs_reach_the_artists():
    import matplotlib.pyplot as plt

    fig = stp.build_figure(
        _synthetic_panel_data(), title_fontsize=22, water_color="#0000ff", water_alpha=1.0
    )
    ax = fig.axes_by_name["rr"]
    assert ax.title.get_fontsize() == 22
    assert ax.get_facecolor()[:3] == pytest.approx((0.0, 0.0, 1.0))
    plt.close(fig)


def test_prepare_arrays_plumbing(monkeypatch):
    """Synthetic loaders: totals and titles follow the documented formulas."""
    import xarray as xr

    inten = xr.DataArray(
        np.array([[30.0, np.nan], [np.nan, 40.0]]),
        coords={"lat": [1.5, 0.5], "lon": [0.5, 1.5]},
        dims=("lat", "lon"),
        attrs={"storm_name": "KATRINA", "start_date": "2005-08-23T18:00:00"},
    )
    curve = __import__("pandas").DataFrame({"windspeed": [10.0, 100.0], "mean": [1.0, 2.0]})
    lat_p = np.array([1.75, 1.25, 0.75, 0.25])
    lon_p = np.array([0.25, 0.75, 1.25, 1.75])

    monkeypatch.setattr(stp, "load_intensity", lambda *a, **k: inten)
    monkeypatch.setattr(stp, "load_rr_curve", lambda *a, **k: curve)
    monkeypatch.setattr(
        stp, "read_population_window",
        lambda *a, **k: (np.full((4, 4), 10.0), lat_p, lon_p, [0.0, 2.0, 0.0, 2.0]),
    )
    monkeypatch.setattr(stp, "load_admin1", lambda *a, **k: None)

    data = stp.prepare_arrays(storm_id="2005236N23285", impact_days=365.0, pad_deg=0.0)
    ws = np.array([10.0, 100.0]) * stp.KNOTS_TO_MS
    rr30 = 1 + (30 - ws[0]) / (ws[1] - ws[0])
    rr40 = 1 + (40 - ws[0]) / (ws[1] - ws[0])
    expected_py = 4 * 10 * (rr30 - 1) / rr30 + 4 * 10 * (rr40 - 1) / rr40
    assert data.totals["population"] == pytest.approx(160.0)
    assert data.totals["affected_population"] == pytest.approx(80.0)
    assert data.totals["person_years"] == pytest.approx(expected_py)
    assert data.extent == [0.5, 1.5, 0.5, 1.5]
    assert data.titles[0].startswith("Relative risk, respiratory\nKatrina (August 23, 2005)")
    assert "160 people in extent" in data.titles[1]
    assert data.density.shape == (4, 4)


def test_cli_help_lists_options():
    result = CliRunner().invoke(stp.main, ["--help"])
    assert result.exit_code == 0
    for opt in ("--storm-id", "--outcome", "--impact-days", "--adm1-name", "--out"):
        assert opt in result.output


# --- generic row builder + two-panel exposure figure -------------------------------------
def test_linear_norm_spans_finite_range():
    n = stp._linear_norm(np.array([[np.nan, 2.0], [5.0, 3.0]]))
    assert (n.vmin, n.vmax) == (2.0, 5.0)


def test_build_panel_row_two_panels_uses_ncols_geometry():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    a = rng.uniform(17, 60, size=(30, 20))
    b = rng.uniform(1, 30, size=(30, 20))
    extent = [0.0, 2.0, 0.0, 3.0]
    specs = [
        stp.PanelSpec(a, "viridis", "A", "m/s", "one"),
        stp.PanelSpec(b, "magma", "B", "hours", "two"),
    ]
    fig = stp.build_panel_row(specs, raster_extent=extent, extent=extent, fig_width=13.0)
    assert {"one", "two", "cbar:one", "cbar:two"} <= set(fig.axes_by_name)
    geo = stp.figure_geometry(13.0, 1.5, ncols=2)
    assert fig.get_size_inches()[1] == pytest.approx(geo["fig_height"])
    w, h = fig.get_size_inches()
    pos = fig.axes_by_name["two"].get_position(original=True)
    assert (pos.height * h) / (pos.width * w) == pytest.approx(1.5, rel=0.01)
    plt.close(fig)


def test_grid_edges_from_centres():
    edges = stp.grid_edges(np.array([2.5, 1.5, 0.5]), np.array([10.0, 11.0]))
    assert edges == pytest.approx([9.5, 11.5, 0.0, 3.0])


def _fake_exposure_inputs(monkeypatch, *, flip_lat=False):
    import xarray as xr

    lat = np.array([1.5, 0.5])
    inten_vals = np.array([[30.0, np.nan], [np.nan, 40.0]])
    hours_vals = np.array([[6.0, np.nan], [np.nan, 12.0]])
    if flip_lat:
        lat, inten_vals, hours_vals = lat[::-1], inten_vals[::-1], hours_vals[::-1]
    coords = {"lat": lat, "lon": [0.5, 1.5]}
    inten = xr.DataArray(
        inten_vals, coords=coords, dims=("lat", "lon"),
        attrs={"storm_name": "KATRINA", "start_date": "2005-08-23T18:00:00"},
    )
    hours = xr.DataArray(hours_vals, coords=coords, dims=("lat", "lon"), attrs={"wind_threshold_m_s": 17.0})
    monkeypatch.setattr(stp, "load_intensity", lambda *a, **k: inten)
    monkeypatch.setattr(stp, "load_exposure_hours", lambda *a, **k: hours)
    monkeypatch.setattr(stp, "load_admin1", lambda *a, **k: None)


@pytest.mark.parametrize("flip_lat", [False, True])
def test_prepare_exposure_arrays_plumbing(monkeypatch, flip_lat):
    _fake_exposure_inputs(monkeypatch, flip_lat=flip_lat)
    data = stp.prepare_exposure_arrays(storm_id="2005236N23285", pad_deg=0.0)
    # north-first regardless of the stored orientation
    assert data.intensity[0, 0] == 30.0 and data.exposure_hours[1, 1] == 12.0
    assert np.isnan(data.intensity[0, 1])
    assert data.raster_extent == pytest.approx([0.0, 2.0, 0.0, 2.0])
    assert data.extent == [0.5, 1.5, 0.5, 1.5]
    assert data.titles[0].startswith("Max intensity, windspeed > 17 m/s\nKatrina (August 23, 2005)")
    assert data.titles[1].startswith("Exposure hours, windspeed > 17 m/s")


def test_prepare_exposure_arrays_rejects_grid_mismatch(monkeypatch):
    import xarray as xr

    _fake_exposure_inputs(monkeypatch)
    bad = xr.DataArray(np.ones((3, 2)), coords={"lat": [2.5, 1.5, 0.5], "lon": [0.5, 1.5]}, dims=("lat", "lon"))
    monkeypatch.setattr(stp, "load_exposure_hours", lambda *a, **k: bad)
    with pytest.raises(ValueError, match="grids differ"):
        stp.prepare_exposure_arrays(storm_id="2005236N23285")


def test_build_exposure_figure_names_and_knobs():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2)
    extent = [0.0, 4.0, 0.0, 5.0]
    data = stp.ExposureData(
        intensity=rng.uniform(17, 60, size=(50, 40)),
        exposure_hours=rng.uniform(1, 30, size=(50, 40)),
        raster_extent=extent, extent=extent, titles=("i\nx", "h\ny"),
    )
    fig = stp.build_exposure_figure(data, title_fontsize=20, hours_cmap="inferno")
    assert {"intensity", "exposure_hours", "cbar:intensity", "cbar:exposure_hours"} <= set(fig.axes_by_name)
    assert fig.axes_by_name["intensity"].title.get_fontsize() == 20
    assert fig.get_size_inches()[0] == 13.0
    plt.close(fig)
