"""Basin-definition map: box data integrity, painters on a GeoAxes, explicit layout."""

import matplotlib
import pytest

from idd_tc_mortality.viz import basin_map as bm

matplotlib.use("Agg")


def _geoaxes(central_longitude=180.0):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=central_longitude))
    ax.set_global()
    return fig, ax


def test_no_degenerate_boxes():
    for b in bm.IBTRACS_BOXES + bm.TCRISK_BOXES:
        assert not b.is_degenerate(), b
    assert len(bm.TCRISK_BOXES) == 7  # the zero-width SP entry of the original is gone


def test_boxes_painter_rejects_degenerate_box():
    import matplotlib.pyplot as plt

    fig, ax = _geoaxes()
    with pytest.raises(ValueError, match="degenerate"):
        bm.boxes_painter(ax, boxes=[bm.BasinBox("bad", 180, 180, -45, 0, "#000")])
    plt.close(fig)


def test_boxes_painter_adds_two_artists_per_solid_box_one_per_dashed():
    import matplotlib.pyplot as plt

    fig, ax = _geoaxes()
    n0 = len(ax.get_children())
    bm.boxes_painter(ax, boxes=bm.IBTRACS_BOXES, solid=True)
    n1 = len(ax.get_children())
    bm.boxes_painter(ax, boxes=bm.TCRISK_BOXES, solid=False)
    n2 = len(ax.get_children())
    assert n1 - n0 == 2 * len(bm.IBTRACS_BOXES)
    assert n2 - n1 == len(bm.TCRISK_BOXES)
    plt.close(fig)


def test_labels_and_split_line_painters():
    import matplotlib.pyplot as plt

    fig, ax = _geoaxes()
    bm.labels_painter(ax, labels=bm.NH_LABELS + bm.SH_LABELS)
    assert len(ax.texts) == len(bm.NH_LABELS) + len(bm.SH_LABELS)
    bm.split_line_painter(ax)
    assert len(ax.lines) == 1 and len(ax.texts) == len(bm.NH_LABELS) + len(bm.SH_LABELS) + 1
    plt.close(fig)


def test_legend_handles_cover_keys_and_basins():
    labels = [h.get_label() for h in bm.legend_handles()]
    assert labels[:2] == ["IBTrACS basins (solid)", "tc-risk basins (dashed)"]
    assert len(labels) == 3 + len(bm.BASIN_LEGEND)


def test_build_basin_map_layout_offline():
    """Synthetic continents as land: no shapefile, no network; explicit geometry holds."""
    import matplotlib.pyplot as plt
    from idd_figures.lib.geo_fixture import make_synthetic_continents

    fig = bm.build_basin_map(make_synthetic_continents(), fig_width=12.0, gridlines=False)
    assert {"title", "map", "legend"} <= set(fig.axes_by_name)
    w, h = fig.get_size_inches()
    assert w == 12.0
    pos = fig.axes_by_name["map"].get_position(original=True)
    assert (pos.height * h) / (pos.width * w) == pytest.approx(140 / 360, rel=0.01)
    leg = fig.axes_by_name["legend"].get_legend()
    assert leg is not None and len(leg.texts) == 3 + len(bm.BASIN_LEGEND)
    assert fig.axes_by_name["title"].texts[0].get_text() == bm.TITLE
    fig.canvas.draw()  # the original failed only at draw time; make sure this one renders
    plt.close(fig)


def test_cli_help():
    from click.testing import CliRunner

    r = CliRunner().invoke(bm.main, ["--help"])
    assert r.exit_code == 0 and "--land" in r.output


def test_gaps_are_independent_inches():
    import matplotlib.pyplot as plt
    from idd_figures.lib.geo_fixture import make_synthetic_continents

    fig = bm.build_basin_map(make_synthetic_continents(), fig_width=12.0, title_h=0.5, title_gap_h=0.2,
                             legend_gap_h=0.8, legend_h=0.9, gridlines=False)
    h = fig.get_size_inches()[1]
    top = fig.axes_by_name["title"].get_position(original=True)
    mp = fig.axes_by_name["map"].get_position(original=True)
    lg = fig.axes_by_name["legend"].get_position(original=True)
    assert (top.y0 - mp.y1) * h == pytest.approx(0.2, abs=0.01)
    assert (mp.y0 - lg.y1) * h == pytest.approx(0.8, abs=0.01)
    assert top.height * h == pytest.approx(0.5, abs=0.01) and lg.height * h == pytest.approx(0.9, abs=0.01)
    plt.close(fig)
