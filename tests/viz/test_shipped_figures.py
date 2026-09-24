"""Summaries and figure builders for the shipped-blend figures, on synthetic draws."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.viz import shipped_figures as sf

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def draws():
    rng = np.random.default_rng(0)
    rows = []
    for loc in (1, 4, 64):
        for sc, years in (("historical", range(2000, 2015)), ("ssp126", range(2015, 2101)),
                          ("ssp245", range(2015, 2101)), ("ssp585", range(2015, 2101))):
            for d in range(20):
                base = 1000 if loc == 1 else 300
                for y in years:
                    rows.append((d, sc, y, loc, base * rng.lognormal(0, 0.3)))
    return pd.DataFrame(rows, columns=["storm_draw", "ssp_scenario", "year_id", "location_id", "deaths"])


def test_summarize_shape_and_order(draws):
    s = sf.summarize(draws, 1)
    assert list(s.columns) == ["ssp_scenario", "year_id", "mean", "lower", "upper"]
    assert (s["lower"] <= s["mean"]).all() and (s["mean"] <= s["upper"]).all()
    assert len(s) == 15 + 3 * 86
    assert s.iloc[0]["ssp_scenario"] == "historical" and s.iloc[0]["year_id"] == 2000


def test_cumulative_by_draw_and_paired_difference(draws):
    cum = sf.cumulative_by_draw(draws, 1, 2023, 2100)
    assert set(cum["ssp_scenario"]) == set(sf.SCENARIOS)  # historical excluded
    assert len(cum) == 3 * 20
    one = draws[(draws.location_id == 1) & (draws.ssp_scenario == "ssp126") & (draws.storm_draw == 0)
                & draws.year_id.between(2023, 2100)]["deaths"].sum()
    assert cum[(cum.ssp_scenario == "ssp126") & (cum.storm_draw == 0)]["deaths"].iloc[0] == pytest.approx(one)
    diff = sf.paired_difference(cum)
    assert len(diff) == 20
    w = cum.pivot(index="storm_draw", columns="ssp_scenario", values="deaths")
    assert diff[3] == pytest.approx(w.loc[3, "ssp585"] - w.loc[3, "ssp126"])


def test_plot_global_2x2(draws):
    import matplotlib.pyplot as plt

    obs = pd.DataFrame({"location_id": 1, "year": range(2000, 2024), "deaths": 500.0})
    fig = sf.plot_global_2x2(sf.summarize(draws, 1), obs, start_year=2000, zoom_year=2020)
    axes = [a for a in fig.axes if a.get_xlabel() == "Year"]
    assert len(axes) == 4
    assert axes[2].get_yscale() == "log" and axes[0].get_yscale() == "linear"
    labels = [t.get_text() for t in fig.legends[0].texts]
    assert labels == ["Observed Deaths", "Historical", "RCP2.6", "RCP4.5", "RCP8.5"]
    assert axes[0].get_xlim()[0] < 2000 < axes[1].get_xlim()[0]  # zoom column starts later
    plt.close(fig)


def test_plot_sr_grid(draws):
    import matplotlib.pyplot as plt

    fig = sf.plot_sr_grid({4: sf.summarize(draws, 4), 64: sf.summarize(draws, 64)}, {4: "A", 64: "B"},
                          start_year=2020, ncols=2)
    titled = [a for a in fig.axes if a.get_title()]
    assert [a.get_title() for a in titled] == ["A", "B"]
    assert [t.get_text() for t in fig.legends[0].texts] == ["RCP2.6", "RCP4.5", "RCP8.5"]  # no historical from 2020
    plt.close(fig)


def test_plot_cumulative_boxes(draws):
    import matplotlib.pyplot as plt

    cum = sf.cumulative_by_draw(draws, 1, 2023, 2100)
    fig = sf.plot_cumulative_boxes(cum, start=2023, end=2100)
    ax1, ax2 = fig.axes[:2]
    assert [t.get_text() for t in ax1.get_xticklabels()] == ["RCP2.6", "RCP4.5", "RCP8.5"]
    assert "100,000s" in ax1.get_ylabel()
    assert len(ax2.lines) >= 1  # the zero line
    plt.close(fig)


def test_cli_help():
    from click.testing import CliRunner

    r = CliRunner().invoke(sf.main, ["--help"])
    assert r.exit_code == 0 and "--cum-start" in r.output and "--tag" in r.output
