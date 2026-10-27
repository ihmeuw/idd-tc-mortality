"""Fit-draw averaging, blend + rake + rollup, observed roll-up."""

import pandas as pd
import pytest

from idd_tc_mortality.viz.cmip6_ibtracs_comparison import mortality as mo


class _FakeModel:
    def __init__(self, scale):
        self.scale = scale

    def predict(self, df, *, outcome_draw, expected_bernoulli, seed):
        assert not outcome_draw and expected_bernoulli
        return pd.DataFrame({"deaths": df["exposed"].to_numpy() * self.scale})


@pytest.fixture
def hier():
    # Global 1 > SR 4 > region 5 > China 6 > two provinces 491, 492 ; SR 4 > region 5 > Fiji 22
    return pd.DataFrame({
        "location_id": [1, 4, 5, 6, 491, 492, 22],
        "level": [0, 1, 2, 3, 4, 4, 3],
        "parent_id": [0, 1, 4, 5, 6, 6, 5],
        "path_to_top_parent": ["1", "1,4", "1,4,5", "1,4,5,6", "1,4,5,6,491", "1,4,5,6,492", "1,4,5,22"],
        "region_id": [-1, -1, 5, 5, 5, 5, 5],
        "super_region_id": [-1, 4, 4, 4, 4, 4, 4],
        "location_name": ["Global", "SR", "R", "China", "P1", "P2", "Fiji"],
    })


def test_predict_deaths_average_and_per_draw():
    sl = pd.DataFrame({"draw": [0, 0, 1], "basin": ["WP"] * 3, "location_id": [6, 6, 22],
                       "year": [2000] * 3, "exposed": [1.0, 2.0, 4.0]})
    models = {1: _FakeModel(1.0), 2: _FakeModel(3.0)}
    avg = mo.predict_deaths(sl, models, mo.KEYS, average=True).set_index(["draw", "location_id"])["deaths"]
    assert avg.loc[(0, 6)] == pytest.approx(3.0 * 2.0)   # (1+3)/2 × (1+2)
    assert avg.loc[(1, 22)] == pytest.approx(8.0)
    per = mo.predict_deaths(sl, models, mo.KEYS, average=False)
    assert list(per.columns) == ["fit_draw"] + mo.KEYS + ["deaths"]
    assert len(per) == 4 and per.loc[per.fit_draw == 2, "deaths"].sum() == pytest.approx(21.0)


def test_blend_rollup_uses_a1_for_subnationalized_and_rakes(hier):
    a0 = pd.DataFrame({"draw": [0, 0], "basin": ["WP", "WP"], "location_id": [6, 22], "year": [2000, 2000],
                       "deaths": [100.0, 10.0]})
    a1 = pd.DataFrame({"draw": [0, 0], "basin": ["WP", "WP"], "location_id": [491, 492], "year": [2000, 2000],
                       "deaths": [30.0, 20.0]})
    out = mo.blend_rollup({"a0": a0, "a1": a1}, hier, pd.Series({4: 0.5}))
    t = out.set_index("location_id")
    assert t.loc[6, "deaths_unadj"] == 50.0           # A1 provinces replace the A0 China row
    assert t.loc[22, "deaths_unadj"] == 10.0
    assert t.loc[1, "deaths_unadj"] == 60.0 and t.loc[4, "deaths_unadj"] == 60.0
    assert t.loc[1, "deaths_adj"] == 30.0 and t.loc[6, "deaths_adj"] == 25.0
    assert set(out.columns) == set(mo.KEYS + ["deaths_unadj", "deaths_adj"])


def test_observed_deaths_rolls_up(hier, tmp_path):
    d = pd.DataFrame({"basin": ["WP", "WP", "NA"], "location_id": [6, 22, 22], "year": [2000, 2000, 2001],
                      "deaths": [5, 7, 1], "storm_id": ["a", "b", "c"]})
    p = tmp_path / "input.parquet"
    d.to_parquet(p, index=False)
    out = mo.observed_deaths(hier, p).set_index(["basin", "location_id", "year"])["deaths"]
    assert out.loc[("WP", 1, 2000)] == 12 and out.loc[("NA", 1, 2001)] == 1 and out.loc[("WP", 5, 2000)] == 12


def test_prepare_requires_rows():
    ctx = {"is_island": pd.DataFrame({"location_id": [6], "is_island": [0]}),
           "sdi": pd.DataFrame({"location_id": [6], "year": [2000], "sdi": [0.5]})}
    df = pd.DataFrame({"location_id": [6], "year": [2000], "max_wind_speed": [30.0], "person_storm_hours": [0.0],
                       "total_population": [1.0], "total_population_exposed": [1.0]})
    with pytest.raises(RuntimeError, match="dropped every row"):
        mo.prepare(df, ctx)
    df["person_storm_hours"] = 5.0
    sl = mo.prepare(df, ctx)
    assert "exposed_population" in sl.columns and sl["exposed"].iloc[0] == 5.0
