"""Window transforms, measures, surrogates, draw summaries, ceiling, cell assembly, densification."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from idd_tc_mortality.viz.cmip6_ibtracs_comparison import stats as s
from idd_tc_mortality.viz.cmip6_ibtracs_comparison.constants import (
    CAT_COLS,
    N_DRAWS,
    OUTCOMES,
    RUNS,
)


def test_detrend_removes_line_per_row():
    t = np.arange(10.0)
    y = np.stack([3 + 2 * t, 5 - t])
    out = s.detrend(y)
    assert np.allclose(out, 0.0)
    rng = np.random.default_rng(1)
    noise = rng.normal(size=10)
    res = s.detrend(noise + 7 + 0.5 * t)
    tc = t - t.mean()
    assert abs(res.mean()) < 1e-9 and abs((res * tc).sum()) < 1e-9   # mean and slope removed
    assert np.allclose(res, s.detrend(noise))                          # the line itself is gone


def test_moving_average_valid_only():
    y = np.array([1.0, 2, 3, 4, 5])
    assert np.allclose(s.moving_average(y, 3), [2, 3, 4])
    assert np.allclose(s.moving_average(y, 1), y)
    assert s.moving_average(y, 6).shape == (0,)
    assert np.allclose(s.moving_average(np.stack([y, 2 * y]), 2)[1], [3, 5, 7, 9])


def test_window_spec_moving_averages_only():
    assert s.window_spec("ma10") == ("ma", 10)
    with pytest.raises(ValueError, match="moving averages only"):
        s.window_spec("blk5")


def test_period_bounds():
    years = np.arange(1970, 1982)
    st, en = s.period_bounds(years, "ma5")
    assert st[0] == 1970 and en[0] == 1974 and len(st) == 8
    assert s.period_bounds(years[:3], "ma5")[0].shape == (0,)


def test_bretherton_n_eff():
    n = 40
    white = np.random.default_rng(3).normal(size=(2, n))
    assert s.n_eff_bretherton(white[0], white[1]) == pytest.approx(n * (1 - s.lag1_autocorr(white[0]) * s.lag1_autocorr(white[1]))
                                                                   / (1 + s.lag1_autocorr(white[0]) * s.lag1_autocorr(white[1])))
    smooth = s.moving_average(white, 10)                               # strongly autocorrelated
    assert 1 <= s.n_eff_bretherton(smooth[0], smooth[1]) < smooth.shape[1] / 2
    assert s.n_eff_bretherton(np.ones(10), white[0][:10]) == 10.0      # degenerate series → n
    assert np.isnan(s.n_eff_bretherton(white[0][:2], white[1][:2]))
    assert s.n_eff_bretherton(np.arange(5.0), np.arange(5.0)) >= 1.0   # clipped below at 1


def test_measures_known_values():
    x = np.array([1.0, 2, 3, 4, 5, 6])
    assert s.spearman(x, x) == pytest.approx(1.0) and s.spearman(x, -x) == pytest.approx(-1.0)
    assert s.kendall(x, x[::-1]) == pytest.approx(-1.0)
    assert np.isnan(s.spearman(x, np.ones(6))) and np.isnan(s.spearman(x[:2], x[:2]))
    y = np.array([6.0, 5, 4, 1, 2, 3])
    assert s.hit_rate(x, y) == 0.0            # top-2 of x = {5,6} (idx 4,5); top-2 of y = idx 0,1
    assert s.hit_rate(x, x) == 1.0
    assert s.updown(x, y) == pytest.approx(2 / 5)
    assert s.ratio(x, 2 * x) == pytest.approx(2.0) and np.isnan(s.ratio(np.zeros(3), x[:3]))


def test_percentile_of_with_ties():
    draws = np.array([[0, 1, 2, 3, 4]] * 1).T  # 5 draws, 1 period → shape (5,1)
    assert s.percentile_of(np.array([2.5]), draws)[0] == pytest.approx(0.6)
    assert s.percentile_of(np.array([2.0]), draws)[0] == pytest.approx(0.4 + 0.1)
    assert s.percentile_of(np.array([]), np.zeros((5, 0))).shape == (0,)


def test_pair_metrics_ratio_nan_when_detrended():
    x = np.array([1.0, 2, 3, 4])
    m = s.pair_metrics(x, x, "ma2", True)
    assert np.isnan(m["ratio"]) and m["n"] == 4 and 1 <= m["n_eff"] <= 4
    m = s.pair_metrics(x, x, "ma2", False, x, 3 * x)
    assert m["ratio"] == pytest.approx(3.0)


def test_phase_randomize_keeps_spectrum_and_mean():
    rng = np.random.default_rng(5)
    y = rng.poisson(30, size=37).astype(float) + np.arange(37)
    sur = s.phase_randomize(y, 50, rng)
    assert sur.shape == (50, 37)
    assert np.allclose(sur.mean(axis=1), y.mean())
    assert np.allclose(np.abs(np.fft.rfft(sur - y.mean(), axis=1)), np.abs(np.fft.rfft(y - y.mean())))
    assert not np.allclose(sur[0], sur[1])
    even = s.phase_randomize(y[:36], 3, rng)
    assert np.allclose(np.abs(np.fft.rfft(even - y[:36].mean(), axis=1)), np.abs(np.fft.rfft(y[:36] - y[:36].mean())))
    assert np.allclose(s.phase_randomize(y[:2], 4, rng), np.tile(y[:2], (4, 1)))


def test_spearman_rows_matches_scipy():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 12))
    b = rng.integers(0, 3, size=(3, 12)).astype(float)   # ties
    out = s.spearman_rows(a, b)
    for i in range(4):
        for j in range(3):
            assert out[i, j] == pytest.approx(sps.spearmanr(a[i], b[j]).statistic)
    assert np.isnan(s.spearman_rows(a, np.ones((1, 12)))).all()
    assert np.isnan(s.spearman_rows(a[:, :2], b[:, :2])).all()


def test_surrogate_summary():
    v = np.linspace(-0.5, 0.5, 101)
    r = s.surrogate_summary(v, 0.0)
    assert r["sur_lo"] == pytest.approx(-0.475) and r["sur_hi"] == pytest.approx(0.475)
    assert r["sur_pct"] == pytest.approx(0.5, abs=0.01) and r["sur_p"] == pytest.approx(1.0, abs=0.02)
    assert s.surrogate_summary(v, 0.9)["sur_p"] == pytest.approx(0.0)
    assert np.isnan(s.surrogate_summary(v[:5], 0.1)["sur_pct"]) and np.isnan(s.surrogate_summary(v, np.nan)["sur_p"])


def _draws(n_years=6, seed=0):
    rng = np.random.default_rng(seed)
    runs = np.repeat(np.array(RUNS), 3)
    draws = rng.poisson(5, size=(len(runs), n_years)).astype(float)
    return draws, runs


def test_summarize_draws_pipelines_weight_gcms_equally():
    draws, runs = _draws()
    out = s.summarize_draws(draws, runs)
    assert set(out) == {f"run:{r}" for r in RUNS} | {"gcm:CMCC-ESM2", "gcm:EC-Earth3", "gcm:MPI-ESM1-2-HR", "gcm:MRI-ESM2-0", "ens", "pooled"}
    run_means = np.stack([draws[runs == r].mean(0) for r in RUNS])
    assert np.allclose(out["pooled"]["mean"], run_means.mean(0))
    mri = run_means[3:].mean(0)
    ens = np.stack([run_means[0], run_means[1], run_means[2], mri]).mean(0)
    assert np.allclose(out["ens"]["mean"], ens)
    assert np.allclose(out["gcm:MRI-ESM2-0"]["median"], np.median(np.stack([np.median(draws[runs == r], 0) for r in RUNS[3:]]), 0))


def test_ceiling_summary_pairs_and_cross_gcm():
    rng = np.random.default_rng(11)
    run_w = {r: rng.normal(size=20) for r in RUNS}
    obs = rng.normal(size=20)
    c = s.ceiling_summary(run_w, obs)
    assert c["n_pairs"] == 28 and c["ceil_min"] <= c["ceil_q25"] <= c["ceil_med"] <= c["ceil_q75"] <= c["ceil_max"]
    assert c["xg_min"] >= c["ceil_min"] and c["xg_max"] <= c["ceil_max"]     # 18 cross-GCM pairs ⊂ 28
    assert c["obsrun_min"] <= c["obsrun_med"] <= c["obsrun_max"]
    assert np.isnan(s.ceiling_summary({RUNS[0]: obs}, obs)["ceil_med"])
    same = {r: obs for r in RUNS}
    assert s.ceiling_summary(same, obs)["ceil_med"] == pytest.approx(1.0)


def test_cell_stats_shapes_and_keys():
    draws, runs = _draws(n_years=16)
    obs = draws[0] + 1
    years = np.arange(1970, 1986)
    key = {"domain": "storm", "source": "land", "level": "global", "geo": "1", "outcome": "total"}
    st, bd, se, ce = s.cell_stats(key, obs, draws, runs, years, ("ma1", "ma5"), (1970, 1980), n_sur=50)
    stats = pd.DataFrame(st)
    assert len(stats) == 14 * 2 * 2 * 2 * 2     # 14 series × 2 stats × 2 windows × raw/detrended × 2 starts
    assert set(stats["pair"]) == {"obs_vs_model"} and set(stats.columns) >= set(s.METRIC_COLS) | {"start_year"}
    assert set(stats["start_year"]) == {1970, 1980}
    ok = stats["n"] >= 3                        # 1980 + ma5 on a 16-year series leaves 2 periods → no ρ
    assert stats.loc[ok, "sur_lo"].notna().all() and stats.loc[~ok, "spearman"].isna().all()
    assert ((stats.loc[ok, "sur_p"] >= 0) & (stats.loc[ok, "sur_p"] <= 1)).all()
    band = pd.DataFrame(bd)
    raw = band[(band.window == "ma5") & (~band.detrended)]
    assert set(raw["start_year"]) == {s.RAW_BAND_START} and raw["start"].tolist() == list(range(1970, 1982))
    det = band[(band.window == "ma5") & band.detrended]
    assert set(det["start_year"]) == {1970, 1980} and det[det.start_year == 1980]["start"].tolist() == [1980, 1981]
    series = pd.DataFrame(se)
    assert (series[series.series == "obs"]["value"].to_numpy() == obs).all()
    assert len(series) == 16 * (1 + 14 * 2)
    ceil = pd.DataFrame(ce)
    assert len(ceil) == 2 * 2 * 2 * 2 and set(ceil["stat"]) == {"mean", "median"}
    assert (ceil["n_pairs"].dropna() == 28).all() and ceil["n_pairs"].isna().sum() == 4   # 1980 + ma5: 2 periods


def test_cell_stats_skips_short_starts():
    draws, runs = _draws(n_years=8)
    years = np.arange(1980, 1988)
    st, _bd, _se, _ce = s.cell_stats({"k": 1}, draws[1], draws, runs, years, ("ma1",), (1980, 1986), n_sur=20)
    assert set(pd.DataFrame(st)["start_year"]) == {1980}        # 1986 leaves two years → skipped


def test_cell_stats_with_ibt_adds_pairs():
    draws, runs = _draws(n_years=12)
    obs = draws[1]
    ibt = draws[:5] + 0.5
    years = np.arange(1985, 1997)
    key = {"domain": "mortality", "source": "adjusted", "level": "global", "geo": "1", "outcome": "deaths"}
    st, bd, se, _ce = s.cell_stats(key, obs, draws, runs, years, ("ma5",), (1985,), ibt=ibt, n_sur=30)
    stats = pd.DataFrame(st)
    assert set(stats["pair"]) == {"obs_vs_model", "obs_vs_ibt", "ibt_vs_model"}
    assert len(stats[stats.pair == "obs_vs_ibt"]) == 2 * 2        # mean/median × raw/detrended
    assert len(stats[stats.pair == "ibt_vs_model"]) == 14 * 2 * 2   # stat matched to stat
    assert set(pd.DataFrame(bd)["band"]) == {"model", "ibt"}
    assert "ibt" in set(pd.DataFrame(se)["series"])


def test_densify_model_counts_and_obs():
    years = np.arange(2000, 2003)
    rows = []
    for r in RUNS:
        for d in (0, N_DRAWS - 1):
            rows.append({"run": r, "draw": d, "year": 2001, "level": "global", "geo": "1",
                         **{c: 0 for c in CAT_COLS}, "n6": 2})
    counts = pd.DataFrame(rows)
    arr = s.densify_model_counts(counts, "global", ["1"], years)
    assert arr.shape == (1, 8 * N_DRAWS, 3, len(OUTCOMES))
    oi = list(OUTCOMES).index("cat5")
    assert arr[0, 0, 1, oi] == 2 and arr[0, N_DRAWS - 1, 1, oi] == 2 and arr[0, 1, 1, oi] == 0
    assert arr[0, :, :, list(OUTCOMES).index("total")].sum() == 2 * 16
    obs = pd.DataFrame([{"year": 2002, "level": "basin", "geo": "NA", **{c: 1 for c in CAT_COLS}}])
    o = s.densify_obs_counts(obs, "basin", ["NA", "WP"], years)
    assert o.shape == (2, 3, len(OUTCOMES)) and o[0, 2, list(OUTCOMES).index("total")] == 6 and o[1].sum() == 0


def test_densify_model_counts_rejects_bad_draw_range():
    counts = pd.DataFrame([{"run": RUNS[0], "draw": 1, "year": 2000, "level": "global", "geo": "1",
                            **{c: 0 for c in CAT_COLS}}])
    with pytest.raises(RuntimeError, match="draw ids"):
        s.densify_model_counts(counts, "global", ["1"], np.array([2000]))


def test_deaths_by_level_and_densify():
    df = pd.DataFrame({"run": [RUNS[0]] * 4, "draw": [0] * 4, "basin": ["WP", "NA", "WP", "WP"],
                       "location_id": [1, 1, 6, 6], "year": [2000, 2000, 2000, 2001],
                       "deaths_adj": [10.0, 5.0, 3.0, 4.0]})
    b = s.deaths_by_level(df, "basin", ["WP", "NA"], ["run", "draw", "year"]).set_index("geo")["deaths_adj"]
    assert b["WP"] == 10.0 and b["NA"] == 5.0
    c = s.densify_deaths(df, "country", ["6"], np.array([2000, 2001]), "deaths_adj", ["run", "draw"])
    assert c.shape == (1, 8 * N_DRAWS, 2) and c[0, 0].tolist() == [3.0, 4.0]
    g = s.densify_deaths(df, "global", ["1"], np.array([2000, 2001]), "deaths_adj", ["run", "draw"])
    assert g[0, 0].tolist() == [15.0, 0.0]


def _hier():
    return pd.DataFrame({"location_id": [1, 4, 5, 6], "level": [0, 1, 2, 3], "region_id": [-1, -1, 5, 5],
                         "super_region_id": [-1, 4, 4, 4], "location_name": ["G", "SR", "R", "C"]})


def test_kept_geos_rule():
    obs = pd.DataFrame([
        {"year": 2000, "level": "country", "geo": "6", **{c: 0 for c in CAT_COLS}, "n1": 9},
        {"year": 2001, "level": "country", "geo": "6", **{c: 0 for c in CAT_COLS}, "n2": 1},
        {"year": 2000, "level": "basin", "geo": "NA", **{c: 0 for c in CAT_COLS}, "n1": 3},
    ])
    g = s.kept_geos(obs, (2000, 2001), _hier(), [6], min_storms=10).set_index(["level", "geo"])
    assert g.loc[("country", "6"), "kept"] and g.loc[("country", "6"), "n_obs_land"] == 10
    assert not g.loc[("basin", "NA"), "kept"] and g.loc[("global", "1"), "kept"]
    assert not g.loc[("region", "5"), "kept"]


def test_kept_geos_by_start_reevaluates_the_rule():
    obs = pd.DataFrame([{"year": y, "level": "country", "geo": "6", **{c: 0 for c in CAT_COLS}, "n1": 1}
                        for y in range(1990, 2002)])
    g = s.kept_geos_by_start(obs, (1990, 2000), 2001, _hier(), [6], min_storms=10).set_index(["start_year", "level", "geo"])
    assert g.loc[(1990, "country", "6"), "kept"] and g.loc[(1990, "country", "6"), "n_obs_land"] == 12
    assert not g.loc[(2000, "country", "6"), "kept"] and g.loc[(2000, "country", "6"), "n_obs_land"] == 2
    assert s._starts_kept(g.reset_index(), "6") == (1990,)


def test_densify_places_every_run_without_int8_overflow():
    """Regression: Categorical codes are int8; run index × 100 must not wrap for runs ≥ 2."""
    years = np.arange(2000, 2001)
    rows = [{"run": r, "draw": 99, "year": 2000, "level": "global", "geo": "1", **{c: 0 for c in CAT_COLS}, "n1": i + 1}
            for i, r in enumerate(RUNS)] + [{"run": RUNS[0], "draw": 0, "year": 2000, "level": "global", "geo": "1",
                                            **{c: 0 for c in CAT_COLS}, "n1": 50}]
    arr = s.densify_model_counts(pd.DataFrame(rows), "global", ["1"], years)
    oi = list(OUTCOMES).index("total")
    for i in range(len(RUNS)):
        assert arr[0, i * N_DRAWS + 99, 0, oi] == i + 1
    assert arr[0, 0, 0, oi] == 50 and arr[0, :, 0, oi].sum() == 50 + sum(range(1, 9))
    df = pd.DataFrame({"run": list(RUNS), "draw": [99] * 8, "basin": ["WP"] * 8, "location_id": [1] * 8,
                       "year": [2000] * 8, "deaths_adj": [float(i + 1) for i in range(8)]})
    d = s.densify_deaths(df, "global", ["1"], years, "deaths_adj", ["run", "draw"])
    assert [d[0, i * N_DRAWS + 99, 0] for i in range(8)] == [float(i + 1) for i in range(8)]
