"""_tail_mean branches for the 20260722 tail-variant families (lognormal,
log_logistic_cens) must reproduce the corresponding distributions.*.predict
formulas exactly — the deterministic DrawModel forecast has to match
evaluate-time predictions."""

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.distributions.log_logistic_cens import loglogistic_censored_mean
from idd_tc_mortality.tail_cap import excess_cap
from idd_tc_mortality.uncertainty.draw_models import StageDraw, _tail_mean

THRESHOLD = 2e-6


def _stage(family, scale):
    return StageDraw(
        stage="tail",
        family=family,
        exposure_mode="free+weight",
        covariate_combo={"sdi": True},
        params=np.array([0.1, -0.5]),
        param_names=["const", "sdi"],
        scale=scale,
        threshold_rate=THRESHOLD,
    )


@pytest.fixture
def eta():
    rng = np.random.default_rng(7)
    return rng.normal(-12.0, 1.0, size=6)


@pytest.fixture
def storm_df():
    rng = np.random.default_rng(8)
    exposed = rng.uniform(1e4, 1e7, size=6)
    return pd.DataFrame({
        "exposed": exposed,
        "exposed_population": exposed * rng.uniform(0.5, 50.0, size=6),
    })


def test_lognormal_tail_mean_matches_mean_formula(eta):
    sigma2 = 1.7
    out = _tail_mean(_stage("lognormal", sigma2), eta, THRESHOLD)
    expected = np.exp(eta + 0.5 * sigma2) + THRESHOLD
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_lognormal_tail_mean_requires_scale(eta):
    with pytest.raises(ValueError, match="sigma"):
        _tail_mean(_stage("lognormal", None), eta, THRESHOLD)


def test_cens_tail_mean_matches_censored_mean(eta, storm_df):
    k = 1.8
    out = _tail_mean(_stage("log_logistic_cens", k), eta, THRESHOLD, storm_df)
    H = excess_cap(storm_df, THRESHOLD)
    expected = loglogistic_censored_mean(np.exp(eta), k, np.asarray(H)) + THRESHOLD
    np.testing.assert_allclose(out, expected, rtol=1e-12)
    assert np.isfinite(out).all()


def test_cens_tail_mean_requires_storm_df(eta):
    with pytest.raises(ValueError, match="storm_df"):
        _tail_mean(_stage("log_logistic_cens", 1.8), eta, THRESHOLD, None)


def test_unknown_family_still_raises(eta):
    with pytest.raises(NotImplementedError):
        _tail_mean(_stage("weibull", 1.0), eta, THRESHOLD)


# ---- outcome-draw (_tail_draw) branches -------------------------------------

from idd_tc_mortality.uncertainty.draw_models import _tail_draw


def test_lognormal_tail_draw_mean_matches_tail_mean(eta):
    sigma2 = 0.8
    stage = _stage("lognormal", sigma2)
    rng = np.random.default_rng(11)
    big_eta = np.repeat(eta[:1], 200_000)
    draws = _tail_draw(stage, big_eta, pd.DataFrame({"exposed": np.ones(len(big_eta))}),
                       THRESHOLD, rng)
    expected = np.exp(big_eta[0] + 0.5 * sigma2) + THRESHOLD
    np.testing.assert_allclose(draws.mean(), expected, rtol=0.02)
    # log-scale spread matches sigma
    np.testing.assert_allclose(np.log(draws - THRESHOLD).std(), np.sqrt(sigma2), rtol=0.02)


def test_cens_tail_draw_capped_and_mean_matches(eta, storm_df):
    from idd_tc_mortality.distributions.log_logistic_cens import loglogistic_censored_mean
    k = 2.0
    stage = _stage("log_logistic_cens", k)
    rng = np.random.default_rng(12)
    n = 200_000
    big_eta = np.repeat(eta[:1], n)
    sdf = pd.DataFrame({"exposed": np.full(n, 1e5),
                        "exposed_population": np.full(n, 5e5)})
    draws = _tail_draw(stage, big_eta, sdf, THRESHOLD, rng)
    H = excess_cap(sdf, THRESHOLD)
    assert (draws <= H + THRESHOLD + 1e-15).all(), "draws exceed the physical cap"
    expected = loglogistic_censored_mean(np.exp(big_eta[:1]), k, np.asarray(H[:1]))[0] + THRESHOLD
    np.testing.assert_allclose(draws.mean(), expected, rtol=0.02)


def test_tail_draw_deterministic_under_seed(eta, storm_df):
    stage = _stage("log_logistic_cens", 2.0)
    d1 = _tail_draw(stage, eta, storm_df, THRESHOLD, np.random.default_rng(5))
    d2 = _tail_draw(stage, eta, storm_df, THRESHOLD, np.random.default_rng(5))
    np.testing.assert_array_equal(d1, d2)


# ---- gpd_cens + log_logistic_trunc branches ----------------------------------

def test_gpd_cens_tail_mean_matches(eta, storm_df):
    from idd_tc_mortality.distributions.gpd_cens import gpd_censored_mean
    xi = 0.4
    out = _tail_mean(_stage("gpd_cens", xi), eta, THRESHOLD, storm_df)
    H = excess_cap(storm_df, THRESHOLD)
    expected = gpd_censored_mean(np.exp(eta), xi, np.asarray(H)) + THRESHOLD
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_ll_trunc_tail_mean_matches(eta, storm_df):
    from idd_tc_mortality.distributions.log_logistic_trunc import truncated_mean
    k = 1.6
    out = _tail_mean(_stage("log_logistic_trunc", k), eta, THRESHOLD, storm_df)
    H = excess_cap(storm_df, THRESHOLD)
    expected = truncated_mean(np.exp(eta), k, np.asarray(H)) + THRESHOLD
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_gpd_cens_tail_draw_capped_and_mean(eta):
    from idd_tc_mortality.distributions.gpd_cens import gpd_censored_mean
    xi = 0.4
    stage = _stage("gpd_cens", xi)
    n = 200_000
    big_eta = np.repeat(eta[:1], n)
    sdf = pd.DataFrame({"exposed": np.full(n, 1e5), "exposed_population": np.full(n, 5e5)})
    draws = _tail_draw(stage, big_eta, sdf, THRESHOLD, np.random.default_rng(21))
    H = excess_cap(sdf, THRESHOLD)
    assert (draws <= H + THRESHOLD + 1e-15).all()
    expected = gpd_censored_mean(np.exp(big_eta[:1]), xi, np.asarray(H[:1]))[0] + THRESHOLD
    np.testing.assert_allclose(draws.mean(), expected, rtol=0.02)


def test_ll_trunc_tail_draw_truncated_and_mean(eta):
    from idd_tc_mortality.distributions.log_logistic_trunc import truncated_mean
    k = 1.6
    stage = _stage("log_logistic_trunc", k)
    n = 200_000
    big_eta = np.repeat(eta[:1], n)
    sdf = pd.DataFrame({"exposed": np.full(n, 1e5), "exposed_population": np.full(n, 5e5)})
    draws = _tail_draw(stage, big_eta, sdf, THRESHOLD, np.random.default_rng(22))
    H = excess_cap(sdf, THRESHOLD)
    assert (draws <= H + THRESHOLD + 1e-15).all()
    expected = truncated_mean(np.exp(big_eta[:1]), k, np.asarray(H[:1]))[0] + THRESHOLD
    np.testing.assert_allclose(draws.mean(), expected, rtol=0.02)
