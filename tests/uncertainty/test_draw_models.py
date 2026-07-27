"""
Tests for the coefficient-draw uncertainty module.

Covers the TOPSIS-winner family combo:
  S1=logit/free, S2=logit/free, bulk=scaled_logit/free, tail=gamma/free+weight.

Tests verify:
  - Toggle 1 OFF -> every draw's β identical to MLE (per stage).
  - Toggle 2 OFF -> every draw's scale identical to MLE (per scale-bearing stage).
  - Toggle 1 ON  -> draws differ across N and, in expectation over many draws,
                    their column-wise mean is close to the MLE.
  - Toggle 2 ON  -> dispersion draws differ.
  - predict reproducibility: same seed -> identical output.
  - predict with outcome_draw=True/False produces rate/deaths Series of the
    expected shape and respects S1=0 -> deaths=0.
  - Save/load round-trip preserves the list of DrawModels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.refit_with_objects import refit_model_with_objects
from idd_tc_mortality.uncertainty import (
    DrawModel,
    build_draw_models,
    load_draw_models,
    save_draw_models,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic training data + a focus_model dict mirroring the TOPSIS
# winner family/exposure combo.
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Tiny synthetic dataset with the columns the model expects."""
    rng = np.random.default_rng(0)
    n = 400
    basin_levels = ["NA", "WP", "SI"]
    basin = rng.choice(basin_levels, size=n)
    is_island = rng.integers(0, 2, n).astype(int)
    sdi = rng.uniform(0.3, 0.9, n)
    wind_speed = rng.uniform(20.0, 80.0, n)
    exposed = rng.uniform(5e4, 5e6, n)

    # DGP -- aim for ~30-40% rows with deaths>=1 and a healthy tail share.
    log_p_event = -12.0 + 0.03 * wind_speed + 1.5 * is_island + np.log(exposed)
    p_event = 1.0 / (1.0 + np.exp(-log_p_event))
    p_event = np.clip(p_event, 1e-9, 1 - 1e-9)
    has_deaths = rng.binomial(1, p_event)

    # death rate among events: gamma-shaped on log scale
    base_rate = np.exp(-12.0 + 0.04 * wind_speed - 1.2 * sdi)
    rate = rng.gamma(shape=2.0, scale=base_rate / 2.0)
    rate = np.where(has_deaths == 1, rate, 0.0)
    deaths = np.maximum(np.floor(rate * exposed), 0).astype(int)

    df = pd.DataFrame(
        {
            "deaths":     deaths,
            "exposed":    exposed,
            "basin":      basin,
            "is_island":  is_island,
            "sdi":        sdi,
            "wind_speed": wind_speed,
        }
    )
    return df


@pytest.fixture
def fold_assignments(synthetic_data):
    """Minimum fold table satisfying refit_with_objects's n_seeds × n_folds loop."""
    rng = np.random.default_rng(1)
    n = len(synthetic_data)
    folds = pd.DataFrame(
        {"seed_0": rng.integers(0, 2, n)},
        index=synthetic_data.index,
    )
    return folds


@pytest.fixture
def focus_model_topsis():
    """Match the half-coupled TOPSIS winner family/exposure combo (covs minimal)."""
    cov = {"basin": True, "is_island": True, "sdi": True, "wind_speed": True}
    import json as _j
    return {
        "threshold_quantile":    0.70,
        "s1_family":             "logit",
        "s1_exposure_mode":      "free",
        "s2_family":             "logit",
        "s2_exposure_mode":      "free",
        "bulk_family":           "scaled_logit",
        "bulk_exposure_mode":    "free",
        "tail_family":           "gamma",
        "tail_exposure_mode":    "free+weight",
        "s1_cov":                _j.dumps(cov),
        "s2_cov":                _j.dumps(cov),
        "bulk_cov":              _j.dumps(cov),
        "tail_cov":              _j.dumps(cov),
    }


@pytest.fixture
def refit_out(synthetic_data, fold_assignments, focus_model_topsis):
    return refit_model_with_objects(
        focus_model=focus_model_topsis,
        data=synthetic_data,
        fold_assignments=fold_assignments,
        n_seeds=1,
        n_folds=2,
    )


# ---------------------------------------------------------------------------
# Toggle behaviour: coefficient draws
# ---------------------------------------------------------------------------

def test_toggle1_off_all_betas_identical(refit_out, focus_model_topsis, synthetic_data):
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=20, draw_coefs=False, draw_scale=False, seed=42,
    )
    assert len(models) == 20
    for stage in ("s1", "s2", "bulk", "tail"):
        baseline = getattr(models[0], stage).params
        for m in models[1:]:
            np.testing.assert_array_equal(getattr(m, stage).params, baseline)


def test_toggle1_on_betas_differ(refit_out, focus_model_topsis, synthetic_data):
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=20, draw_coefs=True, draw_scale=False, seed=42,
    )
    # At least one stage's β should differ across draws (in practice all four do).
    differs = False
    for stage in ("s1", "s2", "bulk", "tail"):
        baseline = getattr(models[0], stage).params
        for m in models[1:]:
            if not np.array_equal(getattr(m, stage).params, baseline):
                differs = True
                break
        if differs:
            break
    assert differs, "draw_coefs=True did not produce any cross-draw variation."


def test_toggle1_consistency_mean_near_mle(refit_out, focus_model_topsis, synthetic_data):
    """With many draws, the column-wise mean of drawn β should approach the MLE."""
    n = 1500
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=n, draw_coefs=True, draw_scale=False, seed=42,
    )
    # MLE β is the value used when toggle 1 is off.
    mle_models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=1, draw_coefs=False, draw_scale=False, seed=42,
    )
    for stage in ("s1", "s2", "bulk", "tail"):
        mle_params = getattr(mle_models[0], stage).params
        stacked = np.vstack([getattr(m, stage).params for m in models])
        mean_params = stacked.mean(axis=0)
        # Loose tolerance; this is a Monte Carlo check, not exact.
        max_dev = np.max(np.abs(mean_params - mle_params))
        scale = max(np.max(np.abs(mle_params)), 1.0)
        assert max_dev / scale < 0.15, (
            f"Stage {stage}: drawn-mean β strays from MLE by {max_dev:.3g} "
            f"(scale {scale:.3g}). Expected < 0.15× scale."
        )


# ---------------------------------------------------------------------------
# Toggle behaviour: scale draws
# ---------------------------------------------------------------------------

def test_toggle2_off_scale_identical(refit_out, focus_model_topsis, synthetic_data):
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=20, draw_coefs=False, draw_scale=False, seed=42,
    )
    for stage in ("bulk", "tail"):
        baseline = getattr(models[0], stage).scale
        for m in models[1:]:
            assert getattr(m, stage).scale == baseline


def test_toggle2_on_scale_differs(refit_out, focus_model_topsis, synthetic_data):
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=20, draw_coefs=False, draw_scale=True, seed=42,
    )
    bulk_scales = [m.bulk.scale for m in models]
    tail_scales = [m.tail.scale for m in models]
    assert len(set(bulk_scales)) > 1, "bulk scale draws should vary."
    assert len(set(tail_scales)) > 1, "tail scale draws should vary."


def test_logit_stages_have_no_scale(refit_out, focus_model_topsis, synthetic_data):
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=5, draw_coefs=True, draw_scale=True, seed=42,
    )
    for m in models:
        assert m.s1.scale is None
        assert m.s2.scale is None


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def test_predict_reproducible(refit_out, focus_model_topsis, synthetic_data):
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=5, draw_coefs=True, draw_scale=True, seed=42,
    )
    storms = synthetic_data.head(50).copy()
    pred1 = models[0].predict(storms, outcome_draw=True, seed=123)
    pred2 = models[0].predict(storms, outcome_draw=True, seed=123)
    pd.testing.assert_frame_equal(pred1, pred2)


def test_predict_shape_and_columns(refit_out, focus_model_topsis, synthetic_data):
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=3, draw_coefs=False, draw_scale=False, seed=42,
    )
    storms = synthetic_data.head(30).copy()
    pred = models[0].predict(storms, outcome_draw=True, seed=7)
    assert len(pred) == 30
    assert set(pred.columns) == {"p_s1", "s1_flip", "p_s2", "s2_flip", "rate", "deaths"}


def test_predict_s1_zero_implies_deaths_zero(refit_out, focus_model_topsis, synthetic_data):
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=3, draw_coefs=False, draw_scale=False, seed=42,
    )
    storms = synthetic_data.copy()
    pred = models[0].predict(storms, outcome_draw=True, seed=7)
    s1_zero = pred["s1_flip"] == 0
    assert (pred.loc[s1_zero, "rate"] == 0).all()
    assert (pred.loc[s1_zero, "deaths"] == 0).all()
    # And the s2 fields should be NaN where s1 fired zero.
    assert pred.loc[s1_zero, "p_s2"].isna().all()
    assert pred.loc[s1_zero, "s2_flip"].isna().all()


def test_predict_mean_vs_draw_differs(refit_out, focus_model_topsis, synthetic_data):
    """Toggle 3 ON vs OFF must produce different rate vectors for rows that
    actually flow through bulk/tail (i.e. s1_flip = 1)."""
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=1, draw_coefs=False, draw_scale=False, seed=42,
    )
    storms = synthetic_data.copy()
    # Same seed for both, so S1/S2 flips are identical; only bulk/tail behaviour differs.
    pred_mean = models[0].predict(storms, outcome_draw=False, seed=11)
    pred_draw = models[0].predict(storms, outcome_draw=True,  seed=11)
    fired = pred_mean["s1_flip"] == 1
    assert fired.sum() > 0, "fixture too sparse to test outcome toggle."
    # On rows that fired, rates should not all be equal between mean and draw.
    diff_rows = (pred_mean.loc[fired, "rate"] != pred_draw.loc[fired, "rate"]).sum()
    assert diff_rows > 0, "outcome_draw=True did not change any rate predictions."


# ---------------------------------------------------------------------------
# Toggle 4: expected_bernoulli
# ---------------------------------------------------------------------------

def test_expected_bernoulli_matches_assemble_predictions(
    refit_out, focus_model_topsis, synthetic_data,
):
    """(T, F, F, F) reproduces evaluate.assemble.assemble_predictions exactly.

    With outcome_draw=False, expected_bernoulli=True, draw_coefs=False,
    draw_scale=False, DrawModel.predict is the same closed-form arithmetic
    the evaluate stage uses for full_*_oos metrics.
    """
    from idd_tc_mortality.evaluate.assemble import assemble_predictions

    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=1, draw_coefs=False, draw_scale=False, seed=42,
    )
    pred = models[0].predict(
        synthetic_data, outcome_draw=False, expected_bernoulli=True, seed=42,
    )

    expected_rate = assemble_predictions(
        s1_result  = refit_out["is"]["s1"]["fit_result"],
        s1_spec    = refit_out["is"]["s1"]["spec"],
        s2_result  = refit_out["is"]["s2"]["fit_result"],
        s2_spec    = refit_out["is"]["s2"]["spec"],
        bulk_result= refit_out["is"]["bulk"]["fit_result"],
        bulk_spec  = refit_out["is"]["bulk"]["spec"],
        tail_result= refit_out["is"]["tail"]["fit_result"],
        tail_spec  = refit_out["is"]["tail"]["spec"],
        df         = synthetic_data,
    )

    np.testing.assert_allclose(
        pred["rate"].values, expected_rate.values, rtol=1e-10, atol=1e-12,
    )


def test_expected_bernoulli_nullifies_flips(
    refit_out, focus_model_topsis, synthetic_data,
):
    """expected_bernoulli=True leaves s1_flip and s2_flip NaN; p_s2 is no
    longer s1-masked."""
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=1, draw_coefs=False, draw_scale=False, seed=42,
    )
    pred = models[0].predict(
        synthetic_data, outcome_draw=False, expected_bernoulli=True, seed=42,
    )
    assert pred["s1_flip"].isna().all()
    assert pred["s2_flip"].isna().all()
    assert pred["p_s2"].notna().all()


def test_expected_bernoulli_soft_hurdle(
    refit_out, focus_model_topsis, synthetic_data,
):
    """expected_bernoulli=True with outcome_draw=True is the 'soft hurdle'
    cell: probability-weighted convex combo of sampled bulk and tail. Same
    seed produces different rates than the hard-hurdle cell, and produces
    strictly fewer exactly-zero rates."""
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=1, draw_coefs=False, draw_scale=False, seed=42,
    )
    pred_soft = models[0].predict(
        synthetic_data, outcome_draw=True, expected_bernoulli=True, seed=11,
    )
    pred_hard = models[0].predict(
        synthetic_data, outcome_draw=True, expected_bernoulli=False, seed=11,
    )
    different = (pred_soft["rate"].values != pred_hard["rate"].values).sum()
    assert different > 0
    hard_zeros = int((pred_hard["rate"].values == 0).sum())
    soft_zeros = int((pred_soft["rate"].values == 0).sum())
    assert soft_zeros < hard_zeros, (
        f"Soft hurdle should have fewer zero-rate rows than hard "
        f"(soft={soft_zeros}, hard={hard_zeros})."
    )


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path, refit_out, focus_model_topsis, synthetic_data):
    models = build_draw_models(
        refit_out, focus_model_topsis, synthetic_data,
        n_draws=5, draw_coefs=True, draw_scale=True, seed=42,
    )
    path = tmp_path / "draws.pkl"
    save_draw_models(models, path)
    loaded = load_draw_models(path)
    assert len(loaded) == len(models)
    for orig, back in zip(models, loaded):
        assert isinstance(back, DrawModel)
        np.testing.assert_array_equal(orig.s1.params, back.s1.params)
        np.testing.assert_array_equal(orig.bulk.params, back.bulk.params)
        assert orig.bulk.scale == back.bulk.scale
        assert orig.tail.scale == back.tail.scale
        assert orig.threshold_rate == back.threshold_rate


# ---------------------------------------------------------------------------
# GPD tail: draw builder, draw_scale no-op, 16-toggle predict
# ---------------------------------------------------------------------------
#
# A GPD tail fits via scipy BFGS and stores the joint (beta, xi) inverse Hessian
# in fit_result.meta['hess_inv']; fit_result.cov is None. The generic
# _prepare_stage branch (which calls raw.cov_params()) cannot serve it, so
# _prepare_stage dispatches to _prepare_stage_gpd. Semantics under test:
#   * draw_coefs draws BOTH beta and xi jointly from the full hess_inv MVN.
#   * draw_scale is a documented NO-OP for GPD.
#   * all 16 (c, s, o, b) toggle cells build + predict finite rate/deaths.


@pytest.fixture
def focus_model_gpd():
    """Same S1/S2/bulk as the TOPSIS winner, but a GPD tail (free exposure).

    'free' (not 'free+weight') keeps the tail MLE well-scaled for a small
    fixture; the weight path is exercised elsewhere and by Problem-B work.
    """
    cov = {"basin": True, "is_island": True, "sdi": True, "wind_speed": True}
    import json as _j
    return {
        "threshold_quantile":    0.70,
        "s1_family":             "logit",
        "s1_exposure_mode":      "free",
        "s2_family":             "logit",
        "s2_exposure_mode":      "free",
        "bulk_family":           "scaled_logit",
        "bulk_exposure_mode":    "free",
        "tail_family":           "gpd",
        "tail_exposure_mode":    "free",
        "s1_cov":                _j.dumps(cov),
        "s2_cov":                _j.dumps(cov),
        "bulk_cov":              _j.dumps(cov),
        "tail_cov":              _j.dumps(cov),
    }


@pytest.fixture
def refit_out_gpd(synthetic_data, fold_assignments, focus_model_gpd):
    return refit_model_with_objects(
        focus_model=focus_model_gpd,
        data=synthetic_data,
        fold_assignments=fold_assignments,
        n_seeds=1,
        n_folds=2,
    )


def test_gpd_tail_is_fit_not_sentinel(refit_out_gpd):
    """Sanity: the GPD tail IS fit must not be the hard-failure sentinel, else
    build_draw_models can't run. converged may be False (that's fine)."""
    tail_entry = refit_out_gpd["is"]["tail"]
    assert not tail_entry["failed"], (
        f"GPD tail IS fit sentinel'd: {tail_entry['metrics']}"
    )
    fr = tail_entry["fit_result"]
    assert fr.family == "gpd"
    assert "hess_inv" in fr.meta
    assert "shape_param" in fr.meta


def test_gpd_build_draw_models_runs(refit_out_gpd, focus_model_gpd, synthetic_data):
    """A GPD-tail focus model builds N draw models without raising."""
    models = build_draw_models(
        refit_out_gpd, focus_model_gpd, synthetic_data,
        n_draws=25, draw_coefs=True, draw_scale=True, seed=42,
    )
    assert len(models) == 25
    for m in models:
        assert m.tail.family == "gpd"
        # xi is carried in StageDraw.scale.
        assert m.tail.scale is not None
        assert np.isfinite(m.tail.scale)


def test_gpd_coefs_off_beta_and_xi_fixed(refit_out_gpd, focus_model_gpd, synthetic_data):
    """draw_coefs=False -> every draw shares the MLE beta AND MLE xi."""
    models = build_draw_models(
        refit_out_gpd, focus_model_gpd, synthetic_data,
        n_draws=15, draw_coefs=False, draw_scale=False, seed=42,
    )
    beta0 = models[0].tail.params
    xi0 = models[0].tail.scale
    for m in models[1:]:
        np.testing.assert_array_equal(m.tail.params, beta0)
        assert m.tail.scale == xi0


def test_gpd_coefs_on_beta_and_xi_vary(refit_out_gpd, focus_model_gpd, synthetic_data):
    """draw_coefs=True -> both beta and xi vary across draws (xi is drawn
    jointly with beta, not held fixed)."""
    models = build_draw_models(
        refit_out_gpd, focus_model_gpd, synthetic_data,
        n_draws=30, draw_coefs=True, draw_scale=False, seed=42,
    )
    betas = np.vstack([m.tail.params for m in models])
    xis = np.array([m.tail.scale for m in models])
    assert np.any(betas.std(axis=0) > 0), "GPD beta draws did not vary."
    assert xis.std() > 0, "GPD xi draws did not vary (should be drawn with beta)."


@pytest.mark.parametrize("draw_coefs", [False, True])
def test_gpd_draw_scale_is_noop(refit_out_gpd, focus_model_gpd, synthetic_data, draw_coefs):
    """draw_scale is a documented no-op for GPD: toggling it (with draw_coefs
    held) yields byte-identical draw models (beta AND xi unchanged)."""
    common = dict(n_draws=20, draw_coefs=draw_coefs, seed=42)
    models_s_off = build_draw_models(
        refit_out_gpd, focus_model_gpd, synthetic_data, draw_scale=False, **common,
    )
    models_s_on = build_draw_models(
        refit_out_gpd, focus_model_gpd, synthetic_data, draw_scale=True, **common,
    )
    for a, b in zip(models_s_off, models_s_on):
        np.testing.assert_array_equal(a.tail.params, b.tail.params)
        assert a.tail.scale == b.tail.scale


def test_gpd_all_16_toggle_cells_predict(refit_out_gpd, focus_model_gpd, synthetic_data):
    """Every (draw_coefs, draw_scale, outcome_draw, expected_bernoulli) cell —
    all 16 — must build draws and predict finite rate/deaths for a GPD tail."""
    import itertools

    storms = synthetic_data.head(60).copy()
    cells = list(itertools.product([False, True], repeat=4))
    assert len(cells) == 16
    for draw_coefs, draw_scale, outcome_draw, expected_bernoulli in cells:
        models = build_draw_models(
            refit_out_gpd, focus_model_gpd, synthetic_data,
            n_draws=3, draw_coefs=draw_coefs, draw_scale=draw_scale, seed=42,
        )
        pred = models[0].predict(
            storms,
            outcome_draw=outcome_draw,
            expected_bernoulli=expected_bernoulli,
            seed=7,
        )
        label = (draw_coefs, draw_scale, outcome_draw, expected_bernoulli)
        assert len(pred) == len(storms), label
        assert set(pred.columns) == {
            "p_s1", "s1_flip", "p_s2", "s2_flip", "rate", "deaths"
        }, label
        assert np.isfinite(pred["rate"].values).all(), f"non-finite rate in cell {label}"
        assert np.isfinite(pred["deaths"].values).all(), f"non-finite deaths in cell {label}"
        assert (pred["rate"].values >= 0).all(), f"negative rate in cell {label}"


def test_gpd_prepare_stage_recovers_and_centers():
    """Controlled recovery: generate excess rates from a known GPD log-linear
    DGP, fit via gpd.fit, then check (a) the MLE recovers the truth and (b) the
    _prepare_stage GPD draws center on the MLE for both beta and xi."""
    from scipy.stats import genpareto

    from idd_tc_mortality.distributions import gpd
    from idd_tc_mortality.uncertainty.draw_models import _prepare_stage

    rng = np.random.default_rng(77)
    n = 600
    x = rng.normal(0.0, 1.0, n)
    X = pd.DataFrame({"const": 1.0, "wind_speed": x})
    log_sigma = -10.0 + 0.3 * x       # sigma ~ exp(-10)
    sigma = np.exp(log_sigma)
    xi_true = 0.3
    y = genpareto.rvs(c=xi_true, scale=sigma, random_state=rng)
    weights = np.ones(n)

    fit_result = gpd.fit(X, y, weights)
    beta_mle = np.asarray(fit_result.params)
    xi_mle = float(fit_result.meta["shape_param"])

    # (a) MLE recovers the DGP truth within tolerance.
    assert abs(beta_mle[0] - (-10.0)) < 0.5, f"intercept off: {beta_mle[0]}"
    assert abs(beta_mle[1] - 0.3) < 0.25, f"wind coef off: {beta_mle[1]}"
    assert abs(xi_mle - xi_true) < 0.25, f"xi off: {xi_mle}"

    # (b) draws center on the MLE. Build a minimal refit_entry; raw_object is a
    # non-None sentinel (the GPD branch reads hess_inv from fit_result.meta,
    # not raw). exposure_mode='free' -> train_weight_mean short-circuits without
    # touching `data`.
    refit_entry = {
        "raw_object": object(),
        "spec": {"family": "gpd", "covariate_combo": {"wind_speed": True}},
        "fit_result": fit_result,
    }
    focus_model = {"tail_exposure_mode": "free"}

    n_draws = 5000
    kit = _prepare_stage(
        stage="tail",
        refit_entry=refit_entry,
        focus_model=focus_model,
        data=pd.DataFrame(),
        threshold_rate=1e-6,
        n_draws=n_draws,
        draw_coefs=True,
        draw_scale=True,      # no-op for GPD
        seed=np.random.SeedSequence(123),
    )

    # Per-parameter Monte-Carlo tolerance: |mean - mle| < 5 * SE / sqrt(n_draws),
    # where SE is the asymptotic sd from the (psd-projected) joint covariance.
    from idd_tc_mortality.uncertainty.draw_models import _psd_project
    cov = _psd_project(np.asarray(fit_result.meta["hess_inv"]))
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    tol = 5.0 * se / np.sqrt(n_draws)

    mean_beta = kit["params_draws"].mean(axis=0)
    mean_xi = kit["scale_draws"].mean()
    for j in range(len(beta_mle)):
        assert abs(mean_beta[j] - beta_mle[j]) < tol[j], (
            f"beta[{j}] drawn-mean {mean_beta[j]} strays from MLE {beta_mle[j]} "
            f"(tol {tol[j]})"
        )
    assert abs(mean_xi - xi_mle) < tol[-1], (
        f"xi drawn-mean {mean_xi} strays from MLE {xi_mle} (tol {tol[-1]})"
    )
