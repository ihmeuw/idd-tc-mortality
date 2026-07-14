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
