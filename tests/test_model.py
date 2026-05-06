"""
Tests for model.DoubleHurdleModel.

Covers:
  - predict returns a non-negative array of length len(df).
  - predict output is consistent with manual component predictions.
  - metrics, diagnostics, plot all raise NotImplementedError.
  - Missing required constructor arguments raise TypeError.

Fixture design
--------------
Uses gamma for both bulk and tail (simplest predict: exp(X @ params), no extra
meta needed). covariate_combo={"wind_speed": True} with log_exposed as free
covariate in S2/bulk/tail; S1 uses log_exposed as offset. Parameters chosen to
give plausible probabilities and rates without needing to run actual MLE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.distributions.base import FitResult
from idd_tc_mortality.model import DoubleHurdleModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COV_COMBO = {"wind_speed": True, "sdi": False, "basin": False, "is_island": False}
N_OBS = 20


@pytest.fixture
def prediction_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "wind_speed": rng.normal(50, 10, N_OBS),
        "sdi":        rng.uniform(0, 1, N_OBS),
        "basin":      ["NA"] * N_OBS,
        "is_island":  np.zeros(N_OBS),
        "exposed":    rng.uniform(10_000, 1_000_000, N_OBS),
    })


@pytest.fixture
def model():
    """DoubleHurdleModel with synthetic FitResult objects and gamma bulk/tail."""
    # S1: param_names = ["const", "wind_speed"] (no log_exposed — it's an offset)
    # S2: param_names = ["const", "wind_speed", "log_exposed"] (free covariate)
    s1_result = FitResult(
        params=np.array([-3.0, 0.02]),
        param_names=["const", "wind_speed"],
        fitted_values=np.zeros(1),
        family="s1",
    )
    s2_result = FitResult(
        params=np.array([-2.0, 0.01, 0.5]),
        param_names=["const", "wind_speed", "log_exposed"],
        fitted_values=np.zeros(1),
        family="cloglog",
        meta={"threshold_rate": 1e-4, "link": "cloglog"},
    )
    # Bulk/tail: param_names include log_exposed as free covariate
    bulk_result = FitResult(
        params=np.array([-10.0, 0.02, 1.0]),
        param_names=["const", "wind_speed", "log_exposed"],
        fitted_values=np.zeros(1),
        family="gamma",
    )
    tail_result = FitResult(
        params=np.array([-9.0, 0.02, 1.0]),
        param_names=["const", "wind_speed", "log_exposed"],
        fitted_values=np.zeros(1),
        family="gamma",
    )
    return DoubleHurdleModel(
        s1_result=s1_result,
        s2_result=s2_result,
        bulk_result=bulk_result,
        bulk_family="gamma",
        tail_result=tail_result,
        tail_family="gamma",
        threshold=1e-4,
        covariate_combo=COV_COMBO,
    )


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_output_length(model, prediction_df):
    result = model.predict(prediction_df)
    assert len(result) == N_OBS


def test_predict_returns_ndarray(model, prediction_df):
    result = model.predict(prediction_df)
    assert isinstance(result, np.ndarray)


def test_predict_nonnegative(model, prediction_df):
    result = model.predict(prediction_df)
    assert np.all(result >= 0), f"Negative predictions: {result[result < 0]}"


def test_predict_finite(model, prediction_df):
    result = model.predict(prediction_df)
    assert np.all(np.isfinite(result)), "predict returned non-finite values"


def test_predict_consistent_with_components(model, prediction_df):
    """Verify predict output matches manual assembly of components."""
    from idd_tc_mortality.combine import assemble_dh_prediction
    from idd_tc_mortality.distributions import get_family
    from idd_tc_mortality.features import align_X, build_X
    from idd_tc_mortality.distributions.binomial_cloglog import predict_binomial_cloglog
    from idd_tc_mortality import s2 as s2_mod

    df = prediction_df
    log_exposed = np.log(df["exposed"].to_numpy(dtype=float))

    # S1: log_exposed is offset, not in X
    X_s1 = align_X(
        build_X(df, COV_COMBO, include_log_exposed=False),
        model.s1_result.param_names,
    )
    p_s1 = predict_binomial_cloglog(model.s1_result, X_s1, log_exposed)

    # S2: log_exposed is a free covariate in X
    X_s2 = align_X(
        build_X(df, COV_COMBO, include_log_exposed=True),
        model.s2_result.param_names,
    )
    p_s2 = s2_mod.predict(model.s2_result, X_s2)

    X_bulk = align_X(
        build_X(df, COV_COMBO, include_log_exposed=True),
        model.bulk_result.param_names,
    )
    rate_bulk = get_family("gamma")["predict"](model.bulk_result, X_bulk)
    rate_tail = get_family("gamma")["predict"](model.tail_result, X_bulk)

    expected = assemble_dh_prediction(p_s1, p_s2, rate_bulk, rate_tail)
    np.testing.assert_allclose(model.predict(df), expected, rtol=1e-12)


def test_predict_single_row(model, prediction_df):
    """Single-row DataFrame produces a length-1 array."""
    result = model.predict(prediction_df.iloc[:1])
    assert result.shape == (1,)


# ---------------------------------------------------------------------------
# Stubs raise NotImplementedError
# ---------------------------------------------------------------------------

def test_metrics_raises_not_implemented(model, prediction_df):
    with pytest.raises(NotImplementedError):
        model.metrics(prediction_df, np.zeros(N_OBS))


def test_diagnostics_raises_not_implemented(model):
    with pytest.raises(NotImplementedError):
        model.diagnostics()


def test_plot_raises_not_implemented(model):
    with pytest.raises(NotImplementedError):
        model.plot()


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_missing_required_argument_raises():
    """Omitting any required argument raises TypeError."""
    with pytest.raises(TypeError):
        DoubleHurdleModel(
            s1_result=FitResult(np.array([0.0]), ["const"], np.zeros(1), "s1"),
            # s2_result omitted
            bulk_result=FitResult(np.array([0.0]), ["const"], np.zeros(1), "gamma"),
            bulk_family="gamma",
            tail_result=FitResult(np.array([0.0]), ["const"], np.zeros(1), "gamma"),
            tail_family="gamma",
            threshold=1e-4,
            covariate_combo={},
        )


def test_all_required_args_accepted(model):
    """Constructor succeeds when all required arguments are provided."""
    assert model.threshold == 1e-4
    assert model.bulk_family == "gamma"
    assert model.tail_family == "gamma"
    assert model.covariate_combo == COV_COMBO
