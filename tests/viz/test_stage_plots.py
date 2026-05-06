"""
Minimal tests for StagePlotter.

Verifies instantiation, predict_df shape and columns, and that vet_stage
returns a matplotlib Figure without error.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from idd_tc_mortality.viz.stage_plots import StagePlotter


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_instantiation(stage_plotter):
    assert isinstance(stage_plotter, StagePlotter)


def test_instantiation_without_output_dir(synth_data, results_dir):
    plotter = StagePlotter(data=synth_data, results_dir=results_dir)
    assert plotter._output_dir is None


def test_predict_df_oos_raises_without_output_dir(stage_plotter, model_row):
    with pytest.raises(ValueError, match="output_dir"):
        stage_plotter.predict_df_oos(model_row)


# ---------------------------------------------------------------------------
# predict_df
# ---------------------------------------------------------------------------

def test_predict_df_returns_dataframe(stage_plotter, model_row):
    result = stage_plotter.predict_df(model_row)
    assert isinstance(result, pd.DataFrame)


def test_predict_df_length(stage_plotter, model_row, synth_data):
    result = stage_plotter.predict_df(model_row)
    assert len(result) == len(synth_data)


def test_predict_df_required_columns(stage_plotter, model_row):
    result = stage_plotter.predict_df(model_row)
    for col in ["pred_s1", "pred_s2", "pred_bulk", "pred_tail", "pred_rate",
                "observed_rate", "threshold_rate", "is_bulk", "is_tail"]:
        assert col in result.columns, f"Missing column: {col}"


def test_predict_df_pred_s1_in_unit_interval(stage_plotter, model_row):
    result = stage_plotter.predict_df(model_row)
    assert (result["pred_s1"].values >= 0).all()
    assert (result["pred_s1"].values <= 1).all()


def test_predict_df_no_nans(stage_plotter, model_row):
    result = stage_plotter.predict_df(model_row)
    for col in ["pred_s1", "pred_s2", "pred_bulk", "pred_tail", "pred_rate"]:
        assert not result[col].isna().any(), f"NaN in {col}"


def test_predict_df_pred_rate_nonnegative(stage_plotter, model_row):
    result = stage_plotter.predict_df(model_row)
    assert (result["pred_rate"].values >= 0).all()


# ---------------------------------------------------------------------------
# vet_stage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage", ["s1", "s2", "bulk", "tail"])
def test_vet_stage_returns_figure(stage_plotter, model_row, stage):
    fig = stage_plotter.vet_stage(model_row, stage)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_vet_stage_3x3_layout(stage_plotter, model_row):
    fig = stage_plotter.vet_stage(model_row, "s1")
    # 3 rows × 3 cols = 9 axes
    assert len(fig.axes) == 9
    plt.close(fig)


# ---------------------------------------------------------------------------
# vet_model
# ---------------------------------------------------------------------------

def test_vet_model_returns_dict_of_figures(stage_plotter, model_row):
    figs = stage_plotter.vet_model(model_row)
    assert set(figs.keys()) == {"s1", "s2", "bulk", "tail"}
    for stage, fig in figs.items():
        assert isinstance(fig, plt.Figure), f"Stage {stage} did not return a Figure"
        plt.close(fig)


# ---------------------------------------------------------------------------
# _reconstruct_is_spec
# ---------------------------------------------------------------------------

def test_reconstruct_is_spec_s1_has_no_threshold(stage_plotter, model_row):
    spec = stage_plotter._reconstruct_is_spec(model_row, "s1")
    assert spec["threshold_quantile"] is None
    assert spec["threshold_rate"] is None
    assert spec["family"] == "cloglog"
    assert spec["exposure_mode"] == "offset"
    assert spec["fold_tag"] == "is"


def test_reconstruct_is_spec_bulk_has_family(stage_plotter, model_row):
    spec = stage_plotter._reconstruct_is_spec(model_row, "bulk")
    assert spec["family"] == "gamma"
    assert spec["threshold_quantile"] == 0.75
    assert spec["threshold_rate"] is not None


def test_reconstruct_is_spec_threshold_rate_matches_data(stage_plotter, model_row, synth_data):
    spec = stage_plotter._reconstruct_is_spec(model_row, "bulk")
    # threshold_rate should be computed from the data, not arbitrary
    dr = synth_data["deaths"].values / synth_data["exposed"].values
    pos = dr[dr > 0]
    expected = float(np.quantile(pos, 0.75))
    assert abs(spec["threshold_rate"] - expected) < 1e-10


import pandas as pd
