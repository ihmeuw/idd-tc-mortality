"""Censored-GPD tail support in StagePlotter: density panel, cap-aware predict,
result injection. Uses the shared viz fixtures (synthetic data, gamma-tail fits)
and a hand-built gpd_cens FitResult so the tests do not depend on the GPD fitter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

from idd_tc_mortality.model import FitResult
from idd_tc_mortality.uncertainty.draw_models import StageDraw
from idd_tc_mortality.viz import stage_plots as sp

matplotlib.use("Agg")

XI = 0.8
LOG_SIGMA = -6.0
NO_COV = json.dumps({"basin": False, "is_island": False, "sdi": False, "wind_speed": False}, sort_keys=True)


def _gpd_result(threshold_rate):
    return FitResult(
        params=np.array([LOG_SIGMA]),
        param_names=["const"],
        fitted_values=np.empty(0),
        family="gpd_cens",
        converged=True,
        meta={"shape_param": XI, "threshold_rate": threshold_rate},
    )


@pytest.fixture(scope="module")
def data_with_cap(synth_data):
    df = synth_data.copy()
    df["exposed_population"] = df["exposed"] * 0.05  # any positive headcount
    return df


@pytest.fixture(scope="module")
def model_row_gpd(model_row):
    row = model_row.copy()
    row["tail_family"] = "gpd_cens"
    row["tail_exposure_mode"] = "excluded"
    row["tail_cov"] = NO_COV
    return row


@pytest.fixture(scope="module")
def plotter_gpd(data_with_cap, results_dir, model_row_gpd):
    plotter = sp.StagePlotter(data=data_with_cap, results_dir=results_dir)
    q = float(model_row_gpd["threshold_quantile"])
    plotter.register_results(model_row_gpd, {"tail": _gpd_result(plotter._compute_threshold_rate(q))})
    return plotter


# --- gpd_rate_pdf ---------------------------------------------------------------------
def test_gpd_rate_pdf_integrates_to_one_and_matches_exponential_limit():
    T, sigma = 0.01, 2.0
    w = np.geomspace(1e-9, 1e9, 400_000)
    for xi in (0.5, -0.3):
        f = sp.gpd_rate_pdf(w + T, sigma, xi, T)
        assert np.trapezoid(f, w) == pytest.approx(1.0, abs=2e-3)
    f0 = sp.gpd_rate_pdf(np.array([T + 1.0, T + 3.0]), sigma, 0.0, T)
    np.testing.assert_allclose(f0, np.exp(-np.array([1.0, 3.0]) / sigma) / sigma)


def test_gpd_rate_pdf_zero_beyond_finite_support():
    f = sp.gpd_rate_pdf(np.array([0.0 + 5.0]), sigma=1.0, xi=-0.5, threshold_rate=0.0)  # support w < 2
    assert f[0] == 0.0


# --- register_results / cap-aware _predict -----------------------------------------------
def test_register_results_is_what_load_result_returns(plotter_gpd, model_row_gpd):
    spec = plotter_gpd._reconstruct_is_spec(model_row_gpd, "tail")
    assert plotter_gpd._load_result(spec).family == "gpd_cens"
    assert plotter_gpd._load_result(spec).meta["shape_param"] == XI


def test_register_results_rejects_unknown_stage(plotter_gpd, model_row_gpd):
    with pytest.raises(ValueError, match="unknown stage"):
        plotter_gpd.register_results(model_row_gpd, {"s3": _gpd_result(0.1)})


def test_predict_gpd_cens_passes_cap_and_adds_threshold(plotter_gpd, model_row_gpd, data_with_cap):
    spec = plotter_gpd._reconstruct_is_spec(model_row_gpd, "tail")
    T = spec["threshold_rate"]
    pred_df = plotter_gpd._make_pred_df(data_with_cap, n=7)
    assert "exposed_population" in pred_df.columns
    preds = plotter_gpd._predict(plotter_gpd._load_result(spec), pred_df, spec, T)
    assert preds.shape == (7,)
    assert np.all(np.isfinite(preds)) and np.all(preds >= T)
    # censored mean is bounded by the physical cap c_i = exposed_population / exposed
    cap = (pred_df["exposed_population"] / pred_df["exposed"]).to_numpy()
    assert np.all(preds <= cap + 1e-12)


# --- density panel ------------------------------------------------------------------------
def _tail_pdf_config(uncertainty=False):
    panel = {
        "kind": "cont_pdf_quintile", "row": 0, "r_grid_lo": 1, "r_grid_hi": 100,
        "x_label": "", "y_label": "Relative likelihood", "inline_legends": False,
        "uncertainty": {"show": uncertainty, "percentiles": (5, 95), "alpha": 0.3},
    }
    return {
        "layout": {
            "figsize": (8, 3), "suptitle": "Stage 4", "tight_layout": True,
            "panels": {
                "lin": {**panel, "col": 0, "xscale": "linear"},
                "log": {**panel, "col": 1, "xscale": "log"},
            },
        },
        "legend": {"mode": "none"},
    }


def test_tail_pdf_panel_draws_decreasing_gpd_density(plotter_gpd, model_row_gpd):
    import matplotlib.pyplot as plt

    fig = plotter_gpd.vet_stage(model_row_gpd, stage="tail", config=_tail_pdf_config())
    axes = [ax for ax in fig.axes if ax.lines]
    assert len(axes) == 2
    for ax in axes:
        y = ax.lines[0].get_ydata()
        assert np.all(np.isfinite(y)) and y[0] > 0
        assert np.all(np.diff(y) <= 1e-12)  # xi > 0 GPD excess density is monotone decreasing
    assert axes[1].get_xscale() == "log"
    plt.close(fig)


def test_tail_pdf_panel_uncertainty_band_from_stage_draws(plotter_gpd, model_row_gpd):
    import matplotlib.pyplot as plt

    T = plotter_gpd._compute_threshold_rate(float(model_row_gpd["threshold_quantile"]))
    rng = np.random.default_rng(0)
    draws = [
        SimpleNamespace(tail=StageDraw(
            stage="tail", family="gpd_cens", exposure_mode="excluded",
            covariate_combo=json.loads(NO_COV), params=np.array([LOG_SIGMA + rng.normal(0, 0.2)]),
            param_names=["const"], scale=XI + rng.normal(0, 0.05), threshold_rate=T,
        ))
        for _ in range(20)
    ]
    fig = plotter_gpd.vet_stage(model_row_gpd, stage="tail", config=_tail_pdf_config(True), draws=draws)
    ax = next(ax for ax in fig.axes if ax.lines)
    assert len(ax.collections) == 1  # the fill_between band
    plt.close(fig)


def test_default_tail_vet_runs_for_gpd_cens(plotter_gpd, model_row_gpd):
    """The historical 3x3 tail vet predicts through the cap-aware path."""
    import matplotlib.pyplot as plt

    fig = plotter_gpd.vet_stage(model_row_gpd, stage="tail")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_pdf_panel_still_rejects_unsupported_family(plotter_gpd, model_row):
    # the shared model_row's gamma tail fit exists in results_dir; gamma is not a pdf-panel family
    with pytest.raises(ValueError, match="cont_pdf_quintile supports"):
        plotter_gpd.vet_stage(model_row, stage="tail", config=_tail_pdf_config())
