"""
Shared fixtures for viz tests.

Fits all four IS components on a small synthetic dataset, saves to a tmp dir,
and provides a model_row Series matching the dh_results.parquet schema.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must happen before any plt import

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.cache import component_id, save_result
from idd_tc_mortality.fit.fit_component import fit_one_component
from idd_tc_mortality.thresholds import compute_thresholds

_RNG = np.random.default_rng(99)
_N = 200

_EXPOSED = _RNG.uniform(10_000, 500_000, _N)
_WIND    = _RNG.normal(0, 1, _N)

_ETA_S1    = -9.5 + 0.5 * _WIND + np.log(_EXPOSED)
_P_S1      = 1.0 - np.exp(-np.exp(_ETA_S1))
_ANY_DEATH = _RNG.binomial(1, _P_S1).astype(float)

_MU_RATE = np.exp(-3.5 + 0.3 * _WIND)
_DR_RAW  = np.zeros(_N)
_S1_MASK = _ANY_DEATH == 1
_DR_RAW[_S1_MASK] = _RNG.gamma(5.0, _MU_RATE[_S1_MASK] / 5.0)
_DEATHS  = np.maximum(np.round(_DR_RAW * _EXPOSED), 0.0)
_DEATHS[_S1_MASK] = np.maximum(_DEATHS[_S1_MASK], 1.0)

_THRESHOLD_Q = 0.75
_COMBO = {"wind_speed": True, "sdi": False, "basin": False, "is_island": False}
_COMBO_JSON = json.dumps(_COMBO, sort_keys=True)

_DF = pd.DataFrame({
    "deaths":     _DEATHS,
    "exposed":    _EXPOSED,
    "wind_speed": _WIND,
    "sdi":        _RNG.uniform(0.3, 0.8, _N),
    "basin":      "NA",
    "is_island":  0.0,
})

_DR_COL = _DEATHS / _EXPOSED
_THRESHOLD_RATE = float(
    compute_thresholds(_DR_COL, np.array([_THRESHOLD_Q]))[_THRESHOLD_Q]
)


def _make_spec(component: str, family: str | None = None) -> dict:
    spec = {
        "component":          component,
        "covariate_combo":    _COMBO,
        "threshold_quantile": None if component == "s1" else _THRESHOLD_Q,
        "threshold_rate":     None if component == "s1" else _THRESHOLD_RATE,
        "family":             family,
        "fold_tag":           "is",
    }
    if component == "s1":
        spec["family"]        = family if family is not None else "cloglog"
        spec["exposure_mode"] = "offset"
    elif component == "s2":
        spec["exposure_mode"] = "free"
    else:
        spec["exposure_mode"] = "free+weight"
    return spec


@pytest.fixture(scope="module")
def synth_data() -> pd.DataFrame:
    return _DF


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    """Fit and save all four IS components; return the directory path."""
    tmp = tmp_path_factory.mktemp("stage_plots_results")
    for component, family in [
        ("s1",   "cloglog"),
        ("s2",   "cloglog"),
        ("bulk", "gamma"),
        ("tail", "gamma"),
    ]:
        spec   = _make_spec(component, family)
        result = fit_one_component(spec, _DF)
        save_result(result, spec, tmp, overwrite=True)
    return tmp


@pytest.fixture(scope="module")
def model_row() -> pd.Series:
    """Synthetic dh_results row for the fitted model configuration."""
    return pd.Series({
        "s1_cov":              _COMBO_JSON,
        "s2_cov":              _COMBO_JSON,
        "bulk_cov":            _COMBO_JSON,
        "tail_cov":            _COMBO_JSON,
        "threshold_quantile":  _THRESHOLD_Q,
        "s1_family":           "cloglog",
        "s1_exposure_mode":    "offset",
        "s2_family":           "cloglog",
        "s2_exposure_mode":    "free",
        "bulk_family":         "gamma",
        "bulk_exposure_mode":  "free+weight",
        "tail_family":         "gamma",
        "tail_exposure_mode":  "free+weight",
        "fold_tag":            "insample",
    })


@pytest.fixture(scope="module")
def stage_plotter(synth_data, results_dir):
    from idd_tc_mortality.viz.stage_plots import StagePlotter
    return StagePlotter(data=synth_data, results_dir=results_dir)
