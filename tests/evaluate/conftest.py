"""
Shared fixtures for evaluate tests.

A single module-scoped 'pipeline' fixture generates a small synthetic DataFrame,
fits all four double-hurdle components, and returns them along with the specs and
pre-computed threshold_rate. Both test files import from this conftest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.cache import component_id, save_result
from idd_tc_mortality.cv import compute_fold_assignments
from idd_tc_mortality.fit.fit_component import fit_one_component
from idd_tc_mortality.thresholds import compute_thresholds

# ---------------------------------------------------------------------------
# Synthetic DGP — module-level constants so the data is deterministic
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(7)
_N   = 300

_EXPOSED    = _RNG.uniform(10_000, 500_000, _N)
_WIND       = _RNG.normal(0, 1, _N)
_LOG_EXP    = np.log(_EXPOSED)

_ETA_S1 = -9.5 + 0.5 * _WIND + _LOG_EXP
_P_S1   = 1.0 - np.exp(-np.exp(_ETA_S1))
_ANY_DEATH = _RNG.binomial(1, _P_S1).astype(float)

_MU_RATE = np.exp(-3.5 + 0.3 * _WIND)
_DEATH_RATE_RAW = np.zeros(_N)
_S1_MASK = _ANY_DEATH == 1
_DEATH_RATE_RAW[_S1_MASK] = _RNG.gamma(5.0, _MU_RATE[_S1_MASK] / 5.0)
_DEATHS_RAW = np.maximum(np.round(_DEATH_RATE_RAW * _EXPOSED), 0.0)
_DEATHS_RAW[_S1_MASK] = np.maximum(_DEATHS_RAW[_S1_MASK], 1.0)

_THRESHOLD_Q = 0.75
_COMBO = {"wind_speed": True, "sdi": False, "basin": False, "is_island": False}

_DF = pd.DataFrame({
    "deaths":     _DEATHS_RAW,
    "exposed":    _EXPOSED,
    "wind_speed": _WIND,
    "sdi":        _RNG.uniform(0.3, 0.8, _N),
    "basin":      "NA",
    "is_island":  0.0,
})

_DEATH_RATE_COL = _DEATHS_RAW / _EXPOSED
_THRESHOLD_RATE = float(
    compute_thresholds(_DEATH_RATE_COL, np.array([_THRESHOLD_Q]))[_THRESHOLD_Q]
)


def _make_spec(component: str, family: str | None = None, fold_tag: str = "is") -> dict:
    spec = {
        "component":          component,
        "covariate_combo":    _COMBO,
        "threshold_quantile": None if component == "s1" else _THRESHOLD_Q,
        "threshold_rate":     None if component == "s1" else _THRESHOLD_RATE,
        "family":             family,
        "fold_tag":           fold_tag,
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
def pipeline():
    """Fit all four double-hurdle components on the shared synthetic dataset."""
    s1_spec   = _make_spec("s1")
    s2_spec   = _make_spec("s2",   family="cloglog")
    bulk_spec = _make_spec("bulk", family="gamma")
    tail_spec = _make_spec("tail", family="gamma")

    s1_result   = fit_one_component(s1_spec,   _DF)
    s2_result   = fit_one_component(s2_spec,   _DF)
    bulk_result = fit_one_component(bulk_spec, _DF)
    tail_result = fit_one_component(tail_spec, _DF)

    return {
        "df":              _DF,
        "death_rate":      _DEATH_RATE_COL,
        "threshold_rate":  _THRESHOLD_RATE,
        "threshold_q":     _THRESHOLD_Q,
        "combo":           _COMBO,
        "s1_spec":         s1_spec,
        "s2_spec":         s2_spec,
        "bulk_spec":       bulk_spec,
        "tail_spec":       tail_spec,
        "s1_result":       s1_result,
        "s2_result":       s2_result,
        "bulk_result":     bulk_result,
        "tail_result":     tail_result,
        "any_death":       _ANY_DEATH,
        "s1_mask":         _S1_MASK,
        "bulk_mask":       _S1_MASK & (_DEATH_RATE_COL < _THRESHOLD_RATE),
        "tail_mask":       _DEATH_RATE_COL >= _THRESHOLD_RATE,
    }


# ---------------------------------------------------------------------------
# OOS pipeline fixture — fits IS + 2-fold OOS components, saves to tmp dir
# ---------------------------------------------------------------------------

_OOS_N_SEEDS = 1
_OOS_N_FOLDS = 2


@pytest.fixture(scope="module")
def oos_pipeline(tmp_path_factory):
    """IS + OOS component fits saved to disk for testing assemble_oos_predictions.

    Uses 1 seed and 2 folds to minimise fitting time. All four components
    (s1, s2, bulk=gamma, tail=gamma) are fitted for each fold.
    """
    tmp = tmp_path_factory.mktemp("oos_results")

    is_specs = {
        "s1":   _make_spec("s1"),
        "s2":   _make_spec("s2",   family="cloglog"),
        "bulk": _make_spec("bulk", family="gamma"),
        "tail": _make_spec("tail", family="gamma"),
    }

    # Fit and save IS components.
    is_results = {}
    for key, spec in is_specs.items():
        result = fit_one_component(spec, _DF)
        save_result(result, spec, tmp, overwrite=True)
        is_results[key] = result

    fold_assignments = compute_fold_assignments(
        _DF, n_seeds=_OOS_N_SEEDS, n_folds=_OOS_N_FOLDS
    )

    # Fit and save OOS components for seed=0, folds 0 and 1.
    for fold in range(_OOS_N_FOLDS):
        fold_tag = f"s0_f{fold}"
        train_mask = fold_assignments["seed_0"].values != fold
        df_train = _DF.iloc[train_mask]

        for key, is_spec in is_specs.items():
            oos_spec = {**is_spec, "fold_tag": fold_tag}
            result = fit_one_component(oos_spec, df_train)
            save_result(result, oos_spec, tmp, overwrite=True)

    model_spec_key = {
        "s1_spec":   is_specs["s1"],
        "s2_spec":   is_specs["s2"],
        "bulk_spec": is_specs["bulk"],
        "tail_spec": is_specs["tail"],
    }

    return {
        "df":               _DF,
        "fold_assignments": fold_assignments,
        "results_dir":      tmp,
        "model_spec_key":   model_spec_key,
        "n_folds":          _OOS_N_FOLDS,
        "n_seeds":          _OOS_N_SEEDS,
        "is_results":       is_results,
        "is_specs":         is_specs,
    }
