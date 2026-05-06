"""
Tests for fit/fit_component.py.

Covers:
  - S1: full dataset, binary y, FitResult family='s1'.
  - S2: deaths>=1 subset, binary y, threshold_rate in meta.
  - Bulk rate models (gamma, lognormal): death_rate outcome, FitResult family matches.
  - Bulk count models (nb, poisson): raw deaths outcome, correct family.
  - Bulk special cases: beta (y/threshold_rate), scaled_logit (threshold_rate arg).
  - Tail rate models (gamma, lognormal, gpd): excess rate outcome.
  - Tail count models (nb, poisson): raw deaths, correct family.
  - Unknown component type raises ValueError.
  - All returned FitResults have params/param_names length parity.
  - S1 fit uses full df (not a subset).
  - S2 fit uses deaths>=1 subset.
  - Bulk/tail fitted_values length matches the relevant subset size.

DGP: n=400, seed=0 synthetic storms with wind_speed, SDI, and gamma bulk rates.
Threshold at 0.75 quantile.  All families are tested on this shared fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.fit.fit_component import fit_one_component
from idd_tc_mortality.thresholds import compute_thresholds


# ---------------------------------------------------------------------------
# Shared synthetic dataset
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_N = 400
_EXPOSED = _RNG.uniform(10_000, 500_000, _N)
_WIND = _RNG.normal(0, 1, _N)
_LOG_EXP = np.log(_EXPOSED)

_ETA_S1 = -10.0 + 0.6 * _WIND + _LOG_EXP
_P_S1 = 1.0 - np.exp(-np.exp(_ETA_S1))
_ANY_DEATH = _RNG.binomial(1, _P_S1).astype(float)

_MU_RATE = np.exp(-3.0 + 0.4 * _WIND)
_DEATH_RATE_RAW = np.zeros(_N)
_S1_MASK = _ANY_DEATH == 1
_DEATH_RATE_RAW[_S1_MASK] = _RNG.gamma(5.0, _MU_RATE[_S1_MASK] / 5.0)

_DEATHS_RAW = np.round(_DEATH_RATE_RAW * _EXPOSED).astype(float)
# Ensure deaths >= 1 wherever we drew a positive rate (avoid zero-rate rows in s1 subset)
_DEATHS_RAW[_S1_MASK] = np.maximum(_DEATHS_RAW[_S1_MASK], 1.0)

_DF = pd.DataFrame({
    "deaths":     _DEATHS_RAW,
    "exposed":    _EXPOSED,
    "wind_speed": _WIND,
    "sdi":        _RNG.uniform(0.3, 0.8, _N),
    "basin":      "NA",
    "is_island":  ((_RNG.uniform(size=_N) < 0.2)).astype(float),
})

_COMBO_ALL = {"wind_speed": True, "sdi": True, "basin": False, "is_island": False}
_COMBO_INT = {"wind_speed": False, "sdi": False, "basin": False, "is_island": False}

_THRESHOLD_Q = 0.75


# ---------------------------------------------------------------------------
# Helper to build a spec dict
# ---------------------------------------------------------------------------

def _spec(component: str, family: str | None = None, q: float | None = _THRESHOLD_Q,
          combo: dict | None = None, exposure_mode: str | None = None) -> dict:
    spec: dict = {
        "component":          component,
        "covariate_combo":    combo if combo is not None else _COMBO_ALL,
        "threshold_quantile": q,
        "family":             family,
    }
    if component == "s1":
        spec["family"]        = family if family is not None else "cloglog"
        spec["exposure_mode"] = exposure_mode if exposure_mode is not None else "offset"
    elif component == "s2":
        spec["exposure_mode"] = exposure_mode if exposure_mode is not None else "free"
    elif component in ("bulk", "tail"):
        spec["exposure_mode"] = exposure_mode if exposure_mode is not None else "free+weight"
    return spec


# ---------------------------------------------------------------------------
# S1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,exposure_mode", [
    ("cloglog", "offset"),
    ("cloglog", "free"),
    ("cloglog", "excluded"),
    ("logit",   "free"),
    ("logit",   "excluded"),
])
def test_s1_returns_fitresult(family, exposure_mode):
    result = fit_one_component(_spec("s1", family=family, exposure_mode=exposure_mode, q=None), _DF)
    assert result.family == "s1"
    assert result.meta["link"] == family
    assert result.meta["exposure_mode"] == exposure_mode


@pytest.mark.parametrize("family,exposure_mode", [
    ("cloglog", "offset"),
    ("cloglog", "free"),
    ("cloglog", "excluded"),
    ("logit",   "free"),
    ("logit",   "excluded"),
])
def test_s1_fitted_length_equals_full_n(family, exposure_mode):
    result = fit_one_component(_spec("s1", family=family, exposure_mode=exposure_mode, q=None), _DF)
    assert len(result.fitted_values) == len(_DF)


@pytest.mark.parametrize("family,exposure_mode", [
    ("cloglog", "offset"),
    ("cloglog", "free"),
    ("logit",   "free"),
    ("logit",   "excluded"),
])
def test_s1_converged(family, exposure_mode):
    result = fit_one_component(_spec("s1", family=family, exposure_mode=exposure_mode, q=None), _DF)
    assert result.converged


@pytest.mark.parametrize("family,exposure_mode", [
    ("cloglog", "offset"),
    ("cloglog", "free"),
    ("cloglog", "excluded"),
    ("logit",   "free"),
    ("logit",   "excluded"),
])
def test_s1_fitted_values_in_0_1(family, exposure_mode):
    result = fit_one_component(_spec("s1", family=family, exposure_mode=exposure_mode, q=None), _DF)
    assert np.all(result.fitted_values >= 0)
    assert np.all(result.fitted_values <= 1)


def test_s1_free_mode_includes_log_exposed_in_X():
    result = fit_one_component(_spec("s1", family="cloglog", exposure_mode="free", q=None,
                                     combo=_COMBO_INT), _DF)
    assert "log_exposed" in result.param_names


def test_s1_offset_mode_excludes_log_exposed_from_X():
    result = fit_one_component(_spec("s1", family="cloglog", exposure_mode="offset", q=None,
                                     combo=_COMBO_INT), _DF)
    assert "log_exposed" not in result.param_names


def test_s1_excluded_mode_excludes_log_exposed_from_X():
    result = fit_one_component(_spec("s1", family="logit", exposure_mode="excluded", q=None,
                                     combo=_COMBO_INT), _DF)
    assert "log_exposed" not in result.param_names


# ---------------------------------------------------------------------------
# S2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["logit", "cloglog"])
def test_s2_returns_fitresult(family):
    result = fit_one_component(_spec("s2", family=family), _DF)
    assert result.family == family


@pytest.mark.parametrize("family", ["logit", "cloglog"])
def test_s2_threshold_rate_in_meta(family):
    result = fit_one_component(_spec("s2", family=family), _DF)
    assert "threshold_rate" in result.meta
    assert result.meta["threshold_rate"] > 0


@pytest.mark.parametrize("family", ["logit", "cloglog"])
def test_s2_fitted_length_equals_s1_subset(family):
    n_s1 = int((_DF["deaths"].values >= 1).sum())
    result = fit_one_component(_spec("s2", family=family), _DF)
    assert len(result.fitted_values) == n_s1


@pytest.mark.parametrize("family", ["logit", "cloglog"])
def test_s2_params_param_names_parity(family):
    result = fit_one_component(_spec("s2", family=family), _DF)
    assert len(result.params) == len(result.param_names)


@pytest.mark.parametrize("family", ["logit", "cloglog"])
def test_s2_fitted_values_in_0_1(family):
    result = fit_one_component(_spec("s2", family=family), _DF)
    assert np.all(result.fitted_values >= 0)
    assert np.all(result.fitted_values <= 1)


@pytest.mark.parametrize("exposure_mode,expect_log_exposed", [
    ("free",     True),
    ("excluded", False),
])
def test_s2_exposure_mode_controls_log_exposed_in_X(exposure_mode, expect_log_exposed):
    result = fit_one_component(
        _spec("s2", family="logit", combo=_COMBO_INT, exposure_mode=exposure_mode), _DF
    )
    if expect_log_exposed:
        assert "log_exposed" in result.param_names
    else:
        assert "log_exposed" not in result.param_names


# ---------------------------------------------------------------------------
# Bulk — rate families
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["gamma", "lognormal"])
def test_bulk_rate_family_returns_correct_family(family):
    result = fit_one_component(_spec("bulk", family=family), _DF)
    assert result.family == family


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
def test_bulk_rate_family_fitted_length(family):
    death_rate = _DF["deaths"].values / _DF["exposed"].values
    threshold_rate = compute_thresholds(death_rate, np.array([_THRESHOLD_Q]))[_THRESHOLD_Q]
    expected_n = int(((_DF["deaths"].values >= 1) & (death_rate < threshold_rate)).sum())
    result = fit_one_component(_spec("bulk", family=family), _DF)
    assert len(result.fitted_values) == expected_n


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
def test_bulk_rate_family_log_exposed_in_param_names(family):
    result = fit_one_component(_spec("bulk", family=family), _DF)
    assert "log_exposed" in result.param_names


# ---------------------------------------------------------------------------
# Bulk — count families
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["nb", "poisson"])
def test_bulk_count_family_returns_correct_family(family):
    result = fit_one_component(_spec("bulk", family=family), _DF)
    assert result.family == family


@pytest.mark.parametrize("family", ["nb", "poisson"])
def test_bulk_count_family_log_exposed_not_in_X(family):
    result = fit_one_component(_spec("bulk", family=family), _DF)
    assert "log_exposed" not in result.param_names


# ---------------------------------------------------------------------------
# Bulk — special cases: beta and scaled_logit
# ---------------------------------------------------------------------------

def test_bulk_beta_returns_beta():
    result = fit_one_component(_spec("bulk", family="beta"), _DF)
    assert result.family == "beta"


def test_bulk_scaled_logit_returns_scaled_logit():
    result = fit_one_component(_spec("bulk", family="scaled_logit"), _DF)
    assert result.family == "scaled_logit"


def test_bulk_rate_family_log_exposed_in_param_names_default_mode():
    """Default exposure_mode='free+weight' includes log_exposed in X for bulk rate families."""
    result = fit_one_component(_spec("bulk", family="lognormal"), _DF)
    assert "log_exposed" in result.param_names


# ---------------------------------------------------------------------------
# Tail — rate families
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", [
    "gamma", "lognormal", "gpd", "truncated_normal", "weibull", "log_logistic",
])
def test_tail_rate_family_returns_correct_family(family):
    result = fit_one_component(_spec("tail", family=family), _DF)
    assert result.family == family


@pytest.mark.parametrize("family", [
    "gamma", "lognormal", "gpd", "truncated_normal", "weibull", "log_logistic",
])
def test_tail_rate_family_fitted_length(family):
    death_rate = _DF["deaths"].values / _DF["exposed"].values
    threshold_rate = compute_thresholds(death_rate, np.array([_THRESHOLD_Q]))[_THRESHOLD_Q]
    expected_n = int((death_rate >= threshold_rate).sum())
    result = fit_one_component(_spec("tail", family=family), _DF)
    assert len(result.fitted_values) == expected_n


@pytest.mark.parametrize("family", [
    "gamma", "lognormal", "gpd", "truncated_normal", "weibull", "log_logistic",
])
def test_tail_rate_family_log_exposed_in_param_names_default_mode(family):
    """Default exposure_mode='free+weight' includes log_exposed in X."""
    result = fit_one_component(_spec("tail", family=family), _DF)
    assert "log_exposed" in result.param_names


# ---------------------------------------------------------------------------
# Tail — count families
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["nb", "poisson"])
def test_tail_count_family_returns_correct_family(family):
    result = fit_one_component(_spec("tail", family=family), _DF)
    assert result.family == family


@pytest.mark.parametrize("family", ["nb", "poisson"])
def test_tail_count_family_log_exposed_not_in_X(family):
    result = fit_one_component(_spec("tail", family=family), _DF)
    assert "log_exposed" not in result.param_names


# ---------------------------------------------------------------------------
# Intercept-only covariate combo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component,family,exposure_mode,expected_names", [
    # S1 offset: log_exposed is a fixed offset, not in X → only intercept.
    ("s1",   "cloglog",       "offset",      ["const"]),
    # S1 free: log_exposed in X → intercept + log_exposed.
    ("s1",   "cloglog",       "free",        ["const", "log_exposed"]),
    # S1 excluded: no log_exposed → only intercept.
    ("s1",   "logit",         "excluded",    ["const"]),
    # S2 free: log_exposed is a free covariate in X → intercept + log_exposed.
    ("s2",   "cloglog",       "free",        ["const", "log_exposed"]),
    # S2 excluded: no log_exposed → only intercept.
    ("s2",   "logit",         "excluded",    ["const"]),
    # Rate-scale tail, default (free+weight): log_exposed in X.
    ("tail", "gamma",         "free+weight", ["const", "log_exposed"]),
    # Rate-scale tail, 'weight' mode: log_exposed NOT in X.
    ("tail", "gamma",         "weight",      ["const"]),
    # Rate-scale tail, 'excluded' mode: log_exposed NOT in X.
    ("tail", "gamma",         "excluded",    ["const"]),
    # Rate-scale tail, 'free' mode: log_exposed in X.
    ("tail", "gamma",         "free",        ["const", "log_exposed"]),
    # Rate-scale bulk, default: log_exposed in X.
    ("bulk", "lognormal",     "free+weight", ["const", "log_exposed"]),
    # Rate-scale bulk, 'weight' mode: no log_exposed.
    ("bulk", "lognormal",     "weight",      ["const"]),
    # Count bulk/tail: log_exposed is an offset → only intercept regardless of mode.
    ("bulk", "nb",            "offset",      ["const"]),
    ("tail", "nb",            "offset",      ["const"]),
])
def test_intercept_only_combo(component, family, exposure_mode, expected_names):
    spec = _spec(component, family=family, combo=_COMBO_INT, exposure_mode=exposure_mode)
    result = fit_one_component(spec, _DF)
    assert result.param_names == expected_names


# ---------------------------------------------------------------------------
# Unknown component raises
# ---------------------------------------------------------------------------

def test_unknown_component_raises():
    spec = _spec("unknown")
    with pytest.raises(ValueError, match="Unknown component type"):
        fit_one_component(spec, _DF)


# ---------------------------------------------------------------------------
# params / param_names parity for all families (parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component,family,exposure_mode,q", [
    ("s1",   "cloglog",         "offset",      None),
    ("s1",   "cloglog",         "free",        None),
    ("s1",   "cloglog",         "excluded",    None),
    ("s1",   "logit",           "free",        None),
    ("s1",   "logit",           "excluded",    None),
    ("s2",   "cloglog",         "free",        _THRESHOLD_Q),
    ("s2",   "cloglog",         "excluded",    _THRESHOLD_Q),
    ("s2",   "logit",           "free",        _THRESHOLD_Q),
    ("s2",   "logit",           "excluded",    _THRESHOLD_Q),
    ("bulk", "lognormal",       "free+weight", _THRESHOLD_Q),
    ("bulk", "beta",            "free+weight", _THRESHOLD_Q),
    ("bulk", "scaled_logit",    "free+weight", _THRESHOLD_Q),
    ("bulk", "nb",              "offset",      _THRESHOLD_Q),
    ("bulk", "poisson",         "offset",      _THRESHOLD_Q),
    ("tail", "gamma",           "free+weight", _THRESHOLD_Q),
    ("tail", "lognormal",       "free+weight", _THRESHOLD_Q),
    ("tail", "gpd",             "free+weight", _THRESHOLD_Q),
    ("tail", "truncated_normal","free+weight", _THRESHOLD_Q),
    ("tail", "weibull",         "free+weight", _THRESHOLD_Q),
    ("tail", "log_logistic",    "free+weight", _THRESHOLD_Q),
    ("tail", "nb",              "offset",      _THRESHOLD_Q),
    ("tail", "poisson",         "offset",      _THRESHOLD_Q),
])
def test_params_param_names_parity(component, family, exposure_mode, q):
    result = fit_one_component(_spec(component, family=family, exposure_mode=exposure_mode, q=q), _DF)
    assert len(result.params) == len(result.param_names), (
        f"{component}/{family}/{exposure_mode}: params length {len(result.params)} != "
        f"param_names length {len(result.param_names)}"
    )


# ---------------------------------------------------------------------------
# exposure_mode dispatch — weights and include_log_exposed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exposure_mode,expect_log_exposed", [
    ("free",        True),
    ("weight",      False),
    ("free+weight", True),
    ("excluded",    False),
])
def test_exposure_mode_controls_log_exposed_in_X(exposure_mode, expect_log_exposed):
    """exposure_mode correctly includes/excludes log_exposed from the design matrix."""
    spec = _spec("tail", family="lognormal", combo=_COMBO_INT, exposure_mode=exposure_mode)
    result = fit_one_component(spec, _DF)
    if expect_log_exposed:
        assert "log_exposed" in result.param_names, (
            f"exposure_mode={exposure_mode!r}: expected log_exposed in param_names"
        )
    else:
        assert "log_exposed" not in result.param_names, (
            f"exposure_mode={exposure_mode!r}: expected log_exposed NOT in param_names"
        )


@pytest.mark.parametrize("exposure_mode", ["free", "weight", "free+weight", "excluded"])
def test_exposure_mode_rate_families_return_positive_fitted_values(exposure_mode):
    """All four rate exposure modes produce positive finite fitted values."""
    spec = _spec("bulk", family="lognormal", exposure_mode=exposure_mode)
    result = fit_one_component(spec, _DF)
    assert np.all(result.fitted_values > 0), f"lognormal/{exposure_mode}: fitted values must be positive"
    assert np.all(np.isfinite(result.fitted_values)), f"lognormal/{exposure_mode}: fitted values must be finite"


# ---------------------------------------------------------------------------
# Graceful non-convergence
# ---------------------------------------------------------------------------

def test_fit_error_returns_non_converged_fitresult(monkeypatch):
    """A fit exception returns a non-converged FitResult rather than raising."""
    import idd_tc_mortality.s1 as s1_mod

    def _bad_fit(*args, **kwargs):
        raise RuntimeError("simulated IRLS explosion")

    monkeypatch.setattr(s1_mod, "fit", _bad_fit)

    spec = _spec("s1", family="cloglog", exposure_mode="offset")
    result = fit_one_component(spec, _DF)

    assert result.converged is False
    assert "fit_error" in result.meta
    assert "simulated IRLS explosion" in result.meta["fit_error"]


def test_fit_error_result_has_nan_params(monkeypatch):
    """Non-converged FitResult has NaN params and a sentinel param name."""
    import idd_tc_mortality.s1 as s1_mod

    monkeypatch.setattr(s1_mod, "fit", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    spec = _spec("s1", family="cloglog", exposure_mode="offset")
    result = fit_one_component(spec, _DF)

    assert np.all(np.isnan(result.params))
    assert result.param_names == ["__failed__"]
