"""
Tests for cache.py.

Covers:
  - Same spec always produces the same component_id.
  - Different specs produce different IDs.
  - Key order in spec does not affect ID.
  - save_result writes both pkl and json files.
  - load_result roundtrips a FitResult exactly.
  - save_result raises FileExistsError without overwrite=True.
  - save_result succeeds with overwrite=True.
  - result_exists returns True after save, False before.
  - load_result raises FileNotFoundError with helpful message.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from idd_tc_mortality.cache import (
    component_id,
    load_result,
    model_id,
    result_exists,
    save_result,
)
from idd_tc_mortality.distributions.base import FitResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spec():
    return {
        "threshold_quantile": 0.85,
        "covariates": ["age", "sex"],
        "family": "gamma",
        "component_type": "bulk",
    }


@pytest.fixture
def fit_result():
    return FitResult(
        params=np.array([0.5, -0.3, 1.2]),
        param_names=["intercept", "age", "sex"],
        fitted_values=np.array([0.01, 0.02, 0.015]),
        family="gamma",
        converged=True,
        cov=None,
        meta={"n_iter": 12, "warning": None},
    )


# ---------------------------------------------------------------------------
# component_id
# ---------------------------------------------------------------------------

def test_same_spec_same_id(spec):
    assert component_id(spec) == component_id(spec)


def test_different_specs_different_ids(spec):
    other = {**spec, "family": "lognormal"}
    assert component_id(spec) != component_id(other)


def test_key_order_irrelevant(spec):
    # Reverse key insertion order
    reversed_spec = dict(reversed(list(spec.items())))
    assert component_id(spec) == component_id(reversed_spec)


def test_id_is_hex_string(spec):
    cid = component_id(spec)
    assert isinstance(cid, str)
    assert len(cid) == 32
    int(cid, 16)  # raises if not valid hex


# ---------------------------------------------------------------------------
# model_id
# ---------------------------------------------------------------------------

@pytest.fixture
def four_specs():
    base = {"covariate_combo": {"wind_speed": True}, "threshold_quantile": 0.80, "fold_tag": "is"}
    return (
        {**base, "component": "s1",   "family": None},
        {**base, "component": "s2",   "family": None},
        {**base, "component": "bulk", "family": "gamma"},
        {**base, "component": "tail", "family": "gpd"},
    )


def test_model_id_is_hex_string(four_specs):
    mid = model_id(*four_specs)
    assert isinstance(mid, str)
    assert len(mid) == 32
    int(mid, 16)  # raises if not valid hex


def test_model_id_deterministic(four_specs):
    assert model_id(*four_specs) == model_id(*four_specs)


def test_model_id_changes_with_component(four_specs):
    s1, s2, bulk, tail = four_specs
    alt_tail = {**tail, "family": "gamma"}
    assert model_id(s1, s2, bulk, tail) != model_id(s1, s2, bulk, alt_tail)


def test_model_id_changes_with_fold_tag(four_specs):
    s1, s2, bulk, tail = four_specs
    oos_bulk = {**bulk, "fold_tag": "s0_f2"}
    assert model_id(s1, s2, bulk, tail) != model_id(s1, s2, oos_bulk, tail)


def test_model_id_distinct_from_component_id(four_specs):
    s1, s2, bulk, tail = four_specs
    mid = model_id(*four_specs)
    for spec in four_specs:
        assert mid != component_id(spec)


# ---------------------------------------------------------------------------
# save_result / result_exists / load_result
# ---------------------------------------------------------------------------

def test_save_writes_pkl_and_json(tmp_path, spec, fit_result):
    save_result(fit_result, spec, tmp_path)
    cid = component_id(spec)
    assert (tmp_path / f"{cid}.pkl").exists()
    assert (tmp_path / f"{cid}.json").exists()


def test_json_content(tmp_path, spec, fit_result):
    save_result(fit_result, spec, tmp_path)
    cid = component_id(spec)
    with open(tmp_path / f"{cid}.json") as f:
        meta = json.load(f)
    assert meta["component_id"] == cid
    assert meta["spec"] == spec
    assert meta["family"] == fit_result.family
    assert meta["converged"] == fit_result.converged
    assert meta["n_obs"] == len(fit_result.fitted_values)
    assert meta["param_names"] == fit_result.param_names


def test_load_result_roundtrip(tmp_path, spec, fit_result):
    save_result(fit_result, spec, tmp_path)
    cid = component_id(spec)
    loaded = load_result(cid, tmp_path)

    np.testing.assert_array_equal(loaded.params, fit_result.params)
    assert loaded.param_names == fit_result.param_names
    assert loaded.family == fit_result.family
    assert loaded.converged == fit_result.converged
    assert loaded.meta == fit_result.meta
    np.testing.assert_array_equal(loaded.fitted_values, fit_result.fitted_values)
    assert loaded.cov is None


def test_save_raises_on_existing_without_overwrite(tmp_path, spec, fit_result):
    save_result(fit_result, spec, tmp_path)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_result(fit_result, spec, tmp_path)


def test_save_succeeds_with_overwrite(tmp_path, spec, fit_result):
    save_result(fit_result, spec, tmp_path)
    # Mutate params so we can verify the overwrite took effect
    updated = FitResult(
        params=np.array([9.9, 8.8, 7.7]),
        param_names=fit_result.param_names,
        fitted_values=fit_result.fitted_values,
        family=fit_result.family,
    )
    save_result(updated, spec, tmp_path, overwrite=True)
    cid = component_id(spec)
    loaded = load_result(cid, tmp_path)
    np.testing.assert_array_equal(loaded.params, updated.params)


def test_result_exists_false_before_save(tmp_path, spec):
    cid = component_id(spec)
    assert result_exists(cid, tmp_path) is False


def test_result_exists_true_after_save(tmp_path, spec, fit_result):
    save_result(fit_result, spec, tmp_path)
    cid = component_id(spec)
    assert result_exists(cid, tmp_path) is True


def test_load_result_missing_raises_file_not_found(tmp_path):
    fake_id = "a" * 32
    with pytest.raises(FileNotFoundError, match=fake_id):
        load_result(fake_id, tmp_path)


def test_load_result_error_message_contains_output_dir(tmp_path):
    fake_id = "b" * 32
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        load_result(fake_id, tmp_path)


def test_save_creates_output_dir_if_missing(tmp_path, spec, fit_result):
    nested = tmp_path / "a" / "b" / "c"
    save_result(fit_result, spec, nested)
    cid = component_id(spec)
    assert (nested / f"{cid}.pkl").exists()
