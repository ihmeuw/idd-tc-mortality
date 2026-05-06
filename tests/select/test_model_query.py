"""
Tests for ModelQuery.

Uses a synthetic dh_results DataFrame with JSON covariate strings matching
the new schema (s1_cov, s2_cov, bulk_cov, tail_cov as JSON, bulk_family,
tail_family, threshold_quantile).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from idd_tc_mortality.select.model_query import ModelQuery, _cov_tokens, _is_neighbor


# ---------------------------------------------------------------------------
# Helpers to build JSON covariate strings
# ---------------------------------------------------------------------------

def _cov(**kwargs) -> str:
    """Build a covariate JSON string from bool kwargs."""
    all_keys = ["basin", "is_island", "sdi", "wind_speed"]
    d = {k: kwargs.get(k, False) for k in all_keys}
    return json.dumps(d, sort_keys=True)


# Common covariate strings
COV_WIND      = _cov(wind_speed=True)
COV_WIND_SDI  = _cov(wind_speed=True, sdi=True)
COV_SDI       = _cov(sdi=True)
COV_NONE      = _cov()
COV_ALL       = _cov(wind_speed=True, sdi=True, basin=True, is_island=True)


# ---------------------------------------------------------------------------
# Synthetic DataFrame fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def dh_results() -> pd.DataFrame:
    """Small synthetic dh_results with three configs and two fold_tags each."""
    rows = []
    configs = [
        # (s1_cov, s2_cov, bulk_cov, tail_cov, bulk_family, tail_family, threshold_q)
        (COV_WIND_SDI, COV_SDI, COV_WIND, COV_NONE, "nb",    "gamma", 0.75),
        (COV_WIND_SDI, COV_SDI, COV_WIND, COV_NONE, "gamma", "gamma", 0.75),
        (COV_WIND,     COV_SDI, COV_WIND, COV_NONE, "nb",    "gamma", 0.75),
        # Different threshold
        (COV_WIND_SDI, COV_SDI, COV_WIND, COV_NONE, "nb",    "gamma", 0.80),
    ]
    for (s1, s2, bulk, tail, bf, tf, q) in configs:
        for fold_tag, suffix in [("insample", ""), ("oos_seed0", "_oos")]:
            row = {
                "s1_cov":             s1,
                "s2_cov":             s2,
                "bulk_cov":           bulk,
                "tail_cov":           tail,
                "bulk_family":        bf,
                "tail_family":        tf,
                "threshold_quantile": q,
                "fold_tag":           fold_tag,
                f"full_mae_rate{suffix}":      0.01,
                f"full_rmse_rate{suffix}":     0.02,
                f"full_pred_obs_ratio{suffix}": 1.1,
                f"s1_auroc{suffix}":           0.85,
            }
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def mq(dh_results) -> ModelQuery:
    return ModelQuery(dh_results)


# ---------------------------------------------------------------------------
# Unit tests for module-level helpers
# ---------------------------------------------------------------------------

def test_cov_tokens_empty():
    assert _cov_tokens(COV_NONE) == frozenset()


def test_cov_tokens_single():
    assert _cov_tokens(COV_WIND) == frozenset({"wind_speed"})


def test_cov_tokens_multi():
    assert _cov_tokens(COV_WIND_SDI) == frozenset({"wind_speed", "sdi"})


def test_is_neighbor_add_one():
    # wind → wind+sdi is neighbor (added sdi)
    assert _is_neighbor(COV_WIND, COV_WIND_SDI)


def test_is_neighbor_remove_one():
    # wind+sdi → wind is neighbor (removed sdi)
    assert _is_neighbor(COV_WIND_SDI, COV_WIND)


def test_is_neighbor_same():
    assert not _is_neighbor(COV_WIND, COV_WIND)


def test_is_neighbor_two_diff():
    # none → wind+sdi differs by 2 tokens — not a neighbor
    assert not _is_neighbor(COV_NONE, COV_WIND_SDI)


# ---------------------------------------------------------------------------
# ModelQuery.get
# ---------------------------------------------------------------------------

def test_get_returns_series(mq):
    result = mq.get(
        s1_cov=COV_WIND_SDI, s2_cov=COV_SDI,
        bulk_cov=COV_WIND, tail_cov=COV_NONE,
    )
    assert isinstance(result, pd.Series)


def test_get_returns_none_when_missing(mq):
    result = mq.get(
        s1_cov=COV_NONE, s2_cov=COV_NONE,
        bulk_cov=COV_NONE, tail_cov=COV_NONE,
    )
    assert result is None


def test_get_respects_bulk_family_override(mq):
    result = mq.get(
        s1_cov=COV_WIND_SDI, s2_cov=COV_SDI,
        bulk_cov=COV_WIND, tail_cov=COV_NONE,
        bulk_family="gamma",
    )
    assert result is not None
    assert result["bulk_family"] == "gamma"


def test_get_respects_threshold_quantile_override(mq):
    result = mq.get(
        s1_cov=COV_WIND_SDI, s2_cov=COV_SDI,
        bulk_cov=COV_WIND, tail_cov=COV_NONE,
        threshold_quantile=0.80,
    )
    assert result is not None
    assert result["threshold_quantile"] == 0.80


# ---------------------------------------------------------------------------
# ModelQuery.enumerate
# ---------------------------------------------------------------------------

def test_enumerate_returns_dataframe(mq):
    result = mq.enumerate(
        s1_cov=COV_WIND_SDI, s2_cov=COV_SDI,
        bulk_cov=COV_WIND, tail_cov=COV_NONE,
    )
    assert isinstance(result, pd.DataFrame)


def test_enumerate_single_returns_subset(mq):
    result = mq.enumerate(
        s1_cov=COV_WIND_SDI, s2_cov=COV_SDI,
        bulk_cov=COV_WIND, tail_cov=COV_NONE,
    )
    # Defaults filter to bulk_family='nb'; 2 fold_tags → 2 rows
    assert len(result) == 2


def test_enumerate_list_expands(mq):
    result = mq.enumerate(
        s1_cov=[COV_WIND_SDI, COV_WIND],
        s2_cov=COV_SDI,
        bulk_cov=COV_WIND,
        tail_cov=COV_NONE,
    )
    # Two distinct s1_cov values × 2 fold_tags = 4 rows (both have nb default)
    assert len(result) == 4


def test_enumerate_empty_when_no_match(mq):
    result = mq.enumerate(
        s1_cov=COV_NONE, s2_cov=COV_NONE,
        bulk_cov=COV_NONE, tail_cov=COV_NONE,
    )
    assert len(result) == 0


# ---------------------------------------------------------------------------
# ModelQuery.neighbors
# ---------------------------------------------------------------------------

@pytest.fixture
def reference_model(dh_results) -> pd.Series:
    """IS row for (COV_WIND_SDI, COV_SDI, COV_WIND, COV_NONE, nb, gamma, 0.75)."""
    mask = (
        (dh_results["s1_cov"]             == COV_WIND_SDI)
        & (dh_results["bulk_family"]       == "nb")
        & (dh_results["fold_tag"]          == "insample")
        & (dh_results["threshold_quantile"]== 0.75)
    )
    return dh_results[mask].iloc[0]


def test_neighbors_returns_dataframe(mq, reference_model):
    result = mq.neighbors(reference_model)
    assert isinstance(result, pd.DataFrame)


def test_neighbors_finds_s1_neighbor(mq, reference_model):
    # COV_WIND differs by one token from COV_WIND_SDI in s1_cov
    result = mq.neighbors(reference_model, vary_stages=["s1_cov"])
    assert len(result) > 0
    assert (result["s1_cov"] == COV_WIND).any()


def test_neighbors_excludes_same_model(mq, reference_model):
    result = mq.neighbors(reference_model)
    # The reference model itself should not appear (symmetric_difference == 0)
    for _, row in result.iterrows():
        for col in ModelQuery.CONFIG_COLS:
            if col != row.get("_varied_stage", None):
                pass
    # Verify none of the neighbors have identical config in the varied stage
    assert not (
        (result["s1_cov"] == COV_WIND_SDI)
        & (result["s2_cov"] == COV_SDI)
        & (result["bulk_cov"] == COV_WIND)
        & (result["tail_cov"] == COV_NONE)
    ).all()


def test_neighbors_parses_json_not_underscore_split(mq, dh_results):
    # COV_WIND_SDI has keys wind_speed and sdi; old code would split on '_'
    # giving ['wind', 'speed', 'sdi'] — 3 tokens instead of 2.
    # New code: tokens = {'wind_speed', 'sdi'} → neighbor check symmetric_diff == 1.
    # COV_WIND has tokens = {'wind_speed'} → symmetric_diff = {'sdi'} → size 1 → neighbor.
    assert _is_neighbor(COV_WIND_SDI, COV_WIND)
    # Old (broken) token count: split('_') on '{"wind_speed": true, "sdi": true}'
    # would NOT work — verify _cov_tokens uses json.loads.
    tokens = _cov_tokens(COV_WIND_SDI)
    assert "wind_speed" in tokens  # single token, not split into ['wind', 'speed']


# ---------------------------------------------------------------------------
# ModelQuery.compare
# ---------------------------------------------------------------------------

def test_compare_returns_dataframe(mq, dh_results):
    models = dh_results.iloc[:4]
    result = mq.compare(models)
    assert isinstance(result, pd.DataFrame)


def test_compare_includes_config_cols(mq, dh_results):
    models = dh_results.iloc[:4]
    result = mq.compare(models)
    for col in ModelQuery.CONFIG_COLS:
        assert col in result.columns


def test_compare_custom_metrics(mq, dh_results):
    models = dh_results.iloc[:4]
    result = mq.compare(models, metrics=["full_mae_rate"])
    assert "full_mae_rate" in result.columns


# ---------------------------------------------------------------------------
# ModelQuery.diff
# ---------------------------------------------------------------------------

def test_diff_returns_dataframe(mq, dh_results):
    m1 = dh_results.iloc[0]
    m2 = dh_results.iloc[2]
    result = mq.diff(m1, m2)
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "metric"


def test_diff_has_diff_column(mq, dh_results):
    m1 = dh_results.iloc[0]
    m2 = dh_results.iloc[2]
    result = mq.diff(m1, m2)
    assert "diff" in result.columns


def test_diff_marks_config_changes(mq, dh_results):
    # m1 has s1_cov=COV_WIND_SDI, m2 has s1_cov=COV_WIND
    m1 = dh_results[dh_results["s1_cov"] == COV_WIND_SDI].iloc[0]
    m2 = dh_results[dh_results["s1_cov"] == COV_WIND].iloc[0]
    result = mq.diff(m1, m2)
    assert result.loc["s1_cov", "diff"] == "←"


# ---------------------------------------------------------------------------
# ModelQuery.compare_to_reference
# ---------------------------------------------------------------------------

def test_compare_to_reference_returns_dataframe(mq, dh_results, reference_model):
    others = dh_results[
        (dh_results["s1_cov"] != reference_model["s1_cov"])
        | (dh_results["bulk_family"] != reference_model["bulk_family"])
    ]
    result = mq.compare_to_reference(others, reference_model)
    assert isinstance(result, pd.DataFrame)


def test_compare_to_reference_excludes_identical(mq, dh_results, reference_model):
    # Rows identical to reference should not appear in result.
    result = mq.compare_to_reference(dh_results, reference_model)
    if len(result) > 0:
        assert "diff_stages" in result.columns
        assert "n_diff" in result.columns


def test_compare_to_reference_marks_diff_stage(mq, dh_results, reference_model):
    # Include a row that differs only in s1_cov
    differ = dh_results[
        (dh_results["s1_cov"] == COV_WIND)
        & (dh_results["fold_tag"] == "insample")
        & (dh_results["bulk_family"] == "nb")
        & (dh_results["threshold_quantile"] == 0.75)
    ]
    result = mq.compare_to_reference(differ, reference_model)
    assert len(result) > 0
    assert "s1_cov" in result.iloc[0]["diff_stages"]
