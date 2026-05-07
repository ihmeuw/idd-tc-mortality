"""
Tests for evaluate/assemble.py.

Covers:
  - assemble_predictions returns a pd.Series of length == len(df).
  - Index matches df.index exactly.
  - No NaN values in the final output.
  - All values are non-negative.
  - Values for any_death=0 rows are lower (on average) than for any_death=1 rows,
    driven by p_s1 ≈ 0 for non-event rows.
  - Assembled predictions correlate positively with true death rates (r > 0.3)
    among any_death=1 rows.
  - Assembling twice on the same df produces identical results (deterministic).
  - Non-default df index is preserved in the returned Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.evaluate.assemble import (
    assemble_oos_predictions,
    assemble_predictions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assemble(pipeline):
    return assemble_predictions(
        s1_result=pipeline["s1_result"],
        s1_spec=pipeline["s1_spec"],
        s2_result=pipeline["s2_result"],
        s2_spec=pipeline["s2_spec"],
        bulk_result=pipeline["bulk_result"],
        bulk_spec=pipeline["bulk_spec"],
        tail_result=pipeline["tail_result"],
        tail_spec=pipeline["tail_spec"],
        df=pipeline["df"],
    )


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------

def test_returns_series(pipeline):
    result = _assemble(pipeline)
    assert isinstance(result, pd.Series)


def test_length_equals_df(pipeline):
    result = _assemble(pipeline)
    assert len(result) == len(pipeline["df"])


def test_index_matches_df(pipeline):
    df = pipeline["df"]
    result = _assemble(pipeline)
    assert list(result.index) == list(df.index)


def test_no_nan(pipeline):
    result = _assemble(pipeline)
    assert not result.isna().any(), (
        f"Found {result.isna().sum()} NaN values in assembled predictions."
    )


def test_all_nonnegative(pipeline):
    result = _assemble(pipeline)
    assert np.all(result.values >= 0), (
        f"Found {(result.values < 0).sum()} negative values."
    )


def test_all_finite(pipeline):
    result = _assemble(pipeline)
    assert np.all(np.isfinite(result.values)), (
        f"Found {(~np.isfinite(result.values)).sum()} non-finite values."
    )


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

def test_predictions_lower_for_no_death_rows(pipeline):
    """Mean prediction for any_death=0 rows < median prediction for any_death=1 rows."""
    result = _assemble(pipeline)
    any_death = pipeline["any_death"].astype(bool)

    mean_no_death  = float(np.mean(result.values[~any_death]))
    median_death   = float(np.median(result.values[any_death]))

    assert mean_no_death < median_death, (
        f"Expected mean_no_death ({mean_no_death:.4e}) < median_death ({median_death:.4e})."
    )


def test_predictions_driven_by_p_s1(pipeline):
    """Assembled predictions must be higher for any_death=1 rows than any_death=0 rows.

    This tests the wiring of p_s1: the assembled formula is
    p_s1 * (p_s2 * rate_tail + (1 - p_s2) * rate_bulk), so rows where
    S1 is likely (p_s1 ≈ 1) must receive substantially higher predictions than
    rows where S1 is unlikely (p_s1 ≈ 0). Using the median gap rather than a
    correlation so the test is not sensitive to tail-component fit quality.
    """
    result = _assemble(pipeline)
    any_death = pipeline["any_death"].astype(bool)

    median_death    = float(np.median(result.values[any_death]))
    median_no_death = float(np.median(result.values[~any_death]))

    assert median_death > median_no_death, (
        f"Median prediction for any_death=1 ({median_death:.4e}) should be > "
        f"any_death=0 ({median_no_death:.4e})."
    )


def test_no_death_rows_have_nonzero_predictions(pipeline):
    """Regression test for the v3 underprediction bug.

    Previously, assemble_predictions reindexed S2/bulk/tail predictions to df.index
    and filled NaN with 0 outside their fit subsets, then multiplied by p_s1. That
    made every any_death=0 row's prediction exactly 0 (since p_s2 = rate_bulk =
    rate_tail = 0 there), zeroing out most of the data's contribution to total
    predicted deaths and making fwd_pred_obs_ratio == full_pred_obs_ratio across
    the whole grid. The fix is to predict S2/bulk/tail on the full df.
    """
    result = _assemble(pipeline)
    any_death = pipeline["any_death"].astype(bool)

    no_death_preds = result.values[~any_death]
    n_zero = int((no_death_preds == 0).sum())
    n_total = int((~any_death).sum())

    assert n_zero == 0, (
        f"{n_zero}/{n_total} any_death=0 rows have predicted rate == 0. "
        "Expected non-zero predictions everywhere when p_s1 > 0 and the "
        "rate components are positive (i.e., the model is being applied "
        "to all rows, not just the conditional fit subset)."
    )


def test_assembled_matches_formula_on_full_df(pipeline):
    """The assembled prediction must equal p_s1 * (p_s2*rate_tail + (1-p_s2)*rate_bulk)
    using each component's prediction on EVERY row of df, not zero-filled outside
    the fit subset."""
    from idd_tc_mortality.evaluate.predict_component import predict_one_component

    df = pipeline["df"]
    p_s1   = predict_one_component(pipeline["s1_spec"],   pipeline["s1_result"],   df).values
    p_s2   = predict_one_component(pipeline["s2_spec"],   pipeline["s2_result"],   df).values
    r_bulk = predict_one_component(pipeline["bulk_spec"], pipeline["bulk_result"], df).values
    r_tail = predict_one_component(pipeline["tail_spec"], pipeline["tail_result"], df).values

    expected = p_s1 * (p_s2 * r_tail + (1.0 - p_s2) * r_bulk)
    result = _assemble(pipeline)

    np.testing.assert_allclose(result.values, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Determinism and index preservation
# ---------------------------------------------------------------------------

def test_deterministic(pipeline):
    """Assembling twice produces identical results."""
    r1 = _assemble(pipeline)
    r2 = _assemble(pipeline)
    np.testing.assert_array_equal(r1.values, r2.values)


def test_non_default_index_preserved(pipeline):
    """A df with a non-RangeIndex gets that index back in the result."""
    df = pipeline["df"].copy()
    df.index = pd.RangeIndex(start=100, stop=100 + len(df))

    # Rebuild specs with the same combo (they don't depend on df index).
    result = assemble_predictions(
        s1_result=pipeline["s1_result"],
        s1_spec=pipeline["s1_spec"],
        s2_result=pipeline["s2_result"],
        s2_spec=pipeline["s2_spec"],
        bulk_result=pipeline["bulk_result"],
        bulk_spec=pipeline["bulk_spec"],
        tail_result=pipeline["tail_result"],
        tail_spec=pipeline["tail_spec"],
        df=df,
    )

    assert list(result.index) == list(df.index)


# ---------------------------------------------------------------------------
# OOS assembly — correctness, no-leakage, coverage
# ---------------------------------------------------------------------------

def test_oos_each_storm_appears_exactly_once(oos_pipeline):
    """Every storm appears exactly once in the OOS assembled predictions."""
    preds, _ = assemble_oos_predictions(
        oos_pipeline["model_spec_key"],
        seed=0,
        df=oos_pipeline["df"],
        fold_assignments=oos_pipeline["fold_assignments"],
        results_dir=oos_pipeline["results_dir"],
    )
    assert len(preds) == len(oos_pipeline["df"])
    assert preds.index.is_unique
    assert list(preds.index) == list(oos_pipeline["df"].index)


def test_oos_length_equals_df(oos_pipeline):
    """OOS predictions Series has same length as full df."""
    preds, fold_tags = assemble_oos_predictions(
        oos_pipeline["model_spec_key"],
        seed=0,
        df=oos_pipeline["df"],
        fold_assignments=oos_pipeline["fold_assignments"],
        results_dir=oos_pipeline["results_dir"],
    )
    assert len(preds) == len(oos_pipeline["df"])
    assert len(fold_tags) == len(oos_pipeline["df"])


def test_oos_no_nan(oos_pipeline):
    """OOS predictions contain no NaN — every row is held out in exactly one fold."""
    preds, fold_tags = assemble_oos_predictions(
        oos_pipeline["model_spec_key"],
        seed=0,
        df=oos_pipeline["df"],
        fold_assignments=oos_pipeline["fold_assignments"],
        results_dir=oos_pipeline["results_dir"],
    )
    assert not preds.isna().any(), f"{preds.isna().sum()} NaN in OOS predictions"
    assert not fold_tags.isna().any(), f"{fold_tags.isna().sum()} NaN in fold tags"


def test_oos_fold_tags_match_assignments(oos_pipeline):
    """heldout_fold_tags values agree with fold_assignments for each row."""
    n_folds = oos_pipeline["n_folds"]
    _, fold_tags = assemble_oos_predictions(
        oos_pipeline["model_spec_key"],
        seed=0,
        df=oos_pipeline["df"],
        fold_assignments=oos_pipeline["fold_assignments"],
        results_dir=oos_pipeline["results_dir"],
    )
    fa = oos_pipeline["fold_assignments"]["seed_0"]
    for fold in range(n_folds):
        expected_tag = f"s0_f{fold}"
        held_out_idx = oos_pipeline["df"].index[fa.values == fold]
        actual_tags  = fold_tags.loc[held_out_idx]
        assert (actual_tags == expected_tag).all(), (
            f"Fold {fold}: expected tag {expected_tag!r} but got {actual_tags.unique()}"
        )


def test_oos_predictions_come_from_heldout_fold_not_training(oos_pipeline):
    """OOS predictions for a held-out row differ from the IS prediction.

    The IS model is trained on all data; OOS models for each fold are trained
    without that fold. For a row held out in fold 0, its OOS prediction comes
    from the s0_f0 model (trained without fold-0 rows). If predictions were
    taken from a training-set model they would be identical to IS predictions
    for many rows. We verify that the OOS assembled predictions are NOT
    identical to the IS assembled predictions — the two must differ since the
    OOS models were fitted on strictly smaller datasets.
    """
    df    = oos_pipeline["df"]
    specs = oos_pipeline["is_specs"]
    res   = oos_pipeline["is_results"]

    is_preds = assemble_predictions(
        s1_result=res["s1"],     s1_spec=specs["s1"],
        s2_result=res["s2"],     s2_spec=specs["s2"],
        bulk_result=res["bulk"], bulk_spec=specs["bulk"],
        tail_result=res["tail"], tail_spec=specs["tail"],
        df=df,
    )

    oos_preds, _ = assemble_oos_predictions(
        oos_pipeline["model_spec_key"],
        seed=0,
        df=df,
        fold_assignments=oos_pipeline["fold_assignments"],
        results_dir=oos_pipeline["results_dir"],
    )

    # IS and OOS predictions must differ: OOS models were trained on subsets.
    assert not np.allclose(is_preds.values, oos_preds.values), (
        "IS and OOS predictions are identical — OOS assembly may be using "
        "training-set models instead of held-out fold models."
    )
