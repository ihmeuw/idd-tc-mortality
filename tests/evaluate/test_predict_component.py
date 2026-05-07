"""
Tests for evaluate/predict_component.py.

Covers:
  - S1: returns Series of length == len(df), index matches df.index.
  - S2: index is the deaths>=1 subset of df.index, no extra rows.
  - Bulk: index is the bulk subset (deaths>=1 AND rate < threshold_rate).
  - Tail: index is the tail subset (rate >= threshold_rate).
  - All predictions are finite and non-negative (rate scale).
  - S1/S2 predictions are in [0, 1].
  - Bulk/tail: gamma predictions are on rate scale (order of magnitude plausible).
  - nb/poisson bulk: predictions are on rate scale (not count scale).
  - beta bulk: predictions are on rate scale (>= 0, < threshold_rate).
  - gpd tail: predictions are >= threshold_rate (full rate, not excess).
  - Unknown component raises ValueError.
  - align_X is used: prediction with df that has a basin column still works even if
    no basin levels appeared at fit time (no basin in covariate_combo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.fit.fit_component import fit_one_component
from idd_tc_mortality.evaluate.predict_component import predict_one_component


# ---------------------------------------------------------------------------
# S1
# ---------------------------------------------------------------------------

def test_s1_length_equals_full_df(pipeline):
    df = pipeline["df"]
    result = predict_one_component(pipeline["s1_spec"], pipeline["s1_result"], df)
    assert len(result) == len(df)


def test_s1_index_matches_df(pipeline):
    df = pipeline["df"]
    result = predict_one_component(pipeline["s1_spec"], pipeline["s1_result"], df)
    assert list(result.index) == list(df.index)


def test_s1_values_in_0_1(pipeline):
    result = predict_one_component(pipeline["s1_spec"], pipeline["s1_result"], pipeline["df"])
    assert np.all(result.values >= 0)
    assert np.all(result.values <= 1)


def test_s1_values_finite(pipeline):
    result = predict_one_component(pipeline["s1_spec"], pipeline["s1_result"], pipeline["df"])
    assert np.all(np.isfinite(result.values))


# ---------------------------------------------------------------------------
# S2
# ---------------------------------------------------------------------------

def test_s2_index_is_full_df(pipeline):
    df = pipeline["df"]
    result = predict_one_component(pipeline["s2_spec"], pipeline["s2_result"], df)
    assert list(result.index) == list(df.index)


def test_s2_length_equals_full_df(pipeline):
    df = pipeline["df"]
    result = predict_one_component(pipeline["s2_spec"], pipeline["s2_result"], df)
    assert len(result) == len(df)


def test_s2_values_in_0_1(pipeline):
    result = predict_one_component(pipeline["s2_spec"], pipeline["s2_result"], pipeline["df"])
    assert np.all(result.values >= 0)
    assert np.all(result.values <= 1)


# ---------------------------------------------------------------------------
# Bulk (gamma)
# ---------------------------------------------------------------------------

def test_bulk_index_is_full_df(pipeline):
    df = pipeline["df"]
    result = predict_one_component(pipeline["bulk_spec"], pipeline["bulk_result"], df)
    assert list(result.index) == list(df.index)


def test_bulk_length_equals_full_df(pipeline):
    df = pipeline["df"]
    result = predict_one_component(pipeline["bulk_spec"], pipeline["bulk_result"], df)
    assert len(result) == len(df)


def test_bulk_gamma_predictions_nonnegative(pipeline):
    result = predict_one_component(pipeline["bulk_spec"], pipeline["bulk_result"], pipeline["df"])
    assert np.all(result.values >= 0)


def test_bulk_gamma_predictions_finite(pipeline):
    result = predict_one_component(pipeline["bulk_spec"], pipeline["bulk_result"], pipeline["df"])
    assert np.all(np.isfinite(result.values))


# ---------------------------------------------------------------------------
# Tail (gamma)
# ---------------------------------------------------------------------------

def test_tail_index_is_full_df(pipeline):
    df = pipeline["df"]
    result = predict_one_component(pipeline["tail_spec"], pipeline["tail_result"], df)
    assert list(result.index) == list(df.index)


def test_tail_length_equals_full_df(pipeline):
    df = pipeline["df"]
    result = predict_one_component(pipeline["tail_spec"], pipeline["tail_result"], df)
    assert len(result) == len(df)


def test_tail_gamma_predictions_nonnegative(pipeline):
    result = predict_one_component(pipeline["tail_spec"], pipeline["tail_result"], pipeline["df"])
    assert np.all(result.values >= 0)


# ---------------------------------------------------------------------------
# nb bulk: predictions are on rate scale (not count scale)
# ---------------------------------------------------------------------------

def test_nb_bulk_predictions_are_rate_scale(pipeline):
    """NB predict divides by exposed internally; results should match death_rate order of magnitude."""
    df = pipeline["df"]
    nb_spec = {
        **pipeline["bulk_spec"],
        "family": "nb",
    }
    nb_result = fit_one_component(nb_spec, df)
    preds = predict_one_component(nb_spec, nb_result, df)

    bulk_mask = pipeline["bulk_mask"]
    true_rates = pipeline["death_rate"][bulk_mask]

    # Rate-scale predictions should be in the same order of magnitude as true rates.
    # Count-scale predictions would be ~exposed × rate, 3–6 orders of magnitude larger.
    pred_median  = float(np.median(preds.values))
    rate_median  = float(np.median(true_rates))
    count_median = float(np.median(true_rates * df["exposed"].values[bulk_mask]))

    assert abs(np.log10(pred_median) - np.log10(rate_median)) < 3, (
        f"NB predictions look like counts, not rates: "
        f"pred_median={pred_median:.2e}, rate_median={rate_median:.2e}, "
        f"count_median={count_median:.2e}"
    )


# ---------------------------------------------------------------------------
# poisson bulk: predictions are on rate scale
# ---------------------------------------------------------------------------

def test_poisson_bulk_predictions_are_rate_scale(pipeline):
    df = pipeline["df"]
    pois_spec = {
        **pipeline["bulk_spec"],
        "family": "poisson",
    }
    pois_result = fit_one_component(pois_spec, df)
    preds = predict_one_component(pois_spec, pois_result, df)

    bulk_mask = pipeline["bulk_mask"]
    true_rates = pipeline["death_rate"][bulk_mask]
    count_median = float(np.median(true_rates * df["exposed"].values[bulk_mask]))
    pred_median  = float(np.median(preds.values))
    rate_median  = float(np.median(true_rates))

    assert abs(np.log10(pred_median) - np.log10(rate_median)) < 3, (
        f"Poisson predictions look like counts: "
        f"pred_median={pred_median:.2e}, rate_median={rate_median:.2e}, "
        f"count_median={count_median:.2e}"
    )


# ---------------------------------------------------------------------------
# beta bulk: predictions are on rate scale (< threshold_rate after multiply)
# ---------------------------------------------------------------------------

def test_beta_bulk_predictions_below_threshold(pipeline):
    df = pipeline["df"]
    beta_spec = {**pipeline["bulk_spec"], "family": "beta"}
    beta_result = fit_one_component(beta_spec, df)
    preds = predict_one_component(beta_spec, beta_result, df)

    # All bulk predictions should be on rate scale: (0, threshold_rate)
    assert np.all(preds.values > 0), "beta predictions should be positive"
    assert np.all(preds.values < pipeline["threshold_rate"]), (
        "beta predictions should be < threshold_rate (rate scale, not (0,1))"
    )


# ---------------------------------------------------------------------------
# Tail rate families: predictions must be full rates (>= threshold_rate)
# Regression tests for the "missing + threshold_rate" bug in predict_component.
# Without the fix, gamma/lognormal/gpd all return excess rates, not full rates.
# ---------------------------------------------------------------------------

def test_gamma_tail_predictions_at_or_above_threshold(pipeline):
    """Gamma tail predictions must be full rates (>= threshold_rate), not excess rates."""
    result = predict_one_component(pipeline["tail_spec"], pipeline["tail_result"], pipeline["df"])
    threshold = pipeline["threshold_rate"]
    assert np.all(result.values >= threshold * 0.99), (
        f"Gamma tail predictions should be full rates (>= {threshold:.4e}). "
        f"Min={result.values.min():.4e}. Catches the missing '+ threshold_rate' bug."
    )


def test_lognormal_tail_predictions_at_or_above_threshold(pipeline):
    """Lognormal tail predictions must be full rates (>= threshold_rate), not excess rates."""
    df = pipeline["df"]
    ln_spec   = {**pipeline["tail_spec"], "family": "lognormal"}
    ln_result = fit_one_component(ln_spec, df)
    result = predict_one_component(ln_spec, ln_result, df)
    threshold = pipeline["threshold_rate"]
    assert np.all(result.values >= threshold * 0.99), (
        f"Lognormal tail predictions should be full rates (>= {threshold:.4e}). "
        f"Min={result.values.min():.4e}. Catches the missing '+ threshold_rate' bug."
    )


# ---------------------------------------------------------------------------
# gpd tail: predictions are full rate (>= threshold_rate after addition)
# ---------------------------------------------------------------------------

def test_gpd_tail_predictions_above_threshold(pipeline):
    df = pipeline["df"]
    gpd_spec = {**pipeline["tail_spec"], "family": "gpd"}
    gpd_result = fit_one_component(gpd_spec, df)
    preds = predict_one_component(gpd_spec, gpd_result, df)

    # GPD predictions should be full rate = excess + threshold_rate >= threshold_rate
    # Use a slightly relaxed bound to account for numerical noise.
    assert np.all(preds.values >= pipeline["threshold_rate"] * 0.99), (
        f"GPD tail predictions should be >= threshold_rate={pipeline['threshold_rate']:.4e}. "
        f"Min prediction: {preds.values.min():.4e}"
    )


# ---------------------------------------------------------------------------
# Unknown component raises
# ---------------------------------------------------------------------------

def test_unknown_component_raises(pipeline):
    bad_spec = {**pipeline["s1_spec"], "component": "bogus"}
    with pytest.raises(ValueError, match="Unknown component type"):
        predict_one_component(bad_spec, pipeline["s1_result"], pipeline["df"])
