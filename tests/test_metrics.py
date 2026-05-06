"""
Tests for metrics.py.

Covers:
  - Each function returns a dict with exactly the expected keys.
  - Coverage definition is correct: hand-constructed case with known exact overlap.
  - Perfect predictions: Brier=0, AUROC=1, MAE=0, RMSE=0, cor=1, all coverage=1.
  - Worst-case binary: all predicted positives are true negatives → FPR=1, FNR=1.
  - False positive count is correct on a small hand-constructed case.
  - Total deaths arithmetic is correct.
  - Empty input raises.
  - Length mismatches raise.
"""

from __future__ import annotations

import numpy as np
import pytest

from idd_tc_mortality.metrics import (
    calc_continuous_metrics,
    calc_full_model_metrics,
    calc_s1_metrics,
    calc_s2_metrics,
    calc_s2_forward_metrics,
)

# ---------------------------------------------------------------------------
# Expected key sets
# ---------------------------------------------------------------------------

_BINARY_KEYS = {"brier", "auroc", "fpr", "fnr", "predicted_positive_rate"}

_CONTINUOUS_KEYS = {
    "mae_rate", "rmse_rate", "cor_rate",
    "mae_count", "rmse_count", "cor_count",
}

_COVERAGE_RATE_KEYS  = {f"coverage_rate_{x}"  for x in range(1, 21)}
_COVERAGE_COUNT_KEYS = {f"coverage_count_{x}" for x in range(1, 21)}

_S2_FORWARD_KEYS = {"mae_rate", "rmse_rate", "pred_obs_ratio"} | _COVERAGE_RATE_KEYS | _COVERAGE_COUNT_KEYS

_FULL_MODEL_KEYS = (
    {"false_positives", "zero_acc", "total_observed_deaths", "total_predicted_deaths", "pred_obs_ratio"}
    | _CONTINUOUS_KEYS
    | _COVERAGE_RATE_KEYS
    | _COVERAGE_COUNT_KEYS
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """50 observations with balanced binary outcomes."""
    rng = np.random.default_rng(0)
    y = np.array([1.0] * 25 + [0.0] * 25)
    p = np.clip(rng.beta(2, 2, 50), 0, 1)
    return y, p


@pytest.fixture
def continuous_data():
    """100 observations with positive rates and varying exposure."""
    rng = np.random.default_rng(1)
    exposed = rng.uniform(10_000, 1_000_000, 100)
    y_true  = rng.gamma(2, 1e-4, 100)
    y_pred  = y_true * rng.uniform(0.8, 1.2, 100)
    return y_true, y_pred, exposed


# ---------------------------------------------------------------------------
# Key set tests
# ---------------------------------------------------------------------------

def test_s1_metrics_keys(binary_data):
    y, p = binary_data
    assert set(calc_s1_metrics(y, p).keys()) == _BINARY_KEYS


def test_s2_metrics_keys(binary_data):
    y, p = binary_data
    assert set(calc_s2_metrics(y, p).keys()) == _BINARY_KEYS


def test_continuous_metrics_keys(continuous_data):
    yt, yp, e = continuous_data
    assert set(calc_continuous_metrics(yt, yp, e).keys()) == _CONTINUOUS_KEYS


def test_s2_forward_metrics_keys(continuous_data):
    yt, yp, e = continuous_data
    assert set(calc_s2_forward_metrics(yt, yp, e).keys()) == _S2_FORWARD_KEYS


def test_full_model_metrics_keys(continuous_data):
    yt, yp, e = continuous_data
    any_death = np.ones(len(yt))
    assert set(calc_full_model_metrics(yt, yp, e, any_death).keys()) == _FULL_MODEL_KEYS


# ---------------------------------------------------------------------------
# Coverage definition
# ---------------------------------------------------------------------------

def test_coverage_definition_rate():
    """Coverage at X% is correctly computed as |A ∩ B| / |A|.

    n=100, observed = [0..99]. Predicted is identical except indices 98 and 99
    are swapped (predicted[98]=99, predicted[99]=98).

    At 1% (n_top=1):
      top_obs={99}, top_pred={98} → overlap=0, coverage=0.0

    At 2% (n_top=2):
      top_obs={98,99}, top_pred={98,99} → overlap=2, coverage=1.0

    At 10% (n_top=10):
      top_obs={90,..,99}, top_pred={90,..,97,99,98} (same 10 indices) → coverage=1.0
    """
    n = 100
    observed  = np.arange(n, dtype=float)
    predicted = observed.copy()
    predicted[98], predicted[99] = predicted[99], predicted[98]

    exposed = np.ones(n)

    result = calc_s2_forward_metrics(observed, predicted, exposed)

    assert result["coverage_rate_1"]  == 0.0, f"Expected 0.0, got {result['coverage_rate_1']}"
    assert result["coverage_rate_2"]  == 1.0, f"Expected 1.0, got {result['coverage_rate_2']}"
    assert result["coverage_rate_10"] == 1.0, f"Expected 1.0, got {result['coverage_rate_10']}"


def test_coverage_definition_count():
    """Coverage on count scale mirrors rate scale when exposed is uniform."""
    n = 100
    observed  = np.arange(n, dtype=float)
    predicted = observed.copy()
    predicted[98], predicted[99] = predicted[99], predicted[98]
    # Uniform exposure: count ordering == rate ordering
    exposed = np.full(n, 50_000.0)

    result = calc_s2_forward_metrics(observed, predicted, exposed)

    assert result["coverage_count_1"]  == 0.0
    assert result["coverage_count_2"]  == 1.0
    assert result["coverage_count_10"] == 1.0


# ---------------------------------------------------------------------------
# Perfect predictions
# ---------------------------------------------------------------------------

def test_perfect_binary_predictions():
    """Perfect binary predictions: Brier=0, AUROC=1, FPR=0, FNR=0."""
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    p = y.copy()

    result = calc_s1_metrics(y, p)

    assert result["brier"] == pytest.approx(0.0)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["fpr"]   == pytest.approx(0.0)
    assert result["fnr"]   == pytest.approx(0.0)


def test_perfect_continuous_predictions():
    """Perfect continuous predictions: MAE=0, RMSE=0, cor=1, all coverage=1."""
    rng = np.random.default_rng(2)
    n   = 50
    y   = rng.gamma(2, 1e-4, n)
    e   = rng.uniform(10_000, 500_000, n)

    result = calc_s2_forward_metrics(y, y, e)

    assert result["mae_rate"]  == pytest.approx(0.0)
    assert result["rmse_rate"] == pytest.approx(0.0)
    for x in range(1, 21):
        assert result[f"coverage_rate_{x}"]  == pytest.approx(1.0), (
            f"coverage_rate_{x} should be 1.0 for perfect predictions"
        )
        assert result[f"coverage_count_{x}"] == pytest.approx(1.0), (
            f"coverage_count_{x} should be 1.0 for perfect predictions"
        )


def test_perfect_continuous_metrics_cor_and_coverage():
    """calc_continuous_metrics: cor=1 and calc_full_model_metrics coverage=1."""
    rng = np.random.default_rng(3)
    n   = 50
    y   = rng.gamma(2, 1e-4, n) + 1e-6  # strictly positive
    e   = rng.uniform(10_000, 500_000, n)

    cm = calc_continuous_metrics(y, y, e)
    assert cm["cor_rate"]  == pytest.approx(1.0)
    assert cm["cor_count"] == pytest.approx(1.0)
    assert cm["mae_rate"]  == pytest.approx(0.0)

    fm = calc_full_model_metrics(y, y, e, np.ones(n))
    for x in range(1, 21):
        assert fm[f"coverage_rate_{x}"]  == pytest.approx(1.0)
        assert fm[f"coverage_count_{x}"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Worst-case binary: all predicted positives are true negatives
# ---------------------------------------------------------------------------

def test_worst_case_binary_fpr_fnr():
    """FPR=1 and FNR=1 when predictions perfectly invert the true classes."""
    # p >= 0.5 predicts positive. Assign high p to true negatives, low p to true positives.
    y = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([0.1, 0.1, 0.9, 0.9])

    result = calc_s1_metrics(y, p)

    assert result["fpr"] == pytest.approx(1.0), f"FPR={result['fpr']}"
    assert result["fnr"] == pytest.approx(1.0), f"FNR={result['fnr']}"


# ---------------------------------------------------------------------------
# False positive count
# ---------------------------------------------------------------------------

def test_false_positives_correct():
    """False positives = count where predicted count > 1 and any_death=0.

    Hand-constructed:
      n=6, exposed=100_000 for all.
      y_pred_rate = [1e-5, 2e-5, 1e-5, 2e-5, 1e-4, 1e-4]
      predicted counts  = [1.0, 2.0, 1.0, 2.0, 10.0, 10.0]
      any_death         = [0,   0,   1,   1,    0,    1  ]

      Predicted count > 1.0 when any_death=0: indices 1 and 4 → false_positives=2.
    """
    exposed   = np.full(6, 100_000.0)
    y_true    = np.zeros(6)
    y_pred    = np.array([1e-5, 2e-5, 1e-5, 2e-5, 1e-4, 1e-4])
    any_death = np.array([0.0,  0.0,  1.0,  1.0,  0.0,  1.0])

    result = calc_full_model_metrics(y_true, y_pred, exposed, any_death)

    assert result["false_positives"] == 2


# ---------------------------------------------------------------------------
# Total deaths arithmetic
# ---------------------------------------------------------------------------

def test_total_deaths_arithmetic():
    """total_observed_deaths and total_predicted_deaths are sum(rate * exposed)."""
    rates_true = np.array([1e-4, 2e-4, 3e-4])
    rates_pred = np.array([1.5e-4, 1.8e-4, 3.2e-4])
    exposed    = np.array([100_000.0, 200_000.0, 50_000.0])
    any_death  = np.ones(3)

    result = calc_full_model_metrics(rates_true, rates_pred, exposed, any_death)

    expected_obs  = float(np.sum(rates_true * exposed))
    expected_pred = float(np.sum(rates_pred * exposed))

    assert result["total_observed_deaths"]  == pytest.approx(expected_obs)
    assert result["total_predicted_deaths"] == pytest.approx(expected_pred)


# ---------------------------------------------------------------------------
# zero_acc
# ---------------------------------------------------------------------------

def test_zero_acc_correct():
    """zero_acc = fraction of any_death=0 rows where predicted_count < 1.

    Hand-constructed:
      n=6, exposed=100_000 for all.
      y_pred_rate = [1e-5, 2e-5, 1e-4, 1e-5, 2e-5, 1e-4]
      predicted counts  = [1.0, 2.0, 10.0, 1.0, 2.0, 10.0]
      any_death         = [0,   0,   0,    1,   1,   1  ]

      Among any_death=0 rows (indices 0,1,2):
        counts = [1.0, 2.0, 10.0] — none < 1.0, so zero_acc = 0.0.

      Change y_pred_rate[0] = 5e-6 → count=0.5 < 1.0 → zero_acc = 1/3.
    """
    exposed   = np.full(6, 100_000.0)
    y_true    = np.zeros(6)
    any_death = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    # All no-death predicted counts >= 1 → zero_acc = 0
    y_pred_a = np.array([1e-5, 2e-5, 1e-4, 1e-5, 2e-5, 1e-4])
    result_a = calc_full_model_metrics(y_true, y_pred_a, exposed, any_death)
    assert result_a["zero_acc"] == pytest.approx(0.0)

    # One no-death predicted count < 1 → zero_acc = 1/3
    y_pred_b = np.array([5e-6, 2e-5, 1e-4, 1e-5, 2e-5, 1e-4])
    result_b = calc_full_model_metrics(y_true, y_pred_b, exposed, any_death)
    assert result_b["zero_acc"] == pytest.approx(1 / 3)


def test_zero_acc_nan_when_no_zero_rows():
    """zero_acc is NaN when all rows have any_death=1 (no zero rows to evaluate)."""
    rng = np.random.default_rng(99)
    n = 20
    y = rng.gamma(2, 1e-4, n)
    e = np.full(n, 50_000.0)
    any_death = np.ones(n)

    result = calc_full_model_metrics(y, y, e, any_death)
    assert np.isnan(result["zero_acc"])


# ---------------------------------------------------------------------------
# pred_obs_ratio
# ---------------------------------------------------------------------------

def test_pred_obs_ratio_correct():
    """pred_obs_ratio = sum(pred * exposed) / sum(obs * exposed)."""
    rates_obs  = np.array([1e-4, 2e-4, 3e-4])
    rates_pred = np.array([2e-4, 2e-4, 2e-4])
    exposed    = np.array([100_000.0, 100_000.0, 100_000.0])
    any_death  = np.ones(3)

    # obs total = 60, pred total = 60 → ratio = 1.0
    result = calc_full_model_metrics(rates_obs, rates_pred, exposed, any_death)
    assert result["pred_obs_ratio"] == pytest.approx(1.0)


def test_pred_obs_ratio_s2_forward():
    """pred_obs_ratio is also emitted by calc_s2_forward_metrics."""
    rates_obs  = np.array([1e-4, 2e-4])
    rates_pred = np.array([2e-4, 2e-4])
    exposed    = np.array([100_000.0, 100_000.0])

    result = calc_s2_forward_metrics(rates_obs, rates_pred, exposed)
    # obs = 30, pred = 40 → ratio = 4/3
    assert result["pred_obs_ratio"] == pytest.approx(4 / 3)


# ---------------------------------------------------------------------------
# Empty input raises
# ---------------------------------------------------------------------------

def test_s1_metrics_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        calc_s1_metrics(np.array([]), np.array([]))


def test_s2_metrics_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        calc_s2_metrics(np.array([]), np.array([]))


def test_continuous_metrics_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        calc_continuous_metrics(np.array([]), np.array([]), np.array([]))


def test_s2_forward_metrics_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        calc_s2_forward_metrics(np.array([]), np.array([]), np.array([]))


def test_full_model_metrics_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        calc_full_model_metrics(np.array([]), np.array([]), np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# Length mismatches raise
# ---------------------------------------------------------------------------

def test_s1_metrics_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        calc_s1_metrics(np.array([1.0, 0.0]), np.array([0.8]))


def test_continuous_metrics_pred_length_mismatch_raises():
    y = np.array([1e-4, 2e-4, 3e-4])
    e = np.array([1e5, 1e5, 1e5])
    with pytest.raises(ValueError, match="length"):
        calc_continuous_metrics(y, y[:2], e)


def test_continuous_metrics_exposed_length_mismatch_raises():
    y = np.array([1e-4, 2e-4, 3e-4])
    with pytest.raises(ValueError, match="length"):
        calc_continuous_metrics(y, y, np.array([1e5, 1e5]))


def test_full_model_metrics_any_death_length_mismatch_raises():
    y = np.array([1e-4, 2e-4, 3e-4])
    e = np.array([1e5, 1e5, 1e5])
    with pytest.raises(ValueError, match="length"):
        calc_full_model_metrics(y, y, e, np.array([1.0, 0.0]))
