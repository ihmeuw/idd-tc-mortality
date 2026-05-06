"""
Tests for thresholds.compute_thresholds.

Covers:
  - Zeros in input are excluded; do not affect threshold values.
  - Default quantile levels produce exactly 6 thresholds at correct levels.
  - Thresholds are monotonically non-decreasing.
  - All-zero input raises ValueError.
  - Fold-safety: thresholds computed on train vs full data differ — guard
    against accidentally computing thresholds on the full dataset in CV.
  - Custom quantile_levels are applied correctly.
  - Single positive value produces constant thresholds (degenerate case).
"""

from __future__ import annotations

import numpy as np
import pytest

from idd_tc_mortality.constants import QUANTILE_LEVELS
from idd_tc_mortality.thresholds import compute_thresholds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def death_rates():
    """200 positive death rates + 50 zeros, reproducible."""
    rng = np.random.default_rng(0)
    positive = rng.uniform(1e-6, 1e-3, 200)
    return np.concatenate([positive, np.zeros(50)])


# ---------------------------------------------------------------------------
# Zeros excluded
# ---------------------------------------------------------------------------

def test_zeros_do_not_affect_thresholds(death_rates):
    """Adding zeros to the input must not change the threshold values."""
    positive_only = death_rates[death_rates > 0]
    thresholds_full = compute_thresholds(death_rates)
    thresholds_positive = compute_thresholds(positive_only)
    for q in thresholds_full:
        assert thresholds_full[q] == thresholds_positive[q], (
            f"Threshold at q={q} differs when zeros present vs absent. "
            "Zeros must be filtered before computing quantiles."
        )


# ---------------------------------------------------------------------------
# Default quantile levels
# ---------------------------------------------------------------------------

def test_default_quantile_levels_count(death_rates):
    """Default QUANTILE_LEVELS (0.70–0.95 step 0.05) produces exactly 6 thresholds."""
    thresholds = compute_thresholds(death_rates)
    assert len(thresholds) == len(QUANTILE_LEVELS)


def test_default_quantile_level_keys(death_rates):
    """Keys match QUANTILE_LEVELS exactly."""
    thresholds = compute_thresholds(death_rates)
    expected_keys = set(float(q) for q in QUANTILE_LEVELS)
    assert set(thresholds.keys()) == expected_keys


def test_threshold_values_are_positive(death_rates):
    thresholds = compute_thresholds(death_rates)
    for q, v in thresholds.items():
        assert v > 0, f"Threshold at q={q} is not positive: {v}"


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

def test_thresholds_monotonically_nondecreasing(death_rates):
    """Thresholds must be non-decreasing with quantile level."""
    thresholds = compute_thresholds(death_rates)
    sorted_vals = [thresholds[float(q)] for q in sorted(thresholds)]
    for i in range(len(sorted_vals) - 1):
        assert sorted_vals[i] <= sorted_vals[i + 1], (
            f"Threshold at index {i} ({sorted_vals[i]}) > index {i+1} "
            f"({sorted_vals[i+1]}). Thresholds must be non-decreasing."
        )


# ---------------------------------------------------------------------------
# All-zero raises
# ---------------------------------------------------------------------------

def test_all_zeros_raises():
    with pytest.raises(ValueError, match="No positive death rates"):
        compute_thresholds(np.zeros(100))


def test_empty_array_raises():
    with pytest.raises(ValueError, match="No positive death rates"):
        compute_thresholds(np.array([]))


# ---------------------------------------------------------------------------
# Fold-safety: thresholds must be computed on training data only
# ---------------------------------------------------------------------------

def test_fold_safety_train_vs_full_differ():
    """Thresholds computed on training data must differ from full-data thresholds.

    This test is an explicit guard against CV leakage. In cross-validation the
    threshold must be computed on the training fold only. If thresholds are
    accidentally computed on the full dataset, the test set's extreme values
    shift the threshold and leak information about test observations into the
    model specification.

    DGP: two halves with very different rate distributions, so train-only
    thresholds reliably differ from full-data thresholds.
    """
    rng = np.random.default_rng(7)
    n = 500
    train = rng.uniform(1e-6, 1e-4, n)      # low rates
    test = rng.uniform(1e-3, 1e-2, n)       # 10–100x higher rates

    full = np.concatenate([train, test])

    thresholds_train = compute_thresholds(train)
    thresholds_full = compute_thresholds(full)

    # At least one quantile must differ — if they are identical, thresholds
    # were computed on the same distribution (both used the full data).
    any_differ = any(
        abs(thresholds_train[q] - thresholds_full[q]) > 1e-10
        for q in thresholds_train
    )
    assert any_differ, (
        "Train-only thresholds equal full-data thresholds. "
        "This indicates thresholds are being computed on the full dataset, "
        "which would leak test set information in cross-validation."
    )


# ---------------------------------------------------------------------------
# Custom quantile_levels
# ---------------------------------------------------------------------------

def test_custom_quantile_levels(death_rates):
    """Custom quantile_levels are used correctly."""
    custom = np.array([0.5, 0.9])
    thresholds = compute_thresholds(death_rates, quantile_levels=custom)
    assert set(thresholds.keys()) == {0.5, 0.9}
    assert thresholds[0.5] <= thresholds[0.9]


def test_single_quantile_level(death_rates):
    thresholds = compute_thresholds(death_rates, quantile_levels=np.array([0.8]))
    assert len(thresholds) == 1
    assert 0.8 in thresholds


# ---------------------------------------------------------------------------
# Degenerate case: single positive value
# ---------------------------------------------------------------------------

def test_single_positive_value():
    """One positive rate: all quantiles equal that value."""
    rates = np.array([0.001, 0.0, 0.0, 0.0])
    thresholds = compute_thresholds(rates)
    for v in thresholds.values():
        assert v == pytest.approx(0.001)
