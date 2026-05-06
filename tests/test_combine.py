"""
Tests for combine.assemble_dh_prediction.

Covers:
  - Correct formula: hand-computed expected values.
  - p_s1 = 0 → all-zero output regardless of other inputs.
  - p_s2 = 0 → p_s1 * rate_bulk (no tail contribution).
  - p_s2 = 1 → p_s1 * rate_tail (no bulk contribution).
  - Shape mismatch raises.
  - p_s1 outside [0, 1] raises (both below 0 and above 1).
  - p_s2 outside [0, 1] raises.
  - Negative rate_bulk raises.
  - Negative rate_tail raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from idd_tc_mortality.combine import assemble_dh_prediction


# ---------------------------------------------------------------------------
# Correct formula
# ---------------------------------------------------------------------------

def test_correct_formula_scalar_broadcast():
    """Verify E[rate] = p_s1 * (p_s2 * rate_tail + (1 - p_s2) * rate_bulk)
    against hand-computed values for a small array."""
    p_s1 = np.array([0.8, 0.5, 1.0])
    p_s2 = np.array([0.3, 0.7, 0.0])
    rate_bulk = np.array([0.001, 0.002, 0.005])
    rate_tail = np.array([0.010, 0.020, 0.050])

    result = assemble_dh_prediction(p_s1, p_s2, rate_bulk, rate_tail)

    expected = p_s1 * (p_s2 * rate_tail + (1.0 - p_s2) * rate_bulk)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_correct_formula_hand_computed():
    """Single row with known inputs: verify each arithmetic step."""
    # p_s1=0.6, p_s2=0.4, rate_bulk=0.002, rate_tail=0.010
    # mixed = 0.4 * 0.010 + 0.6 * 0.002 = 0.004 + 0.0012 = 0.0052
    # E[rate] = 0.6 * 0.0052 = 0.00312
    result = assemble_dh_prediction(
        np.array([0.6]),
        np.array([0.4]),
        np.array([0.002]),
        np.array([0.010]),
    )
    np.testing.assert_allclose(result, [0.00312], rtol=1e-12)


def test_output_shape_matches_input(  ):
    n = 50
    rng = np.random.default_rng(0)
    result = assemble_dh_prediction(
        rng.uniform(0, 1, n),
        rng.uniform(0, 1, n),
        rng.uniform(0, 0.01, n),
        rng.uniform(0, 0.05, n),
    )
    assert result.shape == (n,)


def test_output_is_nonnegative():
    rng = np.random.default_rng(1)
    n = 100
    result = assemble_dh_prediction(
        rng.uniform(0, 1, n),
        rng.uniform(0, 1, n),
        rng.uniform(0, 0.01, n),
        rng.uniform(0, 0.05, n),
    )
    assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# Boundary cases for p_s1 and p_s2
# ---------------------------------------------------------------------------

def test_p_s1_zero_gives_all_zero_output():
    """p_s1 = 0 means no deaths expected: output must be zero regardless of other inputs."""
    n = 10
    rng = np.random.default_rng(2)
    result = assemble_dh_prediction(
        np.zeros(n),
        rng.uniform(0, 1, n),
        rng.uniform(0, 0.01, n),
        rng.uniform(0, 0.05, n),
    )
    np.testing.assert_array_equal(result, np.zeros(n))


def test_p_s2_zero_gives_p_s1_times_rate_bulk():
    """p_s2 = 0: all weight on bulk, output = p_s1 * rate_bulk."""
    p_s1 = np.array([0.3, 0.7, 1.0])
    rate_bulk = np.array([0.001, 0.004, 0.010])
    rate_tail = np.array([0.050, 0.100, 0.200])  # must not appear in output

    result = assemble_dh_prediction(p_s1, np.zeros(3), rate_bulk, rate_tail)
    np.testing.assert_allclose(result, p_s1 * rate_bulk, rtol=1e-12)


def test_p_s2_one_gives_p_s1_times_rate_tail():
    """p_s2 = 1: all weight on tail, output = p_s1 * rate_tail."""
    p_s1 = np.array([0.3, 0.7, 1.0])
    rate_bulk = np.array([0.001, 0.004, 0.010])  # must not appear in output
    rate_tail = np.array([0.050, 0.100, 0.200])

    result = assemble_dh_prediction(p_s1, np.ones(3), rate_bulk, rate_tail)
    np.testing.assert_allclose(result, p_s1 * rate_tail, rtol=1e-12)


def test_p_s1_one_p_s2_one_gives_rate_tail():
    """p_s1 = p_s2 = 1: output equals rate_tail exactly."""
    rate_tail = np.array([0.05, 0.10, 0.20])
    result = assemble_dh_prediction(
        np.ones(3), np.ones(3), np.array([0.001, 0.002, 0.003]), rate_tail
    )
    np.testing.assert_allclose(result, rate_tail, rtol=1e-12)


# ---------------------------------------------------------------------------
# Shape mismatch
# ---------------------------------------------------------------------------

def test_shape_mismatch_p_s1_raises():
    with pytest.raises(ValueError, match="shape"):
        assemble_dh_prediction(
            np.array([0.5, 0.5]),       # length 2
            np.array([0.3, 0.3, 0.3]), # length 3
            np.array([0.001, 0.001, 0.001]),
            np.array([0.010, 0.010, 0.010]),
        )


def test_shape_mismatch_rate_bulk_raises():
    with pytest.raises(ValueError, match="shape"):
        assemble_dh_prediction(
            np.array([0.5]),
            np.array([0.3]),
            np.array([0.001, 0.002]),  # length 2
            np.array([0.010]),
        )


# ---------------------------------------------------------------------------
# Probability bounds
# ---------------------------------------------------------------------------

def test_p_s1_below_zero_raises():
    with pytest.raises(ValueError, match="p_s1"):
        assemble_dh_prediction(
            np.array([-0.1, 0.5]),
            np.array([0.3, 0.3]),
            np.array([0.001, 0.001]),
            np.array([0.010, 0.010]),
        )


def test_p_s1_above_one_raises():
    with pytest.raises(ValueError, match="p_s1"):
        assemble_dh_prediction(
            np.array([1.1, 0.5]),
            np.array([0.3, 0.3]),
            np.array([0.001, 0.001]),
            np.array([0.010, 0.010]),
        )


def test_p_s2_below_zero_raises():
    with pytest.raises(ValueError, match="p_s2"):
        assemble_dh_prediction(
            np.array([0.5, 0.5]),
            np.array([-0.1, 0.3]),
            np.array([0.001, 0.001]),
            np.array([0.010, 0.010]),
        )


def test_p_s2_above_one_raises():
    with pytest.raises(ValueError, match="p_s2"):
        assemble_dh_prediction(
            np.array([0.5, 0.5]),
            np.array([0.3, 1.2]),
            np.array([0.001, 0.001]),
            np.array([0.010, 0.010]),
        )


# ---------------------------------------------------------------------------
# Non-negative rate constraints
# ---------------------------------------------------------------------------

def test_negative_rate_bulk_raises():
    with pytest.raises(ValueError, match="rate_bulk"):
        assemble_dh_prediction(
            np.array([0.5, 0.5]),
            np.array([0.3, 0.3]),
            np.array([-0.001, 0.001]),
            np.array([0.010, 0.010]),
        )


def test_negative_rate_tail_raises():
    with pytest.raises(ValueError, match="rate_tail"):
        assemble_dh_prediction(
            np.array([0.5, 0.5]),
            np.array([0.3, 0.3]),
            np.array([0.001, 0.001]),
            np.array([0.010, -0.010]),
        )
