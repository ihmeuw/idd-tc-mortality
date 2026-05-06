"""
Double-hurdle prediction assembly.

assemble_dh_prediction(p_s1, p_s2, rate_bulk, rate_tail)
    Computes the unconditional expected death rate from the four component
    predictions of the double-hurdle model:

        E[rate] = p_s1 * (p_s2 * rate_tail + (1 - p_s2) * rate_bulk)

    where:
        p_s1      = P(deaths >= 1)                         — S1 hurdle
        p_s2      = P(rate >= threshold | deaths >= 1)     — S2 hurdle
        rate_bulk = E[rate | deaths >= 1, rate < threshold] — bulk component
        rate_tail = E[rate | deaths >= 1, rate >= threshold]— tail component

    All four arrays must have the same shape. This function performs pure
    arithmetic on pre-computed arrays; it has no knowledge of thresholds,
    distribution families, or data.
"""

from __future__ import annotations

import numpy as np


def assemble_dh_prediction(
    p_s1: np.ndarray,
    p_s2: np.ndarray,
    rate_bulk: np.ndarray,
    rate_tail: np.ndarray,
) -> np.ndarray:
    """Compute unconditional expected death rate from double-hurdle components.

    Parameters
    ----------
    p_s1:
        P(deaths >= 1), values in [0, 1]. Shape (n,).
    p_s2:
        P(rate >= threshold | deaths >= 1), values in [0, 1]. Shape (n,).
    rate_bulk:
        Expected rate for bulk observations (rate < threshold), non-negative.
        Shape (n,).
    rate_tail:
        Expected rate for tail observations (rate >= threshold), non-negative.
        Shape (n,).

    Returns
    -------
    np.ndarray
        Unconditional expected death rate, shape (n,). Non-negative.

    Raises
    ------
    ValueError
        If any two inputs have different shapes, p_s1 or p_s2 contain values
        outside [0, 1], or rate_bulk or rate_tail contain negative values.
    """
    p_s1 = np.asarray(p_s1, dtype=float)
    p_s2 = np.asarray(p_s2, dtype=float)
    rate_bulk = np.asarray(rate_bulk, dtype=float)
    rate_tail = np.asarray(rate_tail, dtype=float)

    _validate(p_s1, p_s2, rate_bulk, rate_tail)

    return p_s1 * (p_s2 * rate_tail + (1.0 - p_s2) * rate_bulk)


def _validate(
    p_s1: np.ndarray,
    p_s2: np.ndarray,
    rate_bulk: np.ndarray,
    rate_tail: np.ndarray,
) -> None:
    shapes = {
        "p_s1": p_s1.shape,
        "p_s2": p_s2.shape,
        "rate_bulk": rate_bulk.shape,
        "rate_tail": rate_tail.shape,
    }
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        raise ValueError(
            f"All inputs must have the same shape. Got: "
            + ", ".join(f"{k}={v}" for k, v in shapes.items())
        )

    if np.any(p_s1 < 0) or np.any(p_s1 > 1):
        raise ValueError(
            f"p_s1 must be in [0, 1]. "
            f"Got min={p_s1.min():.4f}, max={p_s1.max():.4f}."
        )

    if np.any(p_s2 < 0) or np.any(p_s2 > 1):
        raise ValueError(
            f"p_s2 must be in [0, 1]. "
            f"Got min={p_s2.min():.4f}, max={p_s2.max():.4f}."
        )

    if np.any(rate_bulk < 0):
        raise ValueError(
            f"rate_bulk must be non-negative. "
            f"Got min={rate_bulk.min():.4g}."
        )

    if np.any(rate_tail < 0):
        raise ValueError(
            f"rate_tail must be non-negative. "
            f"Got min={rate_tail.min():.4g}."
        )
