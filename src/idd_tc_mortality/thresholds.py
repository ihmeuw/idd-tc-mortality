"""
Threshold computation for the bulk/tail split.

compute_thresholds(death_rates, quantile_levels)
    Computes quantiles of the positive death rate distribution. Zero-death
    rows are filtered before computing quantiles — they represent storms with
    no deaths and must not dilute the threshold estimates for the tail model.

    Returns a dict mapping each quantile level to its threshold value. The
    thresholds define the boundary between bulk and tail observations:
    death_rate >= threshold → tail component; death_rate < threshold → bulk.

Thresholds must always be computed on training data only. Computing on the
full dataset (train + test) leaks information about the test set into the
threshold choice, which determines which component fits each observation.
"""

from __future__ import annotations

import numpy as np

from idd_tc_mortality.constants import QUANTILE_LEVELS


def compute_thresholds(
    death_rates: np.ndarray,
    quantile_levels: np.ndarray = QUANTILE_LEVELS,
) -> dict[float, float]:
    """Compute quantile thresholds from the positive death rate distribution.

    Parameters
    ----------
    death_rates:
        Array of death rates (deaths / exposed). May contain zeros (storms
        with no deaths). Zeros are excluded before computing quantiles.
    quantile_levels:
        1-D array of quantile levels in [0, 1]. Defaults to QUANTILE_LEVELS
        (0.70, 0.75, ..., 0.95). Must be non-empty.

    Returns
    -------
    dict[float, float]
        Maps each quantile level to its threshold value. Keys are floats
        (e.g. 0.7, 0.75, ..., 0.95), values are positive floats.
        Thresholds are monotonically non-decreasing with quantile level.

    Raises
    ------
    ValueError
        If no positive death rates exist after filtering zeros.
    """
    rates = np.asarray(death_rates, dtype=float)
    positive = rates[rates > 0]

    if len(positive) == 0:
        raise ValueError(
            "No positive death rates found after filtering zeros. "
            "Cannot compute thresholds."
        )

    return {
        float(q): float(np.quantile(positive, q))
        for q in quantile_levels
    }
