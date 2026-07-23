"""Per-storm physical cap for the bounded / finite-mean tail variants.

The physical cap on a storm's death *rate* is

    c_i = exposed_population_i / exposed_i

i.e. at most the whole exposed headcount (``exposed_population``, P_i) can die,
spread over the person-storm-hours of exposure (``exposed``, E_i — the rate
denominator, so ``rate = deaths / exposed``). Because both are on the
deaths-per-person-storm-hour scale, ``c_i`` is directly comparable to the tail
threshold ``u = threshold_rate``.

The tail families model the EXCESS rate ``w = death_rate - threshold_rate``, so
the cap on the excess scale is

    H_i = c_i - threshold_rate .

Both helpers operate on the evaluate / predict DataFrame, which must carry an
``exposed_population`` column (present in the re-ingested vintages — e.g.
``00-data/20260722_v1985``). A clear error is raised when it is absent so a
missing column never silently disables the cap and biases the bounded variants
back toward their unbounded behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def physical_cap(df: pd.DataFrame) -> np.ndarray:
    """Per-row physical rate cap ``c_i = exposed_population / exposed``.

    Parameters
    ----------
    df:
        Frame carrying ``exposed`` (person-storm-hours, the rate denominator)
        and ``exposed_population`` (exposed headcount).

    Returns
    -------
    np.ndarray
        The cap ``c_i`` for every row, length ``len(df)``. Strictly positive
        wherever ``exposed_population > 0``.

    Raises
    ------
    ValueError
        If ``exposed_population`` is absent — re-ingest the input data with the
        column present before running the bounded tail variants.
    """
    if "exposed_population" not in df.columns:
        raise ValueError(
            "Tail cap requires an 'exposed_population' column, which is absent from "
            "this frame. Re-ingest the input data with exposed_population present "
            "(scripts/ingest/01_prepare_input_data.py) before running the bounded "
            "tail variants (gpd_cens, log_logistic_cens, gpd_shadow, log_logistic_shadow)."
        )
    return df["exposed_population"].values.astype(float) / df["exposed"].values.astype(float)


def excess_cap(df: pd.DataFrame, threshold_rate: float) -> np.ndarray:
    """Per-row excess-scale cap ``H_i = c_i - threshold_rate``.

    This is the quantity the bounded tail families censor / rescale by, since
    they operate on the excess rate ``w = death_rate - threshold_rate``. May be
    negative for rows whose physical cap sits below the threshold (the tail
    event is impossible there); the consuming family is responsible for treating
    ``H_i <= 0`` as "no admissible excess".
    """
    return physical_cap(df) - float(threshold_rate)
