"""
Basin validation functions for step_00_validate.

check_basins(df)
    Inspects the basin column. Returns canonical and noncanonical value counts.
    Never modifies the DataFrame.

recode_noncanonical_basins(df)
    Replaces any basin value not in BASIN_LEVELS (including NaN, '') with 'NA'.
    Returns a new DataFrame. Logs one line per distinct noncanonical value.
"""

from __future__ import annotations

import logging

import pandas as pd

from idd_tc_mortality.constants import BASIN_LEVELS

logger = logging.getLogger(__name__)


def check_basins(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Inspect the basin column and count canonical vs noncanonical values.

    Parameters
    ----------
    df:
        DataFrame with a 'basin' column.

    Returns
    -------
    dict with two keys:
        'canonical':    {basin_code: count} for values in BASIN_LEVELS.
        'noncanonical': {display_value: count} for everything else.
                        NaN is represented as the string '<NaN>' for display.

    Notes
    -----
    The DataFrame is never modified. The returned counts are always ints.
    An empty 'noncanonical' dict means all basin values are canonical.
    """
    vc = df["basin"].value_counts(dropna=False)
    canonical: dict[str, int] = {}
    noncanonical: dict[str, int] = {}

    for val, count in vc.items():
        if pd.isna(val):
            noncanonical["<NaN>"] = int(count)
        elif val in BASIN_LEVELS:
            canonical[str(val)] = int(count)
        else:
            noncanonical[str(val)] = int(count)

    return {"canonical": canonical, "noncanonical": noncanonical}


def recode_noncanonical_basins(df: pd.DataFrame) -> pd.DataFrame:
    """Replace noncanonical basin values with 'NA'.

    Any basin value not in BASIN_LEVELS — including NaN, empty string, or any
    other unrecognised code — is replaced with 'NA' (North Atlantic, the most
    common catch-all for ambiguous TC basin assignments).

    Parameters
    ----------
    df:
        DataFrame with a 'basin' column. Not modified.

    Returns
    -------
    pd.DataFrame
        New DataFrame with noncanonical basin values replaced by 'NA'.
        All other columns and rows are unchanged.

    Side effects
    ------------
    Logs one INFO message per distinct noncanonical value showing the display
    value and number of affected rows.
    """
    result = df.copy()

    # isin returns False for NaN, so this mask covers NaN, '', and any other
    # non-BASIN_LEVELS value in a single pass.
    mask = ~result["basin"].isin(BASIN_LEVELS)

    if not mask.any():
        return result

    for val in result.loc[mask, "basin"].unique():
        if pd.isna(val):
            n = int(result["basin"].isna().sum())
            logger.info("Replacing %d rows with basin=<NaN> → 'NA'", n)
        else:
            n = int((result["basin"] == val).sum())
            logger.info("Replacing %d rows with basin=%r → 'NA'", n, val)

    result.loc[mask, "basin"] = "NA"
    return result
