"""
Feature engineering: design matrix construction and alignment.

Two public functions:

build_X(df, covariates, include_log_exposed)
    Constructs a design matrix from a DataFrame. The caller controls which
    covariates to include via a boolean flag dict and a separate
    include_log_exposed flag. Column order is deterministic:

        const | wind_speed? | sdi? | basin dummies? | is_island? | log_exposed?

    log_exposed is always last when present, and never included when
    include_log_exposed=False — the binomial_cloglog validator relies on this
    guarantee (log_exposed in X is a bug there, caught at fit time).

align_X(X, param_names)
    Aligns a prediction design matrix to a fitted model's column set and order.
    Adds missing columns as zeros (handles basin levels absent from test data),
    drops extra columns, reorders to param_names order. Raises if const is
    missing from either side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idd_tc_mortality.constants import BASIN_LEVELS, BASIN_REFERENCE


def build_X(
    df: pd.DataFrame,
    covariates: dict[str, bool],
    include_log_exposed: bool,
) -> pd.DataFrame:
    """Build a design matrix from a DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame. Must contain columns for any covariate flagged True
        in `covariates` ('wind_speed', 'sdi', 'basin', 'is_island'), and
        'exposed' when include_log_exposed=True.
    covariates:
        Boolean flags controlling which columns to include. Recognised keys:
        'wind_speed', 'sdi', 'basin', 'is_island'. Unknown keys are ignored.
        Missing keys default to False (covariate excluded).
    include_log_exposed:
        If True, appends log(df['exposed']) as the last column ('log_exposed').
        If False, log_exposed is not included regardless of covariate flags.
        Controlled independently of `covariates` because S1/S2 components
        pass log_exposed as an offset (not a free covariate), while bulk/tail
        components include it as a free covariate.

    Returns
    -------
    pd.DataFrame
        Design matrix with consistent column ordering:
        const, [wind_speed], [sdi], [basin dummies], [is_island], [log_exposed]
        Index matches df.index. Basin dummies use drop_first=True, prefix 'basin_'.

    Raises
    ------
    KeyError
        If a flagged covariate column is absent from df, or 'exposed' is absent
        when include_log_exposed=True.
    """
    parts: list[pd.DataFrame] = []

    # const — always first
    parts.append(pd.DataFrame({"const": 1.0}, index=df.index))

    if covariates.get("wind_speed", False):
        parts.append(df[["wind_speed"]])

    if covariates.get("sdi", False):
        parts.append(df[["sdi"]])

    if covariates.get("basin", False):
        # astype preserves the Series index; pd.Categorical() would drop it
        basin_cat = df["basin"].astype(pd.CategoricalDtype(categories=BASIN_LEVELS))
        dummies = pd.get_dummies(basin_cat, prefix="basin", drop_first=False, dtype=float)
        # Drop the reference level explicitly so EP is always absent regardless of
        # which basins are present in this dataset.
        dummies = dummies.drop(columns=[f"basin_{BASIN_REFERENCE}"])
        parts.append(dummies)

    if covariates.get("is_island", False):
        parts.append(df[["is_island"]].astype(float))

    if include_log_exposed:
        log_exp = np.log(df["exposed"]).rename("log_exposed")
        parts.append(log_exp.to_frame())

    return pd.concat(parts, axis=1)


def align_X(
    X: pd.DataFrame,
    param_names: list[str],
) -> pd.DataFrame:
    """Align a prediction design matrix to a fitted model's column set and order.

    Missing columns (e.g. basin levels unseen at prediction time) are filled
    with zeros. Extra columns are dropped. The result has exactly the columns
    in param_names, in that order.

    Parameters
    ----------
    X:
        Design matrix to align. Must contain a 'const' column.
    param_names:
        Ordered list of column names from a fitted FitResult. Must contain
        'const'.

    Returns
    -------
    pd.DataFrame
        Design matrix with columns exactly matching param_names, in order.
        Index matches X.index.

    Raises
    ------
    ValueError
        If 'const' is missing from X or from param_names.
    """
    if "const" not in X.columns:
        raise ValueError(
            "'const' is missing from X. Design matrices must always include "
            "an intercept column named 'const'."
        )
    if "const" not in param_names:
        raise ValueError(
            "'const' is missing from param_names. The fitted model has no "
            "intercept, which is unexpected."
        )

    # Fill missing columns with zeros (e.g. basin levels absent from prediction data)
    missing = [col for col in param_names if col not in X.columns]
    if missing:
        zeros = pd.DataFrame(0.0, index=X.index, columns=missing)
        X = pd.concat([X, zeros], axis=1)

    return X[param_names]
