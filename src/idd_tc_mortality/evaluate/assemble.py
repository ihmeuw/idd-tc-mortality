"""
Prediction assembly for the double-hurdle model.

assemble_predictions(s1_result, s2_result, bulk_result, bulk_spec,
                     tail_result, tail_spec, df)

    Calls predict_one_component for each of the four components, then assembles
    the full-dataset unconditional expected death rate via combine.assemble_dh_prediction:

        E[rate] = p_s1 * (p_s2 * rate_tail + (1 - p_s2) * rate_bulk)

    predict_one_component returns predictions for every row of df; the fitted
    component models are functions of covariates and apply to every row regardless
    of the row's outcome at fit time. The unconditional formula above is exactly:

        E[rate | row] = P(deaths>=1 | row) *
                        [P(rate>=thresh | row, deaths>=1) * E[rate | row, rate>=thresh]
                         + P(rate<thresh | row, deaths>=1) * E[rate | row, rate<thresh]]

    Every term on the right-hand side is evaluated at each row's covariates. There
    is no "applicable subset only" qualifier on any of them.

assemble_oos_predictions(model_spec_key, seed, df, fold_assignments, results_dir)

    Stitches held-out predictions from each fold for a given seed, producing
    true OOS predictions where each storm appears exactly once with its prediction
    from the fold where it was held out (not the training set).

    For each fold f:
      - Builds OOS spec dicts by cloning IS specs with fold_tag='s{seed}_f{fold}'.
      - Loads the fitted components from results_dir.
      - Calls assemble_predictions on the full df.
      - Extracts only the held-out rows (where fold_assignments['seed_{seed}'] == f).
    Concatenates → full-length Series, original index order preserved.

s1_spec is inferred from the s1_result param_names and the bulk/tail spec's
covariate_combo. Pass a minimal s1_spec with 'covariate_combo' set to the shared
combo; 'component', 'threshold_quantile', 'threshold_rate', and 'family' are
ignored for s1.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd

from idd_tc_mortality.cache import component_id, load_result
from idd_tc_mortality.combine import assemble_dh_prediction
from idd_tc_mortality.distributions.base import FitResult
from idd_tc_mortality.evaluate.predict_component import predict_one_component


def assemble_predictions(
    s1_result: FitResult,
    s1_spec: dict,
    s2_result: FitResult,
    s2_spec: dict,
    bulk_result: FitResult,
    bulk_spec: dict,
    tail_result: FitResult,
    tail_spec: dict,
    df: pd.DataFrame,
) -> pd.Series:
    """Assemble unconditional expected death rates from four fitted components.

    Parameters
    ----------
    s1_result, s2_result, bulk_result, tail_result:
        Fitted FitResult objects for each component.
    s1_spec, s2_spec, bulk_spec, tail_spec:
        Spec dicts for each component (from manifest or enumerate_component_specs).
    df:
        Full DataFrame. Must contain 'deaths', 'exposed', and the covariates
        used in the specs. Index is preserved in the returned Series.

    Returns
    -------
    pd.Series
        Unconditional expected death rates, length == len(df), index == df.index.
        No NaN values — non-event rows receive near-zero rates (driven by p_s1 ≈ 0).

    Raises
    ------
    ValueError
        If s1 predictions contain NaN (s1 must cover the full dataset).
    """
    # Each component predicts for every row of df.
    p_s1_series      = predict_one_component(s1_spec,   s1_result,   df)
    p_s2_series      = predict_one_component(s2_spec,   s2_result,   df)
    rate_bulk_series = predict_one_component(bulk_spec, bulk_result, df)
    rate_tail_series = predict_one_component(tail_spec, tail_result, df)

    assembled = assemble_dh_prediction(
        p_s1=p_s1_series.values,
        p_s2=p_s2_series.values,
        rate_bulk=rate_bulk_series.values,
        rate_tail=rate_tail_series.values,
    )

    return pd.Series(assembled, index=df.index, name="predicted_rate")


def assemble_oos_predictions(
    model_spec_key: dict,
    seed: int,
    df: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    results_dir: str | Path,
) -> tuple[pd.Series, pd.Series]:
    """Assemble true OOS predictions by stitching held-out subsets across folds.

    For each fold, loads the OOS-fitted components for that fold and seed, calls
    assemble_predictions on the full df, then keeps only the held-out rows (those
    where fold_assignments['seed_{seed}'] == fold). The held-out subsets are
    concatenated so that each storm appears exactly once with its true OOS
    prediction — the prediction from the model that never saw that storm.

    Parameters
    ----------
    model_spec_key:
        Dict with keys 's1_spec', 's2_spec', 'bulk_spec', 'tail_spec'. Each value
        is the IS (fold_tag='is') spec dict for that component. OOS specs are
        created by cloning these with fold_tag='s{seed}_f{fold}'.
    seed:
        Integer seed index. Determines which column of fold_assignments to use
        ('seed_{seed}') and which OOS fold tags to load.
    df:
        Full DataFrame. Must contain all covariates and 'deaths'/'exposed'.
        Index is preserved in the returned Series.
    fold_assignments:
        DataFrame returned by compute_fold_assignments. Index must align with
        df.index. Must contain column 'seed_{seed}'.
    results_dir:
        Directory containing fitted .pkl files from run-fit-component.

    Returns
    -------
    predictions : pd.Series
        OOS predicted death rates, length == len(df), index == df.index.
        Each storm's value is its prediction from the fold where it was held out.
        No NaN values — every row is held out in exactly one fold.
    heldout_fold_tags : pd.Series
        String fold tags ('s{seed}_f{fold}'), same length and index as predictions,
        indicating which fold each storm was held out in.

    Raises
    ------
    FileNotFoundError
        If any fitted OOS component is missing from results_dir.
    ValueError
        If fold_assignments does not align with df or lacks the required column.
    """
    col = f"seed_{seed}"
    if col not in fold_assignments.columns:
        raise ValueError(
            f"Column {col!r} not found in fold_assignments. "
            f"Available: {list(fold_assignments.columns)}"
        )
    if len(fold_assignments) != len(df):
        raise ValueError(
            f"fold_assignments length {len(fold_assignments)} != df length {len(df)}."
        )
    if not fold_assignments.index.equals(df.index):
        raise ValueError("fold_assignments.index does not match df.index.")

    s1_spec_is   = model_spec_key["s1_spec"]
    s2_spec_is   = model_spec_key["s2_spec"]
    bulk_spec_is = model_spec_key["bulk_spec"]
    tail_spec_is = model_spec_key["tail_spec"]

    fold_col   = fold_assignments[col]
    fold_values = sorted(fold_col.unique())

    held_out_preds: list[pd.Series] = []
    held_out_tags:  list[pd.Series] = []

    for fold in fold_values:
        fold_tag = f"s{seed}_f{fold}"

        s1_spec_oos   = {**s1_spec_is,   "fold_tag": fold_tag}
        s2_spec_oos   = {**s2_spec_is,   "fold_tag": fold_tag}
        bulk_spec_oos = {**bulk_spec_is, "fold_tag": fold_tag}
        tail_spec_oos = {**tail_spec_is, "fold_tag": fold_tag}

        s1_result   = load_result(component_id(s1_spec_oos),   results_dir)
        s2_result   = load_result(component_id(s2_spec_oos),   results_dir)
        bulk_result = load_result(component_id(bulk_spec_oos), results_dir)
        tail_result = load_result(component_id(tail_spec_oos), results_dir)

        # Predict on full df (model was trained on all-but-held-out rows).
        full_preds = assemble_predictions(
            s1_result=s1_result,     s1_spec=s1_spec_oos,
            s2_result=s2_result,     s2_spec=s2_spec_oos,
            bulk_result=bulk_result, bulk_spec=bulk_spec_oos,
            tail_result=tail_result, tail_spec=tail_spec_oos,
            df=df,
        )

        # Extract only the rows held out in this fold.
        held_out_mask = (fold_col == fold).values
        held_out_idx  = df.index[held_out_mask]
        held_out_preds.append(full_preds.loc[held_out_idx])
        held_out_tags.append(
            pd.Series(fold_tag, index=held_out_idx, dtype=str, name="heldout_fold_tag")
        )

    # Concatenate and restore original index order (concat may reorder).
    predictions       = pd.concat(held_out_preds).reindex(df.index)
    heldout_fold_tags = pd.concat(held_out_tags).reindex(df.index)

    predictions.name       = "predicted_rate"
    heldout_fold_tags.name = "heldout_fold_tag"

    return predictions, heldout_fold_tags
