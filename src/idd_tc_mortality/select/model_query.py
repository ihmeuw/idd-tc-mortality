"""
model_query.py
Utilities for querying and comparing DH models from dh_results.parquet.

Usage in notebook:
    from idd_tc_mortality.select.model_query import ModelQuery

    mq = ModelQuery(df_filtered, df_full=df_unfiltered)

    # Get one model by exact covariate config (JSON strings)
    model = mq.get(
        s1_cov='{"basin": false, "is_island": false, "sdi": true, "wind_speed": true}',
        s2_cov='{"basin": true, "is_island": false, "sdi": true, "wind_speed": false}',
        bulk_cov='{"basin": false, "is_island": false, "sdi": true, "wind_speed": true}',
        tail_cov='{"basin": false, "is_island": false, "sdi": false, "wind_speed": false}',
    )

    # Get models matching combinations of covariate JSON strings
    models = mq.enumerate(
        s1_cov=[s1_a, s1_b],
        s2_cov=s2_fixed,
        bulk_cov=[bulk_a, bulk_b],
        tail_cov=tail_fixed,
    )

    # Find models differing by one covariate from a reference
    mq.neighbors(model)
"""

from __future__ import annotations

import json
from itertools import product
from typing import Dict, List, Optional, Union

import pandas as pd


def _cov_tokens(cov_json: str) -> frozenset:
    """Return frozenset of True-valued keys from a covariate JSON string."""
    d = json.loads(cov_json)
    return frozenset(k for k, v in d.items() if v)


def _is_neighbor(cov_a: str, cov_b: str) -> bool:
    """True if cov_a and cov_b differ by exactly one covariate token."""
    return len(_cov_tokens(cov_a).symmetric_difference(_cov_tokens(cov_b))) == 1


class ModelQuery:
    """Query and compare DH models from dh_results.parquet."""

    # Covariate config columns — JSON strings in the new schema.
    CONFIG_COLS = ["s1_cov", "s2_cov", "bulk_cov", "tail_cov"]

    # Distribution columns in the new schema.
    DIST_COLS = [
        "s1_family", "s1_exposure_mode",
        "s2_family", "s2_exposure_mode",
        "bulk_family", "bulk_exposure_mode",
        "tail_family", "tail_exposure_mode",
    ]

    # Default metrics for comparison (OOS primary, IS secondary).
    DEFAULT_METRICS = [
        "full_mae_rate_oos",
        "full_rmse_rate_oos",
        "full_cor_rate_oos",
        "full_zero_acc_oos",
        "full_pred_obs_ratio_oos",
        "s1_auroc_oos",
        "s1_brier_oos",
        "fwd_mae_rate_oos",
        # IS variants
        "full_mae_rate",
        "full_rmse_rate",
        "full_cor_rate",
        "full_pred_obs_ratio",
        "s1_auroc",
        "fwd_mae_rate",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        df_full: Optional[pd.DataFrame] = None,
        threshold_quantile: float = 0.75,
        s1_family: str = "cloglog",
        s1_exposure_mode: str = "offset",
        s2_family: str = "cloglog",
        s2_exposure_mode: str = "free",
        bulk_family: str = "nb",
        bulk_exposure_mode: str = "free+weight",
        tail_family: str = "gamma",
        tail_exposure_mode: str = "free+weight",
    ):
        """
        Parameters
        ----------
        df : DataFrame (possibly filtered) for display.
        df_full : Optional unfiltered DataFrame for lookups. If None, uses df.
        threshold_quantile : default threshold quantile (e.g. 0.75).
        s1_family : default S1 link function ('cloglog' or 'logit').
        s1_exposure_mode : default S1 exposure mode ('offset', 'free', or 'excluded').
        s2_family : default S2 link function ('logit' or 'cloglog').
        s2_exposure_mode : default S2 exposure mode ('free' or 'excluded').
        bulk_family : default bulk distribution family (e.g. 'nb', 'gamma').
        bulk_exposure_mode : default bulk exposure mode (e.g. 'free+weight').
        tail_family : default tail distribution family (e.g. 'gamma', 'gpd').
        tail_exposure_mode : default tail exposure mode (e.g. 'free+weight').
        """
        self.df = df
        self.df_full = df_full if df_full is not None else df
        self.defaults = {
            "threshold_quantile": threshold_quantile,
            "s1_family":          s1_family,
            "s1_exposure_mode":   s1_exposure_mode,
            "s2_family":          s2_family,
            "s2_exposure_mode":   s2_exposure_mode,
            "bulk_family":        bulk_family,
            "bulk_exposure_mode": bulk_exposure_mode,
            "tail_family":        tail_family,
            "tail_exposure_mode": tail_exposure_mode,
        }

    def get(
        self,
        s1_cov: str,
        s2_cov: str,
        bulk_cov: str,
        tail_cov: str,
        **overrides,
    ) -> Optional[pd.Series]:
        """Get a single model by exact covariate config.

        Searches df_full so filtered-out models can be found.

        Parameters
        ----------
        s1_cov, s2_cov, bulk_cov, tail_cov : JSON covariate strings.
        **overrides : override default threshold_quantile, bulk_family, or tail_family.

        Returns
        -------
        Series with model metrics, or None if not found.
        """
        config = {**self.defaults, **overrides}
        config.update(
            {
                "s1_cov": s1_cov,
                "s2_cov": s2_cov,
                "bulk_cov": bulk_cov,
                "tail_cov": tail_cov,
            }
        )

        mask = pd.Series([True] * len(self.df_full), index=self.df_full.index)
        for col, val in config.items():
            if col in self.df_full.columns:
                mask &= self.df_full[col] == val

        matches = self.df_full[mask]
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            print(f"Warning: {len(matches)} matches found, returning first")
        return matches.iloc[0]

    def enumerate(
        self,
        s1_cov: Union[str, List[str]],
        s2_cov: Union[str, List[str]],
        bulk_cov: Union[str, List[str]],
        tail_cov: Union[str, List[str]],
        **overrides,
    ) -> pd.DataFrame:
        """Get all models matching combinations of covariate specs.

        Searches df_full so filtered-out models can be found.

        Parameters
        ----------
        s1_cov, s2_cov, bulk_cov, tail_cov :
            Single JSON string or list of JSON strings for each stage.
        **overrides : override default threshold_quantile, bulk_family, or tail_family.

        Returns
        -------
        DataFrame with all matching models.
        """

        def to_list(x):
            return x if isinstance(x, list) else [x]

        s1_list   = to_list(s1_cov)
        s2_list   = to_list(s2_cov)
        bulk_list = to_list(bulk_cov)
        tail_list = to_list(tail_cov)

        base_config = {**self.defaults, **overrides}

        mask = pd.Series([True] * len(self.df_full), index=self.df_full.index)
        for col, val in base_config.items():
            if col in self.df_full.columns:
                mask &= self.df_full[col] == val

        subset = self.df_full[mask].copy()

        cov_mask = (
            subset["s1_cov"].isin(s1_list)
            & subset["s2_cov"].isin(s2_list)
            & subset["bulk_cov"].isin(bulk_list)
            & subset["tail_cov"].isin(tail_list)
        )

        return subset[cov_mask].copy()

    def neighbors(
        self,
        model: pd.Series,
        vary_stages: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Find models that differ by exactly one covariate token in one stage.

        Covariate columns are JSON strings; tokens are True-valued keys. Two
        configs are neighbors if their token sets have a symmetric difference
        of size 1.

        Searches df_full so filtered-out models can be found.

        Parameters
        ----------
        model : Series (from get() or df.iloc[i]).
        vary_stages : which stages to vary (default: all CONFIG_COLS).

        Returns
        -------
        DataFrame of neighboring models, sorted by full_mae_rate_oos (or
        full_mae_rate if OOS column is absent).
        """
        if vary_stages is None:
            vary_stages = self.CONFIG_COLS

        target = {col: model[col] for col in self.CONFIG_COLS}

        # Filter to rows matching the model's distributions and threshold.
        base_config = {**self.defaults}
        mask = pd.Series([True] * len(self.df_full), index=self.df_full.index)
        for col in self.DIST_COLS + ["threshold_quantile"]:
            if col in self.df_full.columns and col in model.index:
                mask &= self.df_full[col] == model[col]
        subset = self.df_full[mask]

        neighbors = []
        for stage in vary_stages:
            # Match exactly on the other three stages.
            other_mask = pd.Series([True] * len(subset), index=subset.index)
            for s in self.CONFIG_COLS:
                if s != stage:
                    other_mask &= subset[s] == target[s]

            stage_subset = subset[other_mask].copy()
            stage_subset["_is_neighbor"] = stage_subset[stage].apply(
                lambda x: _is_neighbor(x, target[stage])
            )
            stage_neighbors = stage_subset[stage_subset["_is_neighbor"]].copy()
            stage_neighbors["_varied_stage"] = stage
            stage_neighbors = stage_neighbors.drop(columns=["_is_neighbor"])
            neighbors.append(stage_neighbors)

        if not neighbors:
            return pd.DataFrame()

        result = pd.concat(neighbors, ignore_index=True)

        # Sort by best available primary metric.
        sort_col = next(
            (m for m in ["full_mae_rate_oos", "full_mae_rate"] if m in result.columns),
            None,
        )
        if sort_col is not None:
            result = result.sort_values(sort_col)
        return result

    def compare(
        self,
        models: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        show_config: bool = True,
    ) -> pd.DataFrame:
        """Format models for side-by-side comparison.

        Parameters
        ----------
        models : DataFrame of models (from enumerate() or neighbors()).
        metrics : metric columns to show (default: DEFAULT_METRICS that exist in df).
        show_config : include covariate config and distribution columns.

        Returns
        -------
        Formatted DataFrame.
        """
        if metrics is None:
            metrics = [m for m in self.DEFAULT_METRICS if m in models.columns]

        cols: List[str] = []
        if show_config:
            cols.extend([c for c in self.CONFIG_COLS + self.DIST_COLS if c in models.columns])
        cols.extend([m for m in metrics if m in models.columns])

        return models[cols].copy()

    def diff(
        self,
        model1: pd.Series,
        model2: pd.Series,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Show metric differences between two models.

        Returns DataFrame with model1, model2, and diff columns indexed by metric.
        """
        if metrics is None:
            metrics = [
                m for m in self.DEFAULT_METRICS if m in model1.index and m in model2.index
            ]

        config_cols = [
            c for c in self.CONFIG_COLS + self.DIST_COLS
            if c in model1.index and c in model2.index
        ]

        rows = []
        for col in config_cols + metrics:
            v1, v2 = model1[col], model2[col]
            if col in config_cols:
                diff = "←" if v1 != v2 else ""
            else:
                try:
                    diff = float(v2) - float(v1)
                except (TypeError, ValueError):
                    diff = ""
            rows.append({"metric": col, "model1": v1, "model2": v2, "diff": diff})

        return pd.DataFrame(rows).set_index("metric")

    def compare_to_reference(
        self,
        models: pd.DataFrame,
        reference: pd.Series,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare models to a reference, showing only differing stages.

        Parameters
        ----------
        models : DataFrame of models to compare.
        reference : Series (the reference model to compare against).
        metrics : metric columns to include.

        Returns
        -------
        DataFrame with differing stage(s) and metric deltas, sorted by
        full_mae_rate_oos (or full_mae_rate, or n_diff).
        """
        if metrics is None:
            metrics = [m for m in self.DEFAULT_METRICS if m in models.columns]

        ref_config = {c: reference[c] for c in self.CONFIG_COLS}

        rows = []
        for idx, row in models.iterrows():
            diff_stages = [s for s in self.CONFIG_COLS if row[s] != ref_config[s]]
            if not diff_stages:
                continue

            row_data: dict = {
                "diff_stages": ", ".join(diff_stages),
                "n_diff": len(diff_stages),
            }

            for stage in diff_stages:
                row_data[f"{stage}_ref"]   = ref_config[stage]
                row_data[f"{stage}_model"] = row[stage]

            for m in metrics:
                if m in row.index and m in reference.index:
                    row_data[m] = row[m]
                    try:
                        row_data[f"{m}_delta"] = float(row[m]) - float(reference[m])
                    except (TypeError, ValueError):
                        pass

            rows.append(row_data)

        result = pd.DataFrame(rows)
        if len(result) > 0:
            sort_col = next(
                (
                    c
                    for c in ["full_mae_rate_oos", "full_mae_rate", "n_diff"]
                    if c in result.columns
                ),
                None,
            )
            if sort_col is not None:
                result = result.sort_values(sort_col)
        return result
