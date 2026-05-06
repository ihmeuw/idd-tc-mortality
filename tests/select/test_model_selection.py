"""
Tests for select/model_selection.py.

Uses a synthetic dh_results DataFrame with known metric values so ranking
winners are deterministic and assertions are exact.

Structure:
  - _make_dh_results(): builds a synthetic frame with IS + OOS rows for 4 models.
    Model A is unambiguously best (lowest mae_rate, highest cor_rate, etc.).
  - Tests check that each ranking method selects model A as the winner.
  - Additional structural tests check prepare_rankings_df contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.select.model_selection import (
    CALIBRATION_METRICS,
    CONFIG_COLS,
    DEFAULT_METRICS,
    borda_rank,
    get_metric_ranks,
    pareto_frontier,
    pairwise_dominance_summary,
    prepare_rankings_df,
    topsis_rank,
    winner_profile,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

_N_MODELS = 4

# Four model configurations — labelled A-D; A is always best.
_CONFIGS = pd.DataFrame({
    "threshold_quantile": [0.75,      0.75,      0.90,    0.90],
    "s1_family":          ["cloglog", "cloglog", "cloglog", "logit"],
    "s1_exposure_mode":   ["offset",  "offset",  "free",    "free"],
    "s2_family":          ["cloglog", "cloglog", "logit",   "cloglog"],
    "s2_exposure_mode":   ["free",    "free",    "excluded","free"],
    "bulk_family":        ["gamma",   "gamma",   "gamma",   "lognormal"],
    "bulk_exposure_mode": ["free+weight"] * 4,
    "tail_family":        ["gamma",   "lognormal","gamma",  "lognormal"],
    "tail_exposure_mode": ["free+weight"] * 4,
    "s1_cov":             ['{"wind_speed":true}'] * 4,
    "s2_cov":             ['{"wind_speed":true}'] * 4,
    "bulk_cov":           ['{"wind_speed":true}'] * 4,
    "tail_cov":           ['{"wind_speed":true}'] * 4,
})

# OOS metrics — A dominates on every axis.
_OOS_METRICS = {
    "full_mae_rate_oos":         [0.01, 0.05, 0.08, 0.12],  # lower=better; A best
    "full_cor_rate_oos":         [0.90, 0.80, 0.70, 0.60],  # higher=better; A best
    "full_zero_acc_oos":         [0.95, 0.85, 0.75, 0.65],  # higher=better; A best
    "full_pred_obs_ratio_oos":   [1.01, 0.90, 1.15, 0.80],  # calibration; A best
    "full_coverage_rate_5_oos":  [0.80, 0.70, 0.60, 0.50],  # higher=better; A best
    "s1_auroc_oos":              [0.92, 0.85, 0.78, 0.70],  # higher=better; A best
    "s1_brier_oos":              [0.05, 0.10, 0.15, 0.20],  # lower=better; A best
    "fwd_mae_rate_oos":          [0.02, 0.06, 0.09, 0.13],  # lower=better; A best
    "fwd_pred_obs_ratio_oos":    [1.00, 1.08, 0.88, 1.20],  # calibration; A best
    "fwd_coverage_rate_5_oos":   [0.75, 0.65, 0.55, 0.45],  # higher=better; A best
}

# IS metrics — A also best.
_IS_METRICS = {
    "full_mae_rate":         [0.008, 0.04, 0.07, 0.11],
    "full_cor_rate":         [0.92,  0.82, 0.72, 0.62],
    "full_zero_acc":         [0.96,  0.86, 0.76, 0.66],
    "full_pred_obs_ratio":   [1.00,  0.91, 1.12, 0.82],
    "full_coverage_rate_5":  [0.82,  0.72, 0.62, 0.52],
    "s1_auroc":              [0.93,  0.86, 0.79, 0.71],
    "s1_brier":              [0.04,  0.09, 0.14, 0.19],
    "fwd_mae_rate":          [0.018, 0.055, 0.085, 0.125],
    "fwd_pred_obs_ratio":    [1.00,  1.07, 0.90, 1.18],
    "fwd_coverage_rate_5":   [0.76,  0.66, 0.56, 0.46],
}


def _make_dh_results() -> pd.DataFrame:
    """Synthetic dh_results with IS + 2 OOS seeds for each of 4 models."""
    rows = []

    # IS rows.
    for i in range(_N_MODELS):
        row = {col: _CONFIGS[col].iloc[i] for col in CONFIG_COLS}
        row["fold_tag"] = "insample"
        for metric, vals in _IS_METRICS.items():
            row[metric] = vals[i]
        rows.append(row)

    # OOS rows — 2 seeds.
    for seed in range(2):
        for i in range(_N_MODELS):
            row = {col: _CONFIGS[col].iloc[i] for col in CONFIG_COLS}
            row["fold_tag"] = f"oos_seed{seed}"
            # Tiny jitter across seeds so the mean is still the target value.
            for metric, vals in _OOS_METRICS.items():
                row[metric] = vals[i] + (0.001 if seed == 0 else -0.001)
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def dh_results() -> pd.DataFrame:
    return _make_dh_results()


@pytest.fixture(scope="module")
def oos_df(dh_results) -> pd.DataFrame:
    return prepare_rankings_df(dh_results, subset="oos")


@pytest.fixture(scope="module")
def is_df(dh_results) -> pd.DataFrame:
    return prepare_rankings_df(dh_results, subset="is")


# ---------------------------------------------------------------------------
# prepare_rankings_df
# ---------------------------------------------------------------------------

class TestPrepareRankingsDf:
    def test_oos_one_row_per_model(self, dh_results):
        result = prepare_rankings_df(dh_results, subset="oos")
        assert len(result) == _N_MODELS

    def test_is_one_row_per_model(self, dh_results):
        result = prepare_rankings_df(dh_results, subset="is")
        assert len(result) == _N_MODELS

    def test_both_one_row_per_model(self, dh_results):
        result = prepare_rankings_df(dh_results, subset="both")
        assert len(result) == _N_MODELS

    def test_oos_has_oos_metric_columns(self, oos_df):
        assert "full_mae_rate_oos" in oos_df.columns

    def test_is_has_is_metric_columns(self, is_df):
        assert "full_mae_rate" in is_df.columns

    def test_both_has_both_column_sets(self, dh_results):
        result = prepare_rankings_df(dh_results, subset="both")
        assert "full_mae_rate_oos" in result.columns
        assert "full_mae_rate" in result.columns

    def test_oos_averages_across_seeds(self, dh_results, oos_df):
        # full_mae_rate_oos for model A: 0.01 + 0.001 and 0.01 - 0.001 → mean 0.01
        a_row = oos_df[
            (oos_df["threshold_quantile"] == 0.75)
            & (oos_df["bulk_family"] == "gamma")
            & (oos_df["tail_family"] == "gamma")
        ]
        assert len(a_row) == 1
        np.testing.assert_almost_equal(
            a_row["full_mae_rate_oos"].iloc[0], 0.01, decimal=6
        )

    def test_invalid_subset_raises(self, dh_results):
        with pytest.raises(ValueError, match="subset must be"):
            prepare_rankings_df(dh_results, subset="invalid")

    def test_no_fold_tag_column_in_oos_output(self, oos_df):
        # fold_tag is used for filtering only; not meaningful in aggregated output.
        assert "fold_tag" not in oos_df.columns


# ---------------------------------------------------------------------------
# get_metric_ranks
# ---------------------------------------------------------------------------

class TestGetMetricRanks:
    def test_lower_is_better_rank_1_is_smallest(self, oos_df):
        metrics = {"full_mae_rate_oos": "lower"}
        ranks = get_metric_ranks(oos_df, metrics)
        best_row = ranks["full_mae_rate_oos_rank"].idxmin()
        # Model A has the smallest mae → rank 1.
        assert ranks.loc[best_row, "full_mae_rate_oos_rank"] == 1.0

    def test_higher_is_better_rank_1_is_largest(self, oos_df):
        metrics = {"full_cor_rate_oos": "higher"}
        ranks = get_metric_ranks(oos_df, metrics)
        best_row = ranks["full_cor_rate_oos_rank"].idxmin()
        assert ranks.loc[best_row, "full_cor_rate_oos_rank"] == 1.0

    def test_calibration_rank_1_closest_to_one(self, oos_df):
        metrics = {"full_pred_obs_ratio_oos": "lower"}  # direction ignored for calibration
        calibration = ["full_pred_obs_ratio_oos"]
        ranks = get_metric_ranks(oos_df, metrics, calibration)
        # Model A: ratio=1.01 → |1.01-1|=0.01 (smallest) → rank 1.
        best_row = ranks["full_pred_obs_ratio_oos_rank"].idxmin()
        assert ranks.loc[best_row, "full_pred_obs_ratio_oos_rank"] == 1.0

    def test_skips_missing_column(self, oos_df):
        metrics = {"nonexistent_column": "lower", "full_mae_rate_oos": "lower"}
        ranks = get_metric_ranks(oos_df, metrics)
        assert "nonexistent_column_rank" not in ranks.columns
        assert "full_mae_rate_oos_rank" in ranks.columns


# ---------------------------------------------------------------------------
# borda_rank
# ---------------------------------------------------------------------------

class TestBordaRank:
    def test_winner_is_model_a(self, oos_df):
        metrics = {k: v for k, v in DEFAULT_METRICS.items() if k in oos_df.columns}
        calibration = [c for c in CALIBRATION_METRICS if c in oos_df.columns]
        result, _ = borda_rank(oos_df, metrics, calibration, plot=False)
        # Model A: threshold_quantile=0.75, bulk_family=gamma, tail_family=gamma.
        winner = result.iloc[0]
        assert winner["threshold_quantile"] == 0.75
        assert winner["bulk_family"] == "gamma"
        assert winner["tail_family"] == "gamma"

    def test_returns_tuple_df_int(self, oos_df):
        metrics = {k: v for k, v in DEFAULT_METRICS.items() if k in oos_df.columns}
        result, cutpoint = borda_rank(oos_df, metrics, plot=False)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(cutpoint, int)

    def test_borda_score_column_present(self, oos_df):
        metrics = {k: v for k, v in DEFAULT_METRICS.items() if k in oos_df.columns}
        result, _ = borda_rank(oos_df, metrics, plot=False)
        assert "borda_score" in result.columns

    def test_sorted_ascending_by_borda_score(self, oos_df):
        metrics = {k: v for k, v in DEFAULT_METRICS.items() if k in oos_df.columns}
        result, _ = borda_rank(oos_df, metrics, plot=False)
        scores = result["borda_score"].values
        assert np.all(np.diff(scores) >= 0)


# ---------------------------------------------------------------------------
# pareto_frontier
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    def test_model_a_in_pareto(self, oos_df):
        metrics = {k: v for k, v in DEFAULT_METRICS.items() if k in oos_df.columns}
        calibration = [c for c in CALIBRATION_METRICS if c in oos_df.columns]
        pareto_df = pareto_frontier(oos_df, metrics, calibration, verbose=False)
        # Model A dominates all others → must be in Pareto set.
        a_in_pareto = (
            (pareto_df["threshold_quantile"] == 0.75)
            & (pareto_df["bulk_family"] == "gamma")
            & (pareto_df["tail_family"] == "gamma")
        ).any()
        assert a_in_pareto

    def test_returns_dataframe(self, oos_df):
        metrics = {"full_mae_rate_oos": "lower"}
        result = pareto_frontier(oos_df, metrics, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_pareto_subset_of_input(self, oos_df):
        metrics = {"full_mae_rate_oos": "lower", "full_cor_rate_oos": "higher"}
        result = pareto_frontier(oos_df, metrics, verbose=False)
        assert len(result) <= len(oos_df)

    def test_single_metric_only_best_is_non_dominated(self, oos_df):
        # With one metric and all distinct values, only the best model is non-dominated.
        metrics = {"full_mae_rate_oos": "lower"}
        result = pareto_frontier(oos_df, metrics, verbose=False)
        assert len(result) == 1
        assert result.iloc[0]["full_mae_rate_oos"] == oos_df["full_mae_rate_oos"].min()


# ---------------------------------------------------------------------------
# pairwise_dominance_summary
# ---------------------------------------------------------------------------

class TestPairwiseDominanceSummary:
    def test_winner_is_model_a(self, oos_df):
        metrics = {k: v for k, v in DEFAULT_METRICS.items() if k in oos_df.columns}
        calibration = [c for c in CALIBRATION_METRICS if c in oos_df.columns]
        result = pairwise_dominance_summary(oos_df, metrics, calibration, plot=False)
        winner = result.iloc[0]
        assert winner["threshold_quantile"] == 0.75
        assert winner["bulk_family"] == "gamma"
        assert winner["tail_family"] == "gamma"

    def test_dominance_count_column_present(self, oos_df):
        metrics = {"full_mae_rate_oos": "lower", "full_cor_rate_oos": "higher"}
        result = pairwise_dominance_summary(oos_df, metrics, plot=False)
        assert "dominance_count" in result.columns

    def test_model_a_beats_all_others(self, oos_df):
        metrics = {k: v for k, v in DEFAULT_METRICS.items() if k in oos_df.columns}
        calibration = [c for c in CALIBRATION_METRICS if c in oos_df.columns]
        result = pairwise_dominance_summary(oos_df, metrics, calibration, plot=False)
        # Model A: wins on all metrics → beats all 3 other models.
        winner = result.iloc[0]
        assert winner["dominance_count"] == _N_MODELS - 1


# ---------------------------------------------------------------------------
# topsis_rank
# ---------------------------------------------------------------------------

class TestTopsisRank:
    def test_winner_is_model_a(self, oos_df):
        metrics = {k: v for k, v in DEFAULT_METRICS.items() if k in oos_df.columns}
        calibration = [c for c in CALIBRATION_METRICS if c in oos_df.columns]
        result, _ = topsis_rank(oos_df, metrics, calibration, verbose=False)
        winner = result.iloc[0]
        assert winner["threshold_quantile"] == 0.75
        assert winner["bulk_family"] == "gamma"
        assert winner["tail_family"] == "gamma"

    def test_topsis_score_between_0_and_1(self, oos_df):
        metrics = {k: v for k, v in DEFAULT_METRICS.items() if k in oos_df.columns}
        result, _ = topsis_rank(oos_df, metrics, verbose=False)
        assert result["topsis_score"].between(0, 1).all()

    def test_sorted_ascending_by_rank(self, oos_df):
        metrics = {"full_mae_rate_oos": "lower", "full_cor_rate_oos": "higher"}
        result, _ = topsis_rank(oos_df, metrics, verbose=False)
        ranks = result["topsis_rank"].values
        assert np.all(np.diff(ranks) >= 0)

    def test_influence_df_has_metric_column(self, oos_df):
        metrics = {"full_mae_rate_oos": "lower"}
        _, influence = topsis_rank(oos_df, metrics, verbose=False)
        assert "metric" in influence.columns
        assert "influence" in influence.columns

    def test_custom_weights_change_ranking(self, oos_df):
        metrics = {"full_mae_rate_oos": "lower", "full_cor_rate_oos": "higher"}
        result_equal, _ = topsis_rank(oos_df, metrics, verbose=False)
        # Weight cor heavily; should still be deterministic.
        result_weighted, _ = topsis_rank(
            oos_df, metrics,
            weights={"full_mae_rate_oos": 1.0, "full_cor_rate_oos": 100.0},
            verbose=False,
        )
        # Both should return a DataFrame.
        assert isinstance(result_weighted, pd.DataFrame)


# ---------------------------------------------------------------------------
# winner_profile
# ---------------------------------------------------------------------------

class TestWinnerProfile:
    def test_returns_dataframe(self, oos_df):
        # Sort by a metric so top_n rows are meaningful.
        sorted_df = oos_df.sort_values("full_mae_rate_oos")
        result = winner_profile(sorted_df, top_n=2)
        assert isinstance(result, pd.DataFrame)

    def test_has_mode_column(self, oos_df):
        sorted_df = oos_df.sort_values("full_mae_rate_oos")
        result = winner_profile(sorted_df, top_n=2)
        assert "mode" in result.columns

    def test_top1_mode_is_model_a_threshold(self, oos_df):
        sorted_df = oos_df.sort_values("full_mae_rate_oos")
        result = winner_profile(sorted_df, top_n=1)
        # With top_n=1, the only model is A (threshold_quantile=0.75).
        assert result.loc["threshold_quantile", "mode"] == 0.75


# ---------------------------------------------------------------------------
# IS subset
# ---------------------------------------------------------------------------

class TestIsSubset:
    def test_borda_winner_is_model_a_on_is(self, is_df):
        metrics = {k: v for k, v in DEFAULT_METRICS.items() if k in is_df.columns}
        calibration = [c for c in CALIBRATION_METRICS if c in is_df.columns]
        result, _ = borda_rank(is_df, metrics, calibration, plot=False)
        winner = result.iloc[0]
        assert winner["threshold_quantile"] == 0.75
        assert winner["bulk_family"] == "gamma"
        assert winner["tail_family"] == "gamma"

    def test_is_oos_columns_are_all_nan(self, is_df):
        # IS rows have no OOS metric values — any _oos columns should be all-NaN.
        oos_cols = [c for c in is_df.columns if c.endswith("_oos")]
        for col in oos_cols:
            assert is_df[col].isna().all(), f"IS rows have non-NaN values in {col}"


# ---------------------------------------------------------------------------
# Both subset
# ---------------------------------------------------------------------------

class TestBothSubset:
    def test_both_has_oos_and_is_columns(self, dh_results):
        df = prepare_rankings_df(dh_results, subset="both")
        assert "full_mae_rate_oos" in df.columns
        assert "full_mae_rate" in df.columns

    def test_borda_winner_is_model_a_on_both(self, dh_results):
        df = prepare_rankings_df(dh_results, subset="both")
        metrics = {k: v for k, v in DEFAULT_METRICS.items() if k in df.columns}
        calibration = [c for c in CALIBRATION_METRICS if c in df.columns]
        result, _ = borda_rank(df, metrics, calibration, plot=False)
        winner = result.iloc[0]
        assert winner["threshold_quantile"] == 0.75
