"""
model_selection.py

Multi-criteria model selection pipeline for the double-hurdle TC mortality model.
Operates on dh_results.parquet produced by evaluate.

The parquet has a sparse structure:
  - IS rows (fold_tag='insample'): have non-_oos metric columns only.
  - OOS rows (fold_tag='oos_seed{N}'): have _oos metric columns only.

Use prepare_rankings_df(dh_results, subset) to produce one row per model
configuration before passing to ranking functions.

Seven ranking methods:
1. Borda count
2. Pareto dominance
3. Kendall tau metric agreement
4. Friedman + Nemenyi test
5. Pairwise dominance summary
6. TOPSIS
7. Configuration clustering + winner profile
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist  # noqa: F401 — available for callers


# ---------------------------------------------------------------------------
# Configuration and metric constants
# ---------------------------------------------------------------------------

CONFIG_COLS = [
    "threshold_quantile",
    "s1_family",
    "s1_exposure_mode",
    "s2_family",
    "s2_exposure_mode",
    "bulk_family",
    "bulk_exposure_mode",
    "tail_family",
    "tail_exposure_mode",
    "s1_cov",
    "s2_cov",
    "bulk_cov",
    "tail_cov",
]

DEFAULT_METRICS: Dict[str, str] = {
    # OOS (primary — available after subset='oos' or 'both')
    "full_mae_rate_oos":        "lower",
    "full_rmse_rate_oos":       "lower",
    "full_cor_rate_oos":        "higher",
    "full_zero_acc_oos":        "higher",
    "full_coverage_rate_5_oos": "higher",
    "full_coverage_rate_10_oos":"higher",
    "full_coverage_rate_20_oos":"higher",
    "s1_auroc_oos":             "higher",
    "s1_brier_oos":             "lower",
    "fwd_mae_rate_oos":         "lower",
    "fwd_coverage_rate_5_oos":  "higher",
    "fwd_coverage_rate_10_oos": "higher",
    "fwd_coverage_rate_20_oos": "higher",
    # IS variants (available after subset='is' or 'both')
    "full_mae_rate":            "lower",
    "full_rmse_rate":           "lower",
    "full_cor_rate":            "higher",
    "full_zero_acc":            "higher",
    "full_coverage_rate_5":     "higher",
    "full_coverage_rate_10":    "higher",
    "full_coverage_rate_20":    "higher",
    "s1_auroc":                 "higher",
    "s1_brier":                 "lower",
    "fwd_mae_rate":             "lower",
    "fwd_coverage_rate_5":      "higher",
    "fwd_coverage_rate_10":     "higher",
    "fwd_coverage_rate_20":     "higher",
}

# Calibration metrics: closer to 1.0 is better.
CALIBRATION_METRICS: List[str] = [
    "full_pred_obs_ratio_oos",
    "fwd_pred_obs_ratio_oos",
    "full_pred_obs_ratio",
    "fwd_pred_obs_ratio",
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_rankings_df(
    dh_results: pd.DataFrame,
    subset: str = "oos",
) -> pd.DataFrame:
    """Produce one row per model configuration for ranking.

    Parameters
    ----------
    dh_results:
        Raw dh_results.parquet DataFrame. Must contain 'fold_tag' and CONFIG_COLS.
    subset:
        'is'   — IS rows only; non-_oos metric columns used.
        'oos'  — OOS rows only; averaged across seeds; _oos columns used.
        'both' — OOS averaged (primary) merged with IS (tiebreaker);
                 both _oos and non-_oos columns present on the same rows.
    """
    if subset not in ("is", "oos", "both"):
        raise ValueError(f"subset must be 'is', 'oos', or 'both'. Got: {subset!r}")

    is_mask = dh_results["fold_tag"] == "insample"

    if subset == "is":
        return dh_results[is_mask].copy().reset_index(drop=True)

    oos_df = dh_results[~is_mask].copy()
    oos_cols = [c for c in oos_df.columns if c.endswith("_oos")]
    agg_cols = [c for c in CONFIG_COLS if c in oos_df.columns]
    oos_agg = oos_df.groupby(agg_cols, as_index=False)[oos_cols].mean()

    if subset == "oos":
        return oos_agg.reset_index(drop=True)

    # 'both': merge OOS aggregate with IS rows on config columns
    is_df = dh_results[is_mask].copy()
    merged = oos_agg.merge(is_df, on=agg_cols, how="inner", suffixes=("", "_is_dup"))
    # Drop any accidental duplicate columns from the merge
    dup_cols = [c for c in merged.columns if c.endswith("_is_dup")]
    merged = merged.drop(columns=dup_cols)
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metric ranking helpers
# ---------------------------------------------------------------------------

def get_metric_ranks(
    df: pd.DataFrame,
    metrics: Dict[str, str],
    calibration_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute per-metric ranks for all models (rank 1 = best).

    Parameters
    ----------
    df:
        One row per model.
    metrics:
        Dict mapping metric name -> 'lower' or 'higher' (what is better).
    calibration_metrics:
        Metrics where closer to 1.0 is better (ranked by |value - 1|).

    Returns
    -------
    DataFrame with one column per metric named '{metric}_rank'.
    """
    calibration_metrics = calibration_metrics or []
    ranks = pd.DataFrame(index=df.index)

    for metric, direction in metrics.items():
        if metric not in df.columns:
            continue
        if metric in calibration_metrics:
            ranks[f"{metric}_rank"] = np.abs(df[metric] - 1).rank(method="average")
        elif direction == "lower":
            ranks[f"{metric}_rank"] = df[metric].rank(method="average")
        else:  # 'higher'
            ranks[f"{metric}_rank"] = df[metric].rank(method="average", ascending=False)

    return ranks


# ---------------------------------------------------------------------------
# Step 1: Borda count
# ---------------------------------------------------------------------------

def borda_rank(
    df: pd.DataFrame,
    metrics: Dict[str, str],
    calibration_metrics: Optional[List[str]] = None,
    plot: bool = True,
    figsize: Tuple[int, int] = (12, 6),
) -> Tuple[pd.DataFrame, int]:
    """Sum-of-ranks Borda count with automatic elbow detection.

    Returns
    -------
    (df_with_borda, cutpoint)
        df_with_borda: original df plus 'borda_score' and 'borda_rank', sorted
                       ascending by borda_score (lower = better).
        cutpoint: suggested index where quality drops off (1-based row count).
    """
    ranks = get_metric_ranks(df, metrics, calibration_metrics)

    result = df.copy()
    result["borda_score"] = ranks.sum(axis=1)
    result["borda_rank"] = result["borda_score"].rank(method="average")
    result = result.sort_values("borda_score").reset_index(drop=True)

    scores = result["borda_score"].values
    n = len(scores)

    if n > 10:
        d1 = np.diff(scores)
        d2 = np.diff(d1)
        margin = max(5, n // 20)
        search_range = slice(margin, len(d2) - margin)
        cutpoint = int(margin + np.argmax(d2[search_range]))
    else:
        cutpoint = n // 2

    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(1, n + 1), scores, "b-", linewidth=1.5, label="Borda Score")
        ax.axvline(cutpoint, color="red", linestyle="--", alpha=0.7,
                   label=f"Cut point (rank {cutpoint})")
        ax.fill_between(range(1, cutpoint + 1), 0, scores[:cutpoint],
                        alpha=0.2, color="green", label="Top tier")
        ax.set_xlabel("Model Rank")
        ax.set_ylabel("Borda Score (lower = better)")
        ax.set_title(f"Borda Count Ranking ({len(metrics)} metrics, {n} models)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"\n=== Borda Count Summary ===")
        print(f"Total models: {n}")
        print(f"Metrics used: {len(metrics)}")
        print(f"Suggested cutpoint: rank {cutpoint} (top {cutpoint} models)")
        print(f"Top tier Borda range: {scores[0]:.1f} - {scores[cutpoint - 1]:.1f}")
        print(f"Bottom tier starts at: {scores[cutpoint]:.1f}")

    return result, cutpoint


# ---------------------------------------------------------------------------
# Step 2: Pareto dominance
# ---------------------------------------------------------------------------

def pareto_frontier(
    df: pd.DataFrame,
    metrics: Dict[str, str],
    calibration_metrics: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Find non-dominated (Pareto-optimal) models.

    A model is dominated if another model beats or ties it on ALL metrics
    and strictly beats it on at least one.

    Returns
    -------
    DataFrame containing only the non-dominated models.
    """
    calibration_metrics = calibration_metrics or []
    metric_list = [m for m in metrics if m in df.columns]

    # Normalise so higher = better for all metrics.
    values = np.zeros((len(df), len(metric_list)))
    for i, metric in enumerate(metric_list):
        col = df[metric].values
        if metric in calibration_metrics:
            values[:, i] = -np.abs(col - 1)
        elif metrics[metric] == "lower":
            values[:, i] = -col
        else:
            values[:, i] = col

    n = len(df)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if np.all(values[j] >= values[i]) and np.any(values[j] > values[i]):
                dominated[i] = True
                break

    pareto_df = df.iloc[~dominated].copy()

    if verbose:
        print(f"\n=== Pareto Frontier ===")
        print(f"Total models: {n}")
        print(f"Non-dominated: {len(pareto_df)} ({100 * len(pareto_df) / n:.1f}%)")
        print(f"Dominated: {n - len(pareto_df)}")

    return pareto_df


# ---------------------------------------------------------------------------
# Step 3: Kendall tau metric agreement
# ---------------------------------------------------------------------------

def kendall_tau_heatmap(
    df: pd.DataFrame,
    metrics: Dict[str, str],
    calibration_metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> pd.DataFrame:
    """Pairwise Kendall tau correlations between metric rankings.

    Returns
    -------
    Square correlation matrix DataFrame (metrics × metrics).
    """
    calibration_metrics = calibration_metrics or []
    ranks = get_metric_ranks(df, metrics, calibration_metrics)
    metric_names = [c.replace("_rank", "") for c in ranks.columns]
    n_metrics = len(metric_names)

    tau_matrix = np.zeros((n_metrics, n_metrics))
    for i in range(n_metrics):
        for j in range(n_metrics):
            if i == j:
                tau_matrix[i, j] = 1.0
            elif i < j:
                tau, _ = stats.kendalltau(ranks.iloc[:, i], ranks.iloc[:, j])
                tau_matrix[i, j] = tau
                tau_matrix[j, i] = tau

    tau_df = pd.DataFrame(tau_matrix, index=metric_names, columns=metric_names)

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(tau_matrix, dtype=bool), k=1)
    sns.heatmap(tau_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, vmin=-1, vmax=1, mask=mask)
    ax.set_title("Kendall Tau Correlation Between Metric Rankings")
    plt.tight_layout()
    plt.show()

    avg_tau = tau_matrix[np.triu_indices_from(tau_matrix, k=1)].mean()
    disagreeing = int((tau_matrix < 0).sum() // 2)

    print(f"\n=== Metric Agreement Analysis ===")
    print(f"Average pairwise Kendall tau: {avg_tau:.3f}")
    print(f"Pairs with negative correlation: {disagreeing}")

    if avg_tau > 0.5:
        print("Metrics largely agree. Flat top tier likely reflects genuine equivalence.")
    elif avg_tau > 0.2:
        print("Moderate agreement. Some trade-offs between metrics.")
    else:
        print("Low agreement. Flat top tier may be aggregation artifact (metric cancellation).")

    return tau_df


# ---------------------------------------------------------------------------
# Step 4a: Friedman + Nemenyi
# ---------------------------------------------------------------------------

def friedman_nemenyi(
    df: pd.DataFrame,
    metrics: Dict[str, str],
    calibration_metrics: Optional[List[str]] = None,
    top_n: int = 30,
    alpha: float = 0.05,
    plot_cd: bool = True,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[float, float, Optional[pd.DataFrame]]:
    """Friedman test + Nemenyi post-hoc on the top-N models.

    Treats models as treatments and metrics as blocks.

    Returns
    -------
    (friedman_stat, p_value, nemenyi_df)
        nemenyi_df: None if p_value > alpha. Otherwise DataFrame with columns
                    ['model_idx', 'avg_rank'], sorted best-first.
    """
    calibration_metrics = calibration_metrics or []
    df_top = df.head(top_n).copy()
    ranks = get_metric_ranks(df_top, metrics, calibration_metrics)

    n_models = len(df_top)
    n_metrics = ranks.shape[1]
    rank_matrix = ranks.values.T

    try:
        stat, p_value = stats.friedmanchisquare(*rank_matrix.T.tolist())
    except Exception as exc:
        print(f"Friedman test failed: {exc}")
        return float("nan"), float("nan"), None

    print(f"\n=== Friedman Test (top {n_models} models) ===")
    print(f"Chi-squared statistic: {stat:.3f}")
    print(f"P-value: {p_value:.4f}")

    if p_value > alpha:
        print(f"No significant difference at alpha={alpha} (models are statistically equivalent)")
        return stat, p_value, None

    print(f"Significant difference detected at alpha={alpha}")

    q_alpha_table = {0.10: 2.291, 0.05: 2.569, 0.01: 3.144}
    q_alpha = q_alpha_table.get(alpha, 2.569)
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_metrics))

    avg_ranks = ranks.mean(axis=1)
    print(f"\nNemenyi Critical Difference: {cd:.3f}")

    sorted_idx = avg_ranks.sort_values().index
    sorted_ranks = avg_ranks[sorted_idx].values

    if plot_cd:
        _plot_critical_difference(sorted_idx, sorted_ranks, cd, figsize)

    nemenyi_df = pd.DataFrame({
        "model_idx": sorted_idx,
        "avg_rank": sorted_ranks,
    })
    return stat, p_value, nemenyi_df


def _plot_critical_difference(model_ids, avg_ranks, cd, figsize):
    """Demšar-style critical difference diagram."""
    n = len(model_ids)
    sorted_order = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_order]
    sorted_ids = [model_ids[i] for i in sorted_order]

    fig, ax = plt.subplots(figsize=figsize)
    rank_min, rank_max = sorted_ranks.min(), sorted_ranks.max()
    ax.set_xlim(rank_min - 1, rank_max + 1)

    top_half = n // 2
    for i, (mid, rank) in enumerate(zip(sorted_ids, sorted_ranks)):
        y = 0.9 if i < top_half else 0.1
        va = "bottom" if i < top_half else "top"
        ax.plot(rank, 0.5, "ko", markersize=6, zorder=3)
        ax.plot([rank, rank], [0.5, y], "k-", linewidth=0.5, alpha=0.5)
        ax.annotate(f"{mid}", (rank, y), ha="center", va=va, fontsize=7)

    ax.axhline(0.5, color="black", linewidth=1, zorder=1)

    cliques = _find_cd_cliques(sorted_ranks, cd)
    bar_y_positions = np.linspace(0.55, 0.75, max(len(cliques), 1))
    for bar_idx, (start_idx, end_idx) in enumerate(cliques):
        if end_idx > start_idx:
            left_rank = sorted_ranks[start_idx]
            right_rank = sorted_ranks[end_idx]
            y_bar = bar_y_positions[bar_idx % len(bar_y_positions)]
            ax.plot([left_rank, right_rank], [y_bar, y_bar], "b-",
                    linewidth=3, solid_capstyle="round")

    cd_bar_x = rank_min + 0.5
    cd_bar_y = 0.85
    ax.plot([cd_bar_x, cd_bar_x + cd], [cd_bar_y, cd_bar_y], "k-", linewidth=2)
    ax.annotate(f"CD = {cd:.2f}", (cd_bar_x + cd / 2, cd_bar_y + 0.03),
                ha="center", fontsize=9)

    ax.set_xlabel("Average Rank (lower = better)")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(f"Critical Difference Diagram ({n} models)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    plt.show()


def _find_cd_cliques(sorted_ranks, cd):
    """Greedy clique finder: maximal contiguous groups within CD of the leftmost member."""
    n = len(sorted_ranks)
    cliques = []
    used: set = set()
    for i in range(n):
        if i in used:
            continue
        j = i
        while j + 1 < n and (sorted_ranks[j + 1] - sorted_ranks[i]) <= cd:
            j += 1
        if j > i:
            cliques.append((i, j))
            for k in range(i, j + 1):
                used.add(k)
    return cliques


# ---------------------------------------------------------------------------
# Step 4b: Pairwise dominance summary
# ---------------------------------------------------------------------------

def pairwise_dominance_summary(
    df: pd.DataFrame,
    metrics: Dict[str, str],
    calibration_metrics: Optional[List[str]] = None,
    plot: bool = True,
    figsize: Tuple[int, int] = (12, 6),
) -> pd.DataFrame:
    """Majority-wins pairwise dominance.

    For each pair (A, B): A beats B if A wins on > half the metrics.
    Returns df with 'dominance_count' (how many models beaten),
    'dominance_pct', 'total_metric_wins', 'dominance_rank'.
    """
    calibration_metrics = calibration_metrics or []
    metric_list = [m for m in metrics if m in df.columns]
    n_metrics = len(metric_list)
    n_models = len(df)

    values = np.zeros((n_models, n_metrics))
    for j, metric in enumerate(metric_list):
        col = df[metric].values
        if metric in calibration_metrics:
            values[:, j] = -np.abs(col - 1)
        elif metrics[metric] == "lower":
            values[:, j] = -col
        else:
            values[:, j] = col

    dominance_counts = np.zeros(n_models, dtype=int)
    wins_per_model = np.zeros(n_models, dtype=int)

    for i in range(n_models):
        wins_vs_others = 0
        models_beaten = 0
        for j in range(n_models):
            if i == j:
                continue
            wins_i = int(np.sum(values[i] > values[j]))
            wins_vs_others += wins_i
            if wins_i > n_metrics / 2:
                models_beaten += 1
        dominance_counts[i] = models_beaten
        wins_per_model[i] = wins_vs_others

    result = df.copy()
    result["dominance_count"] = dominance_counts
    result["dominance_pct"] = 100 * dominance_counts / max(n_models - 1, 1)
    result["total_metric_wins"] = wins_per_model
    result["dominance_rank"] = result["dominance_count"].rank(ascending=False, method="average")
    result = result.sort_values("dominance_count", ascending=False).reset_index(drop=True)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        ax.hist(dominance_counts, bins=min(50, max(n_models // 5, 2)),
                edgecolor="black", alpha=0.7)
        ax.axvline(np.median(dominance_counts), color="red", linestyle="--",
                   label=f"Median: {np.median(dominance_counts):.0f}")
        ax.set_xlabel("Models Beaten (on majority of metrics)")
        ax.set_ylabel("Count")
        ax.set_title(f"Dominance Distribution ({n_models} models)")
        ax.legend()

        ax = axes[1]
        if "borda_rank" in result.columns:
            ax.scatter(result["borda_rank"], result["dominance_count"], alpha=0.5, s=10)
            ax.set_xlabel("Borda Rank")
            ax.set_ylabel("Dominance Count")
            ax.set_title("Dominance vs Borda Rank")
        else:
            ax.scatter(range(n_models),
                       dominance_counts[np.argsort(-dominance_counts)],
                       alpha=0.5, s=10)
            ax.set_xlabel("Model (sorted by dominance)")
            ax.set_ylabel("Dominance Count")
            ax.set_title("Dominance Curve")

        plt.tight_layout()
        plt.show()

    print(f"\n=== Pairwise Dominance Summary ===")
    print(f"Models compared: {n_models}")
    print(f"Metrics used: {n_metrics}")
    print(f"Max dominance: {dominance_counts.max()} / {n_models - 1} models beaten")
    print(f"Median dominance: {np.median(dominance_counts):.0f}")

    top_cols = ["dominance_rank", "dominance_count", "dominance_pct"]
    config_show = [c for c in CONFIG_COLS if c in result.columns]
    print(f"\nTop 10 most dominant models:")
    print(result.head(10)[top_cols + config_show].to_string())

    return result


# ---------------------------------------------------------------------------
# Step 5: TOPSIS
# ---------------------------------------------------------------------------

def topsis_rank(
    df: pd.DataFrame,
    metrics: Dict[str, str],
    calibration_metrics: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    normalize_method: str = "minmax",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """TOPSIS ranking.

    Parameters
    ----------
    normalize_method:
        'minmax' (recommended — equal influence per metric) or
        'vector' (original TOPSIS — preserves variance).

    Returns
    -------
    (df_with_topsis, influence_df)
        df_with_topsis: sorted by 'topsis_rank' ascending; includes 'topsis_score'.
        influence_df: per-metric separation power.
    """
    calibration_metrics = calibration_metrics or []
    metric_list = [m for m in metrics if m in df.columns]
    n_metrics = len(metric_list)
    n_models = len(df)

    decision = np.zeros((n_models, n_metrics))
    for i, metric in enumerate(metric_list):
        col = df[metric].values
        if metric in calibration_metrics:
            decision[:, i] = 1 - np.abs(col - 1)
        elif metrics[metric] == "lower":
            decision[:, i] = -col
        else:
            decision[:, i] = col

    if normalize_method == "minmax":
        col_min = decision.min(axis=0)
        col_max = decision.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1
        normalized = (decision - col_min) / col_range
    else:
        norm = np.sqrt((decision ** 2).sum(axis=0))
        norm[norm == 0] = 1
        normalized = decision / norm

    if weights is None:
        w = np.ones(n_metrics) / n_metrics
    else:
        w = np.array([weights.get(m, 1.0) for m in metric_list])
        w = w / w.sum()

    weighted = normalized * w
    ideal = weighted.max(axis=0)
    anti_ideal = weighted.min(axis=0)

    d_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_anti = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
    topsis_score = d_anti / (d_ideal + d_anti + 1e-10)

    result = df.copy()
    result["topsis_score"] = topsis_score
    result["topsis_rank"] = pd.Series(topsis_score, index=df.index).rank(
        ascending=False, method="average"
    ).values
    result = result.sort_values("topsis_rank").reset_index(drop=True)

    raw_range = decision.max(axis=0) - decision.min(axis=0)
    influence = pd.DataFrame({
        "metric": metric_list,
        "raw_range": raw_range,
        "normalized_range": weighted.max(axis=0) - weighted.min(axis=0),
        "normalized_std": weighted.std(axis=0),
        "weight": w,
    })
    influence["influence"] = influence["normalized_range"] * influence["weight"]
    influence = influence.sort_values("influence", ascending=False)

    if verbose:
        print(f"\n=== TOPSIS Ranking (normalize={normalize_method}) ===")
        print(f"Models ranked: {n_models}")
        print(f"Metrics used: {n_metrics}")
        print(f"\nTop 10 by TOPSIS:")
        top_cols = ["topsis_rank", "topsis_score"] + metric_list[:5]
        print(result.head(10)[[c for c in top_cols if c in result.columns]].to_string())
        print(f"\nMetric Influence (separation power):")
        print(influence.to_string(index=False))

    return result, influence


# ---------------------------------------------------------------------------
# Step 6: Configuration clustering
# ---------------------------------------------------------------------------

def cluster_configurations(
    df: pd.DataFrame,
    config_cols: Optional[List[str]] = None,
    n_clusters: Optional[int] = None,
    max_clusters: int = 10,
    plot: bool = True,
    figsize: Tuple[int, int] = (14, 6),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hierarchical clustering on one-hot–encoded model configurations.

    Returns
    -------
    (df_with_clusters, cluster_summary)
        df_with_clusters: original df plus 'config_cluster' (1-based int).
        cluster_summary: DataFrame describing dominant values per cluster.
    """
    if config_cols is None:
        config_cols = CONFIG_COLS
    config_cols = [c for c in config_cols if c in df.columns]

    if not config_cols:
        print("No configuration columns found.")
        return df, pd.DataFrame()

    df_config = df[config_cols].copy()
    dummies = pd.get_dummies(df_config, columns=config_cols, drop_first=False)
    X = dummies.values.astype(float)

    Z = linkage(X, method="ward")

    if n_clusters is None:
        from sklearn.metrics import silhouette_score

        best_k = 2
        best_score = -1.0
        for k in range(2, min(max_clusters + 1, len(df))):
            labels = fcluster(Z, k, criterion="maxclust")
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k

        n_clusters = best_k
        print(f"Auto-detected {n_clusters} clusters (silhouette: {best_score:.3f})")

    labels = fcluster(Z, n_clusters, criterion="maxclust")
    result = df.copy()
    result["config_cluster"] = labels

    cluster_summaries = []
    for cluster_id in sorted(np.unique(labels)):
        cluster_df = result[result["config_cluster"] == cluster_id]
        size = len(cluster_df)

        config_str = cluster_df[config_cols].astype(str).agg(" | ".join, axis=1)
        mode_config = Counter(config_str).most_common(1)[0]

        dominant = {}
        for col in config_cols:
            vc = cluster_df[col].value_counts()
            dominant[col] = f"{vc.index[0]} ({100 * vc.iloc[0] / size:.0f}%)"

        perf = {}
        for metric in ["full_mae_rate_oos", "full_pred_obs_ratio_oos",
                        "topsis_score", "dominance_pct"]:
            if metric in cluster_df.columns:
                perf[f"{metric}_mean"] = float(cluster_df[metric].mean())

        cluster_summaries.append({
            "cluster":            cluster_id,
            "size":               size,
            "pct_of_models":      100 * size / len(df),
            "modal_config":       mode_config[0],
            "modal_config_count": mode_config[1],
            **dominant,
            **perf,
        })

    cluster_summary = pd.DataFrame(cluster_summaries)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        cluster_sizes = [
            len(result[result["config_cluster"] == c])
            for c in sorted(np.unique(labels))
        ]
        ax.bar(range(1, n_clusters + 1), cluster_sizes, edgecolor="black")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Models")
        ax.set_title(f"Configuration Clusters ({len(df)} models → {n_clusters} clusters)")
        ax.set_xticks(range(1, n_clusters + 1))

        ax = axes[1]
        if "topsis_score" in result.columns:
            cluster_perf = [
                result[result["config_cluster"] == c]["topsis_score"].mean()
                for c in sorted(np.unique(labels))
            ]
            ax.bar(range(1, n_clusters + 1), cluster_perf, edgecolor="black",
                   color="green", alpha=0.7)
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Mean TOPSIS Score")
            ax.set_title("Average Performance by Cluster")
            ax.set_xticks(range(1, n_clusters + 1))
        else:
            ax.text(0.5, 0.5, "Run TOPSIS first for performance comparison",
                    ha="center", va="center", transform=ax.transAxes)

        plt.tight_layout()
        plt.show()

    print(f"\n=== Configuration Clusters ===")
    print(f"Models: {len(df)}  Clusters: {n_clusters}")
    for _, row in cluster_summary.iterrows():
        print(f"\nCluster {int(row['cluster'])} ({int(row['size'])} models, "
              f"{row['pct_of_models']:.1f}%):")
        for col in config_cols:
            if col in row:
                print(f"  {col}: {row[col]}")

    return result, cluster_summary


# ---------------------------------------------------------------------------
# Step 7: Winner profile
# ---------------------------------------------------------------------------

def winner_profile(
    df: pd.DataFrame,
    top_n: int = 10,
    id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Analyse what the top models have in common.

    Parameters
    ----------
    df:
        DataFrame sorted by some ranking (e.g., TOPSIS), best first.
    top_n:
        Number of top models to analyse.
    id_cols:
        Columns that identify model configuration.
        Defaults to CONFIG_COLS present in df.

    Returns
    -------
    Summary DataFrame with one row per config column showing mode, count,
    mode_pct, and n_unique values.
    """
    if id_cols is None:
        id_cols = CONFIG_COLS
    id_cols = [c for c in id_cols if c in df.columns]

    top_df = df.head(top_n)

    print(f"\n=== Winner Profile (Top {top_n} Models) ===\n")

    summary = {}
    for col in id_cols:
        vc = top_df[col].value_counts()
        mode = vc.index[0]
        mode_pct = 100 * vc.iloc[0] / top_n
        summary[col] = {
            "mode":         mode,
            "mode_count":   int(vc.iloc[0]),
            "mode_pct":     mode_pct,
            "unique_values": int(vc.shape[0]),
        }
        print(f"{col}:")
        print(f"  Most common: {mode} ({vc.iloc[0]}/{top_n} = {mode_pct:.0f}%)")
        if vc.shape[0] > 1:
            print(f"  Others: {dict(vc.iloc[1:4])}")
        print()

    strong_patterns = {k: v["mode"] for k, v in summary.items() if v["mode_pct"] >= 70}
    if strong_patterns:
        print("Strong patterns (>=70% agreement):")
        for k, v in strong_patterns.items():
            print(f"  {k} = {v}")
    else:
        print("No single configuration dominates the top tier (diverse winners).")

    return pd.DataFrame(summary).T


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    dh_results: pd.DataFrame,
    subset: str = "oos",
    metrics: Optional[Dict[str, str]] = None,
    calibration_metrics: Optional[List[str]] = None,
    borda_cutpoint: Optional[int] = None,
    skip_friedman: bool = False,
    top_n_friedman: int = 30,
    top_n_profile: int = 10,
    n_clusters: Optional[int] = None,
) -> Dict:
    """Run the complete multi-criteria model selection pipeline.

    Parameters
    ----------
    dh_results:
        Raw dh_results.parquet DataFrame.
    subset:
        'is', 'oos', or 'both' — passed to prepare_rankings_df.
    metrics:
        Dict mapping metric -> 'lower'/'higher'. Default: DEFAULT_METRICS.
    calibration_metrics:
        Metrics where closer to 1 is better. Default: CALIBRATION_METRICS.
    borda_cutpoint:
        Manual Borda cutpoint. If None, auto-detect.
    skip_friedman:
        If True, run pairwise dominance instead of Friedman/Nemenyi (faster).
    top_n_friedman:
        Models passed to Friedman test (ignored if skip_friedman=True).
    top_n_profile:
        Models in winner profile.
    n_clusters:
        Config clusters. If None, auto-detect.

    Returns
    -------
    Dict with results from each step.
    """
    df = prepare_rankings_df(dh_results, subset=subset)
    metrics = metrics or DEFAULT_METRICS
    calibration_metrics = calibration_metrics or CALIBRATION_METRICS

    # Filter to columns present in df.
    metrics = {k: v for k, v in metrics.items() if k in df.columns}
    calibration_metrics = [c for c in calibration_metrics if c in df.columns]

    print("=" * 72)
    print("MULTI-CRITERIA MODEL SELECTION PIPELINE")
    print("=" * 72)
    print(f"\nInput: {len(df)} models, {len(metrics)} metrics  (subset={subset!r})")
    print(f"Metrics: {list(metrics.keys())}")

    results = {}

    print("\n" + "─" * 72)
    print("STEP 1: BORDA COUNT RANKING")
    print("─" * 72)
    df_borda, cutpoint = borda_rank(df, metrics, calibration_metrics)
    if borda_cutpoint is not None:
        cutpoint = borda_cutpoint
        print(f"\nUsing manual cutpoint: {cutpoint}")
    results["borda"] = (df_borda, cutpoint)
    top_tier = df_borda.head(cutpoint).copy()
    print(f"\n→ Top tier: {cutpoint} models for remaining analysis")

    print("\n" + "─" * 72)
    print("STEP 2: PARETO DOMINANCE (Top Tier Only)")
    print("─" * 72)
    pareto_df = pareto_frontier(top_tier, metrics, calibration_metrics)
    results["pareto"] = pareto_df

    print("\n" + "─" * 72)
    print("STEP 3: KENDALL TAU METRIC AGREEMENT (Top Tier)")
    print("─" * 72)
    tau_df = kendall_tau_heatmap(top_tier, metrics, calibration_metrics)
    results["kendall_tau"] = tau_df

    print("\n" + "─" * 72)
    if skip_friedman:
        print("STEP 4: PAIRWISE DOMINANCE (replaces Friedman at scale)")
        print("─" * 72)
        dominance_df = pairwise_dominance_summary(top_tier, metrics, calibration_metrics)
        results["pairwise_dominance"] = dominance_df
        results["friedman"] = (None, None, None)
    else:
        print("STEP 4: FRIEDMAN + NEMENYI TEST")
        print("─" * 72)
        stat, pval, nemenyi = friedman_nemenyi(
            df_borda, metrics, calibration_metrics,
            top_n=min(top_n_friedman, cutpoint),
        )
        results["friedman"] = (stat, pval, nemenyi)
        results["pairwise_dominance"] = None

    print("\n" + "─" * 72)
    print("STEP 5: TOPSIS RANKING (Top Tier Only)")
    print("─" * 72)
    topsis_df, influence = topsis_rank(top_tier, metrics, calibration_metrics)
    results["topsis"] = (topsis_df, influence)

    print("\n" + "─" * 72)
    print("STEP 6: CONFIGURATION CLUSTERING")
    print("─" * 72)
    clustered_df, cluster_summary = cluster_configurations(
        topsis_df, n_clusters=n_clusters
    )
    results["clusters"] = (clustered_df, cluster_summary)

    print("\n" + "─" * 72)
    print("STEP 7: WINNER PROFILE")
    print("─" * 72)
    profile = winner_profile(clustered_df, top_n=top_n_profile)
    results["profile"] = profile

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)

    return results
