"""Screening / model-selection diagnostic plots for the double-hurdle grid.

Canonical home for the preliminary/refined screening helpers used by the diagnostics notebooks
and the decision reports: data loading + per-config aggregation, per-stage dedup, family ×
exposure heatmaps, cross-threshold Kendall-tau, the family attainability heatmap, and the
per-family metric dot plot. Import as ``from idd_tc_mortality.viz import screening`` — moved out
of reports/_helpers.py so it lives in the package instead of behind a sys.path hack.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm

CONFIG_COLS = [
    "threshold_quantile",
    "s1_family", "s1_exposure_mode", "s1_cov",
    "s2_family", "s2_exposure_mode", "s2_cov",
    "bulk_family", "bulk_exposure_mode", "bulk_cov",
    "tail_family", "tail_exposure_mode", "tail_cov",
]


# ---------------------------------------------------------------------------
# Data loading + per-config aggregation
# ---------------------------------------------------------------------------

def load_and_aggregate(results_path: str | Path) -> pd.DataFrame:
    """Load dh_results.parquet and median-aggregate to one row per config."""
    dh = pd.read_parquet(results_path)
    is_mask = dh["fold_tag"] == "insample"

    is_metric_cols = [
        c for c in dh.columns
        if c not in CONFIG_COLS + ["fold_tag", "seed"]
        and not c.endswith("_oos")
        and pd.api.types.is_numeric_dtype(dh[c])
    ]
    oos_metric_cols = [
        c for c in dh.columns
        if c.endswith("_oos") and pd.api.types.is_numeric_dtype(dh[c])
    ]

    is_agg = (dh[is_mask]
              .groupby(CONFIG_COLS, dropna=False, as_index=False)[is_metric_cols]
              .median())
    oos_agg = (dh[~is_mask]
               .groupby(CONFIG_COLS, dropna=False, as_index=False)[oos_metric_cols]
               .median())
    return is_agg.merge(oos_agg, on=CONFIG_COLS, how="outer")


def dedup_for_stage(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """One row per unique stage spec (S1 has no threshold dependence)."""
    fam = f"{stage}_family"
    exp = f"{stage}_exposure_mode"
    cov = f"{stage}_cov"
    if stage == "s1":
        keys = [fam, exp, cov]
    else:
        keys = [fam, exp, cov, "threshold_quantile"]
    metric_cols = [c for c in df.columns if c.startswith(f"{stage}_") and c.endswith("_oos")]
    return df[keys + metric_cols].drop_duplicates(subset=keys).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Color / marker palettes
# ---------------------------------------------------------------------------

def _palette_and_markers(families, exposures):
    cmap = plt.get_cmap("tab10" if len(families) <= 10 else "tab20")
    fam_color = {f: cmap(i % cmap.N) for i, f in enumerate(families)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    exp_marker = {e: markers[i % len(markers)] for i, e in enumerate(exposures)}
    return fam_color, exp_marker


# ---------------------------------------------------------------------------
# Family × exposure_mode heatmap (best per cell)
# ---------------------------------------------------------------------------

def _agg_for(lower_better: bool) -> str:
    return "min" if lower_better else "max"


def _short_sci(v: float) -> str:
    if not np.isfinite(v):
        return ""
    if v == 0:
        return "0"
    if 1e-3 <= abs(v) < 1e4:
        return f"{v:.3g}"
    s = f"{v:.1e}"
    mantissa, exp = s.split("e")
    return f"{mantissa}e{int(exp)}"


def _family_exposure_pivot(df, stage, metric, *, lower_better):
    fam_col = f"{stage}_family"
    exp_col = f"{stage}_exposure_mode"
    sub = df[[fam_col, exp_col, metric]].copy()
    sub = sub[np.isfinite(sub[metric])]
    families = sorted(df[fam_col].dropna().unique())
    exposures = sorted(df[exp_col].dropna().unique())
    return (
        sub.pivot_table(values=metric, index=fam_col, columns=exp_col,
                        aggfunc=_agg_for(lower_better))
           .reindex(index=families, columns=exposures)
    )


def _draw_heatmap(ax, pivot, *, log_scale, lower_better, title):
    cmap = "viridis_r" if lower_better else "viridis"
    vals = pivot.values
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        ax.set_title(f"{title} (no data)")
        return
    if log_scale and np.nanmin(finite) > 0:
        im = ax.imshow(vals, aspect="auto",
                       norm=LogNorm(vmin=finite.min(), vmax=finite.max()),
                       cmap=cmap)
    else:
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(title, fontsize=10)
    midpoint = np.nanmedian(finite)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v = vals[r, c]
            if np.isfinite(v):
                bad = (lower_better and v > midpoint) or (not lower_better and v < midpoint)
                ax.text(c, r, _short_sci(v), ha="center", va="center",
                        fontsize=8, color="white" if bad else "black")
    plt.colorbar(im, ax=ax, fraction=0.045)


def _grid_shape(n: int):
    if n <= 1: return (1, 1)
    if n <= 3: return (1, n)
    if n == 4: return (2, 2)
    if n <= 6: return (2, 3)
    if n <= 9: return (3, 3)
    return (4, 3)


def plot_stage_heatmap(df, stage, metric, *, lower_better, log_scale,
                       facet_by_threshold):
    if facet_by_threshold:
        thresholds = sorted(df["threshold_quantile"].dropna().unique())
        rows, cols = _grid_shape(len(thresholds))
        fam_count = df[f"{stage}_family"].nunique()
        per_panel_w = 4.2
        per_panel_h = 0.5 * fam_count + 1.5
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(per_panel_w * cols, per_panel_h * rows),
                                 squeeze=False)
        for j, thr in enumerate(thresholds):
            r, c = divmod(j, cols)
            sub = df[df["threshold_quantile"] == thr]
            pivot = _family_exposure_pivot(sub, stage, metric,
                                           lower_better=lower_better)
            _draw_heatmap(axes[r, c], pivot,
                          log_scale=log_scale, lower_better=lower_better,
                          title=f"threshold={thr:.2f}")
        for j in range(len(thresholds), rows * cols):
            r, c = divmod(j, cols)
            axes[r, c].set_visible(False)
        fig.suptitle(f"{stage}: BEST {metric} by family × exposure_mode",
                     y=1.00, fontsize=11)
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot = _family_exposure_pivot(df, stage, metric,
                                       lower_better=lower_better)
        _draw_heatmap(ax, pivot,
                      log_scale=log_scale, lower_better=lower_better,
                      title=f"{stage}: BEST {metric} by family × exposure_mode")
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Cross-threshold Kendall tau (rank stability per metric)
# ---------------------------------------------------------------------------

def _rank_key(values, direction):
    """Return (rank_key, ascending) for a metric direction.

    'lower'   -> rank the raw value ascending  (smallest = best).
    'higher'  -> rank the raw value descending (largest  = best).
    'calib' / 'closer_to_one' -> rank |value - 1| ascending (nearest 1 = best) — the
        calibration convention, matching select.model_selection's CALIBRATION_METRICS.
    """
    if direction in ("calib", "closer_to_one"):
        return (values - 1.0).abs(), True
    return values, (direction == "lower")


def kendall_threshold_heatmap(df, stage, metrics, *, metric_directions):
    from scipy.stats import kendalltau
    fam = f"{stage}_family"
    exp = f"{stage}_exposure_mode"
    cov = f"{stage}_cov"
    keys = [fam, exp, cov]
    thresholds = sorted(df["threshold_quantile"].dropna().unique())
    n = len(thresholds)
    metrics = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(3.6 * len(metrics), 3.6),
                             squeeze=False)
    for k, metric in enumerate(metrics):
        sub = df[[*keys, "threshold_quantile", metric]].copy()
        sub = sub[np.isfinite(sub[metric])]
        _key, ascending = _rank_key(sub[metric], metric_directions.get(metric, "lower"))
        sub["_rk"] = _key
        sub["rank"] = sub.groupby("threshold_quantile")["_rk"].rank(
            ascending=ascending, method="average")
        sub["config_id"] = sub[keys].astype(str).agg(" / ".join, axis=1)
        rank_mat = sub.pivot_table(index="config_id", columns="threshold_quantile",
                                    values="rank", aggfunc="mean")
        tau = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if i == j:
                    tau[i, j] = 1.0
                    continue
                ti, tj = thresholds[i], thresholds[j]
                both = rank_mat[[ti, tj]].dropna()
                if len(both) >= 2:
                    t, _ = kendalltau(both[ti], both[tj])
                    tau[i, j] = t
        ax = axes[0, k]
        im = ax.imshow(tau, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{t:.2f}" for t in thresholds],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"{t:.2f}" for t in thresholds], fontsize=8)
        ax.set_title(metric, fontsize=9)
        for i in range(n):
            for j in range(n):
                if np.isfinite(tau[i, j]):
                    txt_color = "white" if (tau[i, j] < 0.3 or tau[i, j] > 0.85) else "black"
                    ax.text(j, i, f"{tau[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color=txt_color)
        plt.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(f"{stage}: Kendall tau between thresholds (rank stability per metric)",
                 y=1.02)
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Family attainability heatmap (best rank achievable per family per cell)
# ---------------------------------------------------------------------------

def family_attainability_heatmap(df, stage, metrics, *, metric_directions,
                                 top_n=10, drop_threshold=20, col_order="metric"):
    # col_order: "metric" groups columns by metric then threshold (default);
    # "threshold" groups by threshold then metric. Metrics keep the caller's
    # `metrics` list order within each group either way.
    fam_col = f"{stage}_family"
    thresholds = sorted(df["threshold_quantile"].dropna().unique())
    metrics = [m for m in metrics if m in df.columns]
    families = sorted(df[fam_col].dropna().unique())

    rows = []
    for metric in metrics:
        sub = df[[fam_col, "threshold_quantile", metric]].copy()
        sub = sub[np.isfinite(sub[metric])]
        _key, ascending = _rank_key(sub[metric], metric_directions.get(metric, "lower"))
        sub["_rk"] = _key
        sub["rank"] = sub.groupby("threshold_quantile")["_rk"].rank(
            ascending=ascending, method="average")
        best = sub.groupby([fam_col, "threshold_quantile"])["rank"].min()
        for (fam, thr), v in best.items():
            rows.append({"family": fam, "metric": metric,
                         "threshold": thr, "best_rank": v})

    long = pd.DataFrame(rows)
    pivot = (long.pivot_table(index="family", columns=["metric", "threshold"],
                              values="best_rank", aggfunc="min")
                 .reindex(families))
    thr = sorted({t for (_, t) in pivot.columns})
    mets = [m for m in metrics if m in {c[0] for c in pivot.columns}]
    if col_order == "threshold":
        order = [(m, t) for t in thr for m in mets if (m, t) in pivot.columns]
    else:
        order = [(m, t) for m in mets for t in thr if (m, t) in pivot.columns]
    pivot = pivot[order]

    fig, ax = plt.subplots(figsize=(0.5 * pivot.shape[1] + 4,
                                     0.45 * len(families) + 2))
    vals = pivot.values
    # Discrete bins so top-N is visually distinct from "near top" and "bad".
    max_val = float(np.nanmax(vals))
    bounds = [1, top_n // 2, top_n + 1, drop_threshold + 1, 41, max(max_val + 1, 50)]
    bounds = sorted(set(int(b) for b in bounds))
    norm = BoundaryNorm(bounds, ncolors=256, clip=True)
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn_r", norm=norm)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=9)
    if col_order == "threshold":
        col_labels = [f"q{t:.2f}·{m.replace(stage + '_', '').replace('_oos', '')}"
                      for (m, t) in pivot.columns]
    else:
        col_labels = [f"{m.replace(stage + '_', '').replace('_oos', '')}@{t:.2f}"
                      for (m, t) in pivot.columns]
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            v = vals[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{int(v)}", ha="center", va="center",
                        fontsize=6,
                        color="white" if v > top_n else "black")
    ax.set_title(f"{stage}: BEST rank achievable per family at each (metric × threshold)  "
                 f"(top {top_n} = green; > {drop_threshold} = drop candidate)",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.025, label="best rank (1 = best)")
    plt.tight_layout()
    return pivot


# ---------------------------------------------------------------------------
# Per-family metric spread — dot plot (complements the best-only attainability heatmap)
# ---------------------------------------------------------------------------

def family_metric_dotplot(df, stage, metrics, *, metric_directions=None, threshold=None,
                          color_by=None, log_metrics=None, order_by=None, jitter=0.12):
    """Per-family metric spread: one panel per metric, one dot per stage spec.

    Complements ``family_attainability_heatmap`` (which shows only each family's BEST): here
    every distinct spec (covariate × exposure × [threshold]) is a point, so a tight cluster
    (robust family) is visually distinct from a long spread (good only at one knob setting) —
    even when both share the same best value.

    df:         a stage frame, one row per spec (e.g. ``dedup_for_stage(df, stage)``, optionally
                with extra assembled metrics merged on per stage spec).
    metrics:    metric columns to panel over side by side (absent ones are skipped).
    metric_directions: optional {metric: 'lower'|'higher'|'calib'}; 'calib' draws the x=1 ideal.
    threshold:  restrict to one ``threshold_quantile``; None pools all specs.
    color_by:   column to color dots by (e.g. f'{stage}_exposure_mode').
    log_metrics: metrics drawn on a log x-axis.
    order_by:   metric whose per-family median sets the shared row order (best on top);
                default ``metrics[0]``.
    """
    metric_directions = metric_directions or {}
    log_metrics = set(log_metrics or [])
    fam_col = f"{stage}_family"
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise ValueError("none of the requested metrics are present in df")
    d = df if threshold is None else df[df["threshold_quantile"] == threshold]

    # Shared family row order (best first) by the per-family median of order_by.
    ob = order_by if (order_by in d.columns) else metrics[0]
    obdir = metric_directions.get(ob, "lower")
    keyvals = (d[ob] - 1.0).abs() if obdir in ("calib", "closer_to_one") else d[ob]
    med = keyvals.groupby(d[fam_col]).median()
    families = list(med.sort_values(ascending=(obdir != "higher")).index)
    ypos = {f: i for i, f in enumerate(families)}

    if color_by and color_by in d.columns:
        levels = sorted(d[color_by].dropna().unique())
        cmap = plt.get_cmap("tab10")
        cmap_ = {lv: cmap(i % 10) for i, lv in enumerate(levels)}
    else:
        cmap_, levels = None, []

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(3.4 * len(metrics) + 1, 0.42 * len(families) + 2),
                             squeeze=False, sharey=True)
    for k, metric in enumerate(metrics):
        ax = axes[0, k]
        cols = [fam_col, metric] + ([color_by] if cmap_ else [])
        sub = d[cols].copy()
        sub = sub[np.isfinite(sub[metric])]
        yy = sub[fam_col].map(ypos).astype(float) + rng.uniform(-jitter, jitter, len(sub))
        colors = sub[color_by].map(cmap_) if cmap_ else "steelblue"
        ax.scatter(sub[metric], yy, s=18, c=colors, alpha=0.75, edgecolors="none")
        if metric in log_metrics:
            ax.set_xscale("log")
        if metric_directions.get(metric) in ("calib", "closer_to_one"):
            ax.axvline(1.0, color="red", ls="--", lw=1)
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels(families, fontsize=8)
        ax.set_title(metric.replace(f"{stage}_", "").replace("_oos", ""), fontsize=9)
        ax.grid(axis="x", ls=":", alpha=0.4)
    axes[0, 0].invert_yaxis()  # sharey → best family on top across all panels
    if cmap_:
        handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap_[lv], label=str(lv))
                   for lv in levels]
        axes[0, -1].legend(handles=handles, title=color_by.replace(f"{stage}_", ""),
                           fontsize=7, loc="best")
    span = "pooled thresholds" if threshold is None else f"threshold {threshold}"
    fig.suptitle(f"{stage}: per-family metric spread ({span})", y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-threshold winner-rule panels: one panel per threshold, six trajectories
# per panel, each tracing the single config picked by a different selection
# rule (best @ level 1 / 5 / 10 / 15 / 20 and best on the mean across levels).
# ---------------------------------------------------------------------------

def threshold_winner_panels(
    survivors: pd.DataFrame,
    *,
    levels: list[int] | None = None,
    level_winner_levels: list[int] | None = None,
    ncols: int = 3,
):
    """Per-threshold trajectory of selection-rule winners.

    `survivors` is the set of DH configs that pass all upstream screening
    (calibration filter + S1 / S2 / Bulk / Tail decisions). For each
    threshold present, six configs are identified and their full coverage
    trajectories across `levels` are drawn:

      - best @ L for L in `level_winner_levels` (argmax of full_coverage_rate_L_oos)
      - best avg (argmax of the mean of full_coverage_rate over `levels`)
    """
    levels = levels if levels is not None else list(range(1, 21))
    level_winner_levels = (
        level_winner_levels if level_winner_levels is not None else [1, 5, 10, 15, 20]
    )

    cols = [f"full_coverage_rate_{lvl}_oos" for lvl in levels]
    missing = [c for c in cols if c not in survivors.columns]
    assert not missing, f"missing columns: {missing}"

    config_id_cols = [
        "s1_family", "s1_exposure_mode", "s1_cov",
        "s2_family", "s2_exposure_mode", "s2_cov",
        "bulk_family", "bulk_exposure_mode", "bulk_cov",
        "tail_family", "tail_exposure_mode", "tail_cov",
    ]
    config_id_cols = [c for c in config_id_cols if c in survivors.columns]

    cmap = plt.get_cmap("viridis")
    rule_colors = {f"best @ {l}": cmap(i / max(len(level_winner_levels) - 1, 1))
                   for i, l in enumerate(level_winner_levels)}
    rule_colors["best avg"] = "black"

    thresholds = sorted(survivors["threshold_quantile"].dropna().unique())
    if not thresholds:
        raise ValueError("no thresholds present in survivors")

    nrows = int(np.ceil(len(thresholds) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.0 * ncols, 3.6 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )

    best_records = []
    for j, thr in enumerate(thresholds):
        ax = axes[j // ncols, j % ncols]
        sub = (
            survivors[survivors["threshold_quantile"] == thr][cols + config_id_cols]
            .replace([np.inf, -np.inf], np.nan)
        )
        vals = sub[cols]
        for lvl in level_winner_levels:
            target_col = f"full_coverage_rate_{lvl}_oos"
            best_idx = vals[target_col].idxmax()
            traj = vals.loc[best_idx].values
            color = rule_colors[f"best @ {lvl}"]
            ax.plot(levels, traj, color=color, linewidth=1.8, label=f"best @ {lvl}")
            ax.plot([lvl], [vals.loc[best_idx, target_col]],
                    marker="o", color=color,
                    markersize=8, markeredgecolor="black", markeredgewidth=0.6)
            best_records.append({
                "threshold": thr, "rule": f"best @ {lvl}",
                **sub.loc[best_idx, config_id_cols].to_dict(),
            })
        # best avg
        best_idx = vals.mean(axis=1).idxmax()
        traj = vals.loc[best_idx].values
        ax.plot(levels, traj, color=rule_colors["best avg"],
                linewidth=1.8, label="best avg")
        best_records.append({
            "threshold": thr, "rule": "best avg",
            **sub.loc[best_idx, config_id_cols].to_dict(),
        })

        ax.set_title(f"threshold = {thr:.2f}")
        ax.set_xticks(levels)
        ax.tick_params(axis="x", labelsize=7)
        if j // ncols == nrows - 1:
            ax.set_xlabel("coverage level")
        if j % ncols == 0:
            ax.set_ylabel("full_coverage_rate_oos")
    for j in range(len(thresholds), nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"full coverage trajectories of per-rule winner configs "
        f"across {len(survivors):,} surviving configs",
        y=1.00,
    )
    plt.tight_layout()
    return pd.DataFrame(best_records)


# ---------------------------------------------------------------------------
# Stability attrition under a tightening symmetric-log band
# ---------------------------------------------------------------------------

def stability_attrition_curves(
    trim_df: pd.DataFrame,
    attrs: pd.DataFrame,
    *,
    n_cols: list | None = None,
    group_cols: tuple = ("threshold_quantile", "tail_family"),
    lo_grid=None,
    normalize: bool = False,
):
    """Attrition of configs as a symmetric-log calibration band tightens.

    For each ``lo`` in ``lo_grid`` the band is ``[lo, 1/lo]`` (symmetric in log around 1);
    a config survives ``lo`` iff its trimmed ratio stays in-band at EVERY column in
    ``n_cols`` (every drop-top-N level). As ``lo`` -> 1 the band shrinks to a point and
    everything drops out; the level whose curve collapses at the smallest ``lo`` is the
    most fragile threshold / family -- it needs the widest tolerance to keep any config.

    Parameters
    ----------
    trim_df : configs (index = mid) x trimmed-ratio columns (one per drop-top-N level).
    attrs   : per-mid frame carrying the ``group_cols`` (e.g. threshold_quantile, tail_family).
    n_cols  : which ``trim_df`` columns to require in-band (default: all).
    group_cols : one attrition panel per entry.
    lo_grid : lower bounds to sweep (default 0.05..0.95 step 0.05).
    normalize : plot each level's surviving FRACTION (removes base-rate size differences)
                instead of the raw surviving count.

    Returns
    -------
    dict[str, pd.DataFrame] : per group_col, a table indexed by ``lo`` with one column per
    level giving configs surviving (or the surviving fraction when ``normalize``).
    """
    if n_cols is None:
        n_cols = list(trim_df.columns)
    if lo_grid is None:
        lo_grid = np.round(np.arange(0.05, 1.0, 0.05), 2)

    joined = trim_df.join(attrs, how="left")
    vals = trim_df[n_cols]

    tables: dict[str, pd.DataFrame] = {}
    for gc in group_cols:
        levels = sorted(joined[gc].dropna().unique())
        totals = joined[gc].value_counts()
        rows = {}
        for lo in lo_grid:
            hi = 1.0 / lo
            in_band = ((vals >= lo) & (vals <= hi)).all(axis=1)
            surv = joined.loc[in_band, gc].value_counts()
            rows[float(lo)] = {lv: int(surv.get(lv, 0)) for lv in levels}
        tab = pd.DataFrame(rows).T.reindex(columns=levels)
        if normalize:
            tab = tab.divide([totals.get(lv, np.nan) for lv in levels], axis=1)
        tables[gc] = tab

    fig, axes = plt.subplots(1, len(group_cols),
                             figsize=(6.5 * len(group_cols), 4.8), squeeze=False)
    for ax, gc in zip(axes[0], group_cols):
        tab = tables[gc]
        levels = list(tab.columns)
        cmap = plt.get_cmap("tab10" if len(levels) <= 10 else "tab20")
        for i, lv in enumerate(levels):
            label = f"{lv:.2f}" if isinstance(lv, float) else str(lv)
            ax.plot(tab.index, tab[lv].values, marker="o", markersize=3,
                    color=cmap(i % cmap.N), label=label)
        ax.set_xlabel("band lower bound  (band = [lo, 1/lo]; tightens →)")
        ax.set_ylabel("surviving fraction" if normalize else "configs surviving")
        ax.set_title(f"stability attrition by {gc}")
        ax.legend(title=gc, fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return tables
