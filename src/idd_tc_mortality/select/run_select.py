"""
run_select.py

CLI for select: reads dh_results.parquet, runs selected ranking methods,
writes selected_models.parquet with one row per method showing the winning model
config and its metrics.

Usage:
    run-select \\
        --results-path /path/to/dh_results.parquet \\
        --output-dir   /path/to/output/ \\
        --subset       oos \\
        --methods      borda topsis pareto pairwise_dominance
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List

import click
import pandas as pd

from idd_tc_mortality.select.model_selection import (
    CALIBRATION_METRICS,
    CONFIG_COLS,
    DEFAULT_METRICS,
    borda_rank,
    pareto_frontier,
    pairwise_dominance_summary,
    prepare_rankings_df,
    topsis_rank,
    winner_profile,
)

log = logging.getLogger(__name__)

AVAILABLE_METHODS = [
    "borda",
    "pareto",
    "kendall_tau",
    "friedman_nemenyi",
    "pairwise_dominance",
    "topsis",
    "cluster_configurations",
]


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write df to path atomically: temp file → validate → rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    try:
        import os
        os.close(tmp_fd)
        df.to_parquet(tmp_path, index=False)
        # Metadata validation: row count only (never read full file back).
        import pyarrow.parquet as pq
        meta = pq.read_metadata(tmp_path)
        if meta.num_rows != len(df):
            raise RuntimeError(
                f"Row count mismatch: expected {len(df)}, got {meta.num_rows}"
            )
        if path.exists():
            path.unlink()
        shutil.move(tmp_path, path)
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _run_methods(
    df: pd.DataFrame,
    methods: List[str],
    subset: str,
) -> pd.DataFrame:
    """Run each requested ranking method and collect winning rows."""
    metrics = {k: v for k, v in DEFAULT_METRICS.items() if k in df.columns}
    calibration_metrics = [c for c in CALIBRATION_METRICS if c in df.columns]

    config_cols_present = [c for c in CONFIG_COLS if c in df.columns]

    rows = []

    if "borda" in methods:
        result, _ = borda_rank(df, metrics, calibration_metrics, plot=False)
        winner = result.iloc[0]
        row = {"method": "borda", "borda_score": float(winner.get("borda_score", float("nan")))}
        for col in config_cols_present:
            row[col] = winner[col]
        for col in df.columns:
            if col not in CONFIG_COLS and col not in ("fold_tag",):
                row[col] = winner[col]
        rows.append(row)

    if "pareto" in methods:
        result = pareto_frontier(df, metrics, calibration_metrics, verbose=False)
        if len(result) > 0:
            # Among non-dominated, pick the one with best primary metric if available.
            primary = next(
                (m for m in ["full_mae_rate_oos", "full_mae_rate"] if m in result.columns),
                None,
            )
            if primary is not None:
                winner = result.loc[result[primary].idxmin()]
            else:
                winner = result.iloc[0]
            row = {"method": "pareto", "pareto_size": len(result)}
            for col in config_cols_present:
                row[col] = winner[col]
            for col in df.columns:
                if col not in CONFIG_COLS and col not in ("fold_tag",):
                    row[col] = winner[col]
            rows.append(row)

    if "kendall_tau" in methods:
        # Kendall tau is a diagnostic, not a single-winner selector.
        # Emit a row with the method name but no winner config.
        rows.append({"method": "kendall_tau"})

    if "friedman_nemenyi" in methods:
        from idd_tc_mortality.select.model_selection import friedman_nemenyi
        stat, pval, nemenyi_df = friedman_nemenyi(
            df, metrics, calibration_metrics, top_n=min(30, len(df)), plot_cd=False
        )
        row = {"method": "friedman_nemenyi", "stat": stat, "p_value": pval}
        if nemenyi_df is not None and len(nemenyi_df) > 0:
            best_idx = nemenyi_df.iloc[0]["model_idx"]
            winner = df.loc[best_idx]
            for col in config_cols_present:
                row[col] = winner[col]
            for col in df.columns:
                if col not in CONFIG_COLS and col not in ("fold_tag",):
                    row[col] = winner[col]
        rows.append(row)

    if "pairwise_dominance" in methods:
        result = pairwise_dominance_summary(df, metrics, calibration_metrics, plot=False)
        winner = result.iloc[0]
        row = {
            "method":          "pairwise_dominance",
            "dominance_count": float(winner.get("dominance_count", float("nan"))),
            "dominance_pct":   float(winner.get("dominance_pct", float("nan"))),
        }
        for col in config_cols_present:
            row[col] = winner[col]
        for col in df.columns:
            if col not in CONFIG_COLS and col not in ("fold_tag", "dominance_rank",
                                                       "total_metric_wins"):
                row[col] = winner[col]
        rows.append(row)

    if "topsis" in methods:
        result, _ = topsis_rank(df, metrics, calibration_metrics, verbose=False)
        winner = result.iloc[0]
        row = {
            "method":       "topsis",
            "topsis_score": float(winner.get("topsis_score", float("nan"))),
        }
        for col in config_cols_present:
            row[col] = winner[col]
        for col in df.columns:
            if col not in CONFIG_COLS and col not in ("fold_tag", "topsis_rank"):
                row[col] = winner[col]
        rows.append(row)

    if "cluster_configurations" in methods:
        from idd_tc_mortality.select.model_selection import (
            cluster_configurations,
            winner_profile,
        )
        # Run TOPSIS first if not already done, then cluster.
        if "topsis" not in methods:
            topsis_df, _ = topsis_rank(df, metrics, calibration_metrics, verbose=False)
        else:
            topsis_df, _ = topsis_rank(df, metrics, calibration_metrics, verbose=False)
        clustered_df, _ = cluster_configurations(topsis_df, plot=False)
        profile_df = winner_profile(clustered_df, top_n=min(10, len(clustered_df)))
        winner = clustered_df.iloc[0]
        row = {"method": "cluster_configurations"}
        for col in config_cols_present:
            row[col] = winner[col]
        for col in df.columns:
            if col not in CONFIG_COLS and col not in ("fold_tag",):
                row[col] = winner[col]
        rows.append(row)

    return pd.DataFrame(rows)


@click.command()
@click.option(
    "--results-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to dh_results.parquet produced by evaluate.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory to write selected_models.parquet.",
)
@click.option(
    "--subset",
    default="oos",
    type=click.Choice(["is", "oos", "both"]),
    show_default=True,
    help="Which rows of dh_results to use for ranking.",
)
@click.option(
    "--methods",
    multiple=True,
    default=AVAILABLE_METHODS,
    type=click.Choice(AVAILABLE_METHODS),
    show_default=True,
    help="Ranking methods to run (repeatable). Default: all.",
)
def main(results_path: str, output_dir: str, subset: str, methods: tuple) -> None:
    """Run model selection ranking methods on dh_results.parquet."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    results_path_ = Path(results_path)
    output_dir_ = Path(output_dir)

    log.info("Reading %s", results_path_)
    dh_results = pd.read_parquet(results_path_)

    log.info("Preparing rankings df (subset=%r)", subset)
    df = prepare_rankings_df(dh_results, subset=subset)
    log.info("%d model configurations to rank", len(df))

    methods_list = list(methods) if methods else AVAILABLE_METHODS
    log.info("Running methods: %s", methods_list)
    selected = _run_methods(df, methods_list, subset=subset)

    out_path = output_dir_ / "selected_models.parquet"
    log.info("Writing %s (%d rows)", out_path, len(selected))
    _atomic_write_parquet(selected, out_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
