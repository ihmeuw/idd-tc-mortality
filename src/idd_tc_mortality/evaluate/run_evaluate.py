"""
CLI entry point for evaluate.

Usage:
    run-evaluate \\
        --specs-path    <path/to/manifest.json> \\
        --results-dir   <dir/with/.pkl files> \\
        --data-path     <path/to/data.parquet> \\
        --output-dir    <dir>

For every valid four-way combination of fitted S1 × S2 × bulk × tail components
(valid = same covariate_combo, same threshold_quantile for S2/bulk/tail), this script:

1.  Assembles unconditional predicted death rates via assemble.assemble_predictions.
2.  Computes all metrics (S1 binary, S2 binary, bulk continuous, tail continuous,
    S2-forward, full-model).
3.  Collects one row per combination into dh_results.parquet in output-dir.

Output schema (one row per combination):
    s1_cov                — JSON string of S1 covariate_combo dict
    s2_cov                — JSON string of S2 covariate_combo dict
    bulk_cov              — JSON string of bulk covariate_combo dict
    tail_cov              — JSON string of tail covariate_combo dict
    threshold_quantile    — float quantile level for S2/bulk/tail split
    fold_tag              — 'is' for in-sample, 's{seed}_f{fold}' for OOS
    bulk_family           — string family name for the bulk component
    tail_family           — string family name for the tail component
    IS rows (fold_tag='insample'):
    s1_*                  — S1 binary metrics (brier, auroc, fpr, fnr, predicted_positive_rate)
    s2_*                  — S2 binary metrics (same keys)
    bulk_*                — bulk continuous metrics (mae_rate, rmse_rate, cor_rate,
                            mae_count, rmse_count, cor_count)
    tail_*                — tail continuous metrics (same keys)
    fwd_*                 — forward metrics on any_death=1 rows
    full_*                — full-dataset metrics

    OOS rows (fold_tag='oos_seed{N}') — all metrics get _oos suffix:
    s1_*_oos              — S1 binary OOS metrics (mean across folds)
    s2_*_oos              — S2 binary OOS metrics (mean across folds)
    bulk_*_oos            — bulk continuous OOS metrics (mean across folds)
    tail_*_oos            — tail continuous OOS metrics (mean across folds)
    fwd_*_oos             — forward OOS metrics (assembled held-out predictions)
    full_*_oos            — full-dataset OOS metrics (assembled held-out predictions)

Additional outputs written alongside dh_results.parquet:
    component_predictions/{component_id}_{fold_tag}_predictions.parquet
        — per-component prediction Series (one file per unique component fit).
    model_predictions/{model_id}_predictions.parquet
        — assembled full-model prediction Series (one file per assembled model).

Specs with no matching cached result are silently skipped with a warning.
Combinations where any component is missing are skipped.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from idd_tc_mortality.cache import component_id, load_result, model_id, result_exists
from idd_tc_mortality.metrics import (
    calc_continuous_metrics,
    calc_full_model_metrics,
    calc_s1_metrics,
    calc_s2_forward_metrics,
    calc_s2_metrics,
)
from idd_tc_mortality.evaluate.assemble import assemble_predictions

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _combo_key(covariate_combo: dict) -> str:
    """Full JSON string for grouping specs by covariate_combo."""
    return json.dumps(covariate_combo, sort_keys=True)


def _prefix_keys(d: dict, prefix: str, suffix: str = "") -> dict:
    return {f"{prefix}{k}{suffix}": v for k, v in d.items()}


def _mean_dicts(dicts: list[dict]) -> dict:
    """Mean-aggregate a list of same-keyed metric dicts. NaN values are skipped."""
    if not dicts:
        return {}
    return {k: float(np.nanmean([d[k] for d in dicts])) for k in dicts[0]}


def _is_sentinel(result) -> bool:
    """True if result is a failure sentinel from fit_component's graceful non-convergence.

    A sentinel has param_names=["__failed__"] and meta["fit_error"] set.
    Distinguishes a fit that crashed from one that ran but didn't converge.
    """
    return result.meta.get("fit_error") is not None


_OPTIONAL_COVARIATE_COLS = ("wind_speed", "sdi", "basin", "is_island")


def _build_model_predictions_df(
    pred_series: pd.Series,
    df: pd.DataFrame,
    death_rate: np.ndarray,
    any_death: np.ndarray,
    threshold_rate: float,
    fold_tag: str,
    heldout_fold_tags: pd.Series | None = None,
) -> pd.DataFrame:
    """Build an enriched predictions DataFrame for writing to model_predictions/.

    Includes all columns needed for downstream stage-specific metric computation
    without rejoining to the original data.
    """
    out = pd.DataFrame({
        "predicted_rate":   pred_series.values,
        "observed_rate":    death_rate,
        "any_death":        any_death.astype(float),
        "threshold_rate":   float(threshold_rate),
        "exposed":          df["exposed"].values,
        "fold_tag":         fold_tag,
        "heldout_fold_tag": (
            heldout_fold_tags.values if heldout_fold_tags is not None
            else fold_tag
        ),
    }, index=df.index)
    for col in _OPTIONAL_COVARIATE_COLS:
        if col in df.columns:
            out[col] = df[col].values
    return out


def _save_model_predictions_parquet(df_out: pd.DataFrame, path: Path) -> None:
    """Atomically write an enriched model predictions DataFrame to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df_out.to_parquet(tmp, index=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save_component_predictions_parquet(series: pd.Series, path: Path) -> None:
    """Atomically write a per-component prediction Series to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({"prediction": series})
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df_out.to_parquet(tmp, index=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_available_specs(
    specs_path: str, results_dir: str
) -> dict[str, dict]:
    """Load manifest and return only specs whose result pkl exists."""
    with open(specs_path) as f:
        manifest: dict[str, dict] = json.load(f)

    available = {
        cid: spec
        for cid, spec in manifest.items()
        if result_exists(cid, results_dir)
    }
    n_missing = len(manifest) - len(available)
    if n_missing:
        logger.warning(
            "%d of %d specs have no cached result and will be skipped.",
            n_missing, len(manifest),
        )
    return available


def _group_specs(
    available: dict[str, dict],
) -> tuple[dict, dict, dict, dict]:
    """Split IS specs into four dicts keyed by covariate_combo and threshold.

    Only fold_tag='is' specs are included. OOS assembly is handled separately
    via assemble_oos_predictions, which iterates folds internally.
    """
    s1s:   dict[str,   list] = {}  # combo_key → [spec, ...]  (one per S1 family/exposure_mode)
    s2s:   dict[tuple, list] = {}  # (combo_key, q) → [spec, ...]
    bulks: dict[tuple, list] = {}  # (combo_key, q) → [spec, ...]
    tails: dict[tuple, list] = {}  # (combo_key, q) → [spec, ...]

    for spec in available.values():
        if spec.get("fold_tag", "is") != "is":
            continue
        ck   = _combo_key(spec["covariate_combo"])
        comp = spec["component"]

        if comp == "s1":
            s1s.setdefault(ck, []).append(spec)
        elif comp == "s2":
            s2s.setdefault((ck, spec["threshold_quantile"]), []).append(spec)
        elif comp == "bulk":
            bulks.setdefault((ck, spec["threshold_quantile"]), []).append(spec)
        elif comp == "tail":
            tails.setdefault((ck, spec["threshold_quantile"]), []).append(spec)

    return s1s, s2s, bulks, tails


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--specs-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to manifest.json written by orchestrate.py.",
)
@click.option(
    "--results-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing fitted .pkl result files.",
)
@click.option(
    "--data-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to cleaned training data parquet file.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory where dh_results.parquet is written.",
)
@click.option(
    "--fold-assignments",
    "fold_assignments_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to fold_assignments.parquet written by orchestrate.py. "
         "When provided, OOS evaluation is performed (one row per seed per model config).",
)
@click.option(
    "--s1-spec-id",
    default=None,
    help="When set, evaluate only this S1 component_id. Used by run-evaluate-orchestrate workers.",
)
@click.option(
    "--threshold-quantile",
    "threshold_quantile",
    default=None,
    type=float,
    help="When set, evaluate only this threshold quantile. Used by run-evaluate-orchestrate workers.",
)
def main(
    specs_path: str,
    results_dir: str,
    data_path: str,
    output_dir: str,
    fold_assignments_path: str | None,
    s1_spec_id: str | None,
    threshold_quantile: float | None,
) -> None:
    """Evaluate fitted double-hurdle combinations. With --s1-spec-id and --threshold-quantile,
    processes only that group and writes a partial result to partials/."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    death_rate = df["deaths"].values / df["exposed"].values
    any_death  = (df["deaths"].values >= 1).astype(float)

    fold_df: pd.DataFrame | None = None
    if fold_assignments_path is not None:
        fold_df = pd.read_parquet(fold_assignments_path)
        # Align index to df in case the parquet stored a RangeIndex
        fold_df.index = df.index

    available = _load_available_specs(specs_path, results_dir)
    s1s, s2s, bulks, tails = _group_specs(available)

    # Worker mode: filter to a single (s1_spec_id, threshold_quantile) group.
    if s1_spec_id is not None:
        s1s = {ck: [s for s in specs if component_id(s) == s1_spec_id]
               for ck, specs in s1s.items()}
        s1s = {ck: v for ck, v in s1s.items() if v}
    if threshold_quantile is not None:
        s2s    = {k: v for k, v in s2s.items()    if k[1] == threshold_quantile}
        bulks  = {k: v for k, v in bulks.items()  if k[1] == threshold_quantile}
        tails  = {k: v for k, v in tails.items()  if k[1] == threshold_quantile}

    # Cache loaded FitResults to avoid repeated disk reads.
    result_cache: dict[str, object] = {}

    def _load(spec: dict):
        cid = component_id(spec)
        if cid not in result_cache:
            result_cache[cid] = load_result(cid, results_dir)
        return result_cache[cid]

    from idd_tc_mortality.evaluate.predict_component import predict_one_component

    comp_pred_dir  = output_path / "component_predictions"
    model_pred_dir = output_path / "model_predictions"
    comp_pred_dir.mkdir(parents=True, exist_ok=True)
    model_pred_dir.mkdir(parents=True, exist_ok=True)

    # Track which IS component predictions have already been saved.
    saved_component_ids: set[str] = set()

    rows: list[dict] = []
    n_done = 0

    any_death_mask = any_death.astype(bool)

    for ck, s1_spec_list in s1s.items():
      for s1_spec in s1_spec_list:
        s1_result = _load(s1_spec)
        if _is_sentinel(s1_result):
            logger.warning(
                "S1 sentinel result (combo=%s, family=%s, exposure_mode=%s) — skipping.",
                ck, s1_spec.get("family"), s1_spec.get("exposure_mode"),
            )
            continue

        for (ck2, q), s2_spec_list in s2s.items():
            if ck2 != ck:
                continue

            combo_tup = (ck, q)

            for s2_spec in s2_spec_list:
                s2_result = _load(s2_spec)
                if _is_sentinel(s2_result):
                    logger.warning(
                        "S2 sentinel result (combo=%s, q=%s, family=%s) — skipping.",
                        ck, q, s2_spec.get("family"),
                    )
                    continue

                threshold_rate: float = (
                    s2_spec.get("threshold_rate")
                    or float(np.quantile(death_rate[death_rate > 0], q))
                )
                s2_mask   = any_death_mask
                bulk_mask = (df["deaths"].values >= 1) & (death_rate < threshold_rate)
                tail_mask = death_rate >= threshold_rate

                for bulk_spec in bulks.get(combo_tup, []):
                    bulk_result = _load(bulk_spec)

                    for tail_spec in tails.get(combo_tup, []):
                        tail_result = _load(tail_spec)
                        mid = model_id(s1_spec, s2_spec, bulk_spec, tail_spec)

                        # ---- IS evaluation ----------------------------------------
                        try:
                            pred_series = assemble_predictions(
                                s1_result=s1_result,     s1_spec=s1_spec,
                                s2_result=s2_result,     s2_spec=s2_spec,
                                bulk_result=bulk_result, bulk_spec=bulk_spec,
                                tail_result=tail_result, tail_spec=tail_spec,
                                df=df,
                            )
                        except Exception as exc:
                            logger.warning(
                                "IS assembly failed (s1_cov=%s, q=%s, bulk=%s, tail=%s): %s",
                                _combo_key(s1_spec["covariate_combo"]),
                                q, bulk_spec["family"], tail_spec["family"], exc,
                            )
                            continue

                        pred = pred_series.values

                        s1_pred_series = predict_one_component(s1_spec, s1_result, df)
                        s2_pred_series = predict_one_component(s2_spec, s2_result, df)
                        s2_y_true = (death_rate[s2_mask] >= threshold_rate).astype(float)

                        row_is: dict = {
                            "s1_cov":             _combo_key(s1_spec["covariate_combo"]),
                            "s2_cov":             _combo_key(s2_spec["covariate_combo"]),
                            "bulk_cov":           _combo_key(bulk_spec["covariate_combo"]),
                            "tail_cov":           _combo_key(tail_spec["covariate_combo"]),
                            "threshold_quantile": q,
                            "fold_tag":           "insample",
                            "s1_family":          s1_spec.get("family"),
                            "s1_exposure_mode":   s1_spec.get("exposure_mode"),
                            "s2_family":          s2_spec["family"],
                            "s2_exposure_mode":   s2_spec.get("exposure_mode"),
                            "bulk_family":        bulk_spec["family"],
                            "bulk_exposure_mode": bulk_spec.get("exposure_mode"),
                            "tail_family":        tail_spec["family"],
                            "tail_exposure_mode": tail_spec.get("exposure_mode"),
                            **_prefix_keys(calc_s1_metrics(any_death, s1_pred_series.values), "s1_"),
                            **_prefix_keys(
                                calc_s2_metrics(s2_y_true, s2_pred_series.values[s2_mask]),
                                "s2_",
                            ),
                        }

                        bulk_pred_series: pd.Series | None = None
                        if bulk_mask.any():
                            bulk_pred_series = predict_one_component(bulk_spec, bulk_result, df)
                            row_is.update(_prefix_keys(
                                calc_continuous_metrics(
                                    death_rate[bulk_mask],
                                    bulk_pred_series.values[bulk_mask],
                                    df["exposed"].values[bulk_mask],
                                ),
                                "bulk_",
                            ))

                        tail_pred_series: pd.Series | None = None
                        if tail_mask.any():
                            tail_pred_series = predict_one_component(tail_spec, tail_result, df)
                            row_is.update(_prefix_keys(
                                calc_continuous_metrics(
                                    death_rate[tail_mask],
                                    tail_pred_series.values[tail_mask],
                                    df["exposed"].values[tail_mask],
                                ),
                                "tail_",
                            ))

                        if any_death_mask.any():
                            row_is.update(_prefix_keys(
                                calc_s2_forward_metrics(
                                    death_rate[any_death_mask],
                                    pred[any_death_mask],
                                    df["exposed"].values[any_death_mask],
                                ),
                                "fwd_",
                            ))
                        row_is.update(_prefix_keys(
                            calc_full_model_metrics(
                                death_rate, pred, df["exposed"].values, any_death,
                            ),
                            "full_",
                        ))

                        rows.append(row_is)
                        n_done += 1

                        # Save IS component predictions (once per unique component).
                        for cspec, cpreds in [
                            (s1_spec,   s1_pred_series),
                            (s2_spec,   s2_pred_series),
                            (bulk_spec, bulk_pred_series),
                            (tail_spec, tail_pred_series),
                        ]:
                            if cpreds is None:
                                continue
                            cid = component_id(cspec)
                            if cid not in saved_component_ids:
                                cft = cspec.get("fold_tag", "is")
                                _save_component_predictions_parquet(
                                    cpreds,
                                    comp_pred_dir / f"{cid}_{cft}_predictions.parquet",
                                )
                                saved_component_ids.add(cid)

                        # Save IS model predictions with enriched schema.
                        _save_model_predictions_parquet(
                            _build_model_predictions_df(
                                pred_series, df, death_rate, any_death,
                                threshold_rate, fold_tag="insample",
                            ),
                            model_pred_dir / f"{mid}_insample_predictions.parquet",
                        )

                        # ---- OOS evaluation (per seed) ----------------------------
                        if fold_df is None:
                            continue

                        n_seeds = sum(
                            1 for c in fold_df.columns if c.startswith("seed_")
                        )

                        for seed in range(n_seeds):
                            oos_fold_tag = f"oos_seed{seed}"
                            seed_col     = f"seed_{seed}"
                            n_folds      = int(fold_df[seed_col].nunique())

                            held_out_preds:     list[pd.Series] = []
                            held_out_tags_list: list[pd.Series] = []
                            fold_metrics: dict[str, list[dict]] = {
                                "s1": [], "s2": [], "bulk": [], "tail": [],
                            }
                            fold_loop_ok = True

                            for fold in range(n_folds):
                                ft_str = f"s{seed}_f{fold}"
                                held_out_mask = (fold_df[seed_col].values == fold)
                                df_held = df.loc[df.index[held_out_mask]]

                                s1_oos_spec   = {**s1_spec,   "fold_tag": ft_str}
                                s2_oos_spec   = {**s2_spec,   "fold_tag": ft_str}
                                bulk_oos_spec = {**bulk_spec, "fold_tag": ft_str}
                                tail_oos_spec = {**tail_spec, "fold_tag": ft_str}

                                try:
                                    s1_oos   = load_result(component_id(s1_oos_spec),   results_dir)
                                    s2_oos   = load_result(component_id(s2_oos_spec),   results_dir)
                                    bulk_oos = load_result(component_id(bulk_oos_spec), results_dir)
                                    tail_oos = load_result(component_id(tail_oos_spec), results_dir)
                                except FileNotFoundError as exc:
                                    logger.warning(
                                        "OOS fold missing (seed=%d, fold=%d, "
                                        "s1_cov=%s, q=%s, bulk=%s, tail=%s): %s",
                                        seed, fold,
                                        _combo_key(s1_spec["covariate_combo"]),
                                        q, bulk_spec["family"], tail_spec["family"], exc,
                                    )
                                    fold_loop_ok = False
                                    break

                                if any(_is_sentinel(r) for r in [s1_oos, s2_oos, bulk_oos, tail_oos]):
                                    logger.warning(
                                        "OOS sentinel result (seed=%d, fold=%d, "
                                        "s1_cov=%s, q=%s, bulk=%s, tail=%s) — skipping fold.",
                                        seed, fold,
                                        _combo_key(s1_spec["covariate_combo"]),
                                        q, bulk_spec["family"], tail_spec["family"],
                                    )
                                    fold_loop_ok = False
                                    break

                                # Assembled prediction on full df → extract held-out rows.
                                try:
                                    full_pred = assemble_predictions(
                                        s1_result=s1_oos,   s1_spec=s1_oos_spec,
                                        s2_result=s2_oos,   s2_spec=s2_oos_spec,
                                        bulk_result=bulk_oos, bulk_spec=bulk_oos_spec,
                                        tail_result=tail_oos, tail_spec=tail_oos_spec,
                                        df=df,
                                    )
                                except Exception as exc:
                                    logger.warning(
                                        "OOS assembly failed (seed=%d, fold=%d): %s",
                                        seed, fold, exc,
                                    )
                                    fold_loop_ok = False
                                    break

                                held_out_idx = df.index[held_out_mask]
                                held_out_preds.append(full_pred.loc[held_out_idx])
                                held_out_tags_list.append(
                                    pd.Series(ft_str, index=held_out_idx, dtype=str)
                                )

                                # Per-component metrics on held-out rows only.
                                dr_h = df_held["deaths"].values / df_held["exposed"].values
                                ad_h = (df_held["deaths"].values >= 1).astype(float)

                                # S1: all held-out rows
                                s1_pred_h = predict_one_component(s1_oos_spec, s1_oos, df_held)
                                fold_metrics["s1"].append(
                                    calc_s1_metrics(ad_h, s1_pred_h.values)
                                )

                                # S2: held-out rows where deaths >= 1
                                s2_mask_h = df_held["deaths"].values >= 1
                                if s2_mask_h.any():
                                    s2_pred_h = predict_one_component(
                                        s2_oos_spec, s2_oos, df_held
                                    )
                                    s2_y_h = (dr_h[s2_mask_h] >= threshold_rate).astype(float)
                                    fold_metrics["s2"].append(
                                        calc_s2_metrics(s2_y_h, s2_pred_h.values[s2_mask_h])
                                    )

                                # Bulk: held-out rows in the bulk subset
                                bulk_mask_h = (
                                    (df_held["deaths"].values >= 1)
                                    & (dr_h < threshold_rate)
                                )
                                if bulk_mask_h.any():
                                    bulk_pred_h = predict_one_component(
                                        bulk_oos_spec, bulk_oos, df_held
                                    )
                                    fold_metrics["bulk"].append(
                                        calc_continuous_metrics(
                                            dr_h[bulk_mask_h],
                                            bulk_pred_h.values[bulk_mask_h],
                                            df_held["exposed"].values[bulk_mask_h],
                                        )
                                    )

                                # Tail: held-out rows in the tail subset
                                tail_mask_h = dr_h >= threshold_rate
                                if tail_mask_h.any():
                                    tail_pred_h = predict_one_component(
                                        tail_oos_spec, tail_oos, df_held
                                    )
                                    fold_metrics["tail"].append(
                                        calc_continuous_metrics(
                                            dr_h[tail_mask_h],
                                            tail_pred_h.values[tail_mask_h],
                                            df_held["exposed"].values[tail_mask_h],
                                        )
                                    )

                            if not fold_loop_ok:
                                continue

                            # Stitch held-out subsets → each storm appears exactly once.
                            oos_preds    = pd.concat(held_out_preds).reindex(df.index)
                            heldout_tags = pd.concat(held_out_tags_list).reindex(df.index)
                            oos_pred     = oos_preds.values

                            # All OOS metrics get the _oos suffix so IS and OOS columns
                            # are distinct and self-describing without checking fold_tag.
                            row_oos: dict = {
                                "s1_cov":             _combo_key(s1_spec["covariate_combo"]),
                                "s2_cov":             _combo_key(s2_spec["covariate_combo"]),
                                "bulk_cov":           _combo_key(bulk_spec["covariate_combo"]),
                                "tail_cov":           _combo_key(tail_spec["covariate_combo"]),
                                "threshold_quantile": q,
                                "fold_tag":           oos_fold_tag,
                                "s1_family":          s1_spec.get("family"),
                                "s1_exposure_mode":   s1_spec.get("exposure_mode"),
                                "s2_family":          s2_spec["family"],
                                "s2_exposure_mode":   s2_spec.get("exposure_mode"),
                                "bulk_family":        bulk_spec["family"],
                                "bulk_exposure_mode": bulk_spec.get("exposure_mode"),
                                "tail_family":        tail_spec["family"],
                                "tail_exposure_mode": tail_spec.get("exposure_mode"),
                                # Per-component metrics: mean across folds for this seed.
                                **_prefix_keys(
                                    _mean_dicts(fold_metrics["s1"]),   "s1_",   suffix="_oos"
                                ),
                                **_prefix_keys(
                                    _mean_dicts(fold_metrics["s2"]),   "s2_",   suffix="_oos"
                                ),
                                **_prefix_keys(
                                    _mean_dicts(fold_metrics["bulk"]), "bulk_", suffix="_oos"
                                ),
                                **_prefix_keys(
                                    _mean_dicts(fold_metrics["tail"]), "tail_", suffix="_oos"
                                ),
                            }

                            if any_death_mask.any():
                                row_oos.update(_prefix_keys(
                                    calc_s2_forward_metrics(
                                        death_rate[any_death_mask],
                                        oos_pred[any_death_mask],
                                        df["exposed"].values[any_death_mask],
                                    ),
                                    "fwd_", suffix="_oos",
                                ))
                            row_oos.update(_prefix_keys(
                                calc_full_model_metrics(
                                    death_rate, oos_pred, df["exposed"].values, any_death,
                                ),
                                "full_", suffix="_oos",
                            ))

                            rows.append(row_oos)
                            n_done += 1

                            _save_model_predictions_parquet(
                                _build_model_predictions_df(
                                    oos_preds, df, death_rate, any_death,
                                    threshold_rate, fold_tag=oos_fold_tag,
                                    heldout_fold_tags=heldout_tags,
                                ),
                                model_pred_dir / f"{mid}_{oos_fold_tag}_predictions.parquet",
                            )

    logger.info("Computed metrics for %d combinations.", n_done)

    if not rows:
        logger.warning("No valid combinations found. Output file not written.")
        return

    results_df = pd.DataFrame(rows)

    # Worker mode: write a partial result to partials/ so the orchestrator can aggregate.
    # Serial mode: write directly to dh_results.parquet.
    if s1_spec_id is not None and threshold_quantile is not None:
        q_str = f"{threshold_quantile:.4f}".replace(".", "_")
        partials_dir = output_path / "partials"
        partials_dir.mkdir(parents=True, exist_ok=True)
        out_path = partials_dir / f"dh_{s1_spec_id}_{q_str}.parquet"
        write_dir = partials_dir
    else:
        out_path = output_path / "dh_results.parquet"
        write_dir = output_path

    # Atomic write: temp → validate row count → rename.
    fd, tmp = tempfile.mkstemp(dir=write_dir, suffix=".parquet.tmp")
    os.close(fd)
    try:
        results_df.to_parquet(tmp, index=False)
        meta = pq.read_metadata(tmp)
        if meta.num_rows != len(results_df):
            raise RuntimeError(
                f"Parquet row count mismatch: wrote {len(results_df)}, "
                f"metadata reports {meta.num_rows}."
            )
        if out_path.exists():
            out_path.unlink()
        os.replace(tmp, out_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Wrote %d rows to %s", len(results_df), out_path)
