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
import time
from pathlib import Path


def _rss_mb() -> float | None:
    """Current process resident set size in MB, or None if unreadable."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except Exception:
        return None
    return None


class _Tic:
    """Light-weight tick-tock helper for --verbose progress logs.

    Usage:
        tic = _Tic("worker")
        ...
        tic.log("loaded manifest", n_specs=len(available))
        ...
        with tic.timed("IS assembly"):
            do_is_work()
    """

    def __init__(self, label: str, *, enabled: bool):
        self.label    = label
        self.enabled  = enabled
        self.t_start  = time.perf_counter()
        self.t_prev   = self.t_start

    def log(self, what: str, **extras) -> None:
        if not self.enabled:
            return
        now      = time.perf_counter()
        rss      = _rss_mb()
        delta    = now - self.t_prev
        total    = now - self.t_start
        kv       = " ".join(f"{k}={v}" for k, v in extras.items())
        rss_str  = f"rss={rss:.0f}MB" if rss is not None else "rss=?"
        logger.info(
            "[%s] +%6.2fs (total %7.2fs, %s)  %s  %s",
            self.label, delta, total, rss_str, what, kv,
        )
        self.t_prev = now

    class _Block:
        def __init__(self, tic: "_Tic", what: str, extras: dict):
            self.tic    = tic
            self.what   = what
            self.extras = extras

        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            elapsed = time.perf_counter() - self.t0
            self.tic.log(f"{self.what} done", elapsed=f"{elapsed:.3f}s", **self.extras)

    def timed(self, what: str, **extras):
        return _Tic._Block(self, what, extras) if self.enabled else _NullCtx()


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from idd_tc_mortality.cache import component_id, model_id
from idd_tc_mortality.fit.fit_component import fit_one_component
from idd_tc_mortality.metrics import (
    calc_continuous_metrics,
    calc_full_model_metrics,
    calc_s1_metrics,
    calc_s2_forward_metrics,
    calc_s2_metrics,
    calc_time_series_metrics,
)
from idd_tc_mortality.evaluate.assemble import assemble_predictions
from idd_tc_mortality.combine import assemble_dh_prediction
from idd_tc_mortality.evaluate.predict_component import predict_one_component
from idd_tools.jobmon.atomic_io import (
    AtomicRegistry,
    atomic_write_parquet,
    subtask_skip,
)
from idd_tools.jobmon import inflate_cells

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


def _fold_train_subset(spec: dict, df: pd.DataFrame, fold_df: pd.DataFrame | None) -> pd.DataFrame:
    """Training rows for a spec's fold_tag: full df for IS, ``seed != held-out fold``
    for OOS. Mirrors fit.run_component._subset_for_fold but takes the in-memory
    fold_df directly, since the evaluate worker now fits rather than reads results."""
    fold_tag = spec.get("fold_tag", "is")
    if fold_tag == "is" or fold_df is None:
        return df
    seed_part, fold_part = fold_tag.split("_")
    seed_idx = int(seed_part[1:])
    held_out_fold = int(fold_part[1:])
    train_mask = fold_df[f"seed_{seed_idx}"].values != held_out_fold
    return df.loc[df.index[train_mask]]


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


def _load_manifest_specs(specs_path: str) -> dict[str, dict]:
    """Load the IS spec manifest. Every spec is fit in-place by the worker, so
    nothing is filtered against a results directory — no .pkl reads at all."""
    with open(specs_path) as f:
        manifest: dict[str, dict] = json.load(f)
    return manifest


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
# Core evaluation logic
# ---------------------------------------------------------------------------

def _evaluate_group(
    *,
    s1s: dict,
    s2s: dict,
    bulks: dict,
    tails: dict,
    df: pd.DataFrame,
    death_rate: np.ndarray,
    any_death: np.ndarray,
    fold_df: pd.DataFrame | None,
    results_dir: str,
    output_path: Path,
    decouple_covs: bool,
    skip_model_predictions: bool,
    result_cache: dict,
    pred_cache_full: dict,
    saved_component_ids: set,
    tic: _Tic,
) -> list[dict]:
    """Evaluate all valid DH combinations for a filtered spec set.

    Mutates result_cache, pred_cache_full, and saved_component_ids so callers
    looping over multiple groups (bundle mode) share cached state across iterations.
    """
    def _load(spec: dict):
        # Re-fit in-memory (cached per component+fold) instead of reading a pickled
        # result from NFS. Fitting is ~0.1s and contends nothing, unlike the 100s of
        # concurrent .pkl reads that saturated NFS at scale.
        cid = component_id(spec)
        if cid not in result_cache:
            result_cache[cid] = fit_one_component(spec, _fold_train_subset(spec, df, fold_df))
        return result_cache[cid]

    def _predict_full(spec: dict, result) -> pd.Series:
        cid = component_id(spec)
        if cid not in pred_cache_full:
            pred_cache_full[cid] = predict_one_component(spec, result, df)
        return pred_cache_full[cid]

    def _assemble_cached(
        *,
        s1_result,   s1_spec,
        s2_result,   s2_spec,
        bulk_result, bulk_spec,
        tail_result, tail_spec,
    ) -> pd.Series:
        """Inline cached version of assemble.assemble_predictions. Builds
        the unconditional expected-rate series from per-stage predictions
        served by pred_cache_full instead of re-predicting each call."""
        p_s1   = _predict_full(s1_spec,   s1_result)
        p_s2   = _predict_full(s2_spec,   s2_result)
        p_bulk = _predict_full(bulk_spec, bulk_result)
        p_tail = _predict_full(tail_spec, tail_result)
        assembled = assemble_dh_prediction(
            p_s1=p_s1.values,
            p_s2=p_s2.values,
            rate_bulk=p_bulk.values,
            rate_tail=p_tail.values,
        )
        return pd.Series(assembled, index=df.index, name="predicted_rate")

    comp_pred_dir  = output_path / "component_predictions"
    model_pred_dir = output_path / "model_predictions"
    comp_pred_dir.mkdir(parents=True, exist_ok=True)
    model_pred_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_done = 0

    any_death_mask = any_death.astype(bool)

    # Precompute the set of thresholds present in s2 / bulk / tail keys so
    # decoupled iteration can sweep "all covs at threshold q" cheaply.
    thresholds_present = sorted({q for (_ck, q) in s2s.keys()})

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
            # Coupled mode requires the S2 covariate key to equal the S1's.
            # Decoupled mode lets s2_cov vary independently of s1_cov.
            if (not decouple_covs) and ck2 != ck:
                continue

            for s2_spec in s2_spec_list:
                s2_result = _load(s2_spec)
                if _is_sentinel(s2_result):
                    logger.warning(
                        "S2 sentinel result (combo=%s, q=%s, family=%s) — skipping.",
                        ck2, q, s2_spec.get("family"),
                    )
                    continue

                threshold_rate: float = (
                    s2_spec.get("threshold_rate")
                    or float(np.quantile(death_rate[death_rate > 0], q))
                )
                s2_mask   = any_death_mask
                bulk_mask = (df["deaths"].values >= 1) & (death_rate < threshold_rate)
                tail_mask = death_rate >= threshold_rate

                # Build the bulk / tail iteration sets.
                # Coupled: only specs with the SAME cov_key as S1 at this threshold.
                # Decoupled: all specs at this threshold across every cov_key.
                if decouple_covs:
                    bulk_specs_iter = [
                        b for (bck, bq), blist in bulks.items() if bq == q
                        for b in blist
                    ]
                    tail_specs_iter = [
                        t for (tck, tq), tlist in tails.items() if tq == q
                        for t in tlist
                    ]
                else:
                    bulk_specs_iter = bulks.get((ck, q), [])
                    tail_specs_iter = tails.get((ck, q), [])

                for bulk_spec in bulk_specs_iter:
                    bulk_result = _load(bulk_spec)

                    for tail_spec in tail_specs_iter:
                        tail_result = _load(tail_spec)
                        mid = model_id(s1_spec, s2_spec, bulk_spec, tail_spec)
                        tic.log("DH config start", mid=mid[:8],
                                tail_fam=tail_spec.get("family"),
                                bulk_fam=bulk_spec.get("family"))

                        # ---- IS evaluation ----------------------------------------
                        try:
                            pred_series = _assemble_cached(
                                s1_result=s1_result,     s1_spec=s1_spec,
                                s2_result=s2_result,     s2_spec=s2_spec,
                                bulk_result=bulk_result, bulk_spec=bulk_spec,
                                tail_result=tail_result, tail_spec=tail_spec,
                            )
                        except Exception as exc:
                            logger.warning(
                                "IS assembly failed (s1_cov=%s, q=%s, bulk=%s, tail=%s): %s",
                                _combo_key(s1_spec["covariate_combo"]),
                                q, bulk_spec["family"], tail_spec["family"], exc,
                            )
                            continue

                        pred = pred_series.values
                        tic.log("IS assemble done")

                        s1_pred_series = _predict_full(s1_spec, s1_result)
                        s2_pred_series = _predict_full(s2_spec, s2_result)
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
                            bulk_pred_series = _predict_full(bulk_spec, bulk_result)
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
                            tail_pred_series = _predict_full(tail_spec, tail_result)
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
                        row_is.update(calc_time_series_metrics(
                            df["year"].values, death_rate, pred, df["exposed"].values,
                        ))

                        rows.append(row_is)
                        n_done += 1
                        tic.log("IS metrics + row done", n_done=n_done)

                        # Save IS component predictions (once per unique component).
                        # Per-component prediction parquets: at most one write per
                        # unique component spec across the whole worker, so leave
                        # these enabled even under --skip-model-predictions; the
                        # explosive cost is the per-DH-config model_predictions
                        # writes below, not these.
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

                        # Per-DH-config IS model predictions — one parquet per
                        # config. Gated by --skip-model-predictions because at
                        # refined scale this is the dominant NFS write cost.
                        if not skip_model_predictions:
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
                                tic.log("OOS fold start", fold_tag=ft_str)
                                held_out_mask = (fold_df[seed_col].values == fold)
                                df_held = df.loc[df.index[held_out_mask]]

                                s1_oos_spec   = {**s1_spec,   "fold_tag": ft_str}
                                s2_oos_spec   = {**s2_spec,   "fold_tag": ft_str}
                                bulk_oos_spec = {**bulk_spec, "fold_tag": ft_str}
                                tail_oos_spec = {**tail_spec, "fold_tag": ft_str}

                                # Route OOS pickle loads through the _load cache. The
                                # IS path already uses _load, so each unique component_id
                                # is hit exactly once per worker. Direct load_result()
                                # calls here previously re-read every OOS pickle from
                                # NFS once per DH config × fold — fine when each worker
                                # had ~96 configs (coupled), catastrophic at ~393K
                                # (decoupled).
                                try:
                                    s1_oos   = _load(s1_oos_spec)
                                    s2_oos   = _load(s2_oos_spec)
                                    bulk_oos = _load(bulk_oos_spec)
                                    tail_oos = _load(tail_oos_spec)
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
                                # Per-stage held-out predictions are sliced from the
                                # cached full-df series rather than re-predicting on
                                # df_held — same numerical result, ~one disk-cache hit
                                # per stage per fold instead of N (per DH config).
                                try:
                                    full_pred = _assemble_cached(
                                        s1_result=s1_oos,   s1_spec=s1_oos_spec,
                                        s2_result=s2_oos,   s2_spec=s2_oos_spec,
                                        bulk_result=bulk_oos, bulk_spec=bulk_oos_spec,
                                        tail_result=tail_oos, tail_spec=tail_oos_spec,
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
                                s1_pred_h = _predict_full(s1_oos_spec, s1_oos).loc[df_held.index]
                                fold_metrics["s1"].append(
                                    calc_s1_metrics(ad_h, s1_pred_h.values)
                                )

                                # S2: held-out rows where deaths >= 1
                                s2_mask_h = df_held["deaths"].values >= 1
                                if s2_mask_h.any():
                                    s2_pred_h = _predict_full(s2_oos_spec, s2_oos).loc[df_held.index]
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
                                    bulk_pred_h = _predict_full(bulk_oos_spec, bulk_oos).loc[df_held.index]
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
                                    tail_pred_h = _predict_full(tail_oos_spec, tail_oos).loc[df_held.index]
                                    fold_metrics["tail"].append(
                                        calc_continuous_metrics(
                                            dr_h[tail_mask_h],
                                            tail_pred_h.values[tail_mask_h],
                                            df_held["exposed"].values[tail_mask_h],
                                        )
                                    )
                                tic.log("OOS fold metrics done", fold_tag=ft_str)

                            if not fold_loop_ok:
                                continue

                            # Stitch held-out subsets → each storm appears exactly once.
                            oos_preds    = pd.concat(held_out_preds).reindex(df.index)
                            heldout_tags = pd.concat(held_out_tags_list).reindex(df.index)
                            oos_pred     = oos_preds.values
                            tic.log("OOS stitch done", seed=seed)

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
                            row_oos.update(_prefix_keys(
                                calc_time_series_metrics(
                                    df["year"].values, death_rate, oos_pred, df["exposed"].values,
                                ),
                                "", suffix="_oos",
                            ))

                            rows.append(row_oos)
                            n_done += 1
                            tic.log("OOS metrics + row done",
                                    seed=seed, n_done=n_done)

                            # Gated for the same reason as the IS write above.
                            if not skip_model_predictions:
                                _save_model_predictions_parquet(
                                    _build_model_predictions_df(
                                        oos_preds, df, death_rate, any_death,
                                        threshold_rate, fold_tag=oos_fold_tag,
                                        heldout_fold_tags=heldout_tags,
                                    ),
                                    model_pred_dir / f"{mid}_{oos_fold_tag}_predictions.parquet",
                                )

    return rows


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
    default=None,
    type=click.Path(file_okay=False),
    help="(Deprecated; unused.) Models are now re-fit in-memory rather than read "
         "from disk. Retained as an accepted no-op until the fit stage is removed.",
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
@click.option(
    "--decouple-covs",
    is_flag=True,
    default=False,
    help="If set, assemble DH configurations from the full Cartesian product of "
         "per-stage covariate combos (s1_cov × s2_cov × bulk_cov × tail_cov). "
         "Default behaviour (off) requires all four stage covs to match — "
         "appropriate for the preliminary grid but discards per-stage cov "
         "diversification at the refined stage. Ignored when --task-file is set.",
)
@click.option(
    "--task-file",
    "task_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a task JSON file (produced by run-build-evaluate-tasks). "
         "When provided alongside --task-index, the worker processes only the "
         "specs listed in that task and writes its partial to "
         "partials/dh_task_{task_index:05d}.parquet. Takes precedence over "
         "--s1-spec-id / --threshold-quantile / --decouple-covs (those are "
         "ignored). Decoupling is implicit in task mode — the task's per-stage "
         "spec_id lists may span multiple covs.",
)
@click.option(
    "--cells-file",
    "cells_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a partitioned cell manifest JSON (from run-build-refined-cells). "
         "With --task-index, the worker scores the explicit DH-config cells in that "
         "task — each cell is a (s1, s2, bulk, tail) spec_id 4-tuple — and writes its "
         "partial to partials/dh_task_{task_index:05d}.parquet. Unlike --task-file "
         "(which crosses per-stage spec lists), cells are scored individually, so "
         "arbitrary nesting (e.g. tail ⊆ bulk ⊆ s2) is honoured with no Cartesian.",
)
@click.option(
    "--task-index",
    "task_index",
    default=None,
    type=int,
    help="Index into --task-file's or --cells-file's tasks list. Required if either is set.",
)
@click.option(
    "--bundle-file",
    "bundle_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a bundle JSON file produced by run-evaluate-orchestrate. "
         "Contains a list of [s1_spec_id, s2_spec_id, threshold_quantile] triples to evaluate "
         "serially, sharing caches across iterations. Used with --bundle-index to route the partial output.",
)
@click.option(
    "--bundle-index",
    "bundle_index",
    default=None,
    type=int,
    help="Index of this bundle; used to name the partial output "
         "partials/dh_bundle_{bundle_index:05d}.parquet. Required with --bundle-file.",
)
@click.option(
    "--skip-model-predictions",
    is_flag=True,
    default=False,
    help="Skip writing per-DH-config `model_predictions/*.parquet` files. The "
         "aggregated dh_results.parquet (metric rows) is still produced. With "
         "decoupled-cov runs each worker would otherwise emit hundreds of "
         "thousands of tiny parquets — a major NFS bottleneck — and the "
         "per-config prediction parquets are rarely used at refined scale.",
)
@click.option(
    "--scope",
    is_flag=True,
    default=False,
    help="Scope-out / instrument-mode: emit per-step tick-tock logs with "
         "elapsed time and RSS memory. Intended for diagnostic runs (one or "
         "a handful of DH configs) where you want to see exactly where time "
         "and memory are going. For large tasks the output is millions of "
         "lines — only enable on small probes for measurement.",
)
def main(
    specs_path: str,
    results_dir: str,
    data_path: str,
    output_dir: str,
    fold_assignments_path: str | None,
    s1_spec_id: str | None,
    threshold_quantile: float | None,
    decouple_covs: bool,
    task_file: str | None,
    cells_file: str | None,
    task_index: int | None,
    bundle_file: str | None,
    bundle_index: int | None,
    skip_model_predictions: bool,
    scope: bool,
) -> None:
    """Evaluate fitted double-hurdle combinations.

    Modes (in priority order):
      --bundle-file / --bundle-index: loop over N (s1_spec_id, threshold) groups,
        sharing caches across iterations; write partials/dh_bundle_{i:05d}.parquet.
      --task-file / --task-index: evaluate the specs listed in that task entry;
        write partials/dh_task_{i:05d}.parquet.
      --s1-spec-id + --threshold-quantile: legacy single-group worker mode.
      (no filter): serial full-grid evaluation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tic = _Tic("worker", enabled=scope)
    tic.log("startup", data=data_path, results=results_dir, output=output_dir,
            task_file=task_file, task_index=task_index,
            bundle_file=bundle_file, bundle_index=bundle_index,
            skip_model_predictions=skip_model_predictions)

    if (bundle_file is None) != (bundle_index is None):
        raise click.UsageError("--bundle-file and --bundle-index must be used together.")
    _index_modes = [m for m in (task_file, cells_file) if m is not None]
    if len(_index_modes) > 1:
        raise click.UsageError("--task-file and --cells-file are mutually exclusive.")
    if bool(_index_modes) != (task_index is not None):
        raise click.UsageError(
            "--task-index must be used with exactly one of --task-file / --cells-file."
        )

    df = pd.read_parquet(data_path)
    death_rate = df["deaths"].values / df["exposed"].values
    any_death  = (df["deaths"].values >= 1).astype(float)
    tic.log("loaded input.parquet", n_rows=len(df))

    fold_df: pd.DataFrame | None = None
    if fold_assignments_path is not None:
        fold_df = pd.read_parquet(fold_assignments_path)
        # Align index to df in case the parquet stored a RangeIndex
        fold_df.index = df.index
        tic.log("loaded fold_assignments", n_rows=len(fold_df))

    available = _load_manifest_specs(specs_path)
    s1s, s2s, bulks, tails = _group_specs(available)
    tic.log("loaded + grouped manifest",
            n_available=len(available),
            n_s1=sum(len(v) for v in s1s.values()),
            n_s2=sum(len(v) for v in s2s.values()),
            n_bulk=sum(len(v) for v in bulks.values()),
            n_tail=sum(len(v) for v in tails.values()))

    # Shared caches across all _evaluate_group calls — bundle iterations share state
    # so each component pkl is read from NFS at most once per worker.
    result_cache: dict[str, object] = {}
    pred_cache_full: dict[str, pd.Series] = {}
    saved_component_ids: set[str] = set()

    _eval_kwargs = dict(
        df=df, death_rate=death_rate, any_death=any_death,
        fold_df=fold_df, results_dir=results_dir, output_path=output_path,
        skip_model_predictions=skip_model_predictions,
        result_cache=result_cache,
        pred_cache_full=pred_cache_full,
        saved_component_ids=saved_component_ids,
        tic=tic,
    )

    if bundle_file is not None:
        with open(bundle_file) as f:
            bundle_groups = json.load(f)

        partials_dir = output_path / "partials"
        partials_dir.mkdir(parents=True, exist_ok=True)
        entries_dir = partials_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        bundle_partial_path = partials_dir / f"dh_bundle_{bundle_index:05d}.parquet"

        all_rows: list[dict] = []
        with AtomicRegistry(
            output_path,
            task_id=f"bundle_{bundle_index:05d}",
            task_template="evaluate_worker_bundle",
        ) as registry:
            for entry in bundle_groups:
                s1_id, s2_id, q = entry[0], entry[1], float(entry[2])
                entry_path = entries_dir / f"entry_{s1_id[:8]}_{s2_id[:8]}_{q:.4f}.parquet"

                if subtask_skip(entry_path):
                    logger.info("Skipping completed entry %s/%s/%.4f", s1_id[:8], s2_id[:8], q)
                    all_rows.extend(pd.read_parquet(entry_path).to_dict("records"))
                    continue

                _s1s = {ck: [s for s in specs if component_id(s) == s1_id]
                        for ck, specs in s1s.items()}
                _s1s = {ck: v for ck, v in _s1s.items() if v}
                _s2s   = {k: [s for s in v if component_id(s) == s2_id]
                          for k, v in s2s.items() if k[1] == q}
                _s2s   = {k: v for k, v in _s2s.items() if v}
                _bulks = {k: v for k, v in bulks.items() if k[1] == q}
                _tails = {k: v for k, v in tails.items() if k[1] == q}
                entry_rows = _evaluate_group(
                    s1s=_s1s, s2s=_s2s, bulks=_bulks, tails=_tails,
                    decouple_covs=decouple_covs,
                    **_eval_kwargs,
                )
                if entry_rows:
                    atomic_write_parquet(
                        pd.DataFrame(entry_rows), entry_path, registry=registry,
                    )
                all_rows.extend(entry_rows)

        rows = all_rows
        out_path = bundle_partial_path
        write_dir = partials_dir

    elif task_file is not None:
        with open(task_file) as f:
            task_doc = json.load(f)
        tasks = task_doc["tasks"]
        if not (0 <= task_index < len(tasks)):
            raise click.UsageError(
                f"--task-index {task_index} out of range "
                f"(task file has {len(tasks)} tasks)."
            )
        task = tasks[task_index]
        task_threshold = float(task["threshold_quantile"])
        s1_filter   = set(task["s1_spec_ids"])
        s2_filter   = set(task["s2_spec_ids"])
        bulk_filter = set(task["bulk_spec_ids"])
        tail_filter = set(task["tail_spec_ids"])

        _s1s = {ck: [s for s in specs if component_id(s) in s1_filter]
               for ck, specs in s1s.items()}
        _s1s = {ck: v for ck, v in _s1s.items() if v}
        _s2s = {k: [s for s in specs if component_id(s) in s2_filter]
               for k, specs in s2s.items() if k[1] == task_threshold}
        _s2s = {k: v for k, v in _s2s.items() if v}
        _bulks = {k: [s for s in specs if component_id(s) in bulk_filter]
                 for k, specs in bulks.items() if k[1] == task_threshold}
        _bulks = {k: v for k, v in _bulks.items() if v}
        _tails = {k: [s for s in specs if component_id(s) in tail_filter]
                 for k, specs in tails.items() if k[1] == task_threshold}
        _tails = {k: v for k, v in _tails.items() if v}

        rows = _evaluate_group(
            s1s=_s1s, s2s=_s2s, bulks=_bulks, tails=_tails,
            decouple_covs=True,
            **_eval_kwargs,
        )
        partials_dir = output_path / "partials"
        partials_dir.mkdir(parents=True, exist_ok=True)
        out_path = partials_dir / f"dh_task_{task_index:05d}.parquet"
        write_dir = partials_dir

    elif cells_file is not None:
        with open(cells_file) as f:
            cells_doc = json.load(f)
        tasks = cells_doc["tasks"]
        if not (0 <= task_index < len(tasks)):
            raise click.UsageError(
                f"--task-index {task_index} out of range "
                f"(cells file has {len(tasks)} tasks)."
            )
        cell_list = inflate_cells(tasks[task_index]["task_args"])
        logger.info(
            "Cells mode: task %d, %d explicit DH-config cells.",
            task_index, len(cell_list),
        )

        all_rows: list[dict] = []
        for cell in cell_list:
            s1_spec   = available[cell["s1_spec_id"]]
            s2_spec   = available[cell["s2_spec_id"]]
            bulk_spec = available[cell["bulk_spec_id"]]
            tail_spec = available[cell["tail_spec_id"]]
            q = float(s2_spec["threshold_quantile"])
            # Singleton per-stage groups + decouple_covs=True → _evaluate_group scores
            # exactly this one config (s1/s2 covs may differ; bulk/tail covs are
            # subsets). Shared caches across cells mean each component is fit once
            # per worker, so packing configs that share components amortises the fit.
            _s1s   = {_combo_key(s1_spec["covariate_combo"]): [s1_spec]}
            _s2s   = {(_combo_key(s2_spec["covariate_combo"]), q): [s2_spec]}
            _bulks = {(_combo_key(bulk_spec["covariate_combo"]), q): [bulk_spec]}
            _tails = {(_combo_key(tail_spec["covariate_combo"]), q): [tail_spec]}
            all_rows.extend(_evaluate_group(
                s1s=_s1s, s2s=_s2s, bulks=_bulks, tails=_tails,
                decouple_covs=True,
                **_eval_kwargs,
            ))

        rows = all_rows
        partials_dir = output_path / "partials"
        partials_dir.mkdir(parents=True, exist_ok=True)
        out_path = partials_dir / f"dh_task_{task_index:05d}.parquet"
        write_dir = partials_dir

    else:
        _s1s = dict(s1s)
        _s2s = dict(s2s)
        _bulks = dict(bulks)
        _tails = dict(tails)
        if s1_spec_id is not None:
            _s1s = {ck: [s for s in specs if component_id(s) == s1_spec_id]
                   for ck, specs in _s1s.items()}
            _s1s = {ck: v for ck, v in _s1s.items() if v}
        if threshold_quantile is not None:
            _s2s   = {k: v for k, v in _s2s.items()   if k[1] == threshold_quantile}
            _bulks = {k: v for k, v in _bulks.items() if k[1] == threshold_quantile}
            _tails = {k: v for k, v in _tails.items() if k[1] == threshold_quantile}

        rows = _evaluate_group(
            s1s=_s1s, s2s=_s2s, bulks=_bulks, tails=_tails,
            decouple_covs=decouple_covs,
            **_eval_kwargs,
        )
        if s1_spec_id is not None and threshold_quantile is not None:
            q_str = f"{threshold_quantile:.4f}".replace(".", "_")
            partials_dir = output_path / "partials"
            partials_dir.mkdir(parents=True, exist_ok=True)
            out_path = partials_dir / f"dh_{s1_spec_id}_{q_str}.parquet"
            write_dir = partials_dir
        else:
            out_path = output_path / "dh_results.parquet"
            write_dir = output_path

    logger.info("Computed metrics for %d combinations.", len(rows))

    if not rows:
        logger.warning("No valid combinations found. Output file not written.")
        return

    results_df = pd.DataFrame(rows)

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
