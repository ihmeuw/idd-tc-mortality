"""
Refit a chosen DH model with all stage components, retaining the underlying
statsmodels GLMResults (or per-family custom optimizer result) so SEs,
t-values, p-values, and covariance are available — none of which are stored
in the persisted ``FitResult`` dataclass.

The production fit path (``fit_one_component``) keeps only point estimates
and discards the statsmodels result; this module is a side-channel that
re-runs the same fit code with ``statsmodels.GLM.fit`` monkey-patched so the
full ``GLMResults`` object is captured before it goes out of scope.

Use case
--------
You have a chosen winner model (e.g. ``topsis_df.iloc[0].to_dict()``) and
want to inspect coefficient SEs / p-values for each stage at IS and across
all OOS folds. This refits everything in memory and returns a nested dict
of FitResult + raw model object + per-stage metrics.

Cost: ~ (4 stages × (1 IS + n_seeds × n_folds OOS)) component fits per call.
At the project default (5 × 5) that's 104 fits per chosen model.

Limitations
-----------
- Only families that go through ``sm.GLM`` produce a non-None ``raw_object``.
  Families with custom optimisers (``gpd``, anything else using scipy) leave
  ``raw_object`` as None — point estimates are still in ``fit_result.params``
  and ``fit_result.meta`` may carry optimiser-specific diagnostics.
- The monkey-patch captures every ``sm.GLM.fit()`` call inside ``fit_one_component``.
  We return the last one, which is the canonical stage fit for all known
  families. If a future family adds auxiliary GLM calls, revisit.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize as _scipy_optimize
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.othermod.betareg import BetaModel
from statsmodels.regression.linear_model import WLS

from idd_tc_mortality.evaluate.assemble import assemble_predictions
from idd_tc_mortality.evaluate.predict_component import predict_one_component
from idd_tc_mortality.fit.fit_component import fit_one_component
from idd_tc_mortality.metrics import (
    calc_continuous_metrics,
    calc_full_model_metrics,
    calc_s1_metrics,
    calc_s2_metrics,
)


STAGE_NAMES = ("s1", "s2", "bulk", "tail")


# ---------------------------------------------------------------------------
# Fit result capture across every fitter the family modules use
# ---------------------------------------------------------------------------
#
# The families don't all go through sm.GLM. Survey across the codebase:
#
#   sm.GLM                       -> s1, s2, gamma, poisson, binomial_cloglog
#   sm.WLS                       -> lognormal, scaled_logit
#   BetaModel (othermod.betareg) -> beta
#   NegativeBinomial (discrete)  -> nb
#   scipy.optimize.minimize      -> gpd, weibull, log_logistic, truncated_normal
#
# Patching only sm.GLM (the original version of this helper) missed the WLS
# / BetaModel / NegativeBinomial / scipy paths and left their raw_object as
# None even though full inference objects exist. The block below patches
# every fitter's entry point so the captured list ends up with the actual
# result object regardless of family.

_STATSMODELS_CLASSES: list[type] = [GLM, WLS, BetaModel, NegativeBinomial]


@contextmanager
def _capture_fit_results():
    """Capture every fit-result object produced inside the block.

    Yields a list. Each entry is whatever ``.fit()`` (for statsmodels classes)
    or ``scipy.optimize.minimize`` returned, in call order. List is empty if
    the wrapped code path doesn't touch any of the patched entry points.
    """
    captured: list[Any] = []
    originals: dict[tuple[type, str], Any] = {}

    def make_patched(orig):
        def patched(self, *args, **kwargs):
            result = orig(self, *args, **kwargs)
            captured.append(result)
            return result
        return patched

    for cls in _STATSMODELS_CLASSES:
        original = cls.fit
        originals[(cls, "fit")] = original
        cls.fit = make_patched(original)

    # The scipy-based families use `from scipy import optimize` and call
    # `optimize.minimize(...)`, so patching the module attribute is enough.
    original_minimize = _scipy_optimize.minimize

    def patched_minimize(*args, **kwargs):
        result = original_minimize(*args, **kwargs)
        captured.append(result)
        return result

    _scipy_optimize.minimize = patched_minimize

    try:
        yield captured
    finally:
        for (cls, method_name), orig in originals.items():
            setattr(cls, method_name, orig)
        _scipy_optimize.minimize = original_minimize


# ---------------------------------------------------------------------------
# Spec reconstruction from a focus_model dict
# ---------------------------------------------------------------------------

def _reconstruct_spec(focus_model: dict, stage: str, fold_tag: str = "is") -> dict:
    """Build a component spec dict from a focus_model dict, stage, and fold_tag.

    Accepts focus_model in the shape produced by ``df_oos.iloc[0].to_dict()``
    or ``topsis_df.iloc[0].to_dict()`` — i.e. CONFIG_COLS values as plain
    Python scalars / JSON strings.
    """
    cov = focus_model[f"{stage}_cov"]
    if isinstance(cov, str):
        cov = json.loads(cov)

    spec: dict[str, Any] = {
        "component":          stage,
        "covariate_combo":    cov,
        "threshold_quantile": (
            None if stage == "s1"
            else float(focus_model["threshold_quantile"])
        ),
        "threshold_rate":     None,  # fit_one_component will compute it from data
        "family":             focus_model.get(f"{stage}_family"),
        "exposure_mode":      focus_model.get(f"{stage}_exposure_mode"),
        "fold_tag":           fold_tag,
    }
    return spec


def _fit_stage_with_raw(spec: dict, df: pd.DataFrame):
    """Fit one stage and return (FitResult, raw_model_object_or_None).

    raw_object is the last fit-result captured during the fit_one_component
    call — typically a statsmodels results object (GLMResults, WLSResults,
    BetaModelResults, NegativeBinomialResults) or a scipy OptimizeResult.
    None if no patched fitter was invoked (shouldn't happen for any known
    family; documented as a backstop).
    """
    with _capture_fit_results() as captured:
        fit_result = fit_one_component(spec, df)
    raw_object = captured[-1] if captured else None
    return fit_result, raw_object


def _is_sentinel(fit_result) -> bool:
    """True if fit_one_component returned its hard-failure sentinel.

    fit_one_component wraps fit dispatch in try/except and on exception
    returns ``FitResult(params=[nan], param_names=['__failed__'], ...)`` so
    the pipeline can keep going. Predict on such a result will crash, so
    callers that build per-fold tables need to skip these.

    ``converged=False`` is *not* a sentinel — it can mean the optimizer hit
    its iteration limit but a usable result was still returned. Those fits
    still have valid ``params`` / ``param_names``; we let them through and
    surface the ``converged`` flag to the caller.
    """
    return fit_result.param_names == ["__failed__"]


# ---------------------------------------------------------------------------
# Per-stage metric computation on a chosen evaluation subset
# ---------------------------------------------------------------------------

def _stage_metrics(
    stage: str,
    spec: dict,
    fit_result,
    pred_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    threshold_rate: float | None,
) -> dict:
    """Compute the per-stage metric dict on ``eval_df``.

    ``pred_df`` is the DataFrame used to call ``predict_one_component`` (full
    df for both IS and OOS so the predictor's index matches). ``eval_df`` is
    the subset on which metrics are scored (full df for IS; holdout subset
    for OOS).
    """
    pred = predict_one_component(spec, fit_result, pred_df)
    pred_eval = pred.loc[eval_df.index].values

    deaths    = eval_df["deaths"].values
    exposed   = eval_df["exposed"].values
    rate      = deaths / exposed
    any_death = (deaths >= 1).astype(int)

    if stage == "s1":
        return calc_s1_metrics(any_death.astype(float), pred_eval)

    if stage == "s2":
        ad_mask = any_death.astype(bool)
        if threshold_rate is None or ad_mask.sum() == 0:
            return {}
        y_s2 = (rate[ad_mask] >= threshold_rate).astype(float)
        p_s2 = pred_eval[ad_mask]
        return calc_s2_metrics(y_s2, p_s2)

    if stage == "bulk":
        if threshold_rate is None:
            return {}
        mask = (any_death.astype(bool)) & (rate < threshold_rate)
        if mask.sum() == 0:
            return {}
        return calc_continuous_metrics(rate[mask], pred_eval[mask], exposed[mask])

    if stage == "tail":
        if threshold_rate is None:
            return {}
        mask = (any_death.astype(bool)) & (rate >= threshold_rate)
        if mask.sum() == 0:
            return {}
        return calc_continuous_metrics(rate[mask], pred_eval[mask], exposed[mask])

    raise ValueError(f"unknown stage: {stage!r}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def refit_model_with_objects(
    focus_model: dict,
    data: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    n_seeds: int = 5,
    n_folds: int = 5,
) -> dict:
    """Refit the four stages + combined DH model for one chosen config.

    Parameters
    ----------
    focus_model:
        Dict with CONFIG_COLS keys (e.g. ``topsis_df.iloc[0].to_dict()``).
        Must contain ``threshold_quantile`` plus
        ``{s1,s2,bulk,tail}_{family,exposure_mode,cov}``.
    data:
        Full training DataFrame (the ``input.parquet`` for this run).
    fold_assignments:
        DataFrame of fold assignments (``seed_0`` .. ``seed_{n_seeds-1}``
        columns, integer fold index per row). Index must align with ``data``.
    n_seeds, n_folds:
        Match the values used by the orchestrator at fit time. Defaults (5, 5)
        align with the project default.

    Returns
    -------
    dict with shape::

        {
          "is":  {
              "s1":   {"spec", "fit_result", "raw_object", "metrics"},
              "s2":   ...,
              "bulk": ...,
              "tail": ...,
              "combined": {"predictions", "metrics", "threshold_rate"},
          },
          "oos": {
              "s0_f0": {<same keys as IS>},
              ...
              "s{n_seeds-1}_f{n_folds-1}": {...},
          },
        }

    ``raw_object`` is a statsmodels ``GLMResults`` for families that fit via
    ``sm.GLM``, or ``None`` for custom-optimiser families.

    Notes
    -----
    Cost: ``4 * (1 + n_seeds * n_folds)`` component fits per call (104 at the
    project default). Wall time scales with the slowest family in the four
    stages — usually tail-side iterative fits dominate.
    """
    # Compute the IS threshold_rate once from the full data; OOS uses the
    # same threshold rate so per-fold metrics are commensurable.
    threshold_quantile = float(focus_model["threshold_quantile"])
    death_rate = data["deaths"].values / data["exposed"].values
    pos = death_rate[death_rate > 0]
    threshold_rate = float(np.quantile(pos, threshold_quantile)) if len(pos) else None

    out: dict[str, dict] = {"is": {}, "oos": {}}

    # ----- IS -----
    is_specs:   dict[str, dict] = {}
    is_fits:    dict[str, Any]  = {}
    is_failures: list[str]      = []
    for stage in STAGE_NAMES:
        spec = _reconstruct_spec(focus_model, stage, fold_tag="is")
        spec["threshold_rate"] = threshold_rate if stage != "s1" else None
        fit_result, raw_object = _fit_stage_with_raw(spec, data)
        failed = _is_sentinel(fit_result)
        if failed:
            is_failures.append(stage)
            metrics = {"__failed__": fit_result.meta.get("fit_error", "non-convergence")}
        else:
            metrics = _stage_metrics(stage, spec, fit_result, data, data, threshold_rate)
        is_specs[stage] = spec
        is_fits[stage]  = fit_result
        out["is"][stage] = {
            "spec":       spec,
            "fit_result": fit_result,
            "raw_object": raw_object,
            "metrics":    metrics,
            "failed":     failed,
        }

    # IS combined — only if every stage fit succeeded.
    if is_failures:
        out["is"]["combined"] = {
            "predictions":    None,
            "metrics":        {"__failed__": f"upstream stage failures: {is_failures}"},
            "threshold_rate": threshold_rate,
        }
    else:
        combined_pred = assemble_predictions(
            s1_result=is_fits["s1"],     s1_spec=is_specs["s1"],
            s2_result=is_fits["s2"],     s2_spec=is_specs["s2"],
            bulk_result=is_fits["bulk"], bulk_spec=is_specs["bulk"],
            tail_result=is_fits["tail"], tail_spec=is_specs["tail"],
            df=data,
        )
        combined_metrics = calc_full_model_metrics(
            y_true_rate=death_rate,
            y_pred_rate=combined_pred.values,
            exposed=data["exposed"].values,
            any_death=(data["deaths"].values >= 1).astype(int),
        )
        out["is"]["combined"] = {
            "predictions":    combined_pred,
            "metrics":        combined_metrics,
            "threshold_rate": threshold_rate,
        }

    # ----- OOS over all (seed, fold) -----
    for seed in range(n_seeds):
        seed_col = f"seed_{seed}"
        if seed_col not in fold_assignments.columns:
            raise ValueError(
                f"fold_assignments is missing {seed_col!r}; "
                f"expected columns seed_0..seed_{n_seeds - 1}"
            )
        for fold in range(n_folds):
            fold_tag     = f"s{seed}_f{fold}"
            train_mask   = (fold_assignments[seed_col] != fold).values
            holdout_mask = ~train_mask
            train_df   = data.loc[train_mask].copy()
            holdout_df = data.loc[holdout_mask].copy()

            stage_entries: dict[str, dict] = {}
            oos_specs:  dict[str, dict] = {}
            oos_fits:   dict[str, Any]  = {}
            oos_failures: list[str]     = []
            for stage in STAGE_NAMES:
                spec = _reconstruct_spec(focus_model, stage, fold_tag=fold_tag)
                spec["threshold_rate"] = threshold_rate if stage != "s1" else None
                fit_result, raw_object = _fit_stage_with_raw(spec, train_df)
                failed = _is_sentinel(fit_result)
                if failed:
                    oos_failures.append(stage)
                    metrics = {
                        "__failed__": fit_result.meta.get("fit_error", "non-convergence")
                    }
                else:
                    metrics = _stage_metrics(
                        stage, spec, fit_result,
                        pred_df=data,         # predict on full df so index aligns
                        eval_df=holdout_df,   # score on the held-out rows
                        threshold_rate=threshold_rate,
                    )
                stage_entries[stage] = {
                    "spec":       spec,
                    "fit_result": fit_result,
                    "raw_object": raw_object,
                    "metrics":    metrics,
                    "failed":     failed,
                }
                oos_specs[stage] = spec
                oos_fits[stage]  = fit_result

            # OOS combined: only if every stage fit succeeded on this fold.
            if oos_failures:
                stage_entries["combined"] = {
                    "predictions":    None,
                    "metrics":        {"__failed__": f"upstream stage failures: {oos_failures}"},
                    "threshold_rate": threshold_rate,
                }
            else:
                combined_pred_full = assemble_predictions(
                    s1_result=oos_fits["s1"],     s1_spec=oos_specs["s1"],
                    s2_result=oos_fits["s2"],     s2_spec=oos_specs["s2"],
                    bulk_result=oos_fits["bulk"], bulk_spec=oos_specs["bulk"],
                    tail_result=oos_fits["tail"], tail_spec=oos_specs["tail"],
                    df=data,
                )
                combined_pred_holdout = combined_pred_full.loc[holdout_df.index]
                holdout_rate = (
                    holdout_df["deaths"].values / holdout_df["exposed"].values
                )
                holdout_any_death = (holdout_df["deaths"].values >= 1).astype(int)
                combined_metrics = calc_full_model_metrics(
                    y_true_rate=holdout_rate,
                    y_pred_rate=combined_pred_holdout.values,
                    exposed=holdout_df["exposed"].values,
                    any_death=holdout_any_death,
                )
                stage_entries["combined"] = {
                    "predictions":    combined_pred_holdout,
                    "metrics":        combined_metrics,
                    "threshold_rate": threshold_rate,
                }
            out["oos"][fold_tag] = stage_entries

    return out
