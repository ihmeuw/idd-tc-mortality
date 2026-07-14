"""
stage_plots.py
Vetting plots for Double Hurdle TC mortality model stages.

Usage:
    from idd_tc_mortality.viz.stage_plots import StagePlotter, DEFAULT_VET_CONFIG

    plotter = StagePlotter(data=df, results_dir=results_dir, output_dir=output_dir)

    # Default 3x3 layout (preserved current behavior)
    fig = plotter.vet_stage(model_row, stage='s1')

    # Custom layout via config dict (shallow-merged with DEFAULT_VET_CONFIG;
    # layout.panels is REPLACED when provided, not merged):
    fig = plotter.vet_stage(model_row, stage='s1', config={
        'layout': {'figsize': (10, 12), 'panels': {...}},
        'style':  {'fonts': {'suptitle': 18}},
        'legend': {'mode': 'shared_below', 'ncol': 6},
    })

    # All 4 stages at once
    figs = plotter.vet_model(model_row)

    # Prediction DataFrames
    df_is  = plotter.predict_df(model_row)
    df_oos = plotter.predict_df_oos(model_row, seed=0)

    # Rolling-percentile P(any deaths) curve (module-level, no StagePlotter needed)
    fig, ax = plt.subplots()
    plot_rolling_pct(df, variable='wind_speed', ax=ax, half_window=5)
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from idd_tc_mortality.cache import component_id, load_result, model_id
from idd_tc_mortality.combine import assemble_dh_prediction
from idd_tc_mortality.distributions import get_family
from idd_tc_mortality.distributions.base import FitResult
from idd_tc_mortality.features import align_X, build_X
from idd_tc_mortality.thresholds import compute_thresholds
from idd_tc_mortality import s1 as s1_mod
from idd_tc_mortality import s2 as s2_mod


_LINE_STYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]


DEFAULT_VET_CONFIG: dict = {
    'layout': {
        'figsize':      (16, 12),
        'suptitle':     None,           # None = auto from stage/family/n
        'tight_layout': True,
        'panels': {
            'wind_overall': {'kind': 'cont_overall',   'x_col': 'wind_speed',
                             'row': 0, 'col': 0,
                             'x_label': 'Wind speed', 'y_label': None,
                             'title': 'Wind speed — overall'},
            'wind_basin':   {'kind': 'cont_by_basin',  'x_col': 'wind_speed',
                             'row': 0, 'col': 1,
                             'x_label': 'Wind speed', 'y_label': None,
                             'title': 'Wind speed — by basin'},
            'wind_island':  {'kind': 'cont_by_island', 'x_col': 'wind_speed',
                             'row': 0, 'col': 2,
                             'x_label': 'Wind speed', 'y_label': None,
                             'title': 'Wind speed — by island'},
            'sdi_overall':  {'kind': 'cont_overall',   'x_col': 'sdi',
                             'row': 1, 'col': 0,
                             'x_label': 'SDI', 'y_label': None,
                             'title': 'SDI — overall'},
            'sdi_basin':    {'kind': 'cont_by_basin',  'x_col': 'sdi',
                             'row': 1, 'col': 1,
                             'x_label': 'SDI', 'y_label': None,
                             'title': 'SDI — by basin'},
            'sdi_island':   {'kind': 'cont_by_island', 'x_col': 'sdi',
                             'row': 1, 'col': 2,
                             'x_label': 'SDI', 'y_label': None,
                             'title': 'SDI — by island'},
            'basin_swarm':  {'kind': 'cat_beeswarm',   'cat_col': 'basin',
                             'row': 2, 'col': 0,
                             'x_label': 'basin', 'y_label': None,
                             'title': 'basin — beeswarm'},
            'island_swarm': {'kind': 'cat_beeswarm',   'cat_col': 'is_island',
                             'row': 2, 'col': 1,
                             'x_label': 'is_island', 'y_label': None,
                             'title': 'is_island — beeswarm'},
        },
    },
    'style': {
        'pred_line':     {'color': 'steelblue', 'lw': 2,   'ls': '-'},
        'scatter_obs':   {'color': 'lightgray', 's': 10,   'alpha': 0.2, 'zorder': 1},
        'island': {
            0: {'color': 'steelblue', 'label': 'Non-island', 'lw': 2, 'ls': '-'},
            1: {'color': 'coral',     'label': 'Island',     'lw': 2, 'ls': '-'},
        },
        'basin':         {},   # empty = use tab10 + cycling _LINE_STYLES
        'beeswarm_mean': {'color': 'steelblue', 'lw': 3},
        'fonts': {
            'suptitle':    12,
            'panel_title': None,    # None = matplotlib default
            'axis_label':  None,
            'tick':        None,
            'legend':      7,
        },
    },
    'legend': {
        'mode': 'per_panel',        # 'per_panel' | 'shared_below' | 'shared_right' | 'none'
        'ncol': None,               # for shared modes; None = one row (below) or one col (right)
        'bbox_to_anchor': None,     # fine-position the legend in figure coords
        'pad': None,                # fraction of fig reserved for legend zone
                                    # (None = default 0.15 below / 0.18 right)
    },
    'spacing': {
        # Plain-English names: h = horizontal gap between columns,
        # v = vertical gap between rows. (matplotlib's own subplots_adjust
        # uses 'wspace'/'hspace' which we translate to internally.)
        # Units: fraction of average subplot dimension. None = matplotlib
        # default (~0.2). Setting either disables tight_layout so values
        # aren't overwritten.
        'hspace':       None,
        'vspace':       None,
        # Axis-label distance from the axis, in points (matplotlib labelpad).
        # None = matplotlib default (~4 pt).
        'x_labelpad':   None,
        'y_labelpad':   None,
        # Vertical gap between the panels and the figure-level suptitle,
        # as a fraction of figure height. None = matplotlib default
        # (~0.05). Bigger = more breathing room above the panels.
        'suptitle_pad': None,
    },
    'axes': {
        # 'auto'  = renderer's default (log for rate stages, linear for
        #            logistic). 'linear' / 'log' force the y-scale.
        'y_scale_mode':        'auto',
        # When False, suppress scientific notation on both axes
        # (ScalarFormatter without scientific for log axes; ticklabel_format
        # 'plain' for linear).
        'scientific_notation': True,
        # True / False  : force all rendered panels to share a common
        #                 y-range (computed as min/max across them).
        # list of IDs   : share only those panel IDs (others left alone).
        # Applied AFTER include_zero_y.
        'share_y':             False,
        # When True, expand each panel's y-range to include 0. Silently
        # skipped on log-scaled axes (0 has no place on a log axis).
        'include_zero_y':      False,
    },
}


def _deep_merge(base: dict, override: Optional[dict]) -> dict:
    """Deep-merge override into base; override values win.

    Special case: under any 'panels' key, the override REPLACES the base
    panels dict entirely (not merged). This matches the user intent of
    declaring exactly which panels to render — partial merging would
    silently retain default panels alongside the user's, which is rarely
    desired.

    Implemented iteratively (explicit stack) rather than recursively so
    that `%autoreload 2` cannot strand the function's self-reference.
    """
    if override is None:
        return deepcopy(base)
    out: dict = {}
    stack: list[tuple[dict, dict, dict]] = [(base, override, out)]
    while stack:
        b, o, dest = stack.pop()
        for k in set(b) | set(o):
            if k in b and k in o:
                bv, ov = b[k], o[k]
                if isinstance(bv, dict) and isinstance(ov, dict):
                    if k == 'panels':
                        dest[k] = deepcopy(ov)
                    else:
                        dest[k] = {}
                        stack.append((bv, ov, dest[k]))
                else:
                    dest[k] = deepcopy(ov)
            elif k in o:
                dest[k] = deepcopy(o[k])
            else:
                dest[k] = deepcopy(b[k])
    return out


def _normalize_scatter_kws(scatter_style: Optional[dict]) -> dict:
    """Return a matplotlib-compatible scatter kw dict.

    Accepts 'color' (preferred in config) and normalizes to 'c' (matplotlib).
    Strips the internal 'show' flag so it never reaches matplotlib.
    Caller's overrides win over the defaults baked here.
    """
    kws: dict = {"alpha": 0.2, "s": 10, "c": "lightgray", "zorder": 1}
    if scatter_style:
        ss = dict(scatter_style)
        ss.pop('show', None)
        if 'color' in ss:
            ss['c'] = ss.pop('color')
        kws.update(ss)
    return kws


def _scatter_obs_is_visible(scatter_style: Optional[dict]) -> bool:
    """Internal: honor a `show: False` flag in style.scatter_obs."""
    if not scatter_style:
        return True
    return scatter_style.get('show', True) is not False


def _stage_draw_to_fit_result(stage_draw) -> FitResult:
    """Wrap a StageDraw as a FitResult so the existing `_predict` path works.

    The StageDraw carries one β-sample per draw plus an optional `scale`
    (the shape parameter for log_logistic / weibull / etc., or dispersion
    for gaussian-family stages). Where present, scale is stashed in
    `meta['shape_param']` because tail-family `predict` functions read it
    from there.
    """
    # Each family's `predict` reads its hyperparameters from result.meta:
    #   scaled_logit / truncated_normal  -> 'threshold_rate'
    #   gpd / weibull / log_logistic     -> 'shape_param'
    #   lognormal / truncated_normal     -> 'sigma'
    # We populate them all from the StageDraw so the shim works for any
    # family without family-specific code here.
    meta: dict = {}
    if getattr(stage_draw, 'scale', None) is not None:
        # tail families read shape_param; gaussian-family stages read sigma
        meta['shape_param'] = stage_draw.scale
        meta['sigma']       = stage_draw.scale
    if getattr(stage_draw, 'threshold_rate', None) is not None:
        meta['threshold_rate'] = stage_draw.threshold_rate
    # truncated_normal also needs truncation_side; pull from meta if the
    # StageDraw stashed it (in our codebase it doesn't, so this is just a
    # forward-compat hook).
    if hasattr(stage_draw, 'truncation_side'):
        meta['truncation_side'] = stage_draw.truncation_side
    return FitResult(
        params=np.asarray(stage_draw.params),
        param_names=list(stage_draw.param_names),
        fitted_values=np.empty(0),     # not used by predict paths
        family=stage_draw.family,
        converged=True,
        meta=meta,
    )


def plot_rolling_pct(
    data: pd.DataFrame,
    variable: str,
    ax: plt.Axes,
    half_window: int = 5,
    outcome: str = 'deaths',
    line_kws: Optional[dict] = None,
) -> None:
    """P(any deaths) as a rolling-percentile curve.

    For each integer percentile p in [1, 99], collects all rows whose
    `variable` percentile rank falls within [p − half_window, p + half_window]
    and plots the fraction with deaths > 0 against the variable value at p.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain `variable` and 'deaths' columns.
    variable : str
        Column name for the x-axis variable (e.g., 'wind_speed', 'sdi').
    ax : plt.Axes
        Axes to draw on.
    half_window : int, default 5
        Half-width of the rolling percentile window.
    line_kws : dict, optional
        Forwarded to ax.plot for the curve line.
    """
    vals = data[variable].values.astype(float)
    if outcome == 'deaths':
        y_val =  (data['deaths'].values > 0).astype(float)
    pct_ranks = pd.Series(vals).rank(pct=True).values * 100

    percentiles = np.arange(1, 100)
    x_at_pct = np.percentile(vals, percentiles)
    p_deaths = np.empty(len(percentiles))

    for i, p in enumerate(percentiles):
        mask = (pct_ranks >= p - half_window) & (pct_ranks <= p + half_window)
        p_deaths[i] = y_val[mask].mean() if mask.any() else np.nan

    lkws: dict = {'color': 'steelblue', 'lw': 2}
    if line_kws:
        lkws.update(line_kws)
    ax.plot(x_at_pct, p_deaths, **lkws)
    ax.set_xlabel(variable)
    ax.set_ylabel('P(any deaths)')
    ax.set_ylim(-0.05, 1.05)


class StagePlotter:
    """Diagnostic vetting plots for DH model stages."""

    def __init__(
        self,
        data: pd.DataFrame,
        results_dir: str | Path,
        output_dir: Optional[str | Path] = None,
    ):
        """
        Parameters
        ----------
        data:
            Full training DataFrame. Must contain 'deaths', 'exposed',
            'wind_speed', 'sdi', 'basin', 'is_island'.
        results_dir:
            Directory containing fitted .pkl component result files
            (named by component_id hash).
        output_dir:
            Directory where model_predictions/ parquets live (output of
            run-evaluate). Required only for predict_df_oos().
        """
        self._data = data
        self._results_dir = Path(results_dir)
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._result_cache: dict[str, FitResult] = {}

    # ------------------------------------------------------------------
    # Internal: spec reconstruction and result loading
    # ------------------------------------------------------------------

    def _compute_threshold_rate(self, quantile: float) -> float:
        death_rate = self._data["deaths"].values / self._data["exposed"].values
        return float(
            compute_thresholds(death_rate, np.array([quantile]))[quantile]
        )

    def _reconstruct_is_spec(self, model_row: pd.Series, stage: str) -> dict:
        q = float(model_row["threshold_quantile"]) if stage != "s1" else None
        threshold_rate = self._compute_threshold_rate(q) if q is not None else None

        covariate_combo = json.loads(model_row[f"{stage}_cov"])
        family: str | None = None
        if stage == "s1":
            family = str(model_row.get("s1_family", "cloglog"))
        elif stage == "s2":
            family = str(model_row["s2_family"])
        elif stage == "bulk":
            family = str(model_row["bulk_family"])
        elif stage == "tail":
            family = str(model_row["tail_family"])

        if stage == "s1":
            exposure_mode = str(model_row.get("s1_exposure_mode", "offset"))
        elif stage == "s2":
            exposure_mode = str(model_row.get("s2_exposure_mode", "free"))
        else:
            exposure_mode = str(model_row.get(f"{stage}_exposure_mode", "free+weight"))

        return {
            "component":          stage,
            "covariate_combo":    covariate_combo,
            "threshold_quantile": q,
            "threshold_rate":     threshold_rate,
            "family":             family,
            "exposure_mode":      exposure_mode,
            "fold_tag":           "is",
        }

    def _load_result(self, spec: dict) -> FitResult:
        cid = component_id(spec)
        if cid not in self._result_cache:
            self._result_cache[cid] = load_result(cid, self._results_dir)
        return self._result_cache[cid]

    # ------------------------------------------------------------------
    # Internal: subset for stage and prediction helpers
    # ------------------------------------------------------------------

    def _get_subset_for_stage(
        self, stage: str, threshold_rate: float
    ) -> pd.DataFrame:
        """Return the data subset and y column for a given stage."""
        data = self._data.copy()
        death_rate = data["deaths"].values / data["exposed"].values

        if stage == "s1":
            data["y"] = (data["deaths"].values >= 1).astype(int)
        elif stage == "s2":
            data = data[data["deaths"].values >= 1].copy()
            dr = data["deaths"].values / data["exposed"].values
            data["y"] = (dr >= threshold_rate).astype(int)
        elif stage == "bulk":
            mask = (data["deaths"].values >= 1) & (death_rate < threshold_rate)
            data = data[mask].copy()
            data["y"] = data["deaths"].values / data["exposed"].values
        elif stage == "tail":
            mask = death_rate >= threshold_rate
            data = data[mask].copy()
            data["y"] = data["deaths"].values / data["exposed"].values
        else:
            raise ValueError(f"Unknown stage: {stage!r}")
        return data

    def _make_pred_df(
        self,
        data: pd.DataFrame,
        n: int,
        overrides: Optional[dict] = None,
    ) -> pd.DataFrame:
        ref = {
            "wind_speed": float(data["wind_speed"].median()),
            "sdi":        float(data["sdi"].median()),
            "basin":      data["basin"].mode().iloc[0],
            "is_island":  int(data["is_island"].mode().iloc[0]),
            "exposed":    float(data["exposed"].median()),
        }
        if overrides is not None:
            ref.update(overrides)

        cols: dict = {}
        for k, v in ref.items():
            if isinstance(v, (np.ndarray, pd.Series)):
                cols[k] = np.asarray(v)
            else:
                cols[k] = np.full(n, v)
        return pd.DataFrame(cols)

    def _predict(
        self,
        result: FitResult,
        pred_data: pd.DataFrame,
        spec: dict,
        threshold_rate: float = 0.0,
    ) -> np.ndarray:
        component = spec["component"]
        covariate_combo = spec["covariate_combo"]

        if component in ("s1", "s2"):
            include_log_exposed = (spec["exposure_mode"] == "free")
        else:
            include_log_exposed = not get_family(spec["family"])["log_exposed"]

        X = build_X(pred_data, covariate_combo, include_log_exposed=include_log_exposed)
        X = align_X(X, result.param_names)
        log_exposed = np.log(pred_data["exposed"].values)

        if component == "s1":
            preds = s1_mod.predict(result, X, log_exposed)
        elif component == "s2":
            preds = s2_mod.predict(result, X)
        else:
            family_info = get_family(spec["family"])
            pred_fn = family_info["predict"]
            if family_info["log_exposed"]:
                preds = pred_fn(result, X, log_exposed)
            else:
                preds = pred_fn(result, X)

            if spec["family"] == "beta":
                preds = preds * threshold_rate
            elif spec["family"] == "gpd":
                preds = preds + threshold_rate

        return np.asarray(preds)

    # ------------------------------------------------------------------
    # Observed data scatter (shared across plot types)
    # ------------------------------------------------------------------

    def _scatter_obs(
        self,
        ax: plt.Axes,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        is_logistic: bool,
        scatter_style: Optional[dict] = None,
        rng_seed: int = 42,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
    ) -> None:
        if _scatter_obs_is_visible(scatter_style):
            kws = _normalize_scatter_kws(scatter_style)
            rng = np.random.default_rng(rng_seed)
            if is_logistic:
                jitter = rng.uniform(-0.02, 0.02, len(y_vals)) * y_scale
                ax.scatter(x_vals * x_scale, y_vals * y_scale + jitter, **kws)
            else:
                ax.scatter(x_vals * x_scale,
                           np.maximum(y_vals.astype(float), 1e-9) * y_scale, **kws)
        if not is_logistic:
            ax.set_yscale("log")

    def _style_ax(
        self,
        ax: plt.Axes,
        x_label: str,
        is_logistic: bool,
        plot_rate: bool = True,
        y_label: Optional[str] = None,
        y_scale: float = 1.0,
    ) -> None:
        ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
            if is_logistic:
                ax.set_ylim(-0.05 * y_scale, 1.05 * y_scale)
        elif is_logistic:
            ax.set_ylabel("P(Y=1)")
            ax.set_ylim(-0.05 * y_scale, 1.05 * y_scale)
        else:
            ax.set_ylabel("Death rate (deaths / exposed)" if plot_rate else "Deaths")

    def _apply_panel_title(self, ax: plt.Axes, panel_spec: dict) -> None:
        title = panel_spec.get('title')
        if title:
            ax.set_title(title)

    # ------------------------------------------------------------------
    # Per-axis building blocks. Each returns a list of (handle, label)
    # tuples so vet_stage can collect them for a shared legend.
    # ------------------------------------------------------------------

    def _predict_grid_for_draws(
        self,
        stage_draws: list,
        pred_data: pd.DataFrame,
        base_spec: dict,
        threshold_rate: float,
        outcome_sample: bool = False,
        seed_base: int = 42,
    ) -> np.ndarray:
        """Predict pred_data under each StageDraw. Returns (n_draws, n_pred).

        With `outcome_sample=False` (default), each draw returns the
        analytical mean — band = pure parameter uncertainty around the
        predicted curve.

        With `outcome_sample=True`, bulk/tail draws return a single
        realization from their predictive distribution at each grid point
        (= matches DH-predict's `o=1` toggle for those stages). Band then
        also includes single-storm sampling variability. Falls back to the
        analytical mean for non-bulk/tail stages (S1, S2 are probabilities;
        no analytic-mean-vs-sample distinction at the stage level).
        """
        # Lazy import to avoid pulling the heavy uncertainty module on
        # every stage_plots import.
        if outcome_sample:
            from idd_tc_mortality.uncertainty.draw_models import (
                _bulk_draw, _tail_draw,
            )

        samples = []
        for i, sd in enumerate(stage_draws):
            if outcome_sample and sd.stage in ('bulk', 'tail'):
                include_log_exposed = not get_family(sd.family)["log_exposed"]
                X = build_X(pred_data, sd.covariate_combo,
                            include_log_exposed=include_log_exposed)
                X = align_X(X, list(sd.param_names))
                eta = np.asarray(X) @ np.asarray(sd.params)
                if not include_log_exposed:
                    # Family's predict adds log_exposed internally; mirror that.
                    eta = eta + np.log(pred_data["exposed"].values)
                rng = np.random.default_rng(seed_base + i)
                if sd.stage == 'bulk':
                    y = _bulk_draw(sd, eta, pred_data, rng)
                else:  # tail
                    y = _tail_draw(sd, eta, pred_data, threshold_rate, rng)
                samples.append(np.asarray(y))
            else:
                result = _stage_draw_to_fit_result(sd)
                draw_spec = {
                    **base_spec,
                    'family':          sd.family,
                    'exposure_mode':   sd.exposure_mode,
                    'covariate_combo': sd.covariate_combo,
                }
                samples.append(self._predict(result, pred_data, draw_spec, threshold_rate))
        return np.asarray(samples)

    def _plot_cont_overall(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        result: FitResult,
        spec: dict,
        threshold_rate: float,
        x_raw_col: str,
        is_logistic: bool,
        style: dict,
        panel_spec: dict,
        stage_draws: Optional[list] = None,
        n_grid: int = 100,
        plot_rate: bool = True,
    ) -> list[tuple]:
        x_scale = float(panel_spec.get('x_scale', 1.0))
        y_scale = float(panel_spec.get('y_scale', 1.0))
        x_vals = data[x_raw_col].values.astype(float)
        self._scatter_obs(ax, x_vals, data["y"].values, is_logistic,
                          scatter_style=style.get('scatter_obs'),
                          x_scale=x_scale, y_scale=y_scale)

        x_grid = np.linspace(x_vals.min(), x_vals.max(), n_grid)
        pred_data = self._make_pred_df(data, n_grid, {x_raw_col: x_grid})
        y_pred = self._predict(result, pred_data, spec, threshold_rate)

        line_kws = dict(style.get('pred_line', {'color': 'steelblue', 'lw': 2, 'ls': '-'}))
        if 'ls' in line_kws:
            line_kws['linestyle'] = line_kws.pop('ls')
        if 'lw' in line_kws:
            line_kws['linewidth'] = line_kws.pop('lw')

        unc = panel_spec.get('uncertainty') or {}
        if unc.get('show') and stage_draws:
            samples = self._predict_grid_for_draws(
                stage_draws, pred_data, spec, threshold_rate,
                outcome_sample=bool(unc.get('outcome_sample', False)))
            p_lo, p_hi = unc.get('percentiles', (5, 95))
            lo = np.percentile(samples, p_lo, axis=0) * y_scale
            hi = np.percentile(samples, p_hi, axis=0) * y_scale
            band_color = unc.get('color') or line_kws.get('color', 'steelblue')
            ax.fill_between(x_grid * x_scale, lo, hi,
                            color=band_color,
                            alpha=unc.get('alpha', 0.2),
                            zorder=1.5,
                            linewidth=0)

        ax.plot(x_grid * x_scale, y_pred * y_scale, zorder=2, **line_kws)

        self._style_ax(ax, panel_spec.get('x_label', x_raw_col), is_logistic,
                       plot_rate=plot_rate, y_label=panel_spec.get('y_label'),
                       y_scale=y_scale)
        self._apply_panel_title(ax, panel_spec)
        return []

    def _plot_cont_by_basin(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        result: FitResult,
        spec: dict,
        threshold_rate: float,
        x_raw_col: str,
        is_logistic: bool,
        style: dict,
        panel_spec: dict,
        stage_draws: Optional[list] = None,
        n_grid: int = 100,
        plot_rate: bool = True,
    ) -> list[tuple]:
        x_scale = float(panel_spec.get('x_scale', 1.0))
        y_scale = float(panel_spec.get('y_scale', 1.0))
        x_vals = data[x_raw_col].values.astype(float)
        self._scatter_obs(ax, x_vals, data["y"].values, is_logistic,
                          scatter_style=style.get('scatter_obs'),
                          x_scale=x_scale, y_scale=y_scale)

        x_grid = np.linspace(x_vals.min(), x_vals.max(), n_grid)
        basins = sorted(data["basin"].dropna().unique())
        basin_styles = style.get('basin', {}) or {}
        default_colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(basins), 1)))

        unc = panel_spec.get('uncertainty') or {}
        draw_band = bool(unc.get('show') and stage_draws)
        p_lo, p_hi = unc.get('percentiles', (5, 95))
        band_alpha = unc.get('alpha', 0.2)
        outcome_sample = bool(unc.get('outcome_sample', False))

        handles_labels: list[tuple] = []
        for i, basin in enumerate(basins):
            pred_data = self._make_pred_df(data, n_grid, {x_raw_col: x_grid, "basin": basin})
            y_pred = self._predict(result, pred_data, spec, threshold_rate)
            bs = basin_styles.get(basin, {})
            color = bs.get('color', default_colors[i])
            ls    = bs.get('ls', _LINE_STYLES[i % len(_LINE_STYLES)])
            lw    = bs.get('lw', 1.5)
            label = bs.get('label', basin)

            if draw_band:
                samples = self._predict_grid_for_draws(
                    stage_draws, pred_data, spec, threshold_rate,
                    outcome_sample=outcome_sample)
                lo = np.percentile(samples, p_lo, axis=0) * y_scale
                hi = np.percentile(samples, p_hi, axis=0) * y_scale
                ax.fill_between(x_grid * x_scale, lo, hi,
                                color=unc.get('color') or color,
                                alpha=band_alpha,
                                zorder=1.5,
                                linewidth=0)

            line, = ax.plot(x_grid * x_scale, y_pred * y_scale,
                            color=color, linestyle=ls,
                            linewidth=lw, label=label, zorder=2)
            handles_labels.append((line, label))

        self._style_ax(ax, panel_spec.get('x_label', x_raw_col), is_logistic,
                       plot_rate=plot_rate, y_label=panel_spec.get('y_label'),
                       y_scale=y_scale)
        self._apply_panel_title(ax, panel_spec)
        return handles_labels

    def _plot_cont_by_island(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        result: FitResult,
        spec: dict,
        threshold_rate: float,
        x_raw_col: str,
        is_logistic: bool,
        style: dict,
        panel_spec: dict,
        stage_draws: Optional[list] = None,
        n_grid: int = 100,
        plot_rate: bool = True,
    ) -> list[tuple]:
        x_scale = float(panel_spec.get('x_scale', 1.0))
        y_scale = float(panel_spec.get('y_scale', 1.0))
        x_vals = data[x_raw_col].values.astype(float)
        self._scatter_obs(ax, x_vals, data["y"].values, is_logistic,
                          scatter_style=style.get('scatter_obs'),
                          x_scale=x_scale, y_scale=y_scale)

        x_grid = np.linspace(x_vals.min(), x_vals.max(), n_grid)
        island_styles = style.get('island', {})

        unc = panel_spec.get('uncertainty') or {}
        draw_band = bool(unc.get('show') and stage_draws)
        p_lo, p_hi = unc.get('percentiles', (5, 95))
        band_alpha = unc.get('alpha', 0.2)
        outcome_sample = bool(unc.get('outcome_sample', False))

        handles_labels: list[tuple] = []
        for island_val in (0, 1):
            entry = island_styles.get(island_val, {})
            color = entry.get('color', 'steelblue' if island_val == 0 else 'coral')
            label = entry.get('label', 'Non-island' if island_val == 0 else 'Island')
            lw    = entry.get('lw', 2)
            ls    = entry.get('ls', '-')
            pred_data = self._make_pred_df(
                data, n_grid, {x_raw_col: x_grid, "is_island": island_val}
            )
            y_pred = self._predict(result, pred_data, spec, threshold_rate)

            if draw_band:
                samples = self._predict_grid_for_draws(
                    stage_draws, pred_data, spec, threshold_rate,
                    outcome_sample=outcome_sample)
                lo = np.percentile(samples, p_lo, axis=0) * y_scale
                hi = np.percentile(samples, p_hi, axis=0) * y_scale
                ax.fill_between(x_grid * x_scale, lo, hi,
                                color=unc.get('color') or color,
                                alpha=band_alpha,
                                zorder=1.5,
                                linewidth=0)

            line, = ax.plot(x_grid * x_scale, y_pred * y_scale,
                            color=color, linewidth=lw, linestyle=ls,
                            label=label, zorder=2)
            handles_labels.append((line, label))

        self._style_ax(ax, panel_spec.get('x_label', x_raw_col), is_logistic,
                       plot_rate=plot_rate, y_label=panel_spec.get('y_label'),
                       y_scale=y_scale)
        self._apply_panel_title(ax, panel_spec)
        return handles_labels

    def _plot_cat_beeswarm(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        result: FitResult,
        spec: dict,
        threshold_rate: float,
        cat_col: str,
        is_logistic: bool,
        style: dict,
        panel_spec: dict,
        plot_rate: bool = True,
    ) -> list[tuple]:
        # x is categorical here, so x_scale is intentionally ignored.
        y_scale = float(panel_spec.get('y_scale', 1.0))
        scatter_style = style.get('scatter_obs')
        kws = _normalize_scatter_kws(scatter_style)
        show_scatter = _scatter_obs_is_visible(scatter_style)

        categories = sorted(data[cat_col].dropna().unique())
        cat_to_x = {cat: i for i, cat in enumerate(categories)}
        rng = np.random.default_rng(42)

        x_base = np.array([cat_to_x[c] for c in data[cat_col]])
        x_jittered = x_base + rng.uniform(-0.3, 0.3, len(data))
        y_vals = data["y"].values.astype(float)

        if show_scatter:
            if is_logistic:
                jitter_y = rng.uniform(-0.02, 0.02, len(y_vals)) * y_scale
                ax.scatter(x_jittered, y_vals * y_scale + jitter_y, **kws)
            else:
                ax.scatter(x_jittered,
                           np.maximum(y_vals, 1e-9) * y_scale, **kws)
        if not is_logistic:
            ax.set_yscale("log")

        mean_style = style.get('beeswarm_mean', {'color': 'steelblue', 'lw': 3})
        for cat in categories:
            cat_subset = data[data[cat_col] == cat]
            mean_pred = self._predict(result, cat_subset, spec, threshold_rate).mean()
            x_pos = cat_to_x[cat]
            ax.hlines(mean_pred * y_scale, x_pos - 0.35, x_pos + 0.35,
                      colors=mean_style.get('color', 'steelblue'),
                      linewidth=mean_style.get('lw', 3),
                      zorder=2)

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([str(c) for c in categories], rotation=45, ha="right")
        ax.set_xlabel(panel_spec.get('x_label', cat_col))
        y_label = panel_spec.get('y_label')
        if y_label is not None:
            ax.set_ylabel(y_label)
            if is_logistic:
                ax.set_ylim(-0.05 * y_scale, 1.05 * y_scale)
        elif is_logistic:
            ax.set_ylabel("P(Y=1)")
            ax.set_ylim(-0.05 * y_scale, 1.05 * y_scale)
        else:
            ax.set_ylabel("Death rate (deaths / exposed)" if plot_rate else "Deaths")
        self._apply_panel_title(ax, panel_spec)
        return []

    # ------------------------------------------------------------------
    # PDF-overlay panel: 5x5 quintile cross of two continuous covariates
    # ------------------------------------------------------------------

    _DEFAULT_PDF_LINESTYLES = [(0, (3, 1, 1, 1)), ':', '-.', '--', '-']

    def _plot_cont_pdf_quintile(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        result: FitResult,
        spec: dict,
        threshold_rate: float,
        is_logistic: bool,
        style: dict,
        panel_spec: dict,
        stage_draws: Optional[list] = None,
        plot_rate: bool = True,
    ) -> list[tuple]:
        """5x5 quintile-cross overlay of the bulk PDF on the rate scale.

        Currently bulk/scaled_logit only — fits a Gaussian on the link-scale
        outcome and renders the back-transformed density on `rate`.

        Required panel_spec fields:
            color_col      — covariate whose quintiles drive line color
            style_col      — covariate whose quintiles drive linestyle
            fixed_col      — covariate held fixed at fixed_quantile
            fixed_quantile — quantile (in [0,1]) of `fixed_col` in `data`
                             at which to hold it. Default 0.5.

        Optional:
            quintiles      — five quantile probabilities. Default
                             [0.1, 0.3, 0.5, 0.7, 0.9].
            linestyles     — five matplotlib linestyle specs for style_col Qs.
            x_scale        — multiplier for the rate x-axis. Default 100_000
                             (= deaths per 100k person-storm-hours).
            xscale         — 'log' or 'linear' (default 'log').
            r_grid_n       — number of r-grid points. Default 600.
            r_grid_lo/hi   — endpoints as fractions of T. Default (1e-9, 1-1e-9).
            inline_legends — True (default) to draw two inline legends.
            color_legend_loc / style_legend_loc — matplotlib loc strings.

        Stage_draws (if provided): use mean of stage.scale across draws for
        σ. Otherwise read σ from result.meta if present, else fall back
        to 1.0 with a warning.
        """
        component = spec.get('component')
        family    = spec.get('family')
        _SUPPORTED = {('bulk', 'scaled_logit'), ('tail', 'log_logistic')}
        if (component, family) not in _SUPPORTED:
            raise ValueError(
                f"cont_pdf_quintile supports {_SUPPORTED}. "
                f"Got component={component!r}, family={family!r}."
            )

        from scipy.stats import norm as _norm
        from matplotlib.lines import Line2D

        color_col      = panel_spec.get('color_col')
        style_col      = panel_spec.get('style_col')
        fixed_col      = panel_spec.get('fixed_col')
        fixed_quantile = panel_spec.get('fixed_quantile', 0.5)
        single_curve   = not (color_col or style_col)
        if not single_curve and not (color_col and style_col and fixed_col):
            raise ValueError(
                "cont_pdf_quintile requires either both 'color_col' and 'style_col' "
                "(quintile-cross mode) or neither (single-curve mode)."
            )

        # Values: either explicit (color_values / style_values) or computed
        # from quantiles. Length is free (5 is just the historical default).
        quintiles  = panel_spec.get('quintiles', [0.1, 0.3, 0.5, 0.7, 0.9])
        linestyles = panel_spec.get('linestyles', self._DEFAULT_PDF_LINESTYLES)
        x_scale    = float(panel_spec.get('x_scale', 100_000))
        xscale     = panel_spec.get('xscale', 'log')
        n_grid     = int(panel_spec.get('r_grid_n', 600))
        lo_frac    = float(panel_spec.get('r_grid_lo', 1e-9))
        hi_frac    = float(panel_spec.get('r_grid_hi', 1 - 1e-9))

        T = float(threshold_rate)
        death_rate = data['deaths'].values / data['exposed'].values

        is_tail = (component == 'tail')
        if is_tail:
            stage_data = data[death_rate >= T]
        else:
            stage_data = data[(data['deaths'].values >= 1) & (death_rate < T)]

        if single_curve:
            color_vals = np.array([None])
            style_vals = np.array([None])
        else:
            # color_values / style_values: explicit covariate values to use,
            # bypassing the quantile compute. color_labels / style_labels: legend
            # labels per value (defaults to "f{display} Q{i+1}" if not given).
            color_vals = panel_spec.get('color_values')
            if color_vals is None:
                color_vals = stage_data[color_col].quantile(quintiles).values
            color_vals = np.asarray(color_vals)

            style_vals = panel_spec.get('style_values')
            if style_vals is None:
                style_vals = stage_data[style_col].quantile(quintiles).values
            style_vals = np.asarray(style_vals)

            if len(style_vals) > len(linestyles):
                raise ValueError(
                    f"Need at least {len(style_vals)} linestyles; only {len(linestyles)} given. "
                    "Pass a longer 'linestyles' list in panel_spec."
                )

            fixed_val = float(stage_data[fixed_col].quantile(fixed_quantile))

        # β + distribution parameter (σ for bulk, k for tail)
        if stage_draws is not None and len(stage_draws) > 0:
            beta        = np.asarray(stage_draws[0].params)
            param_names = list(stage_draws[0].param_names)
            scales      = [sd.scale for sd in stage_draws if sd.scale is not None]
            if is_tail:
                # sd.scale stores shape_param k directly for log_logistic
                k_shape = float(np.mean(scales)) if scales else float(result.meta.get('shape_param', 1.0))
            else:
                sigma = float(np.mean(np.sqrt(scales))) if scales else 1.0
        else:
            beta        = np.asarray(result.params)
            param_names = list(result.param_names)
            if is_tail:
                k_shape = float(result.meta.get('shape_param', 1.0))
            else:
                sigma = float(np.sqrt(result.meta.get('scale', 1.0)))

        def _build_x(scenario: dict) -> np.ndarray:
            row = {n: 0.0 for n in param_names}
            row['const'] = 1.0
            for col_name, v in scenario.items():
                if col_name == 'exposed':
                    if 'log_exposed' in row:
                        row['log_exposed'] = float(np.log(v))
                elif col_name in row:
                    row[col_name] = float(v)
            return np.array([row[n] for n in param_names])

        def _pdf_rate(r, eta, sigma_val, T_val):
            z        = np.log(r / (T_val - r))
            abs_dzdr = T_val / (r * (T_val - r))
            return _norm.pdf(z, loc=eta, scale=sigma_val) * abs_dzdr

        def _pdf_rate_loglogistic(r, alpha, k, T_val):
            x = r - T_val  # excess above threshold
            return (k / alpha) * (x / alpha) ** (k - 1) / (1 + (x / alpha) ** k) ** 2

        if is_tail:
            # Grid over excess above T; use geomspace so log x-axis looks good.
            excess_lo = T * lo_frac
            excess_hi = T * (hi_frac / lo_frac) * lo_frac  # symmetric decades above T
            # Fallback: use a sensible absolute spread if T is tiny.
            excess_hi = max(excess_hi, T * 100)
            r_grid = np.geomspace(max(excess_lo, 1e-15), excess_hi, n_grid) + T
        else:
            r_grid = np.geomspace(T * lo_frac, T * hi_frac, n_grid)

        # Colormap name (any matplotlib cmap). Use '<name>_r' to reverse, e.g.,
        # 'viridis_r' to put darkest at the last (highest-value) entry.
        cmap = plt.get_cmap(panel_spec.get('color_cmap', 'viridis'))
        n_color = len(color_vals)

        if single_curve:
            # Build scenario at stage_data medians for all model covariates.
            scenario: dict = {}
            for n in param_names:
                if n == 'const':
                    continue
                elif n == 'log_exposed':
                    scenario['exposed'] = float(stage_data['exposed'].median())
                elif n in stage_data.columns:
                    scenario[n] = float(stage_data[n].median())
            x_vec = _build_x(scenario)
            eta   = float(x_vec @ beta)
            line_color = style.get('pred_line', {}).get('color', '#4477AA')
            if is_tail:
                x_vals  = r_grid * x_scale
                density = _pdf_rate_loglogistic(r_grid, np.exp(eta), k_shape, T)
            else:
                x_vals  = r_grid * x_scale
                density = _pdf_rate(r_grid, eta, sigma, T)

            unc = panel_spec.get('uncertainty', {})
            if unc.get('show', False) and stage_draws is not None and len(stage_draws) > 0:
                pct_lo, pct_hi = unc.get('percentiles', (5, 95))
                band_color     = unc.get('color') or line_color
                band_alpha     = unc.get('alpha', 0.3)
                all_densities  = []
                for sd in stage_draws:
                    eta_d = float(x_vec @ np.asarray(sd.params))
                    if is_tail:
                        k_d   = float(sd.scale) if sd.scale is not None else k_shape
                        all_densities.append(_pdf_rate_loglogistic(r_grid, np.exp(eta_d), k_d, T))
                    else:
                        sig_d = float(np.sqrt(sd.scale)) if sd.scale is not None else sigma
                        all_densities.append(_pdf_rate(r_grid, eta_d, sig_d, T))
                densities_arr = np.stack(all_densities, axis=0)
                lo = np.percentile(densities_arr, pct_lo, axis=0)
                hi = np.percentile(densities_arr, pct_hi, axis=0)
                ax.fill_between(x_vals, lo, hi, color=band_color, alpha=band_alpha, linewidth=0)

            ax.plot(x_vals, density, color=line_color, lw=panel_spec.get('lw', 2.0))
        else:
            for ci, cv in enumerate(color_vals):
                for si, sv in enumerate(style_vals):
                    scenario = {color_col: cv, style_col: sv, fixed_col: fixed_val}
                    eta = float(_build_x(scenario) @ beta)
                    if is_tail:
                        alpha   = np.exp(eta)
                        density = _pdf_rate_loglogistic(r_grid, alpha, k_shape, T)
                        x_vals  = r_grid * x_scale
                    else:
                        density = _pdf_rate(r_grid, eta, sigma, T)
                        x_vals  = r_grid * x_scale
                    ax.plot(
                        x_vals, density,
                        color=cmap(ci / max(n_color - 1, 1)),
                        linestyle=linestyles[si],
                        lw=panel_spec.get('lw', 1.4),
                    )

        if xscale == 'log':
            ax.set_xscale('log')
        ax.set_ylim(0, None)
        ax.set_xlabel(panel_spec.get('x_label',
                      f'rate × {int(x_scale)} (deaths per {int(x_scale)} person-storm-hours)'))
        ax.set_ylabel(panel_spec.get('y_label', 'Relative likelihood'))
        ax.set_yticks([])
        self._apply_panel_title(ax, panel_spec)

        if not single_curve and panel_spec.get('inline_legends', True):
            color_display = panel_spec.get('color_legend_label', color_col)
            style_display = panel_spec.get('style_legend_label', style_col)
            color_labels  = panel_spec.get('color_labels')
            if color_labels is None:
                color_labels = [f'{color_display} Q{q+1}' for q in range(len(color_vals))]
            style_labels  = panel_spec.get('style_labels')
            if style_labels is None:
                style_labels = [f'{style_display} Q{q+1}' for q in range(len(style_vals))]
            if len(color_labels) != len(color_vals):
                raise ValueError(
                    f"color_labels has {len(color_labels)} entries, but color_values has {len(color_vals)}."
                )
            if len(style_labels) != len(style_vals):
                raise ValueError(
                    f"style_labels has {len(style_labels)} entries, but style_values has {len(style_vals)}."
                )
            color_handles = [
                Line2D([], [], color=cmap(q / max(n_color - 1, 1)), lw=2,
                       label=color_labels[q])
                for q in range(n_color)
            ]
            style_handles = [
                Line2D([], [], color='black', linestyle=linestyles[q], lw=2,
                       label=style_labels[q])
                for q in range(len(style_vals))
            ]
            color_loc = panel_spec.get('color_legend_loc', 'upper left')
            style_loc = panel_spec.get('style_legend_loc', 'upper right')
            fs        = panel_spec.get('legend_fontsize', 10)
            leg1 = ax.legend(handles=color_handles, loc=color_loc,
                             frameon=False, fontsize=fs)
            ax.add_artist(leg1)
            ax.legend(handles=style_handles, loc=style_loc,
                      frameon=False, fontsize=fs)
        return []  # legends rendered inline; nothing to bubble up

    # ------------------------------------------------------------------
    # Panel dispatch + font sizing
    # ------------------------------------------------------------------

    _PANEL_KINDS = ('cont_overall', 'cont_by_basin', 'cont_by_island',
                    'cat_beeswarm', 'cont_pdf_quintile')

    def _render_panel(
        self,
        ax: plt.Axes,
        panel_spec: dict,
        stage_ctx: dict,
    ) -> list[tuple]:
        """Dispatch one panel to its renderer. Returns (handle, label) tuples."""
        kind = panel_spec.get('kind')
        if kind not in self._PANEL_KINDS:
            raise ValueError(
                f"Unknown panel kind {kind!r}; expected one of {self._PANEL_KINDS}"
            )

        common = (
            stage_ctx['data'], stage_ctx['result'], stage_ctx['spec'],
            stage_ctx['threshold_rate'],
        )
        is_logistic = stage_ctx['is_logistic']
        style       = stage_ctx['style']
        plot_rate   = stage_ctx['plot_rate']

        if kind in ('cont_overall', 'cont_by_basin', 'cont_by_island'):
            x_col = panel_spec.get('x_col')
            if x_col is None:
                raise ValueError(f"panel kind {kind!r} requires 'x_col'")
            method = {
                'cont_overall':   self._plot_cont_overall,
                'cont_by_basin':  self._plot_cont_by_basin,
                'cont_by_island': self._plot_cont_by_island,
            }[kind]
            return method(ax, *common, x_col, is_logistic, style, panel_spec,
                          stage_draws=stage_ctx.get('stage_draws'),
                          plot_rate=plot_rate)
        elif kind == 'cat_beeswarm':
            cat_col = panel_spec.get('cat_col')
            if cat_col is None:
                raise ValueError(f"panel kind 'cat_beeswarm' requires 'cat_col'")
            return self._plot_cat_beeswarm(
                ax, *common, cat_col, is_logistic, style, panel_spec,
                plot_rate=plot_rate,
            )
        else:  # cont_pdf_quintile
            return self._plot_cont_pdf_quintile(
                ax, *common, is_logistic, style, panel_spec,
                stage_draws=stage_ctx.get('stage_draws'),
                plot_rate=plot_rate,
            )

    def _apply_fonts(self, ax: plt.Axes, fonts: dict) -> None:
        if fonts.get('axis_label') is not None:
            ax.xaxis.label.set_fontsize(fonts['axis_label'])
            ax.yaxis.label.set_fontsize(fonts['axis_label'])
        if fonts.get('panel_title') is not None and ax.get_title():
            ax.title.set_fontsize(fonts['panel_title'])
        if fonts.get('tick') is not None:
            ax.tick_params(labelsize=fonts['tick'])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def vet_stage(
        self,
        model_row: pd.Series,
        stage: str,
        config: Optional[dict] = None,
        figsize: Optional[tuple] = None,
        scatter_kws: Optional[dict] = None,
        plot_rate: bool = True,
        draws: Optional[list] = None,
    ) -> plt.Figure:
        """Produce vetting plots for one stage of one model.

        Layout, colors, fonts, and legend placement are controlled by `config`
        (shallow-merged with `DEFAULT_VET_CONFIG`). With no config, produces
        the historical 3×3 grid.

        Parameters
        ----------
        model_row : pd.Series
            A row from dh_results.parquet (fold_tag='insample').
        stage : str
            One of 's1', 's2', 'bulk', 'tail'.
        config : dict, optional
            See DEFAULT_VET_CONFIG for the shape. Top-level keys:
              - 'layout' : figsize, suptitle, tight_layout, panels (dict;
                replaces base panels entirely when provided)
              - 'style'  : pred_line, scatter_obs, island, basin,
                beeswarm_mean, fonts
              - 'legend' : mode in {'per_panel','shared_below','shared_right',
                'none'}, ncol, bbox_to_anchor
        figsize : tuple, optional
            Backward-compat override; overrides config['layout']['figsize'].
        scatter_kws : dict, optional
            Backward-compat override; merged into config['style']['scatter_obs'].
        plot_rate : bool, default True
            If False, bulk/tail y-axis is raw counts instead of death rate.
        draws : list, optional
            List of `DrawModel` objects (e.g. unpickled from
            03-draws/.../draws_c{c}_s{s}.pkl). When provided, panels whose
            spec carries `'uncertainty': {'show': True, ...}` will plot a
            percentile band around the prediction line. Each panel pulls
            this stage's StageDraw from each DrawModel and samples the grid.
            Currently honored by `cont_overall` panels only.
        """
        cfg = _deep_merge(DEFAULT_VET_CONFIG, config)
        if figsize is not None:
            cfg['layout']['figsize'] = figsize
        if scatter_kws is not None:
            base_scatter = cfg['style'].get('scatter_obs') or {}
            cfg['style']['scatter_obs'] = {**base_scatter, **scatter_kws}

        spec = self._reconstruct_is_spec(model_row, stage)
        threshold_rate = spec["threshold_rate"] or 0.0
        result = self._load_result(spec)
        data = self._get_subset_for_stage(stage, threshold_rate)
        is_logistic = stage in ("s1", "s2")

        panels = cfg['layout']['panels']
        if not panels:
            raise ValueError("No panels defined in config['layout']['panels'].")
        # Each panel can carry optional 'rowspan' / 'colspan' (default 1) to
        # occupy a multi-row or multi-col block. Grid dims account for those.
        nrows = max(p['row'] + p.get('rowspan', 1) for p in panels.values())
        ncols = max(p['col'] + p.get('colspan', 1) for p in panels.values())
        fig = plt.figure(figsize=cfg['layout']['figsize'])
        gs  = fig.add_gridspec(nrows, ncols)

        panel_axes: dict[str, plt.Axes] = {}
        for panel_id, p in panels.items():
            r0    = p['row']
            c0    = p['col']
            rspan = p.get('rowspan', 1)
            cspan = p.get('colspan', 1)
            panel_axes[panel_id] = fig.add_subplot(gs[r0:r0 + rspan, c0:c0 + cspan])

        suptitle = cfg['layout'].get('suptitle')
        if suptitle is None:
            family_str = spec["family"] or stage
            suptitle = f"Stage: {stage.upper()}  |  dist: {family_str}  |  n={len(data)}"
        if suptitle:
            fig.suptitle(suptitle, fontsize=cfg['style']['fonts'].get('suptitle', 12))

        # Extract per-stage StageDraw list if the caller passed draws.
        stage_draws: Optional[list] = None
        if draws is not None:
            stage_draws = [getattr(d, stage) for d in draws]

        stage_ctx = {
            'data': data, 'result': result, 'spec': spec,
            'threshold_rate': threshold_rate,
            'is_logistic': is_logistic,
            'style': cfg['style'],
            'plot_rate': plot_rate,
            'stage_draws': stage_draws,
        }
        legend_mode = cfg['legend'].get('mode', 'per_panel')
        legend_font = cfg['style']['fonts'].get('legend', 7)

        spacing = cfg.get('spacing', {}) or {}
        x_labelpad = spacing.get('x_labelpad')
        y_labelpad = spacing.get('y_labelpad')

        all_handles_labels: list[tuple] = []
        rendered_axes: list[plt.Axes] = []
        for panel_id, panel_spec in panels.items():
            ax = panel_axes[panel_id]
            rendered_axes.append(ax)
            hls = self._render_panel(ax, panel_spec, stage_ctx)
            all_handles_labels.extend(hls)

            if legend_mode == 'per_panel' and hls:
                ax.legend(fontsize=legend_font, frameon=False, loc="best")
            self._apply_fonts(ax, cfg['style']['fonts'])

            if x_labelpad is not None:
                ax.xaxis.labelpad = x_labelpad
            if y_labelpad is not None:
                ax.yaxis.labelpad = y_labelpad

        # Post-render axis controls. Ordered so each step sees the prior
        # step's effect: y-scale override -> include-zero -> share-y ->
        # scientific-notation suppression.
        axes_cfg = cfg.get('axes', {}) or {}

        y_scale_mode = axes_cfg.get('y_scale_mode', 'auto')
        if y_scale_mode in ('linear', 'log'):
            for ax in rendered_axes:
                ax.set_yscale(y_scale_mode)

        if axes_cfg.get('include_zero_y'):
            for ax in rendered_axes:
                if ax.get_yscale() == 'log':
                    continue
                ylo, yhi = ax.get_ylim()
                ax.set_ylim(min(ylo, 0.0), max(yhi, 0.0))

        share_y_cfg = axes_cfg.get('share_y', False)
        if share_y_cfg and rendered_axes:
            # share_y may be True (share across all rendered axes) or a list
            # of panel IDs (share only those, ignore the rest).
            if isinstance(share_y_cfg, (list, tuple, set)):
                share_axes = [panel_axes[pid] for pid in share_y_cfg
                              if pid in panel_axes]
            else:
                share_axes = rendered_axes
            if share_axes:
                ylims = [ax.get_ylim() for ax in share_axes]
                ymin = min(l[0] for l in ylims)
                ymax = max(l[1] for l in ylims)
                for ax in share_axes:
                    ax.set_ylim(ymin, ymax)

        if axes_cfg.get('scientific_notation') is False:
            from matplotlib.ticker import ScalarFormatter
            for ax in rendered_axes:
                if ax.get_yscale() == 'log':
                    fmt = ScalarFormatter()
                    fmt.set_scientific(False)
                    ax.yaxis.set_major_formatter(fmt)
                else:
                    try:
                        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
                    except (AttributeError, ValueError):
                        pass
                try:
                    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
                except (AttributeError, ValueError):
                    pass

        # Final pass: hide y-ticks where requested. Runs AFTER all scale
        # changes so a y_scale_mode override doesn't re-add the ticks.
        # Auto-applied to cont_pdf_quintile panels (their y axis is
        # dimensionless "relative likelihood"); opt-in for any panel via
        # panel_spec['hide_y_ticks'] = True.
        for panel_id, panel_spec in panels.items():
            if (panel_spec.get('hide_y_ticks')
                    or panel_spec.get('kind') == 'cont_pdf_quintile'):
                panel_axes[panel_id].set_yticks([])

        # tight_layout is disabled when any explicit panel-spacing field is
        # set, because tight_layout would overwrite the user's hspace/vspace/
        # suptitle_pad values.
        spacing_set = any(spacing.get(k) is not None
                          for k in ('hspace', 'vspace', 'suptitle_pad'))
        if cfg['layout'].get('tight_layout', True) and not spacing_set:
            plt.tight_layout()

        adjust_kw: dict = {}
        if spacing.get('hspace') is not None:
            adjust_kw['wspace'] = spacing['hspace']     # user hspace -> mpl wspace
        if spacing.get('vspace') is not None:
            adjust_kw['hspace'] = spacing['vspace']     # user vspace -> mpl hspace
        if spacing.get('suptitle_pad') is not None:
            adjust_kw['top']    = 1.0 - spacing['suptitle_pad']
        if adjust_kw:
            fig.subplots_adjust(**adjust_kw)

        if legend_mode in ('shared_below', 'shared_right') and all_handles_labels:
            seen: set = set()
            unique: list[tuple] = []
            for h, lab in all_handles_labels:
                if lab not in seen:
                    seen.add(lab)
                    unique.append((h, lab))
            handles = [h for h, _ in unique]
            labels  = [lab for _, lab in unique]
            bbox    = cfg['legend'].get('bbox_to_anchor')

            pad = cfg['legend'].get('pad')
            if legend_mode == 'shared_below':
                ncol = cfg['legend'].get('ncol') or len(unique)
                fig.legend(handles, labels,
                           loc='lower center',
                           ncol=ncol,
                           frameon=False,
                           fontsize=legend_font,
                           bbox_to_anchor=bbox or (0.5, -0.02))
                fig.subplots_adjust(bottom=pad if pad is not None else 0.15)
            else:  # shared_right
                ncol = cfg['legend'].get('ncol') or 1
                fig.legend(handles, labels,
                           loc='center right',
                           ncol=ncol,
                           frameon=False,
                           fontsize=legend_font,
                           bbox_to_anchor=bbox or (1.0, 0.5))
                # For shared_right, pad means "fraction of figure reserved on
                # the right" — mirrors the shared_below semantics.
                fig.subplots_adjust(right=1.0 - pad if pad is not None else 0.82)

        return fig

    def vet_model(
        self,
        model_row: pd.Series,
        config: Optional[dict] = None,
        scatter_kws: Optional[dict] = None,
        plot_rate: bool = True,
    ) -> dict:
        """Produce vetting plots for all four DH model stages."""
        return {
            stage: self.vet_stage(model_row, stage, config=config,
                                  scatter_kws=scatter_kws, plot_rate=plot_rate)
            for stage in ("s1", "s2", "bulk", "tail")
        }

    def predict_df(self, model_row: pd.Series) -> pd.DataFrame:
        """IS predictions for every row in the dataset."""
        data = self._data

        s1_spec   = self._reconstruct_is_spec(model_row, "s1")
        s2_spec   = self._reconstruct_is_spec(model_row, "s2")
        bulk_spec = self._reconstruct_is_spec(model_row, "bulk")
        tail_spec = self._reconstruct_is_spec(model_row, "tail")

        threshold_rate: float = bulk_spec["threshold_rate"] or 0.0

        s1_result   = self._load_result(s1_spec)
        s2_result   = self._load_result(s2_spec)
        bulk_result = self._load_result(bulk_spec)
        tail_result = self._load_result(tail_spec)

        pred_s1   = self._predict(s1_result,   data, s1_spec,   threshold_rate)
        pred_s2   = self._predict(s2_result,   data, s2_spec,   threshold_rate)
        pred_bulk = self._predict(bulk_result, data, bulk_spec, threshold_rate)
        pred_tail = self._predict(tail_result, data, tail_spec, threshold_rate)
        pred_rate = assemble_dh_prediction(pred_s1, pred_s2, pred_bulk, pred_tail)

        death_rate = data["deaths"].values / data["exposed"].values
        is_bulk = (
            (data["deaths"].values >= 1) & (death_rate < threshold_rate)
        ).astype(int)
        is_tail = (death_rate >= threshold_rate).astype(int)

        out = data.copy()
        out["pred_s1"]        = pred_s1
        out["pred_s2"]        = pred_s2
        out["pred_bulk"]      = pred_bulk
        out["pred_tail"]      = pred_tail
        out["pred_rate"]      = pred_rate
        out["observed_rate"]  = death_rate
        out["threshold_rate"] = threshold_rate
        out["is_bulk"]        = is_bulk
        out["is_tail"]        = is_tail
        return out

    def predict_df_oos(
        self, model_row: pd.Series, seed: int = 0
    ) -> pd.DataFrame:
        """OOS predictions from the stored model_predictions parquet."""
        if self._output_dir is None:
            raise ValueError(
                "output_dir must be provided at construction to use predict_df_oos()."
            )

        s1_spec   = self._reconstruct_is_spec(model_row, "s1")
        s2_spec   = self._reconstruct_is_spec(model_row, "s2")
        bulk_spec = self._reconstruct_is_spec(model_row, "bulk")
        tail_spec = self._reconstruct_is_spec(model_row, "tail")

        mid = model_id(s1_spec, s2_spec, bulk_spec, tail_spec)
        fold_tag = f"oos_seed{seed}"
        parquet_path = (
            self._output_dir / "model_predictions" / f"{mid}_{fold_tag}_predictions.parquet"
        )

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"OOS predictions not found at {parquet_path}. "
                "Run run-evaluate with fold assignments first."
            )
        return pd.read_parquet(parquet_path)
