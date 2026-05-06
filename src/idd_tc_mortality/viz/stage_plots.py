"""
stage_plots.py
Vetting plots for Double Hurdle TC mortality model stages.

Usage:
    from idd_tc_mortality.viz.stage_plots import StagePlotter

    plotter = StagePlotter(data=df, results_dir=results_dir, output_dir=output_dir)

    # All vetting plots for one stage (model_row is a row from dh_results.parquet)
    fig = plotter.vet_stage(model_row, stage='s1')
    fig = plotter.vet_stage(model_row, stage='bulk')

    # All 4 stages at once
    figs = plotter.vet_model(model_row)

    # Prediction DataFrames
    df_is  = plotter.predict_df(model_row)          # IS predictions on full data
    df_oos = plotter.predict_df_oos(model_row, seed=0)  # OOS from stored parquet
"""

from __future__ import annotations

import json
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
        """Compute threshold rate from self._data at the given quantile."""
        death_rate = self._data["deaths"].values / self._data["exposed"].values
        return float(
            compute_thresholds(death_rate, np.array([quantile]))[quantile]
        )

    def _reconstruct_is_spec(self, model_row: pd.Series, stage: str) -> dict:
        """Rebuild the IS component spec from a dh_results model row.

        Computes threshold_rate from self._data so the component_id matches
        the one produced by orchestrate.py at fit time.
        """
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
        """Load a FitResult from disk with in-memory caching."""
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
        """Return the data subset and y column for a given stage.

        S1 : all rows,   y = 1 if deaths >= 1 else 0
        S2 : deaths >= 1,  y = 1 if death_rate >= threshold_rate else 0
        bulk: deaths >= 1 AND death_rate < threshold_rate, y = death_rate
        tail: death_rate >= threshold_rate, y = death_rate
        """
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
        """Build a synthetic prediction DataFrame of length n.

        Reference values: median for continuous, mode for categorical.
        The 'exposed' column is set to data['exposed'].median() so that
        log_exposed offsets/covariates are computed correctly.
        Overrides are applied after setting reference values.
        """
        ref = {
            "wind_speed": float(data["wind_speed"].median()),
            "sdi": float(data["sdi"].median()),
            "basin": data["basin"].mode().iloc[0],
            "is_island": int(data["is_island"].mode().iloc[0]),
            "exposed": float(data["exposed"].median()),
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
        """Predict on pred_data using the fitted component result.

        No subset masking — applies the model to all rows of pred_data.
        pred_data must contain 'exposed' and all covariate columns
        flagged True in spec['covariate_combo'].

        Parameters
        ----------
        result : FitResult from fit_one_component.
        pred_data : DataFrame to predict on. Must include 'exposed'.
        spec : component spec dict (component, covariate_combo, family).
        threshold_rate : used for gpd/beta post-processing.
        """
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

            # Post-process to rate scale.
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
        rng_seed: int = 42,
        scatter_kws: Optional[dict] = None,
    ) -> None:
        kws = {"alpha": 0.2, "s": 10, "c": "lightgray", "zorder": 1}
        if scatter_kws:
            kws.update(scatter_kws)
        rng = np.random.default_rng(rng_seed)
        if is_logistic:
            jitter = rng.uniform(-0.02, 0.02, len(y_vals))
            ax.scatter(x_vals, y_vals + jitter, **kws)
        else:
            ax.scatter(x_vals, np.maximum(y_vals.astype(float), 1e-9), **kws)
            ax.set_yscale("log")

    def _style_ax(
        self,
        ax: plt.Axes,
        x_label: str,
        is_logistic: bool,
        plot_rate: bool = True,
    ) -> None:
        ax.set_xlabel(x_label)
        if is_logistic:
            ax.set_ylabel("P(Y=1)")
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylabel("Death rate (deaths / exposed)" if plot_rate else "Deaths")

    # ------------------------------------------------------------------
    # Continuous-covariate sub-plots
    # ------------------------------------------------------------------

    def _plot_cont_overall(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        result: FitResult,
        spec: dict,
        threshold_rate: float,
        x_raw_col: str,
        x_label: str,
        is_logistic: bool,
        n_grid: int = 100,
        scatter_kws: Optional[dict] = None,
        plot_rate: bool = True,
    ) -> None:
        """Single prediction line; all other covariates at reference values."""
        x_vals = data[x_raw_col].values.astype(float)
        self._scatter_obs(ax, x_vals, data["y"].values, is_logistic, scatter_kws=scatter_kws)

        x_grid = np.linspace(x_vals.min(), x_vals.max(), n_grid)
        pred_data = self._make_pred_df(data, n_grid, {x_raw_col: x_grid})
        y_pred = self._predict(result, pred_data, spec, threshold_rate)
        ax.plot(x_grid, y_pred, color="steelblue", linewidth=2, zorder=2)

        self._style_ax(ax, x_label, is_logistic, plot_rate=plot_rate)
        ax.set_title(f"{x_label} — overall")

    def _plot_cont_by_basin(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        result: FitResult,
        spec: dict,
        threshold_rate: float,
        x_raw_col: str,
        x_label: str,
        is_logistic: bool,
        n_grid: int = 100,
        scatter_kws: Optional[dict] = None,
        plot_rate: bool = True,
    ) -> None:
        """Separate prediction line per basin level."""
        x_vals = data[x_raw_col].values.astype(float)
        self._scatter_obs(ax, x_vals, data["y"].values, is_logistic, scatter_kws=scatter_kws)

        x_grid = np.linspace(x_vals.min(), x_vals.max(), n_grid)
        basins = sorted(data["basin"].dropna().unique())
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(basins)))

        for i, basin in enumerate(basins):
            pred_data = self._make_pred_df(data, n_grid, {x_raw_col: x_grid, "basin": basin})
            y_pred = self._predict(result, pred_data, spec, threshold_rate)
            ax.plot(
                x_grid, y_pred,
                color=colors[i],
                linestyle=_LINE_STYLES[i % len(_LINE_STYLES)],
                linewidth=1.5, label=basin, zorder=2,
            )

        self._style_ax(ax, x_label, is_logistic, plot_rate=plot_rate)
        ax.set_title(f"{x_label} — by basin")
        ax.legend(fontsize=7, loc="best")

    def _plot_cont_by_island(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        result: FitResult,
        spec: dict,
        threshold_rate: float,
        x_raw_col: str,
        x_label: str,
        is_logistic: bool,
        n_grid: int = 100,
        scatter_kws: Optional[dict] = None,
        plot_rate: bool = True,
    ) -> None:
        """Prediction lines for is_island=0 and is_island=1."""
        x_vals = data[x_raw_col].values.astype(float)
        self._scatter_obs(ax, x_vals, data["y"].values, is_logistic, scatter_kws=scatter_kws)

        x_grid = np.linspace(x_vals.min(), x_vals.max(), n_grid)
        for island_val, label, color in [(0, "Non-island", "steelblue"), (1, "Island", "coral")]:
            pred_data = self._make_pred_df(
                data, n_grid, {x_raw_col: x_grid, "is_island": island_val}
            )
            y_pred = self._predict(result, pred_data, spec, threshold_rate)
            ax.plot(x_grid, y_pred, color=color, linewidth=2, label=label, zorder=2)

        self._style_ax(ax, x_label, is_logistic, plot_rate=plot_rate)
        ax.set_title(f"{x_label} — by island")
        ax.legend(fontsize=7, loc="best")

    # ------------------------------------------------------------------
    # Categorical covariate beeswarm
    # ------------------------------------------------------------------

    def _plot_cat_beeswarm(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        result: FitResult,
        spec: dict,
        threshold_rate: float,
        cat_col: str,
        is_logistic: bool,
        scatter_kws: Optional[dict] = None,
        plot_rate: bool = True,
    ) -> None:
        """Jittered dots = observed; thick horizontal line = mean prediction."""
        kws = {"alpha": 0.2, "s": 10, "c": "lightgray", "zorder": 1}
        if scatter_kws:
            kws.update(scatter_kws)

        categories = sorted(data[cat_col].dropna().unique())
        cat_to_x = {cat: i for i, cat in enumerate(categories)}
        rng = np.random.default_rng(42)

        x_base = np.array([cat_to_x[c] for c in data[cat_col]])
        x_jittered = x_base + rng.uniform(-0.3, 0.3, len(data))
        y_vals = data["y"].values.astype(float)

        if is_logistic:
            jitter_y = rng.uniform(-0.02, 0.02, len(y_vals))
            ax.scatter(x_jittered, y_vals + jitter_y, **kws)
        else:
            ax.scatter(x_jittered, np.maximum(y_vals, 1e-9), **kws)
            ax.set_yscale("log")

        for cat in categories:
            cat_subset = data[data[cat_col] == cat]
            mean_pred = self._predict(result, cat_subset, spec, threshold_rate).mean()
            x_pos = cat_to_x[cat]
            ax.hlines(mean_pred, x_pos - 0.35, x_pos + 0.35,
                      colors="steelblue", linewidth=3, zorder=2)

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([str(c) for c in categories], rotation=45, ha="right")
        ax.set_xlabel(cat_col)
        if is_logistic:
            ax.set_ylabel("P(Y=1)")
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylabel("Death rate (deaths / exposed)" if plot_rate else "Deaths")
        ax.set_title(f"{cat_col} — beeswarm")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def vet_stage(
        self,
        model_row: pd.Series,
        stage: str,
        figsize: tuple = (16, 12),
        scatter_kws: Optional[dict] = None,
        plot_rate: bool = True,
    ) -> plt.Figure:
        """Produce all vetting plots for one stage of one model.

        Layout (3 rows × 3 cols):
          Row 0: wind speed overall | wind speed by basin | wind speed by island
          Row 1: SDI overall        | SDI by basin        | SDI by island
          Row 2: basin beeswarm     | island beeswarm     | (empty)

        Parameters
        ----------
        model_row : pd.Series
            A row from dh_results.parquet (fold_tag='insample').
        stage : str
            One of 's1', 's2', 'bulk', 'tail'.
        scatter_kws : dict, optional
            Passed to every ax.scatter() call for observed data points.
        plot_rate : bool, default True
            If True, bulk/tail y-axis is death rate. If False, raw counts.
        """
        spec = self._reconstruct_is_spec(model_row, stage)
        threshold_rate = spec["threshold_rate"] or 0.0
        result = self._load_result(spec)

        data = self._get_subset_for_stage(stage, threshold_rate)
        is_logistic = stage in ("s1", "s2")

        fig, axes = plt.subplots(3, 3, figsize=figsize)
        family_str = spec["family"] or stage
        fig.suptitle(
            f"Stage: {stage.upper()}  |  dist: {family_str}  |  n={len(data)}",
            fontsize=12,
        )

        # Row 0: wind speed
        self._plot_cont_overall(
            axes[0, 0], data, result, spec, threshold_rate,
            "wind_speed", "Wind speed", is_logistic,
            scatter_kws=scatter_kws, plot_rate=plot_rate,
        )
        self._plot_cont_by_basin(
            axes[0, 1], data, result, spec, threshold_rate,
            "wind_speed", "Wind speed", is_logistic,
            scatter_kws=scatter_kws, plot_rate=plot_rate,
        )
        self._plot_cont_by_island(
            axes[0, 2], data, result, spec, threshold_rate,
            "wind_speed", "Wind speed", is_logistic,
            scatter_kws=scatter_kws, plot_rate=plot_rate,
        )

        # Row 1: SDI
        self._plot_cont_overall(
            axes[1, 0], data, result, spec, threshold_rate,
            "sdi", "SDI", is_logistic,
            scatter_kws=scatter_kws, plot_rate=plot_rate,
        )
        self._plot_cont_by_basin(
            axes[1, 1], data, result, spec, threshold_rate,
            "sdi", "SDI", is_logistic,
            scatter_kws=scatter_kws, plot_rate=plot_rate,
        )
        self._plot_cont_by_island(
            axes[1, 2], data, result, spec, threshold_rate,
            "sdi", "SDI", is_logistic,
            scatter_kws=scatter_kws, plot_rate=plot_rate,
        )

        # Row 2: categorical beeswarms
        self._plot_cat_beeswarm(
            axes[2, 0], data, result, spec, threshold_rate,
            "basin", is_logistic,
            scatter_kws=scatter_kws, plot_rate=plot_rate,
        )
        self._plot_cat_beeswarm(
            axes[2, 1], data, result, spec, threshold_rate,
            "is_island", is_logistic,
            scatter_kws=scatter_kws, plot_rate=plot_rate,
        )
        axes[2, 2].axis("off")

        plt.tight_layout()
        return fig

    def vet_model(
        self,
        model_row: pd.Series,
        scatter_kws: Optional[dict] = None,
        plot_rate: bool = True,
    ) -> dict:
        """Produce vetting plots for all four DH model stages."""
        return {
            stage: self.vet_stage(model_row, stage, scatter_kws=scatter_kws, plot_rate=plot_rate)
            for stage in ("s1", "s2", "bulk", "tail")
        }

    def predict_df(self, model_row: pd.Series) -> pd.DataFrame:
        """IS predictions for every row in the dataset.

        All four component models are applied to all rows (no subset masking) so
        the returned DataFrame covers the full dataset.

        Columns returned from self._data plus:
          pred_s1       — P(deaths >= 1) for every row
          pred_s2       — P(rate >= threshold | deaths >= 1) for every row
          pred_bulk     — bulk rate prediction for every row
          pred_tail     — tail rate prediction for every row
          pred_rate     — assembled: p_s1*(p_s2*tail + (1-p_s2)*bulk)
          observed_rate — deaths / exposed
          threshold_rate
          is_bulk       — 1 if deaths>=1 and rate < threshold_rate, else 0
          is_tail       — 1 if rate >= threshold_rate, else 0
        """
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
        """OOS predictions from the stored model_predictions parquet.

        Reads output_dir/model_predictions/{model_id}_oos_seed{seed}_predictions.parquet.

        Parameters
        ----------
        model_row : pd.Series
            A row from dh_results.parquet (fold_tag='insample') identifying the
            model configuration.
        seed : int
            OOS seed index (default 0).

        Returns
        -------
        pd.DataFrame
            Columns: predicted_rate, observed_rate, any_death, threshold_rate,
            exposed, fold_tag, heldout_fold_tag, and optional covariate columns.

        Raises
        ------
        ValueError
            If output_dir was not provided at construction time.
        FileNotFoundError
            If the parquet for this model/seed does not exist.
        """
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
