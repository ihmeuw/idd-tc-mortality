"""Figures from the SHIPPED blended deliverable (plus its historical extension).

Inputs: ``direct_risk/2026_08_20_v2000_{adjusted_mean,unadjusted}_direct_deaths.parquet``
(ssp scenarios 2015-2100, 100 paired draws, FHS hierarchy incl. Global=1 and the
super-regions) and ``historical_blend_<tag>.parquet`` from
:mod:`idd_tc_mortality.predict.blend_historical`.

Three figures, one function each, all pure matplotlib on summarised frames:

* :func:`plot_global_2x2`  — Global deaths: rows linear / log, columns full period /
  zoom; observed dots, historical mean + 95% band, one line + band per RCP.
* :func:`plot_sr_grid`     — one panel per super-region, RCP lines + bands.
* :func:`plot_cumulative_boxes` — per-draw cumulative deaths by scenario, and the
  paired RCP8.5 - RCP2.6 difference.
"""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib
import numpy as np
import pandas as pd

from idd_tc_mortality.predict.blend_historical import DR, HIER_PATH, OUT_DIR
from idd_tc_mortality.viz.predict_plots import SSP_SCENARIO_MAP

OBS_CSV = ("/mnt/team/rapidresponse/pub/tropical-storms/data/ibtracs_deaths/"
           "ibtracs_stage4b_pafs_admin0_with_deaths_PSH_20260603_sdi_isisland.csv")
SHIPPED = {tag: DR / f"2026_08_20_v2000_{tag}_direct_deaths.parquet" for tag in ("adjusted_mean", "unadjusted")}
GLOBAL_ID = 1
SCENARIOS = ("ssp126", "ssp245", "ssp585")
CENTRAL_EUROPE_SR = 31  # guard-raked to ~0 deaths; excluded from the SR grid by default


# --- data -----------------------------------------------------------------------------
def load_deaths(tag: str, location_ids, historical_path: Path | None = None) -> pd.DataFrame:
    """Draw-level deaths for ``location_ids``: shipped ssp rows plus (optionally) historical rows."""
    cols = ["storm_draw", "ssp_scenario", "year_id", "location_id", "deaths"]
    ids = [int(i) for i in location_ids]
    frames = [pd.read_parquet(SHIPPED[tag], columns=cols, filters=[("location_id", "in", ids)])]
    if historical_path is not None:
        frames.append(pd.read_parquet(historical_path, columns=cols, filters=[("location_id", "in", ids)]))
    return pd.concat(frames, ignore_index=True)


def load_observed(hier: pd.DataFrame) -> pd.DataFrame:
    """Observed deaths by (location_id, year), rolled up the hierarchy."""
    from idd_tc_mortality.predict.postprocess import (
        build_ancestor_map,
        build_observed_deaths_from_source_csv,
    )

    return build_observed_deaths_from_source_csv(OBS_CSV, build_ancestor_map(hier))


def summarize(df: pd.DataFrame, location_id: int, lower: float = 2.5, upper: float = 97.5) -> pd.DataFrame:
    """Per (ssp_scenario, year_id): mean and central interval of deaths over draws."""
    sub = df[df["location_id"] == location_id]
    g = sub.groupby(["ssp_scenario", "year_id"])["deaths"]
    out = g.agg(mean="mean",
                lower=lambda x: np.percentile(x, lower),
                upper=lambda x: np.percentile(x, upper)).reset_index()
    return out.sort_values(["ssp_scenario", "year_id"], ignore_index=True)


def cumulative_by_draw(df: pd.DataFrame, location_id: int, start: int, end: int) -> pd.DataFrame:
    """Per (ssp_scenario, storm_draw) sum of deaths over ``start..end`` inclusive (ssp scenarios only)."""
    sub = df[(df["location_id"] == location_id) & df["year_id"].between(start, end)
             & df["ssp_scenario"].isin(SCENARIOS)]
    return sub.groupby(["ssp_scenario", "storm_draw"], as_index=False)["deaths"].sum()


def paired_difference(cum: pd.DataFrame, a: str = "ssp585", b: str = "ssp126") -> pd.Series:
    """Per-draw ``a - b`` cumulative deaths, paired on storm_draw."""
    w = cum.pivot(index="storm_draw", columns="ssp_scenario", values="deaths")
    return (w[a] - w[b]).dropna()


# --- drawing --------------------------------------------------------------------------
def _draw_series(ax, stats: pd.DataFrame, *, xmin: int, band_alpha: float = 0.25):
    """Historical (black) and RCP lines with 95% bands onto ``ax``; returns handles."""
    handles = []
    h = stats[(stats["ssp_scenario"] == "historical") & (stats["year_id"] >= xmin)]
    if not h.empty:
        ax.fill_between(h["year_id"], h["lower"], h["upper"], color="black", alpha=0.15, linewidth=0)
        handles += ax.plot(h["year_id"], h["mean"], color="black", lw=1.5, label="Historical")
    for sc in SCENARIOS:
        s = stats[(stats["ssp_scenario"] == sc) & (stats["year_id"] >= xmin)]
        if s.empty:
            continue
        info = SSP_SCENARIO_MAP[sc]
        ax.fill_between(s["year_id"], s["lower"], s["upper"], color=info["color"], alpha=band_alpha, linewidth=0)
        handles += ax.plot(s["year_id"], s["mean"], color=info["color"], lw=1.5, label=info["name"])
    ax.grid(True, alpha=0.3)
    return handles


def plot_global_2x2(stats: pd.DataFrame, obs: pd.DataFrame | None, *, start_year: int = 2000,
                    zoom_year: int = 2020, end_year: int = 2100, title: str = "Global Tropical Storm Deaths",
                    figsize=(14, 6.5), fontsize: int = 11):
    """Rows: linear, log. Columns: ``start_year``..``end_year``, ``zoom_year``..``end_year``."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    legend_handles = None
    for col, xmin in enumerate((start_year, zoom_year)):
        for row, scale in enumerate(("linear", "log")):
            ax = axes[row, col]
            handles = _draw_series(ax, stats, xmin=xmin)
            if obs is not None and col == 0:
                o = obs[obs["year"] >= xmin]
                handles = [ax.scatter(o["year"], o["deaths"], color="black", s=22, zorder=5, label="Observed Deaths")] + handles
            if legend_handles is None:  # first (full-period, linear) panel carries every series
                legend_handles = handles
            ax.set_yscale(scale)
            ax.set_xlim(xmin - 1, end_year + 1)
            ax.set_xlabel("Year", fontsize=fontsize)
            ax.set_ylabel("Number of Deaths", fontsize=fontsize)
            ax.tick_params(labelsize=fontsize - 1)
    fig.suptitle(title, fontsize=fontsize + 3)
    # "outside" placement reserves space under constrained layout, so the legend survives a non-tight save
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=len(legend_handles), frameon=True,
               fontsize=fontsize)
    return fig


def plot_sr_grid(stats_by_sr: dict[int, pd.DataFrame], names: dict[int, str], *, start_year: int = 2020,
                 end_year: int = 2100, ncols: int = 2, title: str = "Tropical Storm Deaths by Super Region",
                 figsize=(14, 6.5), fontsize: int = 11):
    """One panel per super-region (dict order), RCP lines + bands, shared legend below."""
    import matplotlib.pyplot as plt

    n = len(stats_by_sr)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False)
    handles = []
    for ax, (sr, stats) in zip(axes.ravel(), stats_by_sr.items(), strict=False):
        handles = _draw_series(ax, stats, xmin=start_year)
        ax.set_xlim(start_year - 1, end_year + 1)
        ax.set_title(names.get(sr, str(sr)), fontsize=fontsize, fontweight="bold", y=0.85)
        ax.tick_params(labelsize=fontsize - 1)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Deaths", fontsize=fontsize)
    for ax in axes[-1, :]:
        ax.set_xlabel("Year", fontsize=fontsize)
    fig.suptitle(title, fontsize=fontsize + 3)
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=True, fontsize=fontsize)
    return fig


def plot_cumulative_boxes(cum: pd.DataFrame, *, start: int, end: int, unit: float = 1e5,
                          figsize=(12, 5), fontsize: int = 11):
    """Left: per-draw cumulative deaths by scenario. Right: paired RCP8.5 - RCP2.6."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True,
                                   gridspec_kw={"width_ratios": [3, 1]})
    data = [cum.loc[cum["ssp_scenario"] == sc, "deaths"].to_numpy() / unit for sc in SCENARIOS]
    labels = [SSP_SCENARIO_MAP[sc]["name"] for sc in SCENARIOS]
    bp = ax1.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5, whis=(0, 100),
                     medianprops={"color": "black", "lw": 2})  # whiskers span the full draw range
    for patch, sc in zip(bp["boxes"], SCENARIOS, strict=True):
        patch.set_facecolor(SSP_SCENARIO_MAP[sc]["color"])
        patch.set_alpha(0.8)
    unit_label = f"in {int(unit):,}s" if unit != 1 else ""
    ax1.set_ylabel(f"Deaths ({unit_label})" if unit_label else "Deaths", fontsize=fontsize)
    ax1.set_title(f"Cumulative deaths by scenario, {start}-{end}", fontsize=fontsize + 1)
    ax1.grid(True, axis="y", alpha=0.3)

    diff = paired_difference(cum).to_numpy() / unit
    ax2.boxplot([diff], tick_labels=["RCP 8.5 − RCP 2.6"], patch_artist=True, widths=0.5,
                medianprops={"color": "black", "lw": 2}, boxprops={"facecolor": "#999999"})
    ax2.axhline(0, color="black", ls="--", lw=1)
    ax2.set_ylabel(f"Deaths ({unit_label})" if unit_label else "Deaths", fontsize=fontsize)
    ax2.set_title("Difference in cumulative deaths\nRCP 8.5 − RCP 2.6", fontsize=fontsize + 1)
    ax2.grid(True, axis="y", alpha=0.3)
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=fontsize - 1)
    return fig


# --- CLI ------------------------------------------------------------------------------
@click.command()
@click.option("--tag", type=click.Choice(sorted(SHIPPED)), default="adjusted_mean", show_default=True,
              help="Which shipped series: mean+guard raked (adjusted_mean) or unadjusted.")
@click.option("--historical", type=click.Path(path_type=Path), default=None,
              help="Historical blend parquet [default: OUT_DIR/historical_blend_<tag>.parquet]; 'none' to skip.")
@click.option("--start-year", default=2000, show_default=True, help="Left column start (global) / observed dots from.")
@click.option("--zoom-year", default=2020, show_default=True, help="Right column start (global) and SR grid start.")
@click.option("--cum-start", default=2023, show_default=True, help="First year of the cumulative window.")
@click.option("--cum-end", default=2100, show_default=True)
@click.option("--include-sr31/--exclude-sr31", default=False, show_default=True,
              help="Include Central Europe, Eastern Europe, and Central Asia (guard-raked) in the SR grid.")
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_DIR, show_default=True)
@click.option("--dpi", default=150, show_default=True)
def main(tag, historical, start_year, zoom_year, cum_start, cum_end, include_sr31, out_dir, dpi):
    """Global 2x2, super-region grid and cumulative box plots from the shipped blended deliverable."""
    matplotlib.use("Agg")
    from idd_figures.lib.io import save_figure

    out_dir = Path(out_dir)
    hist_path = None if str(historical) == "none" else (historical or out_dir / f"historical_blend_{tag}.parquet")
    if hist_path is not None and not Path(hist_path).exists():
        raise click.ClickException(f"historical blend not found: {hist_path} (run run-blend-historical first, or --historical none)")

    hier = pd.read_parquet(HIER_PATH)
    sr_ids = sorted(int(i) for i in hier.loc[hier["level"] == 1, "location_id"])
    if not include_sr31:
        sr_ids = [s for s in sr_ids if s != CENTRAL_EUROPE_SR]
    names = hier.set_index("location_id")["location_name"].to_dict()

    df = load_deaths(tag, [GLOBAL_ID, *sr_ids], hist_path)
    obs = load_observed(hier)
    obs_global = obs[obs["location_id"] == GLOBAL_ID]

    written = []
    fig = plot_global_2x2(summarize(df, GLOBAL_ID), obs_global, start_year=start_year, zoom_year=zoom_year)
    written += save_figure(fig, out_dir / f"global_2x2_{tag}.png", dpi=dpi)
    fig = plot_sr_grid({sr: summarize(df, sr) for sr in sr_ids}, names, start_year=zoom_year)
    written += save_figure(fig, out_dir / f"sr_grid_{tag}.png", dpi=dpi)
    cum = cumulative_by_draw(df, GLOBAL_ID, cum_start, cum_end)
    fig = plot_cumulative_boxes(cum, start=cum_start, end=cum_end)
    written += save_figure(fig, out_dir / f"cumulative_boxes_{tag}_{cum_start}_{cum_end}.png", dpi=dpi)

    med = cum.groupby("ssp_scenario")["deaths"].median()
    diff = paired_difference(cum)
    for sc in SCENARIOS:
        click.echo(f"cumulative {cum_start}-{cum_end} {SSP_SCENARIO_MAP[sc]['name']}: median {med[sc]:,.0f}")
    click.echo(f"paired RCP8.5-RCP2.6: median {diff.median():,.0f} (IQR {diff.quantile(.25):,.0f} .. {diff.quantile(.75):,.0f})")
    for p in written:
        click.echo(f"wrote {p}")


if __name__ == "__main__":
    main()
