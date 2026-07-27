"""Read the delivered draw-level sensitivity files back and decompose drivers.

All logic lives here so the comparison notebook only calls functions (no
in-notebook definitions). Each delivered file (baseline + one per sensitivity
mode) is the draw-level frame written by ``build_sr_version_files`` with columns
``[storm_draw, ssp_scenario, year_id, location_id, deaths, population, death_rate]``.

Attribution convention (contribution = "effect of letting this driver evolve"):
    dSDI  = baseline − sdi_const     (deaths from SDI evolving vs frozen at anchor)
    dPOP  = baseline − pop_const     (deaths from population evolving vs frozen)
    dTOTAL= baseline − both
    interaction = dTOTAL − (dSDI + dPOP)   (non-additivity of the two drivers)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from idd_tc_mortality.viz.predict_plots import SSP_SCENARIO_MAP

SCEN, YEAR, LOC, VAL, DRAW = "ssp_scenario", "year_id", "location_id", "deaths", "storm_draw"
MODES = ("sdi_const", "pop_const", "both")

SUPER_REGIONS = {
    1: "Global", 4: "SE/E Asia & Oceania", 31: "C/E Europe & C Asia", 64: "High-income",
    103: "Latin America & Caribbean", 137: "N Africa & Middle East",
    158: "South Asia", 166: "Sub-Saharan Africa",
}

# Friendly legend labels for the level series (vs the raw internal mode names).
LEVEL_LABELS = {
    "baseline": "baseline (both evolving)", "sdi_const": "SDI frozen",
    "pop_const": "population frozen", "both": "both frozen (constant)",
}

# Editable font sizes shared by every plot. Pass ``fonts={"legend": 14, ...}`` to
# override any subset; keys: title (panel), label (axis labels), tick (axis
# numbers), legend, suptitle (overall), annot (in-bar value labels, waterfall).
DEFAULT_FONTS = {"title": 11, "label": 10, "tick": 9, "legend": 9, "suptitle": 14, "annot": 9}


def _fonts(fonts: dict | None) -> dict:
    f = dict(DEFAULT_FONTS)
    if fonts:
        f.update(fonts)
    return f


def _apply_fonts(ax, f: dict) -> None:
    """Apply title/label/tick/legend sizes to an axis after it's been drawn."""
    ax.title.set_fontsize(f["title"])
    ax.xaxis.label.set_fontsize(f["label"])
    ax.yaxis.label.set_fontsize(f["label"])
    ax.tick_params(labelsize=f["tick"])
    lg = ax.get_legend()
    if lg is not None:
        for t in lg.get_texts():
            t.set_fontsize(f["legend"])


def _yr(cum: pd.DataFrame) -> str:
    """The cumulative year range label (e.g. '2023-2050') carried on a cumulative frame."""
    return str(cum["year_range"].iloc[0]) if "year_range" in cum.columns and len(cum) else ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_mean_deaths(path: str | Path) -> pd.DataFrame:
    """Mean deaths across storm draws, per (scenario, year, location)."""
    df = pd.read_parquet(path, columns=[DRAW, SCEN, YEAR, LOC, VAL])
    return df.groupby([SCEN, YEAR, LOC], as_index=False)[VAL].mean()


def deliverable_paths(baseline_dir: str | Path, sens_dir: str | Path,
                      prefix: str, variant: str = "unadjusted") -> dict[str, Path]:
    """Map {baseline, sdi_const, pop_const, both} -> delivered file path.

    `variant` is 'unadjusted' or 'adjusted_mean'. Baseline files use `<prefix>_<variant>`;
    sensitivity files use `<prefix>_<mode>_<variant>`.
    """
    baseline_dir, sens_dir = Path(baseline_dir), Path(sens_dir)
    suffix = f"{variant}_direct_deaths.parquet"
    paths = {"baseline": baseline_dir / f"{prefix}_{suffix}"}
    for m in MODES:
        paths[m] = sens_dir / f"{prefix}_{m}_{suffix}"
    return paths


def load_set(paths: dict[str, str | Path]) -> dict[str, pd.DataFrame]:
    """Load the baseline + 3 sensitivity mean-death frames."""
    return {k: load_mean_deaths(p) for k, p in paths.items()}


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def decompose(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge baseline + 3 modes on (scenario, year, location); compute contributions."""
    key = [SCEN, YEAR, LOC]
    out = sets["baseline"].rename(columns={VAL: "baseline"})
    for m in MODES:
        out = out.merge(sets[m].rename(columns={VAL: m}), on=key, how="inner")
    out["dSDI"] = out["baseline"] - out["sdi_const"]
    out["dPOP"] = out["baseline"] - out["pop_const"]
    out["dTOTAL"] = out["baseline"] - out["both"]
    out["interaction"] = out["dTOTAL"] - (out["dSDI"] + out["dPOP"])
    return out


# ---------------------------------------------------------------------------
# Plots (imported by the notebook; no plotting logic lives in the notebook)
# ---------------------------------------------------------------------------

def plot_levels(sets: dict[str, pd.DataFrame], location_id: int, scenario: str, ax=None,
                fonts: dict | None = None):
    """Overlay baseline vs the three frozen-driver counterfactuals over time."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    f = _fonts(fonts)
    style = {"baseline": ("k", "-"), "sdi_const": ("#1f77b4", "--"),
             "pop_const": ("#d62728", "--"), "both": ("#2ca02c", ":")}
    for name, df in sets.items():
        g = df[(df[LOC] == location_id) & (df[SCEN] == scenario)].sort_values(YEAR)
        if g.empty:
            continue
        c, ls = style.get(name, ("0.5", "-"))
        ax.plot(g[YEAR], g[VAL], color=c, ls=ls, lw=1.6, label=LEVEL_LABELS.get(name, name))
    ax.set_title(f"{SUPER_REGIONS.get(location_id, location_id)} — {SSP_SCENARIO_MAP[scenario]['name']}")
    ax.set_xlabel("year"); ax.set_ylabel("mean deaths/yr"); ax.grid(alpha=0.25)
    ax.legend()
    _apply_fonts(ax, f)
    return ax


def plot_contributions(decomp: pd.DataFrame, location_id: int, scenario: str, ax=None,
                       fonts: dict | None = None):
    """Stacked driver contributions (dSDI, dPOP, interaction) over time; dTOTAL as a line."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    f = _fonts(fonts)
    g = decomp[(decomp[LOC] == location_id) & (decomp[SCEN] == scenario)].sort_values(YEAR)
    if g.empty:
        return ax
    ax.stackplot(g[YEAR], g["dSDI"], g["dPOP"], g["interaction"],
                 labels=["SDI", "population", "interaction"],
                 colors=["#1f77b4", "#d62728", "#bbbbbb"], alpha=0.85)
    ax.plot(g[YEAR], g["dTOTAL"], color="k", lw=1.6, label="total (baseline − both frozen)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{SUPER_REGIONS.get(location_id, location_id)} — {SSP_SCENARIO_MAP[scenario]['name']}")
    ax.set_xlabel("year"); ax.set_ylabel("Δ deaths/yr vs frozen"); ax.grid(alpha=0.25)
    ax.legend()
    _apply_fonts(ax, f)
    return ax


def summary_table(decomp: pd.DataFrame, year: int, scenario: str) -> pd.DataFrame:
    """Per-super-region contributions at one (year, scenario)."""
    g = decomp[(decomp[YEAR] == year) & (decomp[SCEN] == scenario)
               & (decomp[LOC].isin(SUPER_REGIONS))].copy()
    g["region"] = g[LOC].map(SUPER_REGIONS)
    cols = ["region", "baseline", "dSDI", "dPOP", "interaction", "dTOTAL"]
    return g[cols].sort_values("dTOTAL", ascending=False).round(1).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cumulative-over-a-year-range decomposition
# ---------------------------------------------------------------------------

def cumulative(decomp: pd.DataFrame, year_range: tuple[int, int]) -> pd.DataFrame:
    """Sum every level/delta over [y0, y1], per (scenario, location).

    Columns: baseline/sdi_const/pop_const/both = cumulative deaths;
    dSDI/dPOP/interaction/dTOTAL = cumulative driver contributions. Here
    ``both`` is the "held constant at anchor" world and ``baseline`` is the
    actual (drivers evolving) world; baseline − both = dSDI + dPOP + interaction.
    """
    y0, y1 = year_range
    g = decomp[(decomp[YEAR] >= y0) & (decomp[YEAR] <= y1)]
    cols = ["baseline", "sdi_const", "pop_const", "both", "dSDI", "dPOP", "dTOTAL", "interaction"]
    out = g.groupby([SCEN, LOC], as_index=False)[cols].sum()
    out["year_range"] = f"{y0}-{y1}"
    return out


def cumulative_table(cum: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Per-super-region cumulative: constant world, actual, driver split, % change."""
    g = cum[(cum[SCEN] == scenario) & (cum[LOC].isin(SUPER_REGIONS))].copy()
    g["region"] = g[LOC].map(SUPER_REGIONS)
    g = g.rename(columns={"both": "constant", "baseline": "actual"})
    g["pct_change"] = 100.0 * (g["actual"] - g["constant"]) / g["constant"].replace(0, np.nan)
    cols = ["region", "constant", "actual", "dSDI", "dPOP", "interaction", "pct_change"]
    return g[cols].sort_values("actual", ascending=False).round(1).reset_index(drop=True)


def plot_waterfall(cum: pd.DataFrame, location_id: int, scenario: str, ax=None,
                   zero_anchor: bool = False, fonts: dict | None = None):
    """Bridge/waterfall from the constant-at-anchor world to the actual world.

    Middle bars (SDI / population / interaction) FLOAT: each is drawn with
    ``bottom = running_total_before_step`` and a signed height, joined by
    connectors at the running level. `constant` is anchored, and the last
    floating step lands exactly on `actual` because ``interaction`` is defined as
    the residual (``dTOTAL − dSDI − dPOP``), so the decomposition closes by
    construction. `actual` is redrawn as a light **checksum** outline (redundant
    but confirms closure).

    ``zero_anchor=False`` (default) zooms the y-axis to the bridge range so the
    steps are legible — the drivers are only ~10-15% of the level, so a
    zero-anchored axis squishes them into a sliver. ``zero_anchor=True`` gives the
    textbook 0-based waterfall.
    """
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    f = _fonts(fonts)
    g = cum[(cum[LOC] == location_id) & (cum[SCEN] == scenario)]
    if g.empty:
        return ax
    r = g.iloc[0]
    constant, actual = float(r["both"]), float(r["baseline"])
    deltas = [("SDI", float(r["dSDI"]), "#1f77b4"),
              ("population", float(r["dPOP"]), "#d62728"),
              ("interaction", float(r["interaction"]), "#999999")]

    levels = [constant]
    for _, v, _ in deltas:
        levels.append(levels[-1] + v)
    resid = actual - levels[-1]           # ~0 by construction; guard against drift
    allv = levels + [actual]
    lo, hi = min(allv), max(allv)
    span = (hi - lo) or (abs(hi) * 0.1 or 1.0)
    pad, fs = 0.045 * span, f["annot"]     # label offset above/below bars, and font size
    ybot = 0.0 if zero_anchor else lo - 0.20 * span
    ytop = hi + 0.22 * span            # headroom so the top total label clears the panel
    labels = ["constant\n(2023 frozen)"] + [d[0] for d in deltas] + ["actual"]
    n = len(labels)

    ax.bar(0, constant - ybot, bottom=ybot, color="0.55", edgecolor="k", linewidth=0.6)
    ax.text(0, constant + pad, f"{constant:,.0f}", ha="center", va="bottom", fontsize=fs)
    run = constant
    for i, (name, v, c) in enumerate(deltas, start=1):
        ax.bar(i, v, bottom=run, color=c, edgecolor="k", linewidth=0.6)
        ytxt = run + v + (pad if v >= 0 else -pad)
        ax.text(i, ytxt, f"{v:+,.0f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=fs, color=c)
        run += v
    # connectors: staircase line joining each running level across the gaps
    for i in range(len(deltas) + 1):
        ax.plot([i + 0.4, i + 1 - 0.4], [levels[i], levels[i]], color="0.4", lw=1.0)

    # actual = thick black level line only (no bar). Closure is exact, so it sits
    # exactly on the last step's top; the connector runs into it.
    ax.plot([n - 1 - 0.42, n - 1 + 0.42], [actual, actual], color="k", lw=3.2,
            solid_capstyle="butt", zorder=5)
    ax.text(n - 1, actual + pad, f"{actual:,.0f}", ha="center", va="bottom", fontsize=fs)
    if abs(resid) > 1e-6 * max(abs(actual), 1.0):   # should never fire (interaction is the residual)
        ax.text(n - 1, actual, f"  (resid {resid:+.1f})", ha="left", va="center",
                fontsize=6, color="red")

    yr = _yr(cum)
    ax.set_ylim(ybot, ytop)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_title(f"{SUPER_REGIONS.get(location_id, location_id)} — {SSP_SCENARIO_MAP[scenario]['name']}"
                 + (f" — cumulative {yr}" if yr else ""))
    ax.set_ylabel(f"cumulative deaths {yr}".rstrip() + ("" if zero_anchor else " (axis zoomed)"))
    ax.grid(alpha=0.2, axis="y")
    _apply_fonts(ax, f)
    return ax


def plot_constant_vs_actual(cum: pd.DataFrame, scenario: str, ax=None, fonts: dict | None = None):
    """Grouped bars per super-region: constant-world vs actual cumulative deaths."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    f = _fonts(fonts)
    g = cum[(cum[SCEN] == scenario) & (cum[LOC].isin(SUPER_REGIONS)) & (cum[LOC] != 1)].copy()
    g["region"] = g[LOC].map(SUPER_REGIONS)
    g = g.sort_values("baseline", ascending=False)
    x = np.arange(len(g)); w = 0.4
    ax.bar(x - w / 2, g["both"], w, label="constant (2023 frozen)", color="0.6")
    ax.bar(x + w / 2, g["baseline"], w, label="actual", color="#1f77b4")
    yr = _yr(cum)
    ax.set_xticks(x); ax.set_xticklabels(g["region"], rotation=35, ha="right")
    ax.set_ylabel(f"cumulative deaths {yr}".rstrip())
    ax.set_title(f"Constant vs actual — {SSP_SCENARIO_MAP[scenario]['name']}" + (f" ({yr})" if yr else ""))
    ax.legend(); ax.grid(alpha=0.2, axis="y")
    _apply_fonts(ax, f)
    return ax


# ---------------------------------------------------------------------------
# "By scenario" views: fix ONE location, one panel / row per SSP (RCP) scenario
# ---------------------------------------------------------------------------

DEFAULT_SCENARIOS = list(SSP_SCENARIO_MAP)   # ['ssp126', 'ssp245', 'ssp585']


def _panels_by_scenario(plot_fn, data, location_id, scenarios, figsize, fonts,
                        suptitle=None, **kw):
    """Build a 1xN figure (one panel per scenario) for a fixed location, reusing
    the single-panel plotter. Panel titles are the RCP name; the location (plus
    any cumulative year range) is the suptitle."""
    import matplotlib.pyplot as plt
    scenarios = scenarios or DEFAULT_SCENARIOS
    f = _fonts(fonts)
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=figsize or (6 * n, 5), squeeze=False)
    for ax, scen in zip(axes.ravel(), scenarios):
        plot_fn(data, location_id, scen, ax=ax, fonts=f, **kw)
        ax.set_title(SSP_SCENARIO_MAP[scen]["name"], fontsize=f["title"])   # region is in the suptitle
    fig.suptitle(suptitle or SUPER_REGIONS.get(location_id, location_id), fontsize=f["suptitle"])
    fig.tight_layout()
    return fig


def plot_levels_by_scenario(sets, location_id, scenarios=None, figsize=None, fonts=None):
    """One location, a panel per scenario: baseline vs frozen-driver levels over time."""
    return _panels_by_scenario(plot_levels, sets, location_id, scenarios, figsize, fonts)


def plot_contributions_by_scenario(decomp, location_id, scenarios=None, figsize=None, fonts=None):
    """One location, a panel per scenario: stacked driver contributions over time."""
    return _panels_by_scenario(plot_contributions, decomp, location_id, scenarios, figsize, fonts)


def plot_waterfall_by_scenario(cum, location_id, scenarios=None, figsize=None,
                               zero_anchor=False, fonts=None):
    """One location, a panel per scenario: cumulative constant→actual waterfall."""
    yr = _yr(cum)
    region = SUPER_REGIONS.get(location_id, location_id)
    suptitle = f"{region} — cumulative {yr}" if yr else region
    return _panels_by_scenario(plot_waterfall, cum, location_id, scenarios, figsize, fonts,
                               suptitle=suptitle, zero_anchor=zero_anchor)


def plot_constant_vs_actual_by_scenario(cum, location_id, scenarios=None, ax=None, fonts=None):
    """One location, grouped bars over the scenarios: constant-world vs actual cumulative deaths."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    f = _fonts(fonts)
    scenarios = scenarios or DEFAULT_SCENARIOS
    labels, const, act = [], [], []
    for scen in scenarios:
        g = cum[(cum[SCEN] == scen) & (cum[LOC] == location_id)]
        if g.empty:
            continue
        r = g.iloc[0]
        labels.append(SSP_SCENARIO_MAP[scen]["name"])
        const.append(float(r["both"])); act.append(float(r["baseline"]))
    yr = _yr(cum)
    x = np.arange(len(labels)); w = 0.4
    ax.bar(x - w / 2, const, w, label="constant (2023 frozen)", color="0.6")
    ax.bar(x + w / 2, act, w, label="actual", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(f"cumulative deaths {yr}".rstrip())
    ax.set_title(f"{SUPER_REGIONS.get(location_id, location_id)} — constant vs actual by scenario"
                 + (f" ({yr})" if yr else ""))
    # bars fill the panel (no clear corner), so put the legend below it
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    ax.grid(alpha=0.2, axis="y")
    _apply_fonts(ax, f)
    return ax


def cumulative_table_by_scenario(cum: pd.DataFrame, location_id: int,
                                 scenarios=None) -> pd.DataFrame:
    """One location, one row per scenario: constant, actual, driver split, % change."""
    scenarios = scenarios or DEFAULT_SCENARIOS
    rows = []
    for scen in scenarios:
        g = cum[(cum[SCEN] == scen) & (cum[LOC] == location_id)]
        if g.empty:
            continue
        r = g.iloc[0]
        constant, actual = float(r["both"]), float(r["baseline"])
        rows.append({"scenario": SSP_SCENARIO_MAP[scen]["name"],
                     "constant": constant, "actual": actual,
                     "dSDI": float(r["dSDI"]), "dPOP": float(r["dPOP"]),
                     "interaction": float(r["interaction"]),
                     "pct_change": 100.0 * (actual - constant) / constant if constant else float("nan")})
    return pd.DataFrame(rows).round(1)


def plot_waterfall_grid(cum: pd.DataFrame, locations=None, scenarios=None,
                        figsize=None, fonts: dict | None = None, zero_anchor: bool = False):
    """Grid of waterfalls: one ROW per location, one COLUMN per scenario.

    Each cell is ``plot_waterfall`` for that (location, scenario). Panel titles are
    ``region — RCP``; the cumulative year range is in the suptitle."""
    import matplotlib.pyplot as plt
    locations = locations or list(SUPER_REGIONS)
    scenarios = scenarios or DEFAULT_SCENARIOS
    f = _fonts(fonts)
    nr, nc = len(locations), len(scenarios)
    fig, axes = plt.subplots(nr, nc, figsize=figsize or (6 * nc, 3.6 * nr), squeeze=False)
    for i, loc in enumerate(locations):
        for j, scen in enumerate(scenarios):
            plot_waterfall(cum, loc, scen, ax=axes[i, j], zero_anchor=zero_anchor, fonts=f)
            axes[i, j].set_title(
                f"{SUPER_REGIONS.get(loc, loc)} — {SSP_SCENARIO_MAP[scen]['name']}",
                fontsize=f["title"])
    yr = _yr(cum)
    fig.suptitle(f"Waterfall by location × scenario" + (f" — cumulative {yr}" if yr else ""),
                 fontsize=f["suptitle"])
    fig.tight_layout()
    return fig


def plot_constant_vs_actual_grid(cum: pd.DataFrame, locations=None, scenarios=None,
                                 ncols: int = 4, figsize=None, fonts: dict | None = None):
    """Grid of constant-vs-actual-by-scenario bars: one PANEL per location.

    Each panel is ``plot_constant_vs_actual_by_scenario`` for that location; the
    per-panel legends are replaced by one shared legend below the figure."""
    import matplotlib.pyplot as plt
    locations = locations or list(SUPER_REGIONS)
    f = _fonts(fonts)
    nr = int(np.ceil(len(locations) / ncols))
    fig, axes = plt.subplots(nr, ncols, figsize=figsize or (5 * ncols, 4 * nr), squeeze=False)
    axf = axes.ravel()
    shared = None
    for ax, loc in zip(axf, locations):
        plot_constant_vs_actual_by_scenario(cum, loc, scenarios=scenarios, ax=ax, fonts=f)
        ax.set_title(SUPER_REGIONS.get(loc, loc), fontsize=f["title"])   # region only; rest in suptitle
        if shared is None:
            shared = ax.get_legend_handles_labels()
        lg = ax.get_legend()
        if lg is not None:
            lg.remove()
    for ax in axf[len(locations):]:
        ax.set_visible(False)
    if shared and shared[0]:
        fig.legend(shared[0], shared[1], loc="lower center", ncol=2, fontsize=f["legend"])
    yr = _yr(cum)
    fig.suptitle("Constant vs actual by scenario" + (f" — cumulative {yr}" if yr else ""),
                 fontsize=f["suptitle"])
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    return fig


def plot_actual_vs_constant_scenarios(sets: dict[str, pd.DataFrame], location_id: int,
                                      scenarios=None, ax=None, fonts: dict | None = None):
    """One location: for each scenario, the actual (drivers evolving, SOLID) and the
    both-frozen constant (DASHED) death trajectories, colored by scenario (RCP).

    2 x len(scenarios) lines per panel — color encodes scenario, linestyle encodes
    actual vs held-constant. Loop over locations to build the panel grid."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ax = ax or plt.gca()
    f = _fonts(fonts)
    scenarios = scenarios or DEFAULT_SCENARIOS
    for scen in scenarios:
        color = SSP_SCENARIO_MAP[scen]["color"]
        for key, ls in (("baseline", "-"), ("both", "--")):
            g = sets[key][(sets[key][LOC] == location_id) & (sets[key][SCEN] == scen)].sort_values(YEAR)
            if not g.empty:
                ax.plot(g[YEAR], g[VAL], color=color, ls=ls, lw=1.6)
    handles = [Line2D([0], [0], color=SSP_SCENARIO_MAP[s]["color"], lw=2.2,
                      label=SSP_SCENARIO_MAP[s]["name"]) for s in scenarios]
    handles += [Line2D([0], [0], color="0.3", ls="-", lw=1.6, label="actual"),
                Line2D([0], [0], color="0.3", ls="--", lw=1.6, label="both frozen (constant)")]
    ax.legend(handles=handles, ncol=1)
    ax.set_title(SUPER_REGIONS.get(location_id, location_id))
    ax.set_xlabel("year"); ax.set_ylabel("mean deaths/yr"); ax.grid(alpha=0.25)
    _apply_fonts(ax, f)
    return ax
