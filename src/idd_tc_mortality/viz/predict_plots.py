"""Predict-pipeline scenario plots.

Standalone consumer of the postprocess outputs (16 `deaths_*_draws.parquet`
files + `population.parquet` + `observed_deaths.parquet`). No data creation
here — this module reads existing artifacts and renders one PNG per
(location, deaths_col, space, mode, include_observed) tuple.

Public surface:
  - `SSP_SCENARIO_MAP` — color + display-name per SSP.
  - `TOGGLE_LABELS`    — semantic names for the c/s/o/b toggle codes.
  - `decode_toggle`    — pretty-format a `deaths_c{c}_s{s}_o{o}_b{b}` column.
  - `safe_filename`    — strip filename-hostile characters from a location_name.
  - `rate_ylabel`      — y-axis label for rate plots.
  - `plot_scenario_panel` — the main plotting function.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — colors, names, semantic labels
# ---------------------------------------------------------------------------

SSP_SCENARIO_MAP: dict[str, dict] = {
    'ssp126': {'name': 'RCP2.6', 'color': '#046C9A'},
    'ssp245': {'name': 'RCP4.5', 'color': '#E58601'},
    'ssp585': {'name': 'RCP8.5', 'color': '#A42820'},
}

# Toggle semantics from scripts/build_topsis_draws.py:
#   c -> draw_coefs (False = point estimate, True = sampled coefficients)
#   s -> draw_scale (False = point estimate, True = sampled scale parameter)
#   o -> outcome_draw (False = expected outcome, True = stochastic draw)
#   b -> expected_bernoulli (False = realized hurdle, True = expected hurdle)
TOGGLE_LABELS: dict[str, dict] = {
    'c': {'name': 'coefs',     0: 'point-est', 1: 'sampled'},
    's': {'name': 'scale',     0: 'point-est', 1: 'sampled'},
    'o': {'name': 'outcome',   0: 'expected',  1: 'stochastic'},
    'b': {'name': 'bernoulli', 0: 'realized',  1: 'expected'},
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def decode_toggle(deaths_col: str) -> str:
    """deaths_c0_s1_o1_b0 -> 'coefs=point-est, scale=sampled, outcome=stochastic, bernoulli=realized'."""
    parts = deaths_col.replace('deaths_', '').split('_')
    out = []
    for p in parts:
        key = p[0]
        val = int(p[1:])
        lbl = TOGGLE_LABELS[key]
        out.append(f"{lbl['name']}={lbl[val]}")
    return ', '.join(out)


def safe_filename(s: str) -> str:
    """Sanitize a location_name for use as a filename stem."""
    return s.replace(' ', '_').replace('/', '_').replace(',', '').replace("'", '')


def rate_ylabel(rate_per: int) -> str:
    """Y-axis label for rate plots — defaults to GBD convention of per 100,000."""
    if rate_per == 1:
        return 'Mean death rate (deaths per person, 95% UI)'
    return f'Mean death rate (per {rate_per:,}, 95% UI)'


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_scenario_panel(
    *,
    location_id: int,
    location_name: str,
    deaths_col: str,
    draws_dir: Path,
    fig_dir: Path,
    pop_df: pd.DataFrame | None = None,
    obs_df: pd.DataFrame | None = None,
    space: str = 'count',
    mode: str = 'full',
    include_observed: bool = False,
    future_start_year: int = 2023,
    rate_per: int = 100_000,
    figsize: tuple[float, float] = (9, 5),
    dpi: int = 120,
) -> Path:
    """Single-panel plot: historical + 3 SSPs, mean line + 95% UI band.

    Required:
        location_id, location_name : the location to plot.
        deaths_col                 : the toggle column (one of the 16).
        draws_dir                  : directory containing `<deaths_col>_draws.parquet`.
        fig_dir                    : where to save the PNG.

    Optional (with defaults):
        pop_df            : (location_id, year, population) — required for space='rate'.
        obs_df            : (location_id, year, deaths) — required if include_observed=True.
        space             : 'count' or 'rate'.
        mode              : 'full' (historical line/band + SSPs from 2015) or
                            'future' (no historical, SSPs from `future_start_year`).
        include_observed  : overlay observed-deaths dots in matching units.
        future_start_year : start of the SSP slice in 'future' mode.
        rate_per          : rate denominator (default 100,000 = GBD convention).

    Returns:
        Path to the saved PNG. Filename:
            <fig_dir>/<safe_name>__<deaths_col>__<space>_<obs|noobs>_<full|future>.png
    """
    if space not in ('count', 'rate'):
        raise ValueError(f"space must be 'count' or 'rate', got {space!r}")
    if mode not in ('full', 'future'):
        raise ValueError(f"mode must be 'full' or 'future', got {mode!r}")
    if space == 'rate' and pop_df is None:
        raise ValueError("pop_df is required when space='rate'")
    if include_observed and obs_df is None:
        raise ValueError("obs_df is required when include_observed=True")

    df = pd.read_parquet(
        Path(draws_dir) / f'{deaths_col}_draws.parquet',
        columns=['scenario', 'location_id', 'year', 'mean', 'lower', 'upper'],
    )
    sub = df[df['location_id'] == location_id].copy()

    if space == 'rate':
        sub = sub.merge(pop_df, on=['location_id', 'year'], how='left')
        for c in ('mean', 'lower', 'upper'):
            sub[c] = sub[c] / sub['population'] * rate_per
        ylabel = rate_ylabel(rate_per)
    else:
        ylabel = 'Mean deaths (95% UI)'

    fig, ax = plt.subplots(figsize=figsize)

    # Historical scenario line/band — only in 'full' mode.
    if mode == 'full':
        h = sub[sub['scenario'] == 'historical'].sort_values('year')
        if not h.empty:
            ax.plot(h['year'], h['mean'], color='black', label='Historical', lw=1.5)
            ax.fill_between(h['year'], h['lower'], h['upper'],
                            color='black', alpha=0.2, linewidth=0)

    # SSP lines/bands — in 'future' mode, trim to year >= future_start_year.
    for ssp, info in SSP_SCENARIO_MAP.items():
        s = sub[sub['scenario'] == ssp].sort_values('year')
        if mode == 'future':
            s = s[s['year'] >= future_start_year]
        if s.empty:
            continue
        ax.plot(s['year'], s['mean'], color=info['color'], label=info['name'], lw=1.5)
        ax.fill_between(s['year'], s['lower'], s['upper'],
                        color=info['color'], alpha=0.2, linewidth=0)

    # Observed dots (always full range when included).
    if include_observed:
        obs_loc = obs_df[obs_df['location_id'] == location_id].copy()
        if not obs_loc.empty:
            if space == 'rate':
                obs_loc = obs_loc.merge(pop_df, on=['location_id', 'year'], how='left')
                obs_y = obs_loc['deaths'] / obs_loc['population'] * rate_per
            else:
                obs_y = obs_loc['deaths']
            ax.scatter(
                obs_loc['year'], obs_y,
                color='black', s=18, zorder=5,
                edgecolor='white', linewidths=0.6,
                label='Observed',
            )

    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{location_name}  [{space}, {mode}]\n{decode_toggle(deaths_col)}')
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.grid(alpha=0.3)
    ax.margins(x=0.01)
    fig.tight_layout()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    obs_tag = 'obs' if include_observed else 'noobs'
    out_path = fig_dir / f'{safe_filename(location_name)}__{deaths_col}__{space}_{obs_tag}_{mode}.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path
