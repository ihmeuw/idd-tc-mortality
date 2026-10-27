"""Rank-based match measures between observed and model series, raw and detrended, for every
start year.

A *cell* is one (domain, source, level, geo, outcome): an observed annual series and the
model's 800 run × draw annual series. From the draws come the model series (per run, per GCM,
ensemble of the 4 GCMs, pooled over the 8 runs; each in a mean pipeline and a median
pipeline). For every start year X the series are cut to X..end, optionally linearly
detrended (annual, before smoothing), then windowed with a centred moving average (valid
points only). Measures per (start, model series, window, detrend): Spearman, Kendall τ,
top-tercile hit rate, up/down agreement between consecutive periods, model/observed ratio
of period totals (raw only), the observed value's percentile within the 800 windowed draws
(mean over periods and share outside the central 90%), and the chance band: Spearman of
the model series against ``N_SURROGATES`` phase-randomized surrogates of the observed
annual series (same power spectrum, random phases, smoothed identically) — its 2.5 / 97.5
percentiles, the percentile of the real ρ within it and the two-sided p-value. ``n`` is the
number of periods; ``n_eff`` is Bretherton et al. (1999): n (1 − r1 r2) / (1 + r1 r2) with
r1, r2 the lag-1 autocorrelations of the two windowed series. The *ceiling* table holds the
run-vs-run Spearman spread at the same (start, window, detrend): the agreement expected if
the observed record behaved like one more run.
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy import stats as sps

from idd_tc_mortality.viz.cmip6_ibtracs_comparison.constants import (
    BASINS,
    GCMS,
    GLOBAL_ID,
    MIN_OBS_LAND_STORMS,
    MORTALITY_STARTS,
    MORTALITY_WINDOWS,
    MORTALITY_YEARS,
    N_DRAWS,
    N_SURROGATES,
    OUTCOMES,
    RUNS,
    STORM_STARTS,
    STORM_WINDOWS,
    STORM_YEARS,
    SURROGATE_SEED,
    gcm_of,
    geo_label,
    load_hierarchy,
    outcome_counts,
)
from idd_tc_mortality.viz.cmip6_ibtracs_comparison.io import write_parquet_checked

QUANTILES = (5, 25, 50, 75, 95)
METRIC_COLS = ["n", "n_eff", "spearman", "kendall", "hit_rate", "updown", "ratio", "pct_mean", "pct_out90",
               "sur_lo", "sur_hi", "sur_pct", "sur_p"]
CEIL_COLS = ["n_pairs", "ceil_min", "ceil_q25", "ceil_med", "ceil_q75", "ceil_max",
             "xg_min", "xg_med", "xg_max", "obsrun_min", "obsrun_med", "obsrun_max"]
MORTALITY_TAGS = {"adjusted": "deaths_adj", "unadjusted": "deaths_unadj"}
# raw bands do not depend on the start year (a moving average of the raw series is the same
# whatever the first year), so they are stored once under this start_year
RAW_BAND_START = 0


# --- window transforms (numpy, last axis = years) -----------------------------------------
def detrend(y: np.ndarray) -> np.ndarray:
    """Remove the least-squares line (slope and mean) from each series along the last axis."""
    y = np.asarray(y, dtype=float)
    n = y.shape[-1]
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    slope = (y * tc).sum(axis=-1, keepdims=True) / (tc ** 2).sum()
    return y - y.mean(axis=-1, keepdims=True) - slope * tc


def moving_average(y: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average, full windows only: ``n - w + 1`` points."""
    y = np.asarray(y, dtype=float)
    if w == 1:
        return y.copy()
    if y.shape[-1] < w:
        return y[..., :0]
    c = np.cumsum(np.concatenate([np.zeros(y.shape[:-1] + (1,)), y], axis=-1), axis=-1)
    return (c[..., w:] - c[..., :-w]) / w


def window_spec(window: str) -> tuple[str, int]:
    """``'ma5'`` → ``('ma', 5)``. Only moving averages exist (blocks were dropped)."""
    if window.startswith("ma"):
        return "ma", int(window[2:])
    raise ValueError(f"unsupported window {window!r}: moving averages only")


def apply_window(y: np.ndarray, window: str) -> np.ndarray:
    _, w = window_spec(window)
    return moving_average(y, w)


def period_bounds(years: np.ndarray, window: str) -> tuple[np.ndarray, np.ndarray]:
    """(start, end) year of each window point."""
    years = np.asarray(years)
    _, w = window_spec(window)
    n = max(len(years) - w + 1, 0)
    return years[:n], years[w - 1:w - 1 + n]


def lag1_autocorr(y) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) < 3 or np.std(y) == 0:
        return np.nan
    d = y - y.mean()
    return float((d[:-1] * d[1:]).sum() / (d * d).sum())


def n_eff_bretherton(x, y) -> float:
    """Bretherton et al. (1999) effective sample size of two autocorrelated series,
    n (1 − r1 r2) / (1 + r1 r2), clipped to [1, n]; n itself when a series is degenerate."""
    n = len(x)
    if n < 3 or len(y) != n:
        return np.nan
    r1, r2 = lag1_autocorr(x), lag1_autocorr(y)
    if not (np.isfinite(r1) and np.isfinite(r2)):
        return float(n)
    f = (1.0 - r1 * r2) / (1.0 + r1 * r2)
    return float(min(float(n), max(1.0, n * f)))


# --- measures ---------------------------------------------------------------------------
def _degenerate(x, y) -> bool:
    return len(x) < 3 or len(x) != len(y) or np.std(x) == 0 or np.std(y) == 0


def spearman(x, y) -> float:
    if _degenerate(x, y):
        return np.nan
    return float(sps.spearmanr(x, y).statistic)


def kendall(x, y) -> float:
    if _degenerate(x, y):
        return np.nan
    return float(sps.kendalltau(x, y).statistic)


def hit_rate(x, y) -> float:
    """Share of the observed top-tercile periods that are also model top-tercile periods."""
    if _degenerate(x, y):
        return np.nan
    k = max(1, round(len(x) / 3))
    top_x = set(np.argsort(-np.asarray(x), kind="stable")[:k])
    top_y = set(np.argsort(-np.asarray(y), kind="stable")[:k])
    return len(top_x & top_y) / k


def updown(x, y) -> float:
    """Share of consecutive-period changes with the same sign in both series."""
    if len(x) < 2 or len(x) != len(y):
        return np.nan
    return float(np.mean(np.sign(np.diff(x)) == np.sign(np.diff(y))))


def ratio(x, y) -> float:
    """Model / observed ratio of period totals."""
    sx = float(np.sum(x))
    return float(np.sum(y)) / sx if sx > 0 else np.nan


def percentile_of(obs: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Per period: fraction of draws below the observed value (ties count half)."""
    obs = np.asarray(obs, dtype=float)
    draws = np.asarray(draws, dtype=float)
    if draws.shape[-1] == 0:
        return obs[:0]
    return (draws < obs).mean(axis=0) + 0.5 * (draws == obs).mean(axis=0)


def pair_metrics(x: np.ndarray, y: np.ndarray, window: str, detrended: bool,
                 x_raw: np.ndarray | None = None, y_raw: np.ndarray | None = None) -> dict:
    """All pairwise measures; ``ratio`` uses the raw windowed series and is NaN when detrended."""
    return {
        "n": len(x), "n_eff": n_eff_bretherton(x, y),
        "spearman": spearman(x, y), "kendall": kendall(x, y),
        "hit_rate": hit_rate(x, y), "updown": updown(x, y),
        "ratio": np.nan if detrended else ratio(x_raw if x_raw is not None else x,
                                                y_raw if y_raw is not None else y),
    }


# --- surrogates and vectorised Spearman -----------------------------------------------------
def phase_randomize(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` phase-randomized surrogates of ``y`` [n, T]: same Fourier amplitudes (so the same
    power spectrum and autocorrelation) with uniformly random phases; the mean is kept."""
    y = np.asarray(y, dtype=float)
    t = len(y)
    if t < 3:
        return np.tile(y, (n, 1))
    mu = y.mean()
    amp = np.abs(np.fft.rfft(y - mu))
    ph = rng.uniform(0.0, 2.0 * np.pi, size=(n, len(amp)))
    ph[:, 0] = 0.0
    if t % 2 == 0:
        ph[:, -1] = 0.0           # Nyquist component stays real
    return np.fft.irfft(amp * np.exp(1j * ph), n=t) + mu


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman ρ between every row of ``a`` [S, T] and every row of ``b`` [M, T] → [S, M]
    (Pearson of average ranks; NaN where a row is constant or T < 3)."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    if a.shape[-1] < 3 or a.shape[-1] != b.shape[-1]:
        return np.full((a.shape[0], b.shape[0]), np.nan)
    ra = sps.rankdata(a, axis=-1)
    rb = sps.rankdata(b, axis=-1)
    ra = ra - ra.mean(axis=-1, keepdims=True)
    rb = rb - rb.mean(axis=-1, keepdims=True)
    na = np.sqrt((ra ** 2).sum(axis=-1))
    nb = np.sqrt((rb ** 2).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (ra @ rb.T) / (na[:, None] * nb[None, :])
    out[~np.isfinite(out)] = np.nan
    return out


def surrogate_summary(rho_sur: np.ndarray, rho_real: float) -> dict:
    """2.5 / 97.5 percentiles of the surrogate ρ, the percentile of the real ρ within them
    (ties count half) and the two-sided p-value."""
    v = np.asarray(rho_sur, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 10 or rho_real is None or not np.isfinite(rho_real):
        return {"sur_lo": np.nan, "sur_hi": np.nan, "sur_pct": np.nan, "sur_p": np.nan}
    lo, hi = np.percentile(v, [2.5, 97.5])
    pct = float((v < rho_real).mean() + 0.5 * (v == rho_real).mean())
    return {"sur_lo": float(lo), "sur_hi": float(hi), "sur_pct": pct, "sur_p": float(2.0 * min(pct, 1.0 - pct))}


# --- draw summaries ---------------------------------------------------------------------
def _stat(arr: np.ndarray, stat: str) -> np.ndarray:
    return arr.mean(axis=0) if stat == "mean" else np.median(arr, axis=0)


def summarize_draws(draws: np.ndarray, runs: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Model series from ``draws`` [n_draws, n_years] labelled by ``runs`` [n_draws].

    Mean pipeline: mean over draws → mean over runs within GCM → mean over GCMs
    (``ens``) / mean over runs (``pooled``). Median pipeline: the same with medians.
    """
    runs = np.asarray(runs)
    out: dict[str, dict[str, np.ndarray]] = {}
    for stat in ("mean", "median"):
        run_series = {r: _stat(draws[runs == r], stat) for r in RUNS if (runs == r).any()}
        gcm_series = {g: _stat(np.stack([v for r, v in run_series.items() if gcm_of(r) == g]), stat)
                      for g in GCMS if any(gcm_of(r) == g for r in run_series)}
        for r, v in run_series.items():
            out.setdefault(f"run:{r}", {})[stat] = v
        for g, v in gcm_series.items():
            out.setdefault(f"gcm:{g}", {})[stat] = v
        out.setdefault("ens", {})[stat] = _stat(np.stack(list(gcm_series.values())), stat)
        out.setdefault("pooled", {})[stat] = _stat(np.stack(list(run_series.values())), stat)
    return out


# --- ceiling ----------------------------------------------------------------------------
def ceiling_summary(run_w: dict[str, np.ndarray], obs_w: np.ndarray) -> dict:
    """Run-vs-run Spearman spread (all pairs and cross-GCM pairs) and the spread of
    observed-vs-run ρ, for one (start, window, detrend, stat)."""
    names = [r for r in RUNS if r in run_w]
    nan = {c: np.nan for c in CEIL_COLS}
    if len(names) < 2:
        return nan
    m = np.stack([run_w[r] for r in names])
    if m.shape[-1] < 3:
        return nan
    r = spearman_rows(m, m)
    iu = np.triu_indices(len(names), 1)
    vals = r[iu]
    xg = np.array([r[i, j] for i, j in zip(*iu) if gcm_of(names[i]) != gcm_of(names[j])])
    obsr = spearman_rows(obs_w[None, :], m)[0]
    vals, xg, obsr = vals[np.isfinite(vals)], xg[np.isfinite(xg)], obsr[np.isfinite(obsr)]
    if len(vals) == 0:
        return nan
    q = np.percentile(vals, [0, 25, 50, 75, 100])
    return {"n_pairs": len(vals), "ceil_min": q[0], "ceil_q25": q[1], "ceil_med": q[2], "ceil_q75": q[3], "ceil_max": q[4],
            "xg_min": xg.min() if len(xg) else np.nan, "xg_med": float(np.median(xg)) if len(xg) else np.nan,
            "xg_max": xg.max() if len(xg) else np.nan,
            "obsrun_min": obsr.min() if len(obsr) else np.nan, "obsrun_med": float(np.median(obsr)) if len(obsr) else np.nan,
            "obsrun_max": obsr.max() if len(obsr) else np.nan}


# --- cell -------------------------------------------------------------------------------
def _band_rows(key: dict, band: str, window: str, detrended: bool, start_year: int, years, xo, dw) -> list[dict]:
    start, end = period_bounds(years, window)
    if len(start) == 0:
        return []
    q = np.percentile(dw, QUANTILES, axis=0) if dw.shape[0] else np.full((len(QUANTILES), len(xo)), np.nan)
    pct = percentile_of(xo, dw)
    return [dict(key, band=band, window=window, detrended=detrended, start_year=int(start_year),
                 start=int(s), end=int(e), q05=q[0, i], q25=q[1, i], q50=q[2, i], q75=q[3, i], q95=q[4, i],
                 obs=float(xo[i]), obs_pct=float(pct[i]))
            for i, (s, e) in enumerate(zip(start, end))]


def _pct_summary(pct: np.ndarray) -> tuple[float, float]:
    if len(pct) == 0:
        return np.nan, np.nan
    return float(pct.mean()), float(((pct < 0.05) | (pct > 0.95)).mean())


def _identity(a):
    return np.asarray(a, dtype=float)


def cell_stats(key: dict, obs: np.ndarray, draws: np.ndarray, runs: np.ndarray, years: np.ndarray,
               windows: tuple[str, ...], starts: tuple[int, ...], ibt: np.ndarray | None = None,
               rng: np.random.Generator | None = None, n_sur: int = N_SURROGATES) -> tuple[list, list, list, list]:
    """Stats rows, band rows, annual series rows and ceiling rows for one cell.

    ``starts`` are the start years to evaluate (those where the cell passes the sparse-place
    rule). ``ibt`` (mortality only) is the [n_fit_draws, n_years] prediction on IBTrACS
    storms; it adds the ``obs_vs_ibt`` and ``ibt_vs_model`` pairs and an ``ibt`` band.
    """
    rng = rng or np.random.default_rng(SURROGATE_SEED)
    years = np.asarray(years)
    obs = np.asarray(obs, dtype=float)
    draws = np.asarray(draws, dtype=float)
    series = summarize_draws(draws, runs)
    sids = [(sid, st) for sid in series for st in ("mean", "median")]
    series_rows = [dict(key, series="obs", stat="raw", year=int(y), value=float(v)) for y, v in zip(years, obs)]
    for sid, d in series.items():
        for stat, v in d.items():
            series_rows += [dict(key, series=sid, stat=stat, year=int(y), value=float(x)) for y, x in zip(years, v)]
    ibt_series = None
    if ibt is not None:
        ibt = np.asarray(ibt, dtype=float)
        ibt_series = {"mean": ibt.mean(axis=0), "median": np.median(ibt, axis=0)}
        for stat, v in ibt_series.items():
            series_rows += [dict(key, series="ibt", stat=stat, year=int(y), value=float(x)) for y, x in zip(years, v)]

    stats_rows, band_rows, ceil_rows = [], [], []
    # raw bands once, over the full period
    for window in windows:
        xo = apply_window(obs, window)
        band_rows += _band_rows(key, "model", window, False, RAW_BAND_START, years, xo, apply_window(draws, window))
        if ibt is not None:
            band_rows += _band_rows(key, "ibt", window, False, RAW_BAND_START, years, xo, apply_window(ibt, window))

    for start in starts:
        sel = years >= start
        if sel.sum() < 3:
            continue
        ys = years[sel]
        obs_s, draws_s = obs[sel], draws[:, sel]
        ser_s = {sid: {st: v[sel] for st, v in d.items()} for sid, d in series.items()}
        sur_obs = phase_randomize(obs_s, n_sur, rng)
        ibt_s = ibt[:, sel] if ibt is not None else None
        ibt_ser_s = {st: v[sel] for st, v in ibt_series.items()} if ibt_series else None
        sur_ibt = {st: phase_randomize(v, n_sur, rng) for st, v in ibt_ser_s.items()} if ibt_ser_s else None
        for detrended in (False, True):
            tf = detrend if detrended else _identity
            obs_a, draws_a, sur_a = tf(obs_s), tf(draws_s), tf(sur_obs)
            ser_a = {sid: {st: tf(v) for st, v in d.items()} for sid, d in ser_s.items()}
            for window in windows:
                xo, xo_raw = apply_window(obs_a, window), apply_window(obs_s, window)
                if len(xo) == 0:
                    continue
                dw = apply_window(draws_a, window)
                if detrended:
                    band_rows += _band_rows(key, "model", window, True, start, ys, xo, dw)
                pct_mean, pct_out = _pct_summary(percentile_of(xo, dw))
                ym = np.stack([apply_window(ser_a[sid][st], window) for sid, st in sids])
                ym_raw = np.stack([apply_window(ser_s[sid][st], window) for sid, st in sids])
                rho_sur = spearman_rows(apply_window(sur_a, window), ym)          # [n_sur, n_series]
                for j, (sid, st) in enumerate(sids):
                    m = pair_metrics(xo, ym[j], window, detrended, xo_raw, ym_raw[j])
                    m.update(surrogate_summary(rho_sur[:, j], m["spearman"]))
                    stats_rows.append(dict(key, start_year=int(start), pair="obs_vs_model", series=sid, stat=st,
                                           window=window, detrended=detrended, pct_mean=pct_mean, pct_out90=pct_out, **m))
                for st in ("mean", "median"):
                    run_w = {sid[4:]: ym[j] for j, (sid, s2) in enumerate(sids) if sid.startswith("run:") and s2 == st}
                    ceil_rows.append(dict(key, start_year=int(start), window=window, detrended=detrended, stat=st,
                                          **ceiling_summary(run_w, xo)))
                if ibt_s is None:
                    continue
                iw = apply_window(tf(ibt_s), window)
                if detrended:
                    band_rows += _band_rows(key, "ibt", window, True, start, ys, xo, iw)
                ipct_mean, ipct_out = _pct_summary(percentile_of(xo, iw))
                for st in ("mean", "median"):
                    xi = apply_window(tf(ibt_ser_s[st]), window)
                    xi_raw = apply_window(ibt_ser_s[st], window)
                    m = pair_metrics(xo, xi, window, detrended, xo_raw, xi_raw)
                    m.update(surrogate_summary(spearman_rows(apply_window(sur_a, window), xi[None, :])[:, 0], m["spearman"]))
                    stats_rows.append(dict(key, start_year=int(start), pair="obs_vs_ibt", series="ibt", stat=st,
                                           window=window, detrended=detrended, pct_mean=ipct_mean, pct_out90=ipct_out, **m))
                    # IBTrACS-predicted as the reference: hazard error alone; surrogates of that series
                    js = [j for j, (sid, s2) in enumerate(sids) if s2 == st]
                    rho_sur2 = spearman_rows(apply_window(tf(sur_ibt[st]), window), ym[js])
                    p2_mean, p2_out = _pct_summary(percentile_of(xi, dw))
                    for jj, j in enumerate(js):
                        m2 = pair_metrics(xi, ym[j], window, detrended, xi_raw, ym_raw[j])
                        m2.update(surrogate_summary(rho_sur2[:, jj], m2["spearman"]))
                        stats_rows.append(dict(key, start_year=int(start), pair="ibt_vs_model", series=sids[j][0], stat=st,
                                               window=window, detrended=detrended, pct_mean=p2_mean, pct_out90=p2_out, **m2))
    return stats_rows, band_rows, series_rows, ceil_rows


# --- geography universe -----------------------------------------------------------------
def geo_universe(level: str, hier: pd.DataFrame, country_ids: list[int]) -> list[str]:
    if level == "global":
        return [str(GLOBAL_ID)]
    if level == "basin":
        return list(BASINS)
    if level == "super_region":
        return [str(i) for i in sorted(hier.loc[hier["level"] == 1, "location_id"])]
    if level == "region":
        return [str(i) for i in sorted(hier.loc[hier["level"] == 2, "location_id"])]
    if level == "country":
        return [str(i) for i in sorted(country_ids)]
    raise ValueError(level)


def kept_geos(obs_land: pd.DataFrame, years: tuple[int, int], hier: pd.DataFrame, country_ids: list[int],
              min_storms: int = MIN_OBS_LAND_STORMS) -> pd.DataFrame:
    """Per (level, geo): observed land-hitting storms in ``years`` and whether the geo is kept."""
    sub = outcome_counts(obs_land[obs_land["year"].between(*years)])
    n_obs = sub.groupby(["level", "geo"])["total"].sum()
    rows = []
    for level in ("global", "basin", "super_region", "region", "country"):
        for geo in geo_universe(level, hier, country_ids):
            n = int(n_obs.get((level, geo), 0))
            rows.append({"level": level, "geo": geo, "name": geo_label(level, geo, hier),
                         "n_obs_land": n, "kept": level == "global" or n >= min_storms})
    return pd.DataFrame(rows)


def kept_geos_by_start(obs_land: pd.DataFrame, starts: tuple[int, ...], end: int, hier: pd.DataFrame,
                       country_ids: list[int], min_storms: int = MIN_OBS_LAND_STORMS) -> pd.DataFrame:
    """``kept_geos`` for every start year, stacked with a ``start_year`` column."""
    return pd.concat([kept_geos(obs_land, (s, end), hier, country_ids, min_storms).assign(start_year=int(s))
                      for s in starts], ignore_index=True)


# --- densification ----------------------------------------------------------------------
def _check_draws(counts: pd.DataFrame) -> None:
    lo, hi = int(counts["draw"].min()), int(counts["draw"].max())
    if lo != 0 or hi != N_DRAWS - 1:
        raise RuntimeError(f"draw ids {lo}..{hi}, expected 0..{N_DRAWS - 1}")


def densify_model_counts(counts: pd.DataFrame, level: str, geos: list[str], years: np.ndarray) -> np.ndarray:
    """[n_geo, n_runs*N_DRAWS, n_years, n_outcomes] float32 (zeros where no storm)."""
    sub = counts[(counts["level"] == level) & counts["geo"].isin(geos)]
    _check_draws(counts)
    oc = outcome_counts(sub)
    # Categorical codes are int8/int16: cast before any arithmetic (run index × 100 overflows int8)
    gi = pd.Categorical(sub["geo"], categories=geos).codes.astype(np.int64)
    ri = pd.Categorical(sub["run"], categories=list(RUNS)).codes.astype(np.int64)
    if (ri < 0).any():
        raise RuntimeError("unknown run label in counts")
    rd = ri * N_DRAWS + sub["draw"].to_numpy().astype(np.int64)
    yi = sub["year"].to_numpy().astype(np.int64) - years[0]
    arr = np.zeros((len(geos), len(RUNS) * N_DRAWS, len(years), len(OUTCOMES)), dtype=np.float32)
    arr[gi, rd, yi, :] = oc[list(OUTCOMES)].to_numpy(dtype=np.float32)
    return arr


def densify_obs_counts(counts: pd.DataFrame, level: str, geos: list[str], years: np.ndarray) -> np.ndarray:
    """[n_geo, n_years, n_outcomes]."""
    sub = counts[(counts["level"] == level) & counts["geo"].isin(geos) & counts["year"].between(years[0], years[-1])]
    oc = outcome_counts(sub)
    gi = pd.Categorical(sub["geo"], categories=geos).codes.astype(np.int64)
    yi = sub["year"].to_numpy().astype(np.int64) - years[0]
    arr = np.zeros((len(geos), len(years), len(OUTCOMES)), dtype=np.float32)
    arr[gi, yi, :] = oc[list(OUTCOMES)].to_numpy(dtype=np.float32)
    return arr


def deaths_by_level(df: pd.DataFrame, level: str, geos: list[str], keys: list[str]) -> pd.DataFrame:
    """Deaths per (keys, geo): basin level = Global rows split by basin; others sum over basins."""
    if level == "basin":
        sub = df[df["location_id"] == GLOBAL_ID].copy()
        sub["geo"] = sub["basin"].astype(str)
    else:
        ids = [int(g) for g in geos]
        sub = df[df["location_id"].isin(ids)].copy()
        sub["geo"] = sub["location_id"].astype(str)
    sub = sub[sub["geo"].isin(geos)]
    vals = [c for c in ("deaths", "deaths_unadj", "deaths_adj") if c in sub.columns]
    return sub.groupby(keys + ["geo"], as_index=False, observed=True)[vals].sum()


def densify_deaths(df: pd.DataFrame, level: str, geos: list[str], years: np.ndarray, value: str,
                   draw_keys: list[str]) -> np.ndarray:
    """[n_geo, n_draw_combos, n_years] from a per-draw deaths frame (zeros where absent)."""
    sub = deaths_by_level(df, level, geos, draw_keys + ["year"])
    sub = sub[sub["year"].between(years[0], years[-1])]
    gi = pd.Categorical(sub["geo"], categories=geos).codes.astype(np.int64)
    yi = sub["year"].to_numpy().astype(np.int64) - years[0]
    if draw_keys == ["run", "draw"]:
        ri = pd.Categorical(sub["run"], categories=list(RUNS)).codes.astype(np.int64)
        if (ri < 0).any():
            raise RuntimeError("unknown run label in deaths")
        di = ri * N_DRAWS + sub["draw"].to_numpy().astype(np.int64)
        nd = len(RUNS) * N_DRAWS
    elif draw_keys == ["draw"]:
        draws = np.sort(df["draw"].unique())
        di = pd.Categorical(sub["draw"], categories=draws).codes.astype(np.int64)
        nd = len(draws)
    else:
        di, nd = np.zeros(len(sub), dtype=int), 1
    arr = np.zeros((len(geos), nd, len(years)), dtype=np.float64)
    arr[gi, di, yi] = sub[value].to_numpy(dtype=float)
    return arr


# --- stage ------------------------------------------------------------------------------
def _country_ids(obs_land: pd.DataFrame, model_land: pd.DataFrame) -> list[int]:
    ids = set(obs_land.loc[obs_land["level"] == "country", "geo"].astype(int))
    ids |= set(model_land.loc[model_land["level"] == "country", "geo"].astype(int))
    return sorted(ids)


def _starts_kept(g: pd.DataFrame, geo: str) -> tuple[int, ...]:
    return tuple(int(s) for s in sorted(g.loc[(g["geo"] == geo) & g["kept"], "start_year"]))


def run_stats(work_dir: Path, storm_years=STORM_YEARS, mortality_years=MORTALITY_YEARS,
              storm_only: bool = False, n_sur: int = N_SURROGATES) -> None:
    work_dir = Path(work_dir)
    out_dir = work_dir / "stats"
    hier = load_hierarchy()
    t0 = time.time()
    rng = np.random.default_rng(SURROGATE_SEED)
    model_land = pd.read_parquet(work_dir / "counts_model_land.parquet")
    model_all = pd.read_parquet(work_dir / "counts_model_all.parquet")
    obs_land = pd.read_parquet(work_dir / "counts_obs_land.parquet")
    obs_all = pd.read_parquet(work_dir / "counts_obs_all.parquet")
    click.echo(f"[stats] counts loaded ({len(model_land):,} land rows), {time.time() - t0:.0f}s")
    country_ids = _country_ids(obs_land, model_land)
    geos_storm = kept_geos_by_start(obs_land, STORM_STARTS, storm_years[1], hier, country_ids).assign(domain="storm")
    geos_mort = kept_geos_by_start(obs_land, MORTALITY_STARTS, mortality_years[1], hier, country_ids).assign(domain="mortality")
    runs = np.repeat(np.array(RUNS), N_DRAWS)

    stats_rows, band_rows, series_rows, ceil_rows = [], [], [], []
    ys = np.arange(storm_years[0], storm_years[1] + 1)
    for source, mc, oc in (("land", model_land, obs_land), ("all", model_all, obs_all)):
        levels = ("global", "basin", "super_region", "region", "country") if source == "land" else ("global", "basin")
        for level in levels:
            g = geos_storm[geos_storm["level"] == level]
            geos = sorted(g["geo"].unique()) if source == "land" else geo_universe(level, hier, country_ids)
            arr_m = densify_model_counts(mc, level, geos, ys)
            arr_o = densify_obs_counts(oc, level, geos, ys)
            n_cells = 0
            for gi, geo in enumerate(geos):
                starts = _starts_kept(g, geo) if source == "land" else tuple(STORM_STARTS)
                if not starts:
                    continue
                n_cells += 1
                for oi, outcome in enumerate(OUTCOMES):
                    key = {"domain": "storm", "source": source, "level": level, "geo": geo, "outcome": outcome}
                    s, b, se, ce = cell_stats(key, arr_o[gi, :, oi], arr_m[gi, :, :, oi], runs, ys, STORM_WINDOWS,
                                              starts, rng=rng, n_sur=n_sur)
                    stats_rows += s
                    band_rows += b
                    series_rows += se
                    ceil_rows += ce
            click.echo(f"[stats] storm/{source}/{level}: {n_cells} geos, {time.time() - t0:.0f}s")

    ym = np.arange(mortality_years[0], mortality_years[1] + 1)
    mdir = work_dir / "mortality"
    if storm_only:
        click.echo("[stats] --storm-only: mortality cells skipped")
        geos_mort = geos_mort.iloc[0:0]
    if not storm_only:
        model = pd.concat([pd.read_parquet(mdir / f"model_{r}.parquet").assign(run=r) for r in RUNS], ignore_index=True)
        _check_draws(model)
        ibt = pd.read_parquet(mdir / "ibtracs_pred.parquet")
        obs = pd.read_parquet(mdir / "observed.parquet")
        click.echo(f"[stats] mortality loaded ({len(model):,} model rows), {time.time() - t0:.0f}s")
    for level in ("global", "basin", "super_region", "region", "country") if not storm_only else ():
        g = geos_mort[geos_mort["level"] == level]
        geos = sorted(g["geo"].unique())
        arr_o = densify_deaths(obs, level, geos, ym, "deaths", [])[:, 0, :]
        n_cells = 0
        for tag, col in MORTALITY_TAGS.items():
            arr_m = densify_deaths(model, level, geos, ym, col, ["run", "draw"])
            arr_i = densify_deaths(ibt, level, geos, ym, col, ["draw"])
            for gi, geo in enumerate(geos):
                starts = _starts_kept(g, geo)
                if not starts:
                    continue
                n_cells += 1
                key = {"domain": "mortality", "source": tag, "level": level, "geo": geo, "outcome": "deaths"}
                s, b, se, ce = cell_stats(key, arr_o[gi], arr_m[gi], runs, ym, MORTALITY_WINDOWS, starts,
                                          ibt=arr_i[gi], rng=rng, n_sur=n_sur)
                stats_rows += s
                band_rows += b
                series_rows += se
                ceil_rows += ce
        click.echo(f"[stats] mortality/{level}: {n_cells // 2} geos, {time.time() - t0:.0f}s")

    stats = pd.DataFrame(stats_rows)
    band = pd.DataFrame(band_rows)
    series = pd.DataFrame(series_rows)
    ceiling = pd.DataFrame(ceil_rows)
    geos_all = pd.concat([geos_storm, geos_mort], ignore_index=True)
    write_parquet_checked(stats, out_dir / "stats.parquet")
    write_parquet_checked(band, out_dir / "band.parquet")
    write_parquet_checked(series, out_dir / "series.parquet")
    write_parquet_checked(ceiling, out_dir / "ceiling.parquet")
    write_parquet_checked(geos_all, out_dir / "geos.parquet")
    click.echo(f"[stats] wrote {len(stats):,} stats / {len(band):,} band / {len(series):,} series / "
               f"{len(ceiling):,} ceiling rows to {out_dir}, {time.time() - t0:.0f}s")
