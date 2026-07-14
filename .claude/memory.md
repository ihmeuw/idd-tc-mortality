# Session memory
Updated: 2026-07-13

## Current task
Mean-vs-median rake comparison — COMPLETE this session. Added a `statistic` (mean|median) option to the super-region rake and generated a 5-version comparison set for M1 (`6004b730b8d8f96cb6d8443378b04f09`), wired into the notebook as 5 columns.

## Context / why
Bobby wanted to see mean-raked alongside median-raked deliverables. The existing rake matched the observed **median** per super-region; "rake to the mean" = same machinery with mean. For heavy-tailed TC deaths the mean ratio is much larger (SR4: mean 6.23 vs median 0.40) so mean-raking ~doubles global deaths (3679 vs 378 median, vs 1841 unadjusted @2050/ssp245). SR31 has zero observed deaths across all of 2000–2023 under BOTH stats, so the guard is still needed for the mean variant too.

## This session's changes
- `finalize_deliverable.py`: renamed `super_region_median_ratios`→`super_region_ratios` (+ `statistic="median"` param, `.agg(statistic)`); extracted `write_draw_level` + `population_index`; `write_deliverable` gained `adjusted_tag`; CLI gained `--rake-stat median|mean` (mean suffixes the adjusted filename `_mean`; default median stays byte-identical to shipped).
- `tests/predict/test_finalize_deliverable.py`: rename + new `test_super_region_ratios_mean_statistic`. 14/14 predict tests pass.
- `scripts/build_sr_version_files.py` (NEW): reads the unadjusted blend once, derives the 2(stat)×2(guard) matrix + unadjusted, writes 5 date-prefixed files. Default out-dir `DIRECT_RISK_DIR`.
- Shipped to `direct_risk/`: `2026_07_13_{unadjusted,adjusted_mean,adjusted_median,adjusted_mean_sr31_guard,adjusted_median_sr31_guard}_direct_deaths.{parquet,nc}` (10 files, 8,669,768 rows each).
- `notebooks/20260707/sr_version_comparison.ipynb`: now 5 columns; per-version obs/pred hlines use the stat each panel was raked to (`STAT_BY_VERSION`); markdown updated. Verified all cells execute headlessly.
- Full-history observed (plot-only): `postprocess.py` gained `aggregate_observed_deaths` (extracted core), `filter_source_observations`, `build_observed_deaths_from_source_csv`; `build_observed_deaths` now a thin reader over the core. New `tests/predict/test_postprocess.py` (3 tests). Notebook has `FULL_HISTORY=True` + `OBS_SOURCE_CSV` → obs_by_loc spans 1980-2023 from the raw ibtracs CSV; model input.parquet (current→20260608, 2000-2023) is UNTOUCHED (Bobby chose plot-only over re-ingest). Rake ratios/hlines still use the fixed 2000-2023 window. NB: plot still cuts at START_YEAR=2000 until Bobby lowers it in the plot cell (I did not edit his live cell). Key data fact: source CSV = 1980-2023, but current input.parquet = 2000-2023 (built with a 2000 year-floor).

## Prior context / why (still open)
New future+past SDI (s130v89) rerun of M1 landed in `04-predict/20260706/`. The SR-31 hard-zero vs guard question (guard = 0.1 × smallest non-zero ratio) is still the open modeling decision that determines which variant becomes canonical.

## Where we are
- **Vintage `04-predict/20260706/`**: `<mid>_a0`, `<mid>_a1` (run-predict-consolidated, 3G/10m, predict_cells_5sd.json), `<mid>_blend` (finalize default), `<mid>_blend_sr31guard` (finalize `--sr31-guard`).
- **Shipped** (2026-07-07, chmod 775): `direct_risk/20260707_adjusted_direct_deaths.{parquet,nc}` + `20260707_sr31guard_adjusted_direct_deaths.{parquet,nc}`. Canonical `direct_deaths.{parquet,nc}` = June/old-SDI, untouched.
- **jobmon fixed**: installer 10.11.6→10.12.2 (slurmrestd v0040 endpoint removed by cluster upgrade → submit 404s); pyproject pins updated both repos, `pip install -e . --no-deps` refreshed metadata.
- **`notebooks/20260707/sr_version_comparison.ipynb`** (UNCOMMITTED): Global+7 SRs rows × 3 version columns (nested gridspec, `plot_locations(plot_locs, space=...)`); controls START_YEAR / ZOOM(obs|forecast|log) / space(count|rate, FHS pop) / SHOW_CI / SHOW_OBS_MEDIAN / SHOW_PRED_MEDIAN / SHOW_HIST_PRED (historical line, SSPs anchored at 2014). Uses viz.SSP_SCENARIO_MAP + finalize imports (canonical ratios). Bobby co-edited heavily; fixed his hardcoded `space='count'` in the draw_panel call (rates plot was mislabeled counts).
- **Commits `f56f570` + `eba4c43`** (consolidated-predict + uncertainty + finalize_deliverable + blend-on-demand) are on main, **NOT PUSHED**.
- Watch out: notebook edit collisions when Bobby has it open in the IDE — sync (Read) before every NotebookEdit, ask him to close for multi-cell edits.

## Next steps
1. Bobby to review the 5-column comparison (mean vs median vs unadjusted, ±guard) → decide mean-or-median AND guard-or-not for the canonical deliverable.
2. Push `f56f570` + `eba4c43` + this session's finalize/script/test changes (all uncommitted).
3. Commit `notebooks/20260707/sr_version_comparison.ipynb`, `scripts/build_sr_version_files.py`.
4. Once decided → `cp` the winning variant's pair over `direct_risk/direct_deaths.{parquet,nc}`.
5. Carried: delete dead fit stage; stale test_run_component tests.

## Resume prompt
Session 2026-07-13 (idd-tc-mortality, main): Added mean-vs-median raking. `super_region_median_ratios`→`super_region_ratios` now takes `statistic="median"|"mean"`; CLI `run-finalize-deliverable` gained `--rake-stat`; new `scripts/build_sr_version_files.py` emits the 2×2 (stat×guard)+unadjusted matrix. 10 files shipped to `direct_risk/` as `2026_07_13_{unadjusted,adjusted_mean,adjusted_median,adjusted_mean_sr31_guard,adjusted_median_sr31_guard}_direct_deaths.{parquet,nc}`. Notebook `notebooks/20260707/sr_version_comparison.ipynb` now plots all 5 as columns (per-version obs/pred hlines via STAT_BY_VERSION); all cells verified to execute. Key finding: mean-raking ~doubles global deaths vs unadjusted (heavy tail; SR4 ratio 6.23), median-raking shrinks it ~5×; SR31 obs=0 under both stats so guard still needed. 14/14 predict tests pass. ALL uncommitted incl. prior f56f570+eba4c43. Underlying vintage `04-predict/20260706/` (new SDI s130v89). Next: Bobby picks stat+guard for canonical; push; commit notebook+script.
