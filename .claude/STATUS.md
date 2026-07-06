# Project status
Updated: 2026-07-06

## Goals
Build a double-hurdle peaks-over-threshold model for tropical cyclone mortality estimation.
Four-component model: S1 (P(deaths>=1)), S2 (P(rate>=threshold | S1=1)), bulk rate, tail rate.
Pipeline stages: validate → grid → fit → evaluate → select → uncertainty → predict.

## Orientation
**What we're doing now & why** (distinct from Goals — this changes whenever the work or its rationale shifts).

*Staged model selection.* We can't fit every knob combination, so selection is a staged fractional-factorial narrowing:
- **preliminary** — coarse sweep over families × exposures × thresholds × covariates; screen out losers.
- **intermediate** — dig into survivors (drop-top-N, per-storm, per-year) to lock thresholds + families.
- **refined** — tighten the surviving combinations (per-family exposures, narrowed thresholds/covariates).
- **final** — a *more* comprehensive fractional-factorial over the refined winners (more permutations, never "all").

*Where we are:* preliminary + intermediate done on the 20260608 vintage; **refined is the next run.** Refined decision (= 20260517 post-2000): threshold ∈ {0.70, 0.85}; tail ∈ {gpd, log_logistic, weibull} with per-family exposures; bulk = scaled_logit × {free, free+weight}; s1/s2 = logit/free. Encoded in `grid/build_refined_specs_post2000.py` (336 IS specs).

*Architecture.* A DH model = 4 components (s1, s2, bulk, tail), each fit then assembled + scored.
- *Old two-step (still how 20260517 refined ran):* (1) fit all components → save `.pkl`; (2) load `.pkl` in 4s → assemble → score. Fitting is ~0.1s; *concurrent* `.pkl` reads at scale melted NFS.
- *Re-fit-in-evaluate (current, applied to preliminary 20260608):* no fit stage, no `.pkl`. The evaluate worker re-fits each component in-memory (`run_evaluate.py` `_load` → `fit_one_component`), caching each component once per worker — redundant across workers, but contends nothing.

*Refined bridge — BUILT, uncommitted (2026-06-12 work):* the migration gap is CLOSED. The evaluate **worker** re-fits in every mode, IS and OOS (each fold's four components re-fit on its training subset via `_load`→`fit_one_component`; the inline OOS loop replaced `assemble_oos_predictions`, so there is no `.pkl` read and no `load_result` anywhere on the path). Refined **orchestration** is wired end-to-end via registered CLIs: (1) `run-build-refined-specs-post2000` → 336 IS specs; (2) `run-evaluate-orchestrate --refined-specs <specs> --manifest-only` → `manifest.json` + `fold_assignments.parquet`, no submit; (3) `run-build-refined-cells` → structure-C cell manifest, one task per (s1, s2) = 512 tasks, `|s2_cov|`-tiered; (4) `run-evaluate-orchestrate --cells-file <cells> --tier-memory/--tier-runtime` → full submit. Refined cov-structure is **structure C**: s1_cov free, s2_cov free & independent of s1, then `s2 ⊇ bulk ⊇ tail` (NOT the old `build_half_coupled_multiconfig_tasks.py` chain, now superseded). Dead-and-deletable: `assemble_oos_predictions`, `cache.load_result/save_result/result_exists`.

*Task-grouping lever:* under re-fit, cost = (#component fits) + (#DH-config assemblies), and each component-fit is paid once per worker. Packing configs that share fits into one worker amortizes cost — the knob for affording *more* permutations, balanced against ~20–30s/task overhead.

*Prediction & deliverable phase (2026-06 onward).* Selection converged on 3 chosen death models; prediction now runs on the **consolidated CLIMADA frames** via a new pipeline (`consolidated.py` + `build_predict_cells.py` + `predict_cell.py` + `orchestrate_consolidated.py`, jobmon), per storm_draw × scenario, for two hierarchy approaches — **A0-only** and the **A1/A0 blend** (A1 admin-1 for 10 subnationalized countries + A0 for the rest, no double-count). The shipped M1 deliverable (`direct_deaths.{parquet,nc}` at rapidresponse/direct_risk) is the adjusted blend, draw-level `deaths_c1_s1_o0_b1`; its three post-processing steps are now committed as `run-finalize-deliverable`, which also builds the blend `summary.parquet` from the a0/a1 partials if absent — so a rerun is `run-predict-consolidated` (×2, a0+a1) → `run-finalize-deliverable`, no manual rollup step. Open: SR-31 hard-zero guard decision.

## Recent steps
- 2026-07-06: Promoted the deliverable post-processing (blend export → super-region median adjustment → FHS-population rate merge) from ad-hoc heredocs to `src/idd_tc_mortality/predict/finalize_deliverable.py` (`run-finalize-deliverable`), recovered verbatim from the session transcript. Added `super_region_median_ratios`/`apply_super_region_adjustment` + unit test (synthetic 2-SR frame, 3/3 pass). Verified byte-identical reproduction of the shipped M1 `direct_deaths.parquet` (8,669,768 rows). Committing with the rest of the consolidated-predict pipeline.
- 2026-06-08: Ingested new data vintage 20260608 (post-2000 filter, basins_standard, AU=first-class); 1,903 rows → `00-data/20260608/input.parquet`
- 2026-06-08: Preliminary fit: 22,854 IS specs (4 bundles, BUNDLE_SIZE=6570, 1G/9m) → `01-preliminary/20260608/`
- 2026-06-08: Removed AU lon-split from predict pipeline (data_prep, predict_tc, predict_year_bin, predict/orchestrate); fixed stale fit-orchestrate probe gate (0.4→0.8×ask)
- 2026-06-09: Redesigned evaluate: in-memory re-fit (no .pkl reads/writes, no NFS contention). Modifications: `run_evaluate.py` `_load`→`fit_one_component`, `run_evaluate_orchestrate.py` adds `_build_manifest_and_folds` + `BUNDLE_SIZE=1`. Full 360-task preliminary evaluate → `02-evaluate/20260608_refit/` (952,966 rows, 164,520 IS, 788,446 OOS). Validated.
- 2026-06-09: Preliminary screening → same decisions as 20260517 run. 18-task survivors evaluate (WITH model_predictions) → `02-evaluate/20260608_survivors/` (4,952 rows + 4,952 model_predictions). Created `dh_preliminary_diagnostics.ipynb` + `dh_intermediate_diagnostics.ipynb`.
- 2026-06-12: Built the refined orchestration bridge (uncommitted) — `build_refined_specs_post2000.py` (336 IS specs), `build_refined_cells.py` (structure-C cells, 512 tasks), and the `run_evaluate_orchestrate.py` `--refined-specs`/`--manifest-only`/`--cells-file`/`--tier-*` modes. Locked the refined decision (= 20260517 post-2000). Closes the old "refined orchestration not migrated" gap.
- 2026-06-15: Verified the refined cells path reads NO `.pkl` end-to-end (IS + OOS both re-fit in-memory; `assemble_oos_predictions` / `cache.load_result` unreachable from the worker). Reconciled STATUS Orientation + Next steps to match.
- 2026-05-12 to 5-15: All-years preliminary→refined→TOPSIS→draws→predict pipeline. Workflow 577423 ran through end of 5-15.
- 2026-05-16: Predict-pipeline diagnostics. SDI boundary bug fix. ~30× low-SDI overprediction surfaced.
- 2026-05-17 (end-to-end post-2000 cycle): three new TS metrics (`cor_ts`, `beta_0_ts`, `beta_p_ts`) in `metrics.py`; preliminary-screening fork (`dh_preliminary_diagnostics_post2000.ipynb` + `reports/preliminary_decisions_post2000.qmd`); intermediate-survivors trim+IS/OOS analysis (`dh_intermediate_diagnostics_post2000.ipynb`); refined fit at `01-refined/20260517/` (336 IS specs × 26 fold_tags = 8,736 tasks, ~11s median); refined evaluate via new `build_half_coupled_multiconfig_tasks.py` (512 tasks × ~34 configs = 17,500 DH configs); refined-final 96-config explicit-decoupled grid at `02-evaluate/refined-final-20260517/` with `model_predictions` enabled.
- 2026-05-18 (final diagnostics + predict submission):
  - Built `notebooks/dh_final_diagnostics_post2000.ipynb` (35 cells) — fork of `dh_refined_diagnostics_post2000.ipynb` repointed at `refined-final-20260517`, with `mid` reconstruction + prediction loaders + drop-top-N sweep (N ∈ {0,5,10,25}; IS + OOS mean + OOS median) + IS/OOS divergence + picking table.
  - **Extended `src/idd_tc_mortality/uncertainty/draw_models.py` to handle `log_logistic` tail family** — added `_prepare_stage_log_logistic` (pulls cov from scipy BFGS `raw.hess_inv`, joint/marginal draws over `(beta, log_k)` per toggle state), `_psd_project` (symmetrize + clamp non-PSD eigenvalues for BFGS hess_inv cleanup), and log_logistic branches in `_tail_mean` and `_tail_draw`. Smoke test: build script produces all 4 pickles in 0.6s; `DrawModel.predict` runs cleanly for all 4 (o, b) toggles.
  - **Built draws_dir for best-TOPSIS post-2000 mid** (`925484b474c724ff9c6e2da7255bacb3`) at `03-draws/20260517/best_topsis_post2000/` via `/tmp/build_facevalid_draws_best_topsis.py` (parameterizes `scripts/build_topsis_draws.py` against a target mid + the refined-final manifest).
  - **Submitted full 45,676-task predict orchestrator** for that mid at `04-predict/best_topsis_post2000/storm_draws/`. **Jobmon bind has been chugging task-metadata-add for 60+ min** — per-task DB latency is the bottleneck. A concurrent 70-task subset workflow (`--storm-draws 1 --scenarios historical`, output at `04-predict/best_topsis_post2000_sd1_hist/`) bound in seconds and completed cleanly (one expected failure at top-tier aggregate_storm_draw because it requires all 4 scenarios). Diagnostic confirms bind cost is roughly linear in task count, not a DB-availability issue. Workflow 577423 had bound the same scale before — something in the jobmon environment has changed since.

## Next steps
**Currently ready:**
- Refined evaluate run is ready to submit — the bridge (specs → manifest-only → cells → submit) is built; see Orientation. The bridge files are still uncommitted.

**In order:**
1. (Optional but advised) Commit the refined bridge first — 2 untracked files + a modified `run_evaluate_orchestrate.py` are real, at-risk work.
2. Run the refined sequence: `run-build-refined-specs-post2000` → `run-evaluate-orchestrate --refined-specs … --manifest-only` → `run-build-refined-cells` → `run-evaluate-orchestrate --cells-file … --tier-memory/--tier-runtime` (reuse the 20260517 tier asks — same structure C, same 512-task scale). Add `--skip-model-predictions` unless the refined diagnostics need the per-config parquets.
3. Uncertainty + predict for selected winner (compare against prior 20260517 post-2000 winner).
4. Delete dead fit stage (`fit/orchestrate.py`, `run_component.py`, `save_result`/`load_result`/`result_exists` from `cache.py`, and `assemble_oos_predictions` from `evaluate/assemble.py`) — confirmed safe once validated.

## Parking lot
**Queued — open for next longer session:**
- **Delete dead fit stage** after full-pipeline validation: `fit/orchestrate.py`, `run_component.py`, `save_result`/`load_result`/`result_exists` from `cache.py`, and `assemble_oos_predictions` from `evaluate/assemble.py`
- **Fix stale `test_run_component.py` tests** (`--spec-id` vs `--bundle-file`)
- **Basin random effects.** EP basin has very little data. Sketch what a basin-RE spec would look like across S1/S2/bulk/tail stages, what it costs in grid size, and whether it buys anything for EP without overfitting NA/WP/SI.
- **Stochasticity in UI suggests we may need more draws.** Think about: how many draws are actually enough; whether noise is from `_bulk_draw`/`_tail_draw` predictive sampling vs. β-spread; whether a variance-decomposition tells us which lever to pull.
- **Directory hashing or per-task aggregated parquets** in `_save_model_predictions_parquet` (the 1M-files NFS thrash workaround is still `--skip-model-predictions`).
- **Tighten `fit/orchestrate.py`'s `_SLURM_RESOURCES`** (references stale "512M" and "34 specs per bundle").

## Process / skill goals (near-term)
- Documentation skills — decision reports (carried).
- **Probe-first sizing as default.** Reliable move is a single-task probe + sacct, not extrapolation from prior data.
- **Stop padding engineering estimates.** Two new corrections today (parallel-submit wall 45-90 min→15-20 min real; log_logistic extension "half a day with tests"→15 min real). Count actual edits (lines/functions touched) + one verification run, not safety-margin hours.
