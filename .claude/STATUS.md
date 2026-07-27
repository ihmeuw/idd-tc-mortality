# Project status
Updated: 2026-07-22

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

*Architecture.* A DH model = 4 components (s1, s2, bulk, tail), each fit then assembled + scored.
- *Old two-step (still how 20260517 refined ran):* (1) fit all components → save `.pkl`; (2) load `.pkl` in 4s → assemble → score. Fitting is ~0.1s; *concurrent* `.pkl` reads at scale melted NFS.
- *Re-fit-in-evaluate (current, applied to preliminary 20260608):* no fit stage, no `.pkl`. The evaluate worker re-fits each component in-memory (`run_evaluate.py` `_load` → `fit_one_component`), caching each component once per worker — redundant across workers, but contends nothing.

*Task-grouping lever:* under re-fit, cost = (#component fits) + (#DH-config assemblies), and each component-fit is paid once per worker. Packing configs that share fits into one worker amortizes cost — the knob for affording *more* permutations, balanced against ~20–30s/task overhead.

*Prediction & deliverable phase (2026-06 onward).* Selection converged on chosen death models; prediction runs on the **consolidated CLIMADA frames** (`run-predict-consolidated` per frame, jobmon) for **A0-only** and the **A1/A0 blend** (A1 admin-1 for 10 subnationalized countries + A0 for the rest, no double-count). The full deliverable chain is committed, two script types, no manual steps: `run-predict-consolidated` (×2, a0+a1) → `run-finalize-deliverable` (blend export → SR adjustment → FHS-pop rates; builds the blend `summary.parquet` on demand; `--sr31-guard` flag). SDI is applied on-the-fly at predict via `--old-sdi-path`/`--new-sdi-path` (the consolidated pipeline IGNORES paths.py).

*Where we are (2026-07-22):* Three rerun **vintages** (`20260714_v{1985,1990,1995}`, differing only in training-window start year) have gone through staged selection to a **4,608-config final grid** each (`02-evaluate/<V>_final/`). From each final TOPSIS we pick a **best-TOPSIS** model + one **universal** model (best on a combined cross-vintage TOPSIS) → 6 vintage models, run through A0 predict for unadjusted deaths by super-region + global (`notebooks/20260714_unadjusted_plots.ipynb`). **GPD tail models were excluded from selection** (couldn't build coefficient draws for the deliverable at decision time + convergence concerns); all six winners are log_logistic and converged. The **v2000** model (`6004b730`) was **re-delivered from the v89-SDI blend** (`04-predict/20260706/…_blend`) to `direct_risk/` as `2026_07_21_v2000_{unadjusted,adjusted_mean,adjusted_mean_sr31_guard}` (correcting an earlier accidental v66 delivery). A **driver-of-change sensitivity** subsystem (freeze SDI / population / both at anchor 2023) is built + run (`04-predict/20260721_v2000_sensitivity/`, decompose notebook) — SDI improvement is the dominant driver (~31% global reduction 2023→2100), population a smaller upward push. Newest: an **observed-storm-fit** diagnostic (`viz/predict_plots.load_observed_fit`) showing IS/OOS predictions on the actual fitted storms vs observed, quantifying the expected heavy-tail under-prediction (2008 Nargis: 140,500 obs vs ~1,200 pred). **Everything since 2026-07-13 is uncommitted.**

## Recent steps
- 2026-07-22: Built `load_observed_fit`/`plot_observed_fit`/`plot_observed_fit_grid` (model IS/OOS predictions on the actual fitted storms, `rate × exposed`, rolled to SR+global, unadjusted); first in `sensitivity/decompose.py`, then **relocated to `viz/predict_plots.py`** so `20260714_unadjusted_plots.ipynb` can import them. Also fixed `plot_constant_vs_actual_grid` panel titles (region-only). Wrote (did not build) a fresh-CC prompt for a heavy-tail under-prediction teaching vignette.
- 2026-07-14→21: Reset the final grid (3 thresholds, 6 bulk-cov / 8 tail-cov sets) and ran final evaluates for all 3 vintages; excluded gpd, selected best-TOPSIS + universal per vintage; unadjusted SR/global plots; re-delivered v2000 from v89 SDI; built + ran the driver-of-change sensitivity subsystem (`sensitivity/`, `run_sensitivity.py`, `decompose.py` + notebook); wrote the idd-tools `model_selection` consolidation brief.
- 2026-07-07: New-SDI rerun end-to-end. Fixed jobmon submit 404 (cluster dropped slurmrestd v0040; `jobmon_installer_ihme` 10.11.6→10.12.2, pins updated in both repos). Ran `run-predict-consolidated` A0+A1 with s130v89 past+future SDI → `04-predict/20260706/`; `run-finalize-deliverable` ×2 (default + `--sr31-guard` → `…_blend_sr31guard/`). Shipped date-stamped deliverable pairs to `direct_risk/`. Built `notebooks/20260707/sr_version_comparison.ipynb` (uncommitted).
- 2026-07-06: Promoted the deliverable post-processing (blend export → super-region median adjustment → FHS-population rate merge) from ad-hoc heredocs to `src/idd_tc_mortality/predict/finalize_deliverable.py` (`run-finalize-deliverable`), recovered verbatim from the session transcript. Added `super_region_median_ratios`/`apply_super_region_adjustment` + unit test (synthetic 2-SR frame, 3/3 pass). Verified byte-identical reproduction of the shipped M1 `direct_deaths.parquet` (8,669,768 rows). Committed with the rest of the consolidated-predict pipeline (`f56f570`, `eba4c43`).
- 2026-06-08: Ingested new data vintage 20260608 (post-2000 filter, basins_standard, AU=first-class); 1,903 rows → `00-data/20260608/input.parquet`
- 2026-06-08: Preliminary fit: 22,854 IS specs (4 bundles, BUNDLE_SIZE=6570, 1G/9m) → `01-preliminary/20260608/`
- 2026-06-08: Removed AU lon-split from predict pipeline (data_prep, predict_tc, predict_year_bin, predict/orchestrate); fixed stale fit-orchestrate probe gate (0.4→0.8×ask)
- 2026-06-09: Redesigned evaluate: in-memory re-fit (no .pkl reads/writes, no NFS contention). Modifications: `run_evaluate.py` `_load`→`fit_one_component`, `run_evaluate_orchestrate.py` adds `_build_manifest_and_folds` + `BUNDLE_SIZE=1`. Full 360-task preliminary evaluate → `02-evaluate/20260608_refit/` (952,966 rows, 164,520 IS, 788,446 OOS). Validated.
- 2026-06-09: Preliminary screening → same decisions as 20260517 run. 18-task survivors evaluate (WITH model_predictions) → `02-evaluate/20260608_survivors/` (4,952 rows + 4,952 model_predictions). Created `dh_preliminary_diagnostics.ipynb` + `dh_intermediate_diagnostics.ipynb`.
- 2026-06-12: Built the refined orchestration bridge (uncommitted) — `build_refined_specs_post2000.py` (336 IS specs), `build_refined_cells.py` (structure-C cells, 512 tasks), and the `run_evaluate_orchestrate.py` `--refined-specs`/`--manifest-only`/`--cells-file`/`--tier-*` modes. Locked the refined decision (= 20260517 post-2000). Closes the old "refined orchestration not migrated" gap.
- 2026-06-15: Verified the refined cells path reads NO `.pkl` end-to-end (IS + OOS both re-fit in-memory; `assemble_oos_predictions` / `cache.load_result` unreachable from the worker). Reconciled STATUS Orientation + Next steps to match.

## Next steps
1. **Commit + push** the large uncommitted working tree (vintage finals + notebooks, gpd-excluded selection, v89 v2000 re-delivery, whole `sensitivity/` package, `viz/predict_plots` observed-fit, idd-tools brief). Nothing since ~07-13 is committed.
2. Wire the observed-fit cell into `20260714_unadjusted_plots.ipynb`.
3. **GPD follow-up:** the "sort out GPD" agent (draws / non-convergence / convergence-gating) never reported back — resume or re-run and fold its findings in.
4. **SR-31 guard decision** (with data producer): guard vs hard-zero → promote that variant's date-stamped files to the canonical `direct_deaths.{parquet,nc}`.
5. Delete dead fit stage (carried; see parking lot).

## Parking lot
**Queued — open for next longer session:**
- **Consolidate `SUPER_REGIONS`** — now a dict in both `viz/predict_plots.py` and `sensitivity/decompose.py`, and a list inline in `20260714_unadjusted_plots.ipynb`. Make `predict_plots.SUPER_REGIONS` the single source; have decompose + the notebook import it.
- **Delete dead fit stage** after full-pipeline validation: `fit/orchestrate.py`, `run_component.py`, `save_result`/`load_result`/`result_exists` from `cache.py`, and `assemble_oos_predictions` from `evaluate/assemble.py`
- **Fix stale `test_run_component.py` tests** (`--spec-id` vs `--bundle-file`)
- **Basin random effects.** EP basin has very little data. Sketch what a basin-RE spec would look like across S1/S2/bulk/tail stages, what it costs in grid size, and whether it buys anything for EP without overfitting NA/WP/SI.
- **Stochasticity in UI suggests we may need more draws.** Think about: how many draws are actually enough; whether noise is from `_bulk_draw`/`_tail_draw` predictive sampling vs. β-spread; whether a variance-decomposition tells us which lever to pull.
- **Directory hashing or per-task aggregated parquets** in `_save_model_predictions_parquet` (the 1M-files NFS thrash workaround is still `--skip-model-predictions`).
- **Tighten `fit/orchestrate.py`'s `_SLURM_RESOURCES`** (references stale "512M" and "34 specs per bundle").

## Process / skill goals (near-term)
- Documentation skills — decision reports (carried).
- **Probe-first sizing as default.** Reliable move is a single-task probe + sacct, not extrapolation from prior data.
- **Stop padding engineering estimates.** Count actual edits (lines/functions touched) + one verification run, not safety-margin hours.
