# Session memory
Updated: 2026-07-23

## Current task
Honest-mean tail-variant machinery for the double-hurdle model is BUILT, TESTED, and COMMITTED.
Design for the next preliminary sweep is settled (honest-only, medians dropped). Not yet run.

## Context / why
Median-reporting tails (gpd/weibull/log_logistic) make assembled E[rate] a non-expectation; the
SR rake silently corrected it. Fix = bounded/finite-mean tail variants using the per-storm cap
c_i = exposed_population/exposed (excess cap H_i = c_i - u). OOS score selects; NEVER rank
estimators by mechanism-stories ahead of OOS (Bobby's standing correction — threshold-coupling of
C is a parameter the sweep adjudicates, not a defect).

## Where we are — machinery COMPLETE + committed
- Commits (author Bobby Reiner, both on top of 624cabe):
  - 7c8aa6c: A/C/D families — weibull_mean (D), gpd_cens/log_logistic_cens (C),
    gpd_shadow/log_logistic_shadow (A, Cirillo-Taleb dual, shared _shadow.py) + tail_cap.py +
    needs_cap seam (predict_component) + fit_component cap-at-fit branch.
  - 69cfdad: B families — gpd_trunc/log_logistic_trunc (renormalized truncated likelihood, shared
    _trunc.py). Truncated MLE warm-started from untruncated, ANALYTIC gradient (FD-verified),
    truncated mean = (E[min(W,H)] - H*S(H))/(1-S(H)) reusing the C censored-mean helpers. Reports
    shape +/- SE on the n-storm scale (identifiability signal). B validated: shape identifiable on
    hard-truncated real data, converges clean with the gradient.
- Tests: full distributions suite passes except the 2 PRE-EXISTING test_gpd.py failures (missing
  `import warnings`, the separate stalled GPD thread — NOT ours). Tail variants MC-verified;
  shadow mu_g vs independent MC + heterogeneous caps; B gradients FD-verified.
- Working dataset 20260722_v1985 (has exposed_population; c_i>=r_i verified).

## Settled preliminary-sweep design (honest-only)
- Medians (gpd/weibull/log_logistic) DROPPED outright (not even baselines). Honest tail set:
  gamma, lognormal, truncated_normal (already means) + weibull_mean (D) + gpd_cens/log_logistic_cens
  (C) + gpd_shadow/log_logistic_shadow (A) + gpd_trunc/log_logistic_trunc (B).
- A families are covariate-free -> intercept-only, weighted/unweighted only (grid-builder must
  special-case; else redundant enumeration). B/C/D enumerate standard cov x exposure.
- Median-vs-mean contrast recovered POST-HOC on the winner (functional-agnostic: same fit object,
  base-median predict vs variant-mean predict — zero re-fit), NOT a factorial arm.
- TWO vintages: 1985->present (20260722_v1985 ready) and 2000->present (needs re-ingest WITH
  exposed_population; old 20260608 lacks it). Same spec file, two --data-path/--output-dir runs.
- pred_obs_ratio is a MEASURED metric selected per-stage in the notebooks (already used), NOT a
  new column and NOT the sole cull criterion.

## Winner-config pred_obs_ratio tables (IS, illustrative; run on v1985 cap-data)
v1985 winner (weibull base, thr 0.85): median 0.125; weibull_mean 0.240; gpd_cens 0.334;
log_logistic_cens 0.360; gpd_trunc 0.326; log_logistic_trunc 0.324; gpd_shadow 6.357;
log_logistic_shadow 3.838. (High threshold -> small H -> C/B/D cluster low near the median.)
v2000 winner (log_logistic base, thr 0.70; config on v1985 cap-data): median 0.119;
weibull_mean 0.664; gpd_cens 4.861; log_logistic_cens 2.287; gpd_trunc 3.123;
log_logistic_trunc 1.718; gpd_shadow 4.338; log_logistic_shadow 1.991. (Family x variant
interacts strongly; log_logistic_trunc nearest 1 here.) IS only, unraked — OOS selects.

## Run setup DONE (spec list + data + probe)
- Generator committed d772613 (grid/build_tailvariant_specs.py). Spec JSON written:
  01-preliminary/tailvariants/is_specs.json (1047 specs, data-independent, both vintages use it).
- Data: v1985 = 00-data/20260722_v1985 (full 1985-2023, has exposed_population). v2000 =
  00-data/20260722_v2000 = the v1985 data FILTERED to year>=2000 (1903 rows) — no re-ingest,
  current untouched (Bobby's "just start dates" approach).
- PROBE done (workflow 605747, coupled honest specs on v1985, 5 tasks, 4G/30m, --skip-model-
  predictions, --probe-n 5 --max-attempts 1): all 5 DONE. max RSS 0.59 GiB, max runtime 329s
  (~5.5m). B added NO material per-task cost — honest set fits the standard preliminary envelope.
  Resource inspection via idd_tools.jobmon.collect_workflow(wf_id) (NOT sacct; workflow_resource_stats
  gives the template-history aggregate, collect_workflow gives the per-workflow tasks).
- FULL-RUN sizing (<=2x probe): memory 1G, runtime 11m, default retries. 360 coupled bundles/vintage.

## Sweeps DONE + v1985 preliminary DECIDED
- Both full sweeps ran: 02-evaluate/20260722_{v1985,v2000}_tailvariants/dh_results.parquet
  (v1985 1.33M rows, v2000 1.31M; 227,240 / 224,640 assembled configs). v2000 data =
  20260722_v1985 filtered to year>=2000.
- Vetting notebooks in ONE folder notebooks/20260722/: dh_preliminary_diagnostics_{v1985,v2000}.ipynb.
- v1985 PRELIMINARY DECISIONS made + logged (DECISIONS.md 2026-07-24): S1 logit/free; S2 logit/free;
  bulk scaled_logit {free,free+weight}; tail DROP nb/poisson/log_logistic_shadow/gpd_shadow, KEEP the
  8 honest-mean families (gamma,lognormal,truncated_normal,weibull_mean,gpd_cens,log_logistic_cens,
  gpd_trunc,log_logistic_trunc).
- reports/_helpers.py MOVED to src/idd_tc_mortality/viz/screening.py (Bobby: reusable code goes in
  lib, never reports/). reports/_helpers.py is now a temporary re-export SHIM (24 archival notebooks +
  2 qmd still import it; v1985 nb still uses `import _helpers as H` via shim; v2000 nb uses the lib
  import). Added there: calib rank direction, family_metric_dotplot, family_attainability_heatmap
  col_order arg.

## INTERMEDIATE stage set up + probed (v2000 = v1985 decisions, Bobby confirmed)
- survivors.json + intermediate_specs.json in 01-preliminary/tailvariants/. intermediate_specs =
  honest is_specs FILTERED to survivors = 633 specs (s1 3, s2 18, bulk 36, tail 576). Fed via
  --refined-specs (NOT --survivors — that flag filters the default MEDIAN enumeration, not our honest
  families). Coupled, model_predictions ON, all covs/thresholds retained → ~1,152 configs / 18 tasks.
- v1985 intermediate probes: bundle-1 (wf 605952) 80s/task 0.42 GiB; Bobby corrected — NEVER ask
  <5m, and pack work (increase --bundle-size) rather than pad time. Added `--bundle-size` flag to
  run_evaluate_orchestrate (default=BUNDLE_SIZE constant). bundle-4 re-probe (wf 605959, model_pred
  ON): max 288s (4.8min), max RSS 0.75 GiB, per-group ~72s. 18 groups/bundle-4 = 5 tasks/vintage.
- FULL-RUN sizing: **--bundle-size 4 --memory 1G --runtime 6m** (5 tasks/vintage; ~4.8min real work
  fills the mandatory 5m floor, 6m = modest contention margin on the measured 288s, NOT padding).
  model_predictions ~6.8k files/vintage (fine). Bobby: DO NOT run the full runs — commands ready.
- Bundle change => probe partials NOT reusable. Full runs need CLEAN dirs: v2000_intermediate is
  clean; v1985_intermediate has bundle-1 leftovers + there's a v1985_intermediate_probe dir (bundle-4)
  — clear/fresh before the full v1985 run.

## Next steps
1. Launch the two intermediate full runs (Bobby's go, NOT yet): run-evaluate-orchestrate
   --refined-specs 01-preliminary/tailvariants/intermediate_specs.json --data-path
   00-data/20260722_{v1985,v2000}/input.parquet --output-dir 02-evaluate/20260722_{v1985,v2000}_intermediate
   --bundle-size 4 --no-probe --memory 1G --runtime 6m (NO --skip-model-predictions). Use CLEAN dirs
   (bundle change => no partial reuse; clear v1985_intermediate + the _probe dir first).
   Both full intermediate runs DONE (v1985 6794-row dh_results + ~6.8k model_pred; v2000 similar,
   6680 model_pred).
2. Intermediate diagnostics notebooks BUILT: notebooks/20260722/dh_intermediate_diagnostics_{v1985,
   v2000}.ipynb (23 cells). Mirror 20260714_v1985/dh_intermediate_diagnostics.ipynb (drop-top-N
   trimmed pred/obs by storm) with 3 changes: import lib screening (not reports shim); load each
   config's 6 pred parquets ONCE and slice per-N (template re-read per N); fan the per-config loop
   over ProcessPoolExecutor(max_workers=12)+as_completed (Bobby caught that I'd shipped a SERIAL loop
   — the 2026-07-20 dh_final_diagnostics rewrite parallelized it, ~10-12x; pool output verified
   byte-identical to serial); added a per-tail-family view. Smoke-tested on v1985: gate
   822/1148 @ (0.1,2.0), mid reconstruction resolves, 10/10 pred files found, trims discriminate
   (near 1 @ n=0, 5-15x @ n=25). RATIO_BOUNDS/lo,hi are TUNE knobs; ends with a STOP-HERE decision cell.
   Builder: scratchpad/build_intermediate_nbs.py.
2b. Added screening.stability_attrition_curves (lib): sweeps a symmetric-log band [lo,1/lo],
   counts configs in-band at EVERY N, by threshold + family; wired into both notebooks (now 25 cells).
   FINDING (v1985): drop-top-N barely discriminates — deaths hyper-concentrated (top-5 storms=80.6%,
   top-25=89.2% of all deaths / 1584 storms). Median trimmed ratio: n0 0.64, n1 1.01, n2 2.46, n5 3.24,
   n25 5.66 — models carried by top 1-2 storms. All thresholds/families collapse together (lo~0.45-0.5
   at N=[0,1,2,3]; lo~0.35 at the inherited N=[0,5,10,25]). So drop-top-N = LOOSE sanity filter here,
   not a fine cull; informative N is 1,2,3 (not 5/10/25). Cull should lean on the OOS gate + coverage +
   n0/n1 per-family view. Weak signals only: thr 0.95 most robust, lognormal/log_logistic_cens most fragile.
3. NEXT: Bobby runs the two intermediate notebooks + makes the cull (thresholds + families). DONE:
   notebook N_SWEEP default reset to [0,1,2,3,5] + marked TUNE (drop-top-N markdown updated too), both
   notebooks edited DIRECTLY (scratchpad builder build_intermediate_nbs.py did NOT survive the session
   gap — notebooks are now the source artifact; no builder to regenerate from).
4. Finish _helpers migration on Bobby's go: sweep 24 archival notebooks + 2 qmd to lib import, delete shim.
5. Commit when ready: --bundle-size flag (run_evaluate_orchestrate), viz/screening.py move + shim,
   the 20260722 preliminary + intermediate notebooks, survivors.json/intermediate_specs.json, DECISIONS entry.

## Resume prompt
idd-tc-mortality honest-mean TAIL VARIANTS: machinery complete + committed (7c8aa6c A/C/D,
69cfdad B). 9 honest tail families registered (gamma/lognormal/truncated_normal already-mean +
weibull_mean D + gpd_cens/log_logistic_cens C + gpd_shadow/log_logistic_shadow A +
gpd_trunc/log_logistic_trunc B). Medians dropped from the sweep; median-vs-mean contrast recovered
post-hoc on the winner. B validated (identifiable, analytic gradient FD-verified). Cap plumbing:
tail_cap.py + needs_cap (predict) + fit_component cap-at-fit branch; reads exposed_population
(20260722_v1985). NEXT: build_tailvariant_specs.py (honest-only + B, A special-cased) + 2000->
re-ingest + two orchestrated sweeps (all need Bobby's go). OOS selects — never rank by mechanism.
2 pre-existing test_gpd.py failures belong to the separate GPD thread.
