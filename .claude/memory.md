# Session memory
Updated: 2026-07-14

## inner_draws design verification (afternoon)
Bobby wants an `inner_draws` inner-replication loop to average outcome-sampling noise (o1/b0 cells) per (storm_draw, tc_risk_draw) with fixed (beta, scale), before the sum-over-storms/div-100 collapse. Verified (not implemented): (1) coefficient reuse within a storm draw CONFIRMED — one DrawModel per (c,s) per storm draw, params frozen in StageDraw, rebuild bit-identical, predict never mutates; subset-vs-full predictions differ only at ~4.5e-13 rel (BLAS blocking ulp noise, 6/1903 rows; production predicts the whole slice in one call so even that doesn't arise). (2) Slot-in point = consolidated.py:106 (`dm.predict(...)` per toggle cell); row-level mean before groupby == his spec by linearity; better home = a batched `inner_draws` kwarg inside DrawModel.predict (hoist eta, re-sample only) to avoid R x deterministic recompute; keep seed=storm_draw path for R=1 (bit-compat with shipped). s1_flip/s2_flip/p_s2 return columns need semantics decision for R>1 (pipeline only uses 'deaths'). (3) Cost: predict-only, 7.0 ms/call on 1,903 rows (3.7 us/row); CLIMADA frames are static parquet inputs — never regenerated. (4) Convergence diagnostic (MLE params, 3 storms x {o1_b0,o1_b1,o0_b0}, R=1000): NOT converged at 1000 — running means still move >5% at R~850-950 (heavy-tail sawtooth: single draws dominate). Per-storm +/-5% needs >>1000; year/global aggregates average across storms so effective need is lower — rerun diagnostic at aggregate level once implemented. Artifacts: 04-predict/20260608/inner_draws_convergence.png + inner_draws_running_means.json. No `inner_draws` naming collision in repo. LIVE HAZARD flagged: run_model1_a0.py + predict_consolidated_probe.py default to 00-data/current (now 20260714_v1995, 2,278 rows) with folds pinned 20260608_final (1,903) — will mismatch if run as-is.

## Current task
Rerun-cycle groundwork DONE through data ingest. Three sibling vintages ingested from the new PSH_20260711 CSV; next up: the notebook-reorganization discussion, then launching preliminary fit/evaluate per vintage.

## Context / why
Bobby is rerunning the full pipeline (preliminary → intermediate → refined → final diagnostics) on new input data as a training-window sensitivity: same filters, min-year 1985 vs 1990 vs 1995. Earlier today: git baseline locked (tag `cycle-20260608`), vintage contract documented, ingest parameterized.

## Where we are
- **Three vintages on disk** under `00-data/`: `20260714_v1985` (3,024 rows, 1985–2023), `20260714_v1990` (2,658), `20260714_v1995` (2,278). All from `ibtracs_stage4b_pafs_admin0_with_deaths_PSH_20260711_sdi_isisland.csv` (3,882 raw = old 3,621 + 261 `no_wind_exposure`-flagged rows) with `--drop-no-wind-exposure` + `--min-year`. `current` → `20260714_v1995`.
- **is_island COPIED from 20260608, not re-pulled** — no conda env anywhere on the node has `db_queries` (documented GBD_shared_functions env lost it; full env sweep empty). Static covariate 2608/release 16 → content-identical (905 rows verified). Re-pull if db_queries returns.
- **Ingest script** now has `--source-csv`, `--vintage`, `--drop-no-wind-exposure` (bool-dtype hard guard). tests/ingest 6/6.
- Commits through `35ec535` (filter + DECISIONS entry) — check pushed state; earlier baseline (8 commits + tag `cycle-20260608`) pushed.
- File perms: new vintage parquets chmod 664 (June's were 600; everything runs as bcreiner so either works).
- Suite 708/719 earlier: 9 known-stale run_component + 2 undiagnosed (validate basins, stage_plots 3x3).

## Next steps
1. Push `6ee7fd9` + `35ec535` if not yet pushed.
2. **Notebook reorganization discussion** (Bobby: "untenable"): notebooks/20260515 (40M forks, uncommitted), archive/, jobmon_resources/, basin PDFs; repoint-in-place vs cp-per-cycle; preliminary notebook's two config cells; final notebook's hardcoded winner mids.
3. Launch cycle runs per vintage contract: `01-preliminary/20260714_v<Y>/`, `02-evaluate/20260714_v<Y>_refit/` etc. (fit/evaluate orchestrators take explicit --data-path/--output-dir; remember single-task probe first per scaling rules).
4. Carried: SR-31 guard + rake-stat canonical decision; dead fit stage deletion; stale run_component tests; 2 undiagnosed test failures.

## Resume prompt
Session 2026-07-14 (idd-tc-mortality, main): After locking the git baseline (tag cycle-20260608) and parameterizing ingest, ingested the three rerun vintages 20260714_v{1985,1990,1995} from the new PSH_20260711 CSV — all with the new --drop-no-wind-exposure filter (drops 261 flagged rows), differing only in --min-year (3,024/2,658/2,278 rows). is_island.parquet was copied from 20260608 because NO env on the node has db_queries anymore (static covariate, content-identical, flagged in DECISIONS). current → 20260714_v1995. Ingest tests 6/6. Next: notebook-reorg discussion, then preliminary fit/evaluate for the three vintages using the vintage contract (dirs keyed on 20260714_v<Y>, suffixes _refit/_survivors/_refined/_final).
