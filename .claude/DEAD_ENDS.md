# Dead ends

<!-- Append-only. Never delete or overwrite entries. -->

## YYYY-MM-DD: [approach name]
**What I tried:** One or two sentences.
**Why I stopped:** What failed or made it not worth continuing.
**Refs:** File paths or commit hashes if useful.

## 2026-05-13: Full decoupled refined-evaluate grid (12.5M configs)
**What I tried:** Submitted multiple decoupled refined-evaluate runs (32K-task partition, then 768-task partition) intending to evaluate every (s1_cov × s2_cov × bulk_cov × tail_cov) combination at the surviving thresholds × family/exposure choices. Built the supporting infra: `--decouple-covs` flag, task-file orchestration, OOS pickle cache + predict cache, `--skip-model-predictions`, `--scope` tick-tock, jobmon log routing.
**Why I stopped:** ~7 hours of debugging chased NFS contention, per-DH-config parquet writes, jobmon log routing — all valid fixes, none of which addressed the root issue: the analytical question (per-stage cov neighbour analysis of a chosen winner) actually needed ~160 DH configs, not 12.5M. The next-coarser sensible scoping (half-coupled cov chains under the locked-in pins) was 1,875 configs. Switched to those targeted approaches; the 12.5M-config grid was the wrong scale of solution to the wrong question.
**Refs:** Probe jobids: `35381562_1` (n=1, 59s), `35384207_1` (n=384, 5:11). Production failures: `35313205_*` (32K-task array, all TIMEOUT'd at 5h before resource fixes), `35450862_*` (32K-task array after fixes — all hit 15-min cap, every task killed). All infra built for this attempt is kept (`--decouple-covs`, `build_evaluate_tasks.py`, cache fixes, log routing) — none of it is wasted IF a future workload genuinely needs at-scale enumeration. The meta-lesson is preserved as Lesson 12 in `~/.claude/proposed_rules/jobmon-lessons-2026-05-13.md` ("ask whether the grid is the right size before optimising how to run it").

## 2026-05-13: 32K-task partition of the decoupled grid
**What I tried:** Partitioned the decoupled grid into 32,768 tasks (one per `(S1 spec × S2 cov × bulk spec × threshold)`), each running 384 DH configs. Per-task probe in isolation finished in 5:11 / 943 MB MaxRSS — within the 15-min wall and 5G memory budget by a comfortable margin.
**Why I stopped:** At full deployment scale (32K tasks concurrent), per-task wall time exceeded the 15-min Slurm cap on every task. The probe measured isolated cost; deployment added a concurrency factor (NFS contention from 32K simultaneous readers of overlapping pickles, plus Python-import contention from 32K simultaneous module loads off shared NFS) that the single-task probe couldn't see. The 768-task version (one per `(S1, threshold, tail_family, tail_exposure)`, 16,384 configs each) projected to ~3h per task even in isolation — workable but slow, and would still face concurrency overhead.
**Refs:** Failed array jobids in DEAD_ENDS entry above. Lesson preserved as Lesson 11 in the proposed-rules file ("probes measure isolation; deployment costs include concurrency").

## 2026-05-15: 1-tier predict_year_bin bundling 7 basins + inline aggs
**What I tried:** Original predict workflow used one jobmon task per (sd, sc, yb) that bundled the 7-basin predict loop + per-basin basin_agg + final year_bin_agg inline. Workflow 577423 submitted 6,090 such tasks. The orchestrator pre-scanned stage4_v2 to determine which (sd, sc, yb) tuples had basin data and pruned year_bins with no basin folders.
**Why I stopped:** Three structural failures forced a rewrite, not a patch.
  (a) Tasks took 50+ min on slow nodes; the 15m runtime ask wasn't enough and the 22.5m retry ask wasn't either (memory said "still failed"). With basin-bundling, retries redid every basin from scratch.
  (b) SDI loaded inside the tc_draw loop (100× per task). Future SDI (110M, cold-cache) made ssp245 cells run 20× slower than historical ones — invisible at the orchestration level.
  (c) Basin-list bug (SA instead of AU) caused every cross-basin aggregate to silently miss AU. The `if not p.exists(): warn; continue` pattern hid this for the entire workflow's run.
**Refs:** Workflow 577423 (jobmon). The replacement is the 4-tier DAG in [orchestrate.py](src/idd_tc_mortality/predict/orchestrate.py); the bug-by-bug fixes are documented in the 2026-05-15 DECISIONS entries. Implementation lessons in `~/.claude/active_designs/jobmon-shared-package-2026-05-13.md` (Implementation notes section).

## 2026-05-15: Per-leaf NFS walk for orchestrator task enumeration
**What I tried:** `_enumerate_year_bins` in `src/idd_tc_mortality/predict/orchestrate.py` walked every `(storm_draw, scenario, year_bin, basin)` combination and checked `.exists() and any(iterdir())` to decide which tasks to create.
**Why I stopped:** ~40k NFS metadata round-trips, multi-minute pre-scan before jobmon's bind even started. Bobby killed the first orchestrator run because of it. Replaced with a single stat per unique `(model, variant)` + an in-memory cross-join against `time_bins_df`. Trust the time_bins source and let `predict_tc`'s runtime checks handle missing basins.
**Refs:** Pre-fix orchestrate.py before the rewrite committed today.

## 2026-05-15: sacct for jobmon task resource queries
**What I tried:** `notebooks/predict_resource_usage.ipynb` originally pulled per-task runtime + RSS via `sacct -u bcreiner --starttime ...`.
**Why I stopped:** sacct returned 0 rows for our workflow even when given specific known jobids. Cluster context mismatch (jobmon-installer-ihme submits to a Slurm cluster sacct's default config doesn't see). After debugging through `--starttime` format, `--clusters=all`, and `-j <jobid>` permutations with no rows landing, switched to jobmon's own CLI (`jobmon workflow tasks -w 577423 -s done -o json` + `jobmon task status -t ... -o json`) which has direct access to its own DB.
**Refs:** Corrections log entries in `~/.claude/projects/-mnt-share-homes-bcreiner-repos-idd-tc-mortality/corrections/2026-05-15.md` under "bouncing sacct flags off the user".

## 2026-05-15: `pip uninstall jobmon` to clear the resolver warning
**What I tried:** When `pip install jobmon-installer-ihme 10.11.6` produced a dependency-resolver warning about the old `jobmon 3.2.4` shim conflicting with `jobmon_core 3.7.16`, Claude recommended `pip uninstall jobmon` based on inspecting the console_script entry points (both packages declared `jobmon` as a CLI).
**Why I stopped:** Two packages declaring the same console_script entry point SHARE the binary on disk — whichever was installed first owns it. Uninstalling that owner removes the binary even though other packages still mention the same entry in metadata. Result: `/.../bin/jobmon` disappeared and the notebook's subprocess calls broke. Fix: `pip install --force-reinstall --no-deps jobmon_client` to recreate the script.
**Refs:** Corrections log entry "uninstalled jobmon and broke the CLI".

## 2026-05-15: Predicting task duration from mtime spread of basin output files
**What I tried:** As a workload calibration check, computed `max(basin_mean_mtimes) - min(basin_mean_mtimes)` per `(sd, sc, yb)` as a proxy for task wall time.
**Why I stopped:** The proxy captures only the spread between FIRST and LAST basin completion, not actual setup time, not Python startup, not retry restarts. Outliers (e.g., a single (sd=66, historical, 2007-2010) bundle at 6,545 seconds = 109 min) were misleading because they spanned multiple retry attempts. Replaced with: scan input `admin_level_exposure_*.parquet` bytes per (mv, sc, yb) as workload, query `jobmon task status` for per-attempt runtime, regress runtime against workload to fit per-scenario calibration models. Notebook: `notebooks/predict_resource_usage.ipynb`.
**Refs:** Corrections log entry "mtime-spread proxy instead of measuring workload".

## 2026-05-17: Symmetric calibration gate `[0.9, 1.1]` on post-2000 data
**What I tried:** First pass at the post-2000 preliminary screening reused the all-years `full_pred_obs_ratio_oos ∈ [0.9, 1.1]` calibration gate (2026-05-12 decision).
**Why I stopped:** On post-2000 data the gate let through only 7,020 of ~155K configs (4.5%) — and the trim-top-N analysis revealed they were exactly the *worst* configs structurally. Median trim_n=1_oos = 1.20 across surviving configs means they over-predict the bulk-of-storms by 20% to compensate for under-predicted mega-events. The symmetric gate was selecting *for* compensating-error overfitting. Replaced with asymmetric `[0.1, 1.5]` per the 2026-05-17 DECISIONS entry.
**Refs:** `dh_intermediate_diagnostics_post2000.ipynb` trim-sweep cells.

## 2026-05-17: Time-series metrics (`cor_ts`, `beta_p_ts`) as hard screening gates
**What I tried:** Added three time-series metrics to evaluate output and planned to use them as a screening gate analogous to `full_pred_obs_ratio_oos ∈ [0.9, 1.1]` (e.g., `cor_ts_oos >= 0.5`).
**Why I stopped:** Year-to-year death variability is dominated by mega-event-year stochasticity (Nargis 2008 = 138K of ~210K total deaths) that storm-level features can't predict. Achievable `cor_ts_oos` ceiling across ALL configs is ~0.44. Any threshold filter `cor_ts_oos >= 0.5` disqualifies everything; lower thresholds are arbitrary. Metrics stay as horror-spotters in the diagnostic notebooks but are NOT hard gates.
**Refs:** DECISIONS 2026-05-17 entry "Time-series metrics computed but NOT used as hard screens".

## 2026-05-17: Single-config half-coupled task partitioning at refined scale
**What I tried:** Used `build_half_coupled_tasks.py` (existing) to enumerate 17,500 DH configs as 17,500 single-config evaluate tasks for the refined run.
**Why I stopped:** ~20-30s per-task overhead (Python startup + manifest load + parquet load) × 17,500 tasks = ~hours of wasted overhead, ignoring real work. Wrote `build_half_coupled_multiconfig_tasks.py` which partitions by (s1_cov × s2_cov × bulk_cov × threshold) and uses worker Cartesian over bulk_exposures × tail_specs within each task. 512 tasks × ~34 configs each. Per-task wall 19s median in practice. Total evaluate wall ~3-5 min on free cluster.
**Refs:** `src/idd_tc_mortality/evaluate/build_half_coupled_multiconfig_tasks.py`.

## 2026-05-17: 1M-file flat-directory model_predictions writes
**What I tried:** Ran preliminary evaluate (post-2000) with `--skip-model-predictions=False`, expecting ~1M small parquet files (168K configs × 6 fold_tags) in a single flat `model_predictions/` directory.
**Why I stopped:** NFS metadata-server thrash. `ls` on the dir hung mid-run; tasks all hit the 15m runtime cap because each per-file `mkstemp` + `os.replace` was serializing on the directory inode lock under cohort contention. Killed workflow, wiped, resubmitted with `--skip-model-predictions` for the preliminary pool. Did a separate small-scale survivors-only re-run (864 configs × 6 = ~5K files in one flat dir — NFS-fine) to get per-row predictions for the post-cull subset. The proper code-level fix is directory hashing (`model_predictions/<mid[:2]>/<mid>_*.parquet` to spread across 256 subdirs) or per-task aggregated parquets; **not yet implemented**, parked.
**Refs:** Corrections log 2026-05-17 "notebook live-file edit destroyed kernel state" (the cleanup also clobbered the in-progress trimmed-ratio sweep).

## 2026-05-17: Estimating task runtime by extrapolating from prior runs
**What I tried:** Quoted "30-60 min" wall for refined fit, "2 min" max for evaluate-survivors tasks, "10 min" remaining for the live sweep at total=61s. All extrapolated from preliminary-cycle per-config wall estimates.
**Why I stopped:** Off by 1-2 orders of magnitude in every direction. Refined fit median was 11s (not minutes). Evaluate-refined median was 19s (not 1-2 min). The live-sweep mid-task ETA ignored the 40s startup-cost portion of total elapsed time. The pattern: I keep extrapolating from a single prior data point, ignoring that overhead/setup costs don't scale with workload. Rule going forward (logged in corrections): probe-first, sacct-based sizing for any new run, no more extrapolation.
**Refs:** Corrections log 2026-05-17 multiple entries on extrapolation failures.

## 2026-05-19: Global parallel `find` for predict-output cleanup
**What I tried:** First rewrite of `cleanup_au_for_repredict.sh` ran 5 background `find` commands in parallel, each scoped to a different file-name pattern across the full storm_draws tree.
**Why I stopped:** Each `find` still traversed the entire 39K-entry tree independently — five copies of the same walk, 50 minutes wall, no speedup over serial. Replaced with per-storm_draw fan-out via xargs `-P` (each worker handles one sd's ~450 entries only).
**Refs:** DECISIONS 2026-05-19 entry "Per-storm_draw fan-out is the parallelization unit".

## 2026-05-26: recover_partial_run on shared entries_dir
**What I tried:** Called `recover_partial_run(entries_dir, terminal_path=bundle_partial_path, intermediate_glob="entry_*.parquet")` at the top of each bundle worker, with `entries_dir = partials/entries/` shared across all 180 concurrent tasks.
**Why I stopped:** All workers glob `entries_dir` at startup, producing overlapping candidate lists. Worker A selects `entry_X.parquet` as the latest-mtime intermediate and deletes it; worker B then stats it inside `max(candidates, key=lambda p: p.stat().st_mtime)` → `FileNotFoundError`. Also redundant: `atomic_write` + `subtask_skip` already handle the OOM-mid-write case without a pre-flight scrub.
**Refs:** `src/idd_tc_mortality/evaluate/run_evaluate.py` bundle block. Bug report sent to idd-tools CC: document that `out_dir` must be worker-private, or wrap stat in try/except and skip missing files.

## 2026-05-19: STATUS.md as a source of truth for CLI invocations
**What I tried:** Echoed yesterday's `--storm-draw-dir` from STATUS.md back into today's resume context without checking whether the path was correct.
**Why I stopped:** STATUS.md from 2026-05-18 documented a session where the prior CC had passed an explicit `--storm-draw-dir` flag that overrode the (correct) code default. The bad path was preserved in STATUS as if it were canonical; following it lost ~3 hours cleaning a garbage directory. Code default is the source of truth for paths; STATUS.md narrates what happened, including mistakes. Verify against code before re-using a flag value from STATUS.
**Refs:** Today's session — non-_v2 vs `_v2` predict path confusion.

## 2026-06-09: Load-based evaluate at 180–360-way concurrency
**What I tried:** Ran the evaluate (`BUNDLE_SIZE=2`, 180 tasks, then `BUNDLE_SIZE=1`, 360 tasks) using `load_result()` to read pre-fitted .pkl files from NFS per component per fold. Probed at 2, 5, 10, 50-task concurrency levels.
**Why I stopped:** At 10-way: 2.5 rows/s (6× contention slowdown vs 5-way 15 rows/s). At 50-way: startup scan took 30–40s/task (22,854 × N_tasks NFS stat calls) and steady-state throughput collapsed to ~1 row/s. At 360-way: TIMEOUT at 20m, 30m, 45m walls regardless of memory ask. The root cause (22,854 × N stat calls + per-config .pkl reads overwhelming NFS) is structural — concurrency throttling could only manage around it, not fix it.
**Refs:** `02-evaluate/20260608/` (old failed bundles 0–6); `src/idd_tc_mortality/evaluate/run_evaluate_orchestrate.py` (now redesigned).

## 2026-06-09: Big-bundle / concurrency-throttle approach for evaluate
**What I tried:** Explored reducing concurrency (cap `max_concurrently_running`) and/or increasing `BUNDLE_SIZE` to reduce total NFS reads via within-task caching.
**Why I stopped:** Probe data (5 tasks=15 rows/s, 10 tasks=2.5 rows/s, 50 tasks=1 row/s) confirmed the contention was too severe even at low concurrency. The correct fix — re-fitting in-memory — eliminates the read storm entirely rather than throttling around it, with effectively no cost (fitting is ~0.1s vs the NFS read that took arbitrarily long under contention).
