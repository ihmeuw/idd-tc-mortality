# Decisions log

<!-- Append-only. Never delete or overwrite entries. -->

## YYYY-MM-DD: [short title]
**Decision:** What was decided.
**Why:** Reasoning. Alternatives considered if relevant.
**Revisit if:** Conditions that would make this worth reconsidering.

## 2026-04-05: scaled_logit fit via WLS on link-transformed outcome
**Decision:** Implemented scaled_logit.fit() using sm.WLS on z = log(y/(threshold-y)) rather than sm.GLM with ScaledLogitLink as the family link.
**Why:** statsmodels Gaussian family validates links against an approved list [Log, Identity, InversePower] and raises ValueError for any other Link subclass. WLS on the link-transformed outcome is mathematically equivalent to the Gaussian GLM with this link — for Gaussian errors the MLE with any link reduces to WLS on the working response, which equals the link-transformed outcome when the link is correctly specified.
**Revisit if:** statsmodels adds a check_link=False option to Gaussian family, or we need IRLS for a future variant where the residuals are not Gaussian on the link scale.

## 2026-04-06: GPD shape parameter xi stored in meta, not params
**Decision:** xi is stored in `meta["shape_param"]` and excluded from `FitResult.params`. Only the scale model betas (one per X column) are in `params`.
**Why:** Consistent with the NB precedent (alpha in `meta["dispersion_params"]`). The params array is intended for the mean/linear model coefficients. xi is a distributional parameter, not part of the linear predictor for sigma. It also has a different role: betas enter the linear predictor, xi enters the mean formula (sigma/(1-xi)) directly.
**Revisit if:** The uncertainty module is implemented. Uncertainty draws will require sampling from the joint distribution over [beta, xi] using `hess_inv`, which covers all n_params = X.shape[1] + 1 entries. The draw logic must therefore read xi from `meta["shape_param"]` and reconstruct the full parameter vector [beta, xi] before drawing from the multivariate normal. This is non-obvious and must be documented when the uncertainty module is built.

## 2026-04-07: OOS per-component metrics computed fold-by-fold, averaged per seed
**Decision:** For OOS rows in dh_results.parquet, per-component metrics (s1_*_oos, s2_*_oos, bulk_*_oos, tail_*_oos) are computed by calling predict_one_component on each fold's held-out rows, then averaging across the n_folds folds within each seed. All OOS metric columns use an `_oos` suffix to be self-describing without inspecting fold_tag. The full_*_oos and fwd_*_oos metrics use the assembled held-out prediction (each storm's prediction from the single fold where it was held out).
**Why:** Per-component OOS metrics were specified from the start — "metrics of performance at each stage on the data that was used in that stage." An earlier implementation deferred this (leakage), then partially fixed it (only full_/fwd_ were OOS). This is the complete fix. The fold-level loop in run_evaluate.py handles both the assembled OOS prediction stitching and per-component fold metrics in a single pass, avoiding redundant component loading. The _oos suffix prevents IS/OOS column name collisions and makes downstream filtering by column name rather than fold_tag possible.
**Revisit if:** step_04_select finds the sparse parquet (IS rows lack _oos columns, OOS rows lack non-_oos component columns) inconvenient. If so, pivot to same column names with fold_tag as the sole discriminator.

## 2026-04-14: Removed step_0N_ prefixes from pipeline directories
**Decision:** Renamed step_01_grid → grid, step_02_fit → fit, step_03_evaluate → evaluate, step_04_select → select. Created viz/ submodule for visualization.
**Why:** The numbered prefix implies a linear one-pass pipeline. The actual workflow is two iterative cycles (preliminary and refined) that share the same infrastructure. In the preliminary cycle you enumerate → fit → evaluate → select → decide on culling. In the refined cycle you reuse already-fitted components and already-evaluated model combinations — only running fit and evaluate on genuinely new combinations. The cycle logic lives in notebooks, not in the pipeline code. Numbered steps suggested a wrong mental model of how the pipeline is used.
**Revisit if:** The pipeline is restructured into a formal DAG with explicit dependency tracking, at which point numbered steps or named nodes might make sense again.

## 2026-04-14: Basin recoding: empty string → NaN → 'NA'
**Decision:** The raw CSV stores North Atlantic basin as empty string ''. Read with
keep_default_na=False, na_values=[''] so only blanks become NaN, then explicitly
recode NaN → 'NA'. Also force dtype={"basin": str} before any NA inference.
**Why:** pandas default NA parsing converts the literal string 'NA' (North Atlantic)
to NaN. keep_default_na=False prevents that. The na_values=[''] makes empty strings
(actual missing values) NaN so they can be explicitly recoded to canonical 'NA'.
**Revisit if:** Source CSV format changes or a new vintage uses a different encoding.

## 2026-04-15: Gamma GLM IRLS instability with tiny y values
**Decision:** gamma.py fit() now passes explicit start_params to glm.fit():
intercept = log(weighted_mean(y)), all slopes = 0.
**Why:** statsmodels Gamma GLM with log link initializes IRLS with mu = y. When y
is on the order of 1e-8 (tail excess rates) or 1e-7 (bulk death rates), eta=log(y)≈-18.
Combined with log_exposed values up to ~19 in the design matrix, the first WLS step
drives some fitted mu values to zero, making IRLS weights (1/mu²) blow up to inf.
Starting at mu=weighted_mean(y) gives a stable, finite initial eta and avoids the
overflow entirely.
**Revisit if:** statsmodels changes its default initialization behavior, or we switch
to a different optimizer (e.g. BFGS) that doesn't use IRLS.

## 2026-04-05: Weighted Beta MLE via loglikeobs subclass
**Decision:** Subclassed BetaModel and overrode loglikeobs to apply normalized per-observation weights rather than using var_weights.
**Why:** var_weights is silently ignored by BetaModel in statsmodels 0.14.6 — verified by test showing weighted and unweighted fits give identical results (SSE equal to 6 decimal places). loglikeobs scaling is the correct approach: multiplying per-observation log-likelihoods by weights normalized to sum to n_obs gives the weighted MLE, and uniform weights produce identical results to unweighted (sanity check).
**Revisit if:** statsmodels adds native var_weights support to BetaModel in a future version.

## 2026-04-15: Level filter default = 'all' (use all admin levels)
**Decision:** 01_prepare_input_data.py now supports three level-filter modes (all/level3/aggregate), defaulting to 'all'.
**Why:** level-3 only = 1,383 rows; all levels = 9,186 rows. Subnational rows are valid TC observations and dropping them was an arbitrary holdover from the old pipeline. 'aggregate' exists if we later decide subnational rows should be rolled up rather than treated as independent observations.
**Revisit if:** Subnational rows introduce dependence structure that biases CV metrics (e.g. many level-4 rows from the same storm count as multiple OOS observations).

## 2026-04-15: Graceful non-convergence — record failure, don't crash
**Decision:** When a component fit fails to converge (e.g. gamma IRLS NaN weights on a small OOS fold), fit_component.py should catch the error and save a non-converged result rather than raising.
**Why:** OOS folds with extreme log_exposed values (~21.5 from 2.3B exposed) and small tail subsets reliably blow up gamma IRLS even with stable start_params. These failures are informative — that model/fold combination doesn't work — and should propagate to evaluation as missing/NA rather than aborting the whole workflow.
**Revisit if:** Non-convergence rate is high enough to bias model selection (e.g. >10% of OOS folds for a given family are non-converged).

## 2026-05-06: Always verify remote state with `git remote -v` before claiming a repo is unpushed/un-PR-able
**Decision:** Before making any claim about a repo's GitHub/remote state, run `git remote -v` and `git ls-remote origin` directly. Never infer remote configuration from `git status` upstream messages alone.
**Why:** This session lost an hour of unbacked-up work because Claude inferred from `git status` saying "Your branch is based on 'origin/master', but the upstream is gone" that there was no remote configured at all, and told the user "no remote to PR against." In fact, `origin → https://github.com/ihmeuw/idd-tc-mortality.git` was configured the entire time — only the upstream *branch* was gone. "Upstream branch is gone" and "remote does not exist" are different facts. Conflating them led to several hours of additional work being added on top of an unpushed initial commit.
**Revisit if:** Git changes its `git status` messages to make the distinction unambiguous, or a different remote-inspection command becomes the canonical one.

## 2026-05-06: Push to remote at session start when uncommitted/unpushed work exists
**Decision:** At session start, if `git log` shows commits that are not on the remote (or no commits exist on remote), surface this immediately and offer to push. Do not begin substantive work on top of unpushed history without flagging it.
**Why:** Same incident as above. Today's session began with zero commits; we made one large initial commit and then proceeded for ~3 hours of work (notebook rewrites, ingest, fit-code changes, smoke test) before realizing nothing was on GitHub. Even after the initial commit, the natural moment to push was missed because a wrong claim about remote state had been made earlier. Treating "is this code backed up?" as a first-class concern, surfaced loudly, prevents this.
**Revisit if:** The repo moves to a workflow where local-only commits are deliberate (e.g. a private branch), in which case the rule should be conditional on branch name.

## 2026-05-08: FLAG — convergence criteria too liberal (revisit at final-round selection)
**Decision:** Defer fix until the final-round model selection. Note as a known issue, drill in then.
**Why:** Diagnostic on the 2026-05-07 evaluate run found that lognormal tail with `tail_exposure_mode='free'` at threshold=0.80 produces RMSE values 1.4-23 across all OOS folds — finite (not caught by the `np.isfinite` cull) but ~10-100x larger than every other family at the same threshold. Same lognormal-free configs are TOP performers at thresholds 0.75 and 0.85, so it isn't a structural lognormal failure — it's a fitting-stability failure where statsmodels' `glm.fit().converged` returns True but the IS coefficients extrapolate catastrophically on OOS folds. Gamma shows the same pattern (max RMSE 365 at threshold=0.80). The current convergence check is necessary but not sufficient: a fit that "converges" can still produce nonsense predictions.
**Fix candidates (for the final round):**
- Tighten `fit_component.py`: after IRLS, sanity-check fitted IS predictions (e.g., max ≤ 100x max observed) and mark `converged=False` if violated. Cleanest — catches the failure where it happens.
- Drill into a single bad seed/fold for one of the failing configs to confirm the IRLS-extrapolation hypothesis before patching.
- Add a post-evaluate cull: drop any config whose OOS metric exceeds, say, 10x the cross-family median. Easier but doesn't fix the root cause.
**Revisit if:** Building the final-selection notebook, OR if a future preliminary run shows the same failure mode in a different family/threshold/exposure combination (suggesting a more general fitter hardening is warranted sooner).

## 2026-05-08: Tail — drop nb and poisson; gamma + weibull on the bubble
**Decision:** Drop `nb` and `poisson` from the tail family grid. Tentatively keep `gpd`, `log_logistic`, `lognormal`, `truncated_normal`, `weibull`, `gamma` — but flag `gamma` and `weibull` as suspicious pending closer look.
**Why:**
- `nb` and `poisson` are the only two count-based tail families and are uniformly bad: best-rank-anywhere is 11+ for nb and 3 for poisson with most cells in the 26–66 range. They cannot reach top-10 across most (metric × threshold) cells regardless of covariate or exposure_mode. Also structurally the wrong choice for tail — they have no threshold awareness so predicted rates can fall below threshold (mismatch with the conditional eval set).
- `gamma`: best-rank ranges from 2 to 45 across cells. Strong on rate metrics at low thresholds but breaks down in MAE-count at low thresholds (rank 32–45). Suspicious because the ceiling is high but the failure mode is broad.
- `weibull`: best-rank ranges from 1 to 16. Doesn't break top-10 in some cells but never goes truly bad. On the bubble — keep for now, may drop if downstream comparison favors a tighter family set.
- Survivors so far: `gpd`, `log_logistic`, `lognormal`, `truncated_normal` are consistently in or near top-10 across most cells.
**Revisit if:** Closer inspection of gamma's failure cells reveals a fixable issue (e.g. specific exposure_mode), or if weibull's near-but-not-in-top-10 behavior turns out to be a stable secondary option worth keeping for diversity.

## 2026-05-08: Bulk — exposure mode = free or free+weight
**Decision:** For bulk (within the locked-in beta + scaled_logit families), restrict exposure_mode to `free` and `free+weight`. Drop `excluded` and `weight`.
**Why:**
- Across all thresholds and headline metrics, `free` and `free+weight` cells are uniformly the best-performing for both beta and scaled_logit. `excluded` and `weight` produce visibly worse mae_rate / rmse_rate / cor_rate at every threshold.
- Same exposure_mode story as S1 / S2: there IS exposure information that should be used, and forcing the slope to a specific value (or dropping it entirely) hurts. `free` lets the data choose; `free+weight` adds population weighting on top.
- Both kept open because the choice between "weighted vs unweighted IRLS objective" is a downstream call that may depend on calibration goals — both produce similar rankings, so leave the option until a final decision is forced.
**Revisit if:** A new run shows `excluded` or `weight` matching `free` for these two families on a key metric, or downstream calibration analysis prefers one IRLS weighting over the other definitively.

## 2026-05-08: Bulk — narrow to beta + scaled_logit (exposure mode TBD)
**Decision:** Restrict bulk to families `beta` and `scaled_logit`. Drop `gamma`, `lognormal`, `nb`, `poisson` from the bulk grid for downstream selection. Exposure mode and covariate choice still open.
**Why:**
- `beta` and `scaled_logit` consistently dominate the top-10 across all thresholds and headline bulk metrics (mae_rate, rmse_rate, cor_rate, mae_count, cor_count). Trajectories are nearly flat at rank ≤ 10 across the full threshold range.
- These are the only two rate-bulk families that enforce predictions strictly in (0, threshold) by construction (beta normalizes y by threshold; scaled_logit's link maps R → (0, threshold)). gamma/lognormal can predict above threshold; nb/poisson are count families with no threshold awareness. The top-rank dominance and the bound-enforcing structure align — bulk's job is the conditional expectation given rate < threshold, and the bounded families don't extrapolate out of support.
- Cross-threshold Kendall tau matrix shows τ > 0.78 for adjacent thresholds and τ ≥ 0.55 even between extremes (0.70 vs 0.95) — rankings are stable enough that this decision is robust to threshold choice.
**Revisit if:** A new evaluate run shows a different rate-bulk family climbing into the top-10 across multiple thresholds, or downstream analysis flags out-of-support bulk predictions as a non-issue.

## 2026-05-08: S2 — pick logit + free
**Decision:** Use `s2_family='logit'` and `s2_exposure_mode='free'` for S2 across all thresholds. Drop `cloglog` and `excluded`.
**Why:**
- Logit and cloglog agree to within ~0.001 AUROC in every (threshold × exposure × cov) cell. Same identifiability story as S1: AUROC is rank-invariant and the MLE produces nearly-identical fitted probabilities for these two links on this data.
- `free` dominates `excluded` at every threshold (e.g. threshold=0.80: 0.918 vs 0.708; threshold=0.95: 0.957 vs 0.803). Exposure information is informative for S2.
- `free + no_covariates` reaches 0.89 AUROC at threshold=0.80 and 0.95 AUROC at threshold=0.95. Most of S2's discriminative signal is coming from `log_exposed` alone — likely because small populations have high-variance rates and are more likely to land in the tail. Covariate set adds only ~1-2 AUROC points on top of exposure.
**Revisit if:** A new data vintage changes the exposure-rate variance relationship, or downstream interpretation requires hazard-ratio coefficients.

## 2026-05-08: S1 — pick logit + free
**Decision:** Use `s1_family='logit'` and `s1_exposure_mode='free'` for S1. Drop `cloglog` and the `offset` and `excluded` exposure modes for S1 going forward.
**Why:**
- Free-exposure-mode dominates excluded (s1_auroc 0.773 vs 0.682) and offset (0.55) on the 2026-05-07 preliminary run, confirming there IS exposure information in `log_exposed` but the per-person hazard slope is not 1 (offset is misspecified). Drops "more exposed → lower per-person rate".
- Logit and cloglog give identical results in `free` mode on both AUROC (0.773 vs 0.773) and Brier (~0.178 vs ~0.177) because (a) AUROC is rank-invariant and (b) the MLE pins down nearly-identical fitted probabilities when data sits in moderate-p range.
- Logit is canonical link for Bernoulli, gives log-odds interpretation, has cleaner small-sample behavior. The cloglog mechanistic argument (proportional hazards / Poisson process) only pays off with offset, which we've already ruled out.
- Uncertainty machinery is identical for both: both use `sm.GLM` underneath and have access to `cov_params()`. Currently both store `cov=None`; turning it on is a one-line change for either.
**Revisit if:** New data pushes P(any_death) into the saturation regions where logit and cloglog diverge in fitted probability surface, or downstream analysis needs hazard-ratio interpretation specifically.

## 2026-05-06: Slurm resource asks tightened from 2G/30m to 1G/5m
**Decision:** `_SLURM_RESOURCES` in `orchestrate.py` now asks for 1 GiB memory and 5 min runtime per spec (was 2 GiB / 30 min).
**Why:** `seff` on ~18k completed jobs from the 2026-05-06 preliminary run reported median 9 s runtime / 0.24 GiB memory, max 39 s / 0.46 GiB. The original asks were over-allocated by roughly 200× on runtime and 4× on memory, hemorrhaging fairshare. New values give ~8× headroom over observed max runtime and ~2× over max memory — comfortable for outliers without waste. Also: smoke test for cluster-bound code should now include a Slurm-submitted sample (not just `--local`) so `seff` calibration happens before the full submission, not after.
**Revisit if:** A future grid expansion (more covariates, finer thresholds, larger data) pushes per-spec runtime past ~3 min consistently, or memory past ~700 MiB.

---

## 2026-04-14: Old-repo assessment — what to port and how

Re-read 10 files from idd-climate-models. Full findings below. Written to survive context loss.

### Files read

**1. orchestrate_dh_expanded.py**
Jobmon orchestrator for the old stage-based DH run. Uses env-var path injection (STAGE_MODELS_DIR, STAGE_RESULTS_DIR), string stage IDs, pkl cache. Already fully superseded by step_02_fit/orchestrate.py. Nothing to port.

**2. analyze_dh_exhaustive.py (v2)**
Exhaustive enumeration of 786K model combinations from pkl cache. Assembles OOS predictions per seed/fold using raw `np.random.default_rng(seed).integers()` fold assignment — not basin-stratified. Threshold in **count space** (`np.percentile(total_deaths, threshold_pct)` on positive rows). Computes cov_1/5/10/20, pred_obs_ratio. Also computes IS metrics separately. The entire pattern (enumerate → precompute predictions → assemble metrics) is what run_evaluate.py + model_selection.py now handle automatically. Nothing to port.

**3. analyze_dh_exhaustive_v3.py**
V3 variant: rate-based threshold (`death_rate = total_deaths / exposed_population`), same covariate set across all 4 stages, rate-space prediction (predict with exposure=1, clip bulk to (0,u], tail to [u,inf), multiply by exposure). Only 180 models (3 covar × 6 threshold × 4 bulk × 3 tail). This design — rate-based threshold, rate-space clipping — is already implemented in the new repo (predict_component.py). Nothing to port; confirms architecture choice.

**4. model_selection.py (old)**
Multi-criteria pipeline: `borda_rank`, `pareto_frontier`, `kendall_tau_heatmap`, `friedman_nemenyi` / `pairwise_dominance_summary`, `topsis_rank`, `cluster_configurations`, `winner_profile`, `run_full_pipeline`. 7 methods, ~1050 lines. Column naming convention uses old prefixes: `oos_mae_rate`, `is_pred_obs_ratio`, `oos_cov_5` — different from new repo's `mae_rate`, `mae_rate_is`, `cov_5`. The new model_selection.py already covers 7 methods. The Condorcet/pairwise dominance method is the 8th (present in old file as `pairwise_dominance_summary`, also used in the topsis notebook as "Condorcet winner"). The new file's column name mapping must use the new schema. Nothing to port verbatim; new file already exists.

**5. model_query.py (old)**
`ModelQuery` class: `get()`, `enumerate()`, `neighbors()`, `compare()`, `diff()`, `compare_to_reference()`. Key observations:
- `get()` / `enumerate()` filter by column equality — clean, portable
- `neighbors()` uses token-based similarity: splits covariate string on `_`, computes symmetric difference. **This will not work in the new repo.** The new repo stores covariates as JSON strings (e.g. `'["wind","sdi","basin"]'`), not underscore-delimited strings like `wind_sdi_basin`. Must be redesigned.
- `compare()` and `diff()` are clean utility wrappers — port verbatim
- `compare_to_reference()` is useful — port verbatim
- Constructor default distributions use old names: `statsmodels_logistic`, `statsmodels_nb`, `statsmodels_gamma`. New repo uses: `cloglog` for S1/S2, `nb` / `gamma` / `lognormal` for bulk/tail. Also: new schema has `bulk_family` / `tail_family` not `bulk_dist` / `tail_dist`, and `threshold_quantile` not `threshold_pct`.

**6. stage_plots.py (old)**
`StagePlotter` class. Key observations:

*What's solid:* The `vet_stage()` 3×3 plot layout (wind overall/by-basin/by-island, SDI overall/by-basin/by-island, basin beeswarm, island beeswarm). The private `_plot_cont_*` and `_plot_cat_beeswarm` methods. The `_make_pred_df()` reference-value DataFrame approach (median/mode for fixed covariates, then override the varying covariate). These are well-designed and worth porting.

*What's broken (3 issues):*

**Bug 1 — threshold inconsistency (count vs rate space):** `_get_subset_for_stage()` computes the s2/bulk/tail threshold as `np.percentile(nonzero_deaths, threshold_pct)` on raw `total_deaths` (count space). But `predict_df()` computes the threshold as `np.percentile(rates_all[pos_mask], thr_pct)` on death rates (rate space). These give different subsets for the same `threshold_pct`. In the new repo, the threshold is rate-based throughout — `_get_subset_for_stage()` must be rewritten to use rate-based threshold to match.

**Bug 2 — predict() wired to old distributions API:** `_predict()` calls `sm.add_constant(X, has_constant='add')` and `get_feature_names_from_covars()` (an old-repo function that doesn't exist in the new repo), then calls `fitted_model.predict(X)` directly on the statsmodels fitted object. In the new repo, fitted models are `FitResult` objects and prediction goes through the distribution class's `.predict()` method. The prediction pathway must be rewritten from scratch.

**Bug 3 — predict_df_oos() re-implements fold assignment:** `predict_df_oos()` reconstructs fold splits via `np.random.default_rng(seed).integers()` (not basin-stratified) and re-runs prediction from PKL files. In the new repo, component-level OOS predictions are stored in `component_predictions/` parquets and assembled predictions in `model_predictions/`. `predict_df_oos()` should read those instead of re-implementing fold logic.

*What to do:* Port the 3×3 layout and plot helpers. Rewrite `_get_subset_for_stage()` for rate-based threshold. Rewrite `_predict()` to use new distributions module + FitResult. Replace `predict_df_oos()` with a reader over stored parquets. `predict_df()` (IS predictions) can be largely ported with updated distribution calls.

**7. dh_model_analysis.ipynb** (`direct_relative_risk/`)
An older exploratory notebook — pre-dates the final DH design. Contains: non-stratified CV, ML model comparisons, coefficient tables, and early uncertainty estimation work (cells 24–29: drawing from the beta covariance matrix to get draws of the mean function, visualizing distributional uncertainty). The uncertainty approach — load `params` and `hess_inv` from the fitted model, draw `[beta, phi]` jointly, propagate through the prediction formula — is the correct blueprint for the future uncertainty module. Not worth porting to the new vignette. Flag as reference for step_05_uncertainty.

**8. dh_model_selection_topsis.ipynb** (`notebooks/`)
The main v2 selection notebook. 37 cells. Flow: load dh_exhaustive_expanded.csv (593,920 models, filtered to 16,044 by pred_obs_ratio and mae_rate) → `run_full_pipeline()` → Condorcet vs TOPSIS winner comparison → coefficient display → export top 30. TOPSIS winner was: `s1=wind_sdi_basin_island, s2=sdi_basin, bulk=sdi, tail=none` (nb/gamma). Uses old column names throughout. This is the direct template for the new vignette notebook, updated for new schema.

**9. stage_vetting.ipynb** (`notebooks/tc_models/`)
Clean 13-cell notebook. Exact structure: load CSV → filter (pred_obs_ratio in [0.95, 1/0.95], mae_rate <= 5) → build ModelQuery → `mq.get(...)` to specify a model → `plotter.vet_stage()` for each of the 4 stages. This is the minimal vetting workflow and maps directly onto the new vignette.

**10. dh_v3_results.ipynb** (`notebooks/tc_models/`)
V3 analysis notebook. **Critical finding:** With the rate-based threshold and rate-space clipping design, pred_obs_ratio is severely wrong — median values range from 0.10 to 0.39 across thresholds (model predicts only 10–40% of observed total deaths). Only 10/180 models pass the filter even at relaxed thresholds (0.5 ≤ pred_obs_ratio ≤ 2.0), and all 10 are at threshold=90%. This suggests the rate-based clipping design introduced systematic underprediction. The new repo uses the same rate-space design. This needs to be verified against actual new-repo run results before declaring the model works.

---

### Recommendations for new repo

**model_selection.py** (already built — verify these):
- New file already has 7 methods + `prepare_rankings_df`. Confirm column names match new schema (`mae_rate`, `mae_rate_is`, `pred_obs_ratio`, `cov_5` etc. — NOT `oos_mae_rate`).
- Consider adding `pairwise_dominance_summary` as an 8th method (Condorcet voting). Used in the old notebooks alongside TOPSIS; gives a useful complementary view.

**ModelQuery** (new file: `step_04_select/model_query.py`):
- Port `get()`, `enumerate()`, `compare()`, `diff()`, `compare_to_reference()` with column name updates.
- `neighbors()`: redesign for JSON covariate format. Parse JSON strings with `json.loads()`, compute symmetric difference of token sets. The ±1-token logic is the same conceptually.
- Constructor defaults: `threshold_quantile` not `threshold_pct`; `bulk_family` / `tail_family` not `bulk_dist` / `tail_dist`; distribution names updated to new repo convention.
- Config cols: `['s1_cov', 's2_cov', 'bulk_cov', 'tail_cov']` — same as old.

**StagePlotter** (new file: `step_04_select/stage_plots.py`):
- Port: `vet_stage()` layout, `_plot_cont_overall/by_basin/by_island`, `_plot_cat_beeswarm`, `_make_pred_df`, `vet_model()`.
- Rewrite: `_get_subset_for_stage()` — use rate-based threshold matching new repo convention.
- Rewrite: `_predict()` — use new distributions module and FitResult API, not raw statsmodels objects.
- Rewrite: `predict_df_oos()` — read from stored component_predictions/ parquets, don't re-implement fold logic.
- Largely port: `predict_df()` — update distribution calls only.
- The StagePlotter needs access to the output directory (for loading FitResult objects from cache) and the input data (for building X matrices and computing the threshold).

**Vignette notebook** (new file: likely `notebooks/vignette.ipynb`):
Combine the best of dh_model_selection_topsis.ipynb and stage_vetting.ipynb:
1. Load `dh_results.parquet` (IS + OOS rows)
2. Filter to OOS rows, apply pred_obs_ratio and mae_rate filters
3. Build ModelQuery
4. `run_full_pipeline()` → TOPSIS winner
5. `mq.get(...)` + `mq.neighbors(...)` + `mq.compare()` — explore candidates
6. `mq.diff(condorcet_winner, topsis_winner)` — compare top two
7. `plotter.vet_model(topsis_winner)` — 4 stage plots
8. `predict_df()` → calibration scatter, coverage curve
9. Coefficient tables (need to port or re-implement `display_dh_model_coefficients`)

**Pre-flight check before all of the above:**
Verify that pred_obs_ratio in the new-repo run results is near 1.0. The v3 notebook showed severe underprediction with rate-based threshold + clipping. If the new repo has the same issue, fix that before building visualization tooling.

**Revisit if:** The v3 underprediction issue turns out to be v3-specific (e.g., due to Poisson tail distribution or the constrained same-covariate design) and does not affect new-repo results. Also revisit once actual run results exist.

## 2026-05-12: Threshold decision — keep 0.70 and 0.75
**Decision:** `threshold_quantile ∈ {0.70, 0.75}` for the refined grid. Drop 0.80–0.95.
**Why:** 0.70 reaches the highest level-20 coverage ceiling (~0.30) and has the tightest best-10/15/20/avg bundle. 0.75 has the best low-level (best @ 5) config visibly distinct from best-avg, giving a different operating point worth preserving. 0.80 attains a competitive ceiling (~0.32) but offers no behaviour 0.70/0.75 don't already cover; 0.85+ drop their ceilings substantially; 0.95 had zero calibrated configs.
**Revisit if:** A new evaluation run with different bulk/tail family pins shifts the threshold-level coverage curves, or downstream forecasting use prefers an operating point only 0.80+ provides.

## 2026-05-12: Calibration gate as precondition, not post-screening filter
**Decision:** `full_pred_obs_ratio_oos ∈ [0.9, 1.1]` is applied as the very first filter — before any per-stage S1/S2/Bulk/Tail screening. All per-stage heatmaps and decisions operate on the calibrated subset.
**Why:** Miscalibrated DH assemblies cannot be salvaged by family/exposure choices downstream, so screening across them produces meaningless cell maxima. The `reports/preliminary_decisions.qmd` code originally didn't apply this filter — but the screening decisions had been made looking at filtered plots in a prior chat. This brings the qmd's code into agreement with what actually drove the decisions.
**Revisit if:** A new metric reveals that the [0.9, 1.1] band is too strict (excluding analytically-relevant configs) or too lenient (admitting unreliable ones).

## 2026-05-13: run_evaluate's cov-coupling is configurable, not structural
**Decision:** The `s1_cov == s2_cov == bulk_cov == tail_cov` constraint at `run_evaluate.py:352` is a preliminary-stage scope-limiter, not an inherent property of the model. Added `--decouple-covs` flag to `run-evaluate` and `run-evaluate-orchestrate` so refined-stage runs can opt out.
**Why:** The preliminary run had 3 covariate combos; coupling kept the grid tractable and made sense. The refined run has 16 covariate combos and per-stage cov diversification was the explicit motivation for that whole stage of work. Mis-framing the coupling as "structural" rather than "configurable" cost ~2 days of optimisation on the wrong-scope grid (see DEAD_ENDS).
**Revisit if:** A future evaluate path needs to express explicit `(s1_id, s2_id, bulk_id, tail_id)` tuples without Cartesian-product behaviour over per-stage spec lists.

## 2026-05-13: Task-file partitioning — manifest-driven, not orchestrator-driven
**Decision:** `run-evaluate-orchestrate` now accepts `--task-file <JSON>` produced by separate `build_*_tasks` CLIs (`run-build-evaluate-tasks`, `run-build-neighbor-tasks`, `run-build-half-coupled-tasks`). Each task in the manifest carries per-stage `*_spec_ids` lists; the worker filters by those lists and runs the Cartesian product of survivors.
**Why:** Decouples partitioning logic from submission logic. Adding a new partition mode (one_per_bulk_spec_threshold, one_per_tail_fam_exp_threshold, single_config, half_coupled, neighbours) requires only a builder function — no orchestrator or worker changes. Enables tuning per-task granularity (a few tasks of large work vs many tasks of small work) without touching the worker.
**Revisit if:** A future workflow needs explicit (s1, s2, bulk, tail) tuples that don't factor as a Cartesian product (currently supported only via single-config tasks; would benefit from an explicit-tuple list mode in run_evaluate).

## 2026-05-13: Half-coupled covariate chains as the refined-stage search space
**Decision:** For the locked-in (threshold=0.70, S1=logit/free, S2=logit/free, bulk=scaled_logit/free, tail ∈ {gamma:free, gamma:free+weight, weibull:excluded}) corner, search half-coupled cov chains: `s1_cov ⊇ s2_cov ⊇ bulk_cov ⊇ tail_cov`. Yields 5⁴ × 3 = 1,875 DH configs.
**Why:** Allows downstream stages to drop covariates from upstream ones (interpretable: progressively-simpler models as you condition on more) but disallows downstream stages from introducing new covariates not seen upstream (would be modelling a feature without a marginal effect on the conditioning event first). Captures the operationally-meaningful slice of the full 1.5M-config decoupled grid at this corner. ~17 CPU-hours total, tractable as 1,875 single-config jobmon tasks.
**Revisit if:** The half-coupling rule turns out to exclude a genuinely better model that violates it (e.g., a covariate matters in the tail but not in the bulk), at which point selectively relax.

## 2026-05-13: sacct over instrumentation for per-task resource calibration
**Decision:** Per-task resource accounting uses `sacct -j <jobid> --format=...,Elapsed,MaxRSS` rather than the worker's `--scope` tick-tock logging. `--scope` is reserved for diagnostic probes when sacct's aggregate view is insufficient.
**Why:** Sacct is captured automatically by Slurm for every task with zero engineering cost; `--scope` is code-change-required instrumentation that took ~30 min to add and produces millions of lines on production runs. Sacct gave the actually-useful numbers (n=1 probe → 59s/301MB; n=384 probe → 311s/943MB; n=384 production → 15:06/2.46GB) that supported all the scaling/contention diagnoses without ever needing `--scope` output.
**Revisit if:** A new bottleneck requires sub-task tracing that sacct can't provide (e.g., precise per-stage timing within a single DH-config eval).

## 2026-05-13: Slurm cgroup memory ≈ 2× sacct MaxRSS — provision accordingly
**Decision:** When sizing Slurm `--mem` for tasks based on sacct MaxRSS measurements, multiply by 2–3× for headroom. Cgroup memory accounting (which is what Slurm's OOM killer uses) typically runs ~2× higher than the RSS sacct reports.
**Why:** n=384 probe peaked at 943 MB MaxRSS; production at `--mem 2G` (2.1× headroom over MaxRSS) OOM-killed 99.9% of tasks. `--mem 5G` (5.3× headroom) ran cleanly at 41% utilisation. The gap is from cgroup memory.usage_in_bytes including file cache + anonymous mappings + framework overhead that MaxRSS doesn't account for.
**Revisit if:** A future Slurm/cgroup configuration change closes the gap, or a workload's actual memory usage drifts significantly from the sacct-reported peak.

## 2026-05-15: BASIN_LEVELS corrected — AU added, SA removed
**Decision:** `BASIN_LEVELS` in `predict/paths.py` is `('AU', 'EP', 'NA', 'NI', 'SI', 'SP', 'WP')`. SA was in the old tuple but not on disk; AU was on disk but not in the tuple.
**Why:** Climada writes basins by ID: AU, EP, NA, NI, SI, SP, WP. The old tuple had SA (likely a copy from a different basin convention). Every cross-basin aggregate from workflow 577423 silently missed AU and tried (and failed, with a warning) to find SA — corrupting every year_bin/scenario/storm_draw mean.
**Revisit if:** Climada output schema changes the basin naming or adds/removes basins.

## 2026-05-15: 4-tier DAG for predict-and-aggregate
**Decision:** Split the old 1-tier `predict_year_bin` (bundling 7 basins × 100 tc_draws + basin_agg + year_bin_agg inline) into four jobmon tiers: predict_basin (sd,sc,yb,basin) → aggregate_year_bin → aggregate_scenario → aggregate_storm_draw.
**Why:** Bundled tasks took 50+ min on slow nodes, blowing through both initial and retry windows. Per-basin tasks are ~1/7 the size with target ~1-2 min wall. Per-tier skip + atomic writes make jobmon retries cheap. Aggregation moves out of the predict task body and into its own template, simplifying both.
**Revisit if:** Per-basin per-tc work grows enough that per-tc_draw sub-task split becomes needed (currently 1 (sd,sc,yb,basin) bundles 100 tc_draws).

## 2026-05-15: Atomic writes + recovery for resumable jobmon tasks
**Decision:** All parquet writes in the predict pipeline use `atomic_write_parquet`: write to `<path>.tmp`, then `os.replace(tmp, path)`. On task entry, when the terminal output (basin-mean) is absent: scrub all `.tmp` files in the output dir AND delete the latest-mtime intermediate parquet (re-do unconditionally).
**Why:** `os.replace` is atomic on POSIX filesystems, so the existence of the final-named parquet implies a complete write. Killed tasks leave `.tmp` files behind, not half-written final files. The "redo latest-mtime" pattern handles the legacy case where prior (non-atomic) code may have left a half-written final-named parquet — re-doing one tc_draw is cheaper than read-verifying all 100. Crucially, `out_path.with_suffix(out_path.suffix + '.tmp')` (append) not `out_path.with_suffix('.tmp')` (replace).
**Revisit if:** A non-POSIX filesystem becomes the target (Windows/some NFS configurations), or per-tc work grows large enough that re-doing one is no longer cheap.

## 2026-05-15: SDI loaded once per task, not per tc_draw
**Decision:** `predict_basin` loads SDI once (full year range, `location_ids=None` for all locations) and merges it into each tc_draw's prepped df inside the loop. `load_sdi_df` now accepts `location_ids=None`.
**Why:** Old code called `load_sdi_df(year_bin, location_ids=admin_df['location_id'].unique())` per tc_draw — 100× xr.open_dataset + slice + mean per basin task. Dominated wall (~50s/task on cold cache, hidden when historical SDI was warm-cache). Cold-cache effect was the most likely explanation for the 20× perf delta between historical and ssp245 cells in workflow 577423.
**Revisit if:** Memory pressure from holding the full-location SDI slice becomes a constraint (currently ~150MB peak), or per-task SDI selection becomes location-set-dependent (it isn't today).

## 2026-05-15: Hard-fail on missing files in the predict pipeline (never silent-skip)
**Decision:** Every "file missing" check in the predict pipeline raises `FileNotFoundError` instead of warning+continuing. Specifically: stage4 basin folder absent, no `tc_risk_draw_*` dirs, missing per-tc admin parquet, missing basin-mean / year_bin-mean / scenario-mean at the agg tiers.
**Why:** Applies the global "never proceed past failed work" rule to the predict pipeline. Workflow 577423's aggregates were corrupted because `aggregate_year_bin` had `if not p.exists(): logger.warning(...); continue` per basin-mean — and the basin-list bug meant 1 of 7 was *always* missing. A silent-skip pattern can corrupt aggregates indefinitely; a hard fail catches it on the first attempt. Exception: post-filter empty data ("file exists, just empty after filter") stays warn-and-continue because it's a legitimate end state, not a contract violation.
**Revisit if:** A legitimate path emerges where a tier-N input is *expected* to be absent (e.g., a basin with no tropical cyclone activity in a year_bin), in which case the upstream should write an empty marker file rather than producing nothing.

## 2026-05-15: Canonical-source enumeration — no filesystem walk during orchestration
**Decision:** `predict/orchestrate.py` enumerates tasks entirely from canonical sources: `storm_draw_table.csv` + `time_bins.csv` + the `BASIN_LEVELS` tuple. No stage4_v2 directory stats during enumeration. The predict_basin task itself raises if its input folder is absent (hard-fail rule above).
**Why:** Old orchestrator stat'd every (model, variant) dir to gate task creation, then warned-and-pruned. Slow (thousands of stats) and racy (mid-copy directories could prune mid-enumeration). Filesystem is for tasks to consume, not for orchestrators to enumerate against. Manifest from canonical metadata also lets us reason about completeness mathematically: 39,585 = 5,655 × 7 derives from CSVs without touching disk.
**Revisit if:** A canonical source becomes unreliable (e.g., `time_bins.csv` lists year_bins that climada hasn't produced), in which case use a coverage filter (next decision) rather than re-introducing filesystem scans.

## 2026-05-15: Coverage filter via notebooks/model_variants.json
**Decision:** Orchestrator reads `notebooks/model_variants.json` (5 (model, variant) pairs, hand-built from one `ls` of stage4_v2) and filters `storm_draw_table.csv` to those pairs. Drops 13 storm_draws (MRI-ESM2-0 r3/r4/r5i1p1f1) whose climada output doesn't exist. CLI override `--model-variants-path`.
**Why:** `storm_draw_table.csv` lists 100 storm_draws but stage4_v2 only has data for 87. With the new hard-fail rule, those 13 would each generate ~450 predict_basin tasks that all crash with FileNotFoundError. Coverage filter is one tiny JSON read + one set membership per row — vastly cheaper than letting tasks fail at runtime.
**Revisit if:** Climada produces output for the missing variants (regenerate `notebooks/model_variants.json`), or the filter becomes stale enough to warrant moving the cache into the package data instead of a notebook-relative path.

## 2026-05-15: Predict-pipeline resource asks — 2G/15m predict, 1G/10m agg
**Decision:** `_PREDICT_RESOURCES = {memory: '2G', runtime: '15m', cores: 1}`, `_AGG_RESOURCES = {memory: '1G', runtime: '10m', cores: 1}`. CLI overrides on `run-predict-orchestrate`: `--predict-memory`, `--predict-runtime`, `--agg-memory`, `--agg-runtime`.
**Why:** With SDI-once + per-basin split, expected per-task wall is ~1-2 min. 15m runtime is ~7-8× headroom for cold-cache outliers; not so generous that fairshare complaints kick in. 2G memory is ~5× the ~400MB peak (SDI dataset + models + island_cov). Agg tiers are sub-second typical; 1G/10m is generous-but-cheap-on-all.q. The previous CC misled on 5m runtime; 50-minute task failures resulted. Generous-not-tight is the right side to err on after that incident.
**Revisit if:** A future calibration (sacct on the new workflow) shows tier-1 tasks consistently under ~3 min and memory under ~600MB — could dial down. Or if cold-cache tail-latency tasks routinely brush 10+ min, dial up.

## 2026-05-17: Restrict preliminary fits to storms with year ≥ 2000
**Decision:** Re-run the 831-spec preliminary grid on a dataset filtered to storms with year ≥ 2000. Added `--min-year` flag to `scripts/ingest/01_prepare_input_data.py`; current `00-data/current → 20260517/input.parquet` (1,693 rows, 2000-2021). Preliminary fit output → `01-preliminary/20260517/`.
**Why:** EM-DAT (the death-count source) documents that its data is incomplete prior to 2000. Reporting completeness is plausibly correlated with SDI (better-resourced / higher-SDI countries are more likely to record deaths), which would systematically bias the model's SDI-mortality relationship and is a plausible mechanistic cause of the ~30× low-SDI overprediction observed in the 2026-05-16 predict-stage diagnostics (Mozambique 1984 ≈ 26k predicted vs ~3k globally observed). This bias was not detected during prior screening because our calibration gate only checked the aggregate `full_pred_obs_ratio_oos`; see the companion 2026-05-17 entry on screening-metric expansion. Refitting on the post-2000 window cuts the dataset roughly in half (~9k all-years → 1,693 post-2000 rows) but removes the suspected bias source at the input layer.
**Revisit if:** Post-2000 preliminary fits show no improvement in the SDI-stratified calibration, OR an EM-DAT vintage with better pre-2000 coverage becomes available, OR a different completeness threshold (e.g., ≥ 1990) gives a better data-quantity / bias tradeoff.

## 2026-05-17: Expand preliminary screening beyond aggregate mean calibration
**Decision:** Preliminary screening will no longer rely solely on aggregate calibration metrics (`full_pred_obs_ratio_oos ∈ [0.9, 1.1]`, the 2026-05-12 precondition gate). Two new diagnostics will be added alongside the gate before any per-stage screening:
  (a) **Time-resolved calibration:** global predicted vs observed deaths-by-year, plotted as side-by-side time series. Same construction as the aggregate obs/pred ratio but resolved on the year dimension.
  (b) **Synthetic-storm face validity:** for chosen winners (IS + OOS folds), predict on the climada synthetic storms from (storm_draw=1, tc_risk_draw=1), historical period, all basins; aggregate globally per year; plot alongside (a) so that real-storm and synthetic-storm predictions are directly comparable to observed.
The all-years `preliminary_decisions.qmd` and `dh_preliminary_diagnostics.ipynb` will stay frozen as the audit record of the prior decision; forked versions under the 20260517 output node will be built around the expanded metric set.
**Why:** The aggregate `full_pred_obs_ratio_oos` averages over time, locations, and storms — a model that is on-average calibrated globally can be badly miscalibrated in specific (year, location, storm) cells. That is exactly what happened: the 30× low-SDI overprediction wasn't visible in the aggregate ratio and only surfaced two stages downstream during predict-time plotting. "We just cared about the mean of the mean." Time-resolved totals catch regime shifts (e.g., the EM-DAT pre-2000 completeness break); cross-storm-set face-validity catches model behaviour that doesn't generalise from the real storms it was fit on to the synthetic storms it will be applied to. Both are cheaper and earlier ways to surface the same failure mode the predict-stage diagnosis eventually caught.
**Revisit if:** Either diagnostic turns out to be insensitive to the failure modes it was added to catch (e.g., time-resolved looks clean but a per-basin or per-storm break is hidden underneath), in which case add finer stratifications. Or the synthetic-storm face-validity diagnostic is too expensive to run on every screening candidate and a sampled subset suffices.

## 2026-05-14: TOPSIS winner adopted over Borda
**Decision:** Use the TOPSIS winner (gamma/free+weight tail, scaled_logit bulk, logit S1/S2) as the operational pick from the half-coupled run.
**Why:** Three of four ranking methods (TOPSIS, Pareto-first, pairwise-dominance) agreed on this model. Borda's outlier choice (weibull/excluded tail) traces to redundancy bias in the metric set: `full_mae_rate_oos`, `fwd_mae_rate_oos`, `full_rmse_rate_oos`, and `full_zero_acc_oos` all reward "predict near-zero" behaviour. A tail-excluded model literally predicts 0 above threshold, which dominates four metrics in a 21-model pool where each metric is worth ~20 rank-points. The negative `full_cor_rate_oos` (−0.013 vs +0.16 for TOPSIS) only costs ~20 points. So Borda picked a degenerate "no big mortality event will ever happen" model. Not the signal we want.
**Revisit if:** A different metric set (e.g., dropping the most-redundant lower-better metrics) brings the four methods into agreement. Or if we discover the heavy-tail under-prediction is a real failure mode of the gamma tail that the weibull-excluded model accidentally side-steps.

## 2026-05-14: Three-toggle uncertainty taxonomy
**Decision:** The uncertainty-draw machinery has three orthogonal toggles:
- **Toggle 1 (`draw_coefs`)** — do the N draw models differ in their β coefficients per stage, or share the MLE?
- **Toggle 2 (`draw_scale`)** — do the N draw models differ in their dispersion / scale parameters per stage, or share the MLE? Only affects stages with unexplained variance (scaled_logit bulk, gamma tail). No effect on logit S1/S2.
- **Toggle 3 (`outcome_draw`, applied at predict time)** — for bulk/tail, sample a realization from the predictive distribution, or return its analytical mean? S1/S2 always Bernoulli-flip regardless — the 0/1 outcome is structural, not a toggle-3 behaviour. Toggle 3 has no effect on S1/S2 because logit GLMs have no fitted residual variance on the linear-predictor scale (the Bernoulli flip IS the noise).

**Why:** Bobby corrected my framing twice during design. Key fact: with all toggles OFF, the prediction is *not* mathematically deterministic — S1/S2 always Bernoulli-flip, so "deterministic" means *seeded-reproducible*, not "expected-value pass-through". The naive "no-variation predictions" path (passing P(S1=1) × P(S2=1) × E[tail] through analytically) destroys the 0/1 structure the model is built around — most storms produce zero deaths, and Bobby needs that to be preserved in every prediction. The Bernoulli is part of the data-generating process, not a knob.
**Revisit if:** A consumer needs the analytical expected-deaths formula for some downstream use (e.g., for closed-form aggregation across millions of storms where Monte-Carlo cost matters). That would be a separate `predict_mean()` method, not a fourth toggle.
**Superseded 2026-05-15:** see the four-toggle entry below. Toggle 4 (`expected_bernoulli`) was added; the "S1/S2 always Bernoulli-flip" invariant was relaxed because a consumer (the diagnostics/coverage path that compares against the evaluate stage's `full_*_oos` metrics) needed the closed-form expected hurdle. The Revisit-if condition above triggered, but as a fourth toggle on `DrawModel.predict`, not a separate method — keeping the four uncertainty sources in one place is simpler than maintaining two parallel predict APIs.

## 2026-05-15: Four-toggle uncertainty taxonomy
**Decision:** Added a fourth orthogonal toggle to `DrawModel.predict`:
- **Toggle 4 (`expected_bernoulli`, applied at predict time)** — for S1/S2, contribute the stage's probability directly (`p_s1`, `p_s2`), or Bernoulli-flip it? Default False (existing stochastic-hurdle behaviour). When True, the assembly switches to `rate = p_s1 × (p_s2 × tail_rate + (1 − p_s2) × bulk_rate)` — the closed-form expected hurdle, independent of `outcome_draw`. The `s1_flip` and `s2_flip` columns are NaN when toggle 4 is on; `p_s2` is no longer s1-masked.

All 2×2 cells of (`expected_bernoulli`, `outcome_draw`) are meaningful: hard-hurdle deterministic (F, F), fully stochastic (F, T), exact closed-form (T, F), soft hurdle with stochastic bulk/tail (T, T).

The (T, F, F, F) recipe — `expected_bernoulli=True, outcome_draw=False, draw_coefs=False, draw_scale=False` — reproduces `evaluate.assemble.assemble_predictions` exactly. Pinned as a regression test in `tests/uncertainty/test_draw_models.py::test_expected_bernoulli_matches_assemble_predictions`.
**Why:** The previous taxonomy treated the S1/S2 Bernoulli flip as a structural invariant rather than a separable noise source, conflating "where does noise enter" with "how are the four stages combined." Treating it as a fourth toggle makes the four uncertainty sources (β, scale, bulk/tail draw, S1/S2 flip) fully orthogonal and gives the diagnostics path a clean way to reproduce the evaluate-stage closed form for direct apples-to-apples comparison.
**Revisit if:** A new family is added that doesn't fit cleanly under the (probability vs flip) framing — e.g., a continuous S1 surrogate or a hurdle with three outcomes. At that point reconsider whether the toggle should be (probability vs draw) per stage rather than a single global flag.

## 2026-05-14: One pickle per draw set, generic over focus_model
**Decision:** `save_draw_models(models, path)` writes a single pickle containing a list of N `DrawModel` objects. Each `DrawModel` is built generically from a `focus_model` dict (CONFIG_COLS) — not hardcoded to the current TOPSIS winner.
**Why:** Single-file load is simpler for downstream consumers and 100 lightweight objects is small on disk (~62 KB for the current TOPSIS winner with 11-coef S1). Generic-over-focus-model means we can rebuild for any future winner without code changes; family coverage is currently logit S1/S2 + scaled_logit bulk + gamma tail (TOPSIS combo), with clean `NotImplementedError` for others. Output node: `<ROOT>/03-draws/<DATE>/<config_slug>/draws.pkl` plus `focus_model.json` + `metadata.json` for provenance.
**Revisit if:** A consumer needs per-draw parallelisation that would benefit from per-file granularity, or a different winner requires a family not yet in the dispatch (extend `_inverse_link` / `_bulk_*` / `_tail_*` in `draw_models.py`).

## 2026-05-15: Four (c, s) pickle layout for the TOPSIS winner
**Decision:** Ship four pickles per draw-model node, one per `(draw_coefs, draw_scale) ∈ {0,1}²` configuration:
  `03-draws/20260514/topsis_winner_v1/draws_c{c}_s{s}.pkl` + matching `metadata_c{c}_s{s}.json`, plus a shared `focus_model.json`. All built with `seed=42` so they're byte-deterministic.
**Why:** Toggles 1 and 2 (`draw_coefs`, `draw_scale`) are build-time, not predict-time — flipping them requires rebuilding the draws. Producing all four cheaply (4 × ~62 KB total) and serving them side-by-side means downstream consumers can `load_draw_models(DRAWS_DIR / f'draws_c{c}_s{s}.pkl')[DRAW_IDX]` to pick any (c, s) configuration with a single line. The existing `draws.pkl` is preserved untouched for backwards compatibility with the diagnostic notebook.
**Revisit if:** A new focus_model winner replaces the TOPSIS pick, OR a downstream consumer needs per-draw granularity (split each pickle into N files).

## 2026-05-15: Top-3 toggle cells (rows 1, 9, 10 of the 16-cell table) for downstream reporting
**Decision:** From the 2⁴ factorial, the recommended subset for downstream propagation is **3 cells**:
  - **Row 1** `(c=0, s=0, o=0, b=1)` — closed-form expected hurdle. THE point estimate. Equivalent to `evaluate.assemble.assemble_predictions` exactly.
  - **Row 9** `(c=1, s=0, o=0, b=1)` — parameter-uncertainty band around the smooth point. Same scale + interpretation as Row 1, so the band is directly comparable to the point.
  - **Row 10** `(c=1, s=0, o=1, b=0)` — full per-storm predictive band on the hard-hurdle scale. The cell that gives ~1.0 per-storm coverage (b=0 preserves zero-mass).
**Why:** Aggregate reporting needs a smooth point + smooth band → Rows 1+9. Per-storm calibration needs a bimodal (mass-at-zero) predictive distribution → Row 10. The b=0 vs b=1 split is structural — mixing them in one CI band would be incoherent. The Jensen bias on Row 9 (~13% upward shift of the c=1 mean vs Row 1's point) means Row 1 must always be reported as the point estimate and Row 9 only as quantiles, never as a separate central tendency.
**Revisit if:** A consumer demands a single one-number summary (then we'd need to take a stand on whether Row 1 or Row 9-mean is "the answer"), or if `s` becomes worth varying (currently inert at `o=0` and adds ≤20k SD at `o=1` — small relative to other sources).

## 2026-05-15: Bundle predict tier by (storm_draw, scenario, year_bin); fold basin + year_bin agg inline
**Decision:** The jobmon DAG has three tiers per storm_draw:
  1. `predict_year_bin` — one task per `(sd, sc, yb)`. Inner loop over the 7 basins; for each basin, processes ~100 tc_draws, then runs basin aggregation inline; after all basins, runs year_bin aggregation inline.
  2. `aggregate_scenario` — one task per `(sd, sc)`, concats year_bin means.
  3. `aggregate_storm_draw` — one task per sd, concats scenario means with a scenario label column.
Setup (mkdir + info.json) is run inline by the orchestrator before binding the workflow.
Total task count for the full run: 6,090 (5,655 + 348 + 87).
**Why:** Earlier per-leaf granularity (one task per `(sd, sc, yb, basin)`) gave 34k predict + 34k basin agg tasks = 73k total. The aggregation tasks were 5-10 seconds each — almost all overhead. Bundling the basin loop into the predict task and folding the two cheap aggs inline drops the task count 12× without changing the CPU-hours budget. Per-task footprint ~6 min / 0.55 GiB (observed), well under the 15m / 1G ask. The cluster smoke confirmed end-to-end correctness on one storm_draw before the full workflow launched.
**Revisit if:** Per-task wall time grows past ~30 min (then split year_bins back out), or if jobmon's per-task overhead becomes negligible (then per-leaf granularity is fine and gives more parallelism).

## 2026-05-15: Time_bins_df is the source of truth for which (sd, sc, yb) tasks to create
**Decision:** Orchestrator's `_enumerate_year_bins` does ONE stat per unique `(model, variant)` to skip storm_draws whose climada output doesn't exist yet, then expands the cross-join `storm_draw_table × time_bins_df` to produce `(sd, sc, yb)` task tuples. No per-leaf NFS walks.
**Why:** A previous version walked `(sd, sc, yb, basin)` paths with `.exists()` + `iterdir()` (~40k NFS round-trips, multi-minute pre-scan). `time_bins_df` already encodes which (mv, sc, yb) combinations climada planned to produce — trust it. Missing basins at task runtime are handled by `predict_tc`'s graceful warning + skip, not by orchestrator-time enumeration.
**Revisit if:** climada's `time_bins_df` becomes unreliable (regularly references year_bins that don't exist on disk), then add a per-(mv,sc,yb) dir-exists check (still cheap, ~500 stats).

## 2026-05-17: Post-2000 fork — preliminary refit on data with year ≥ 2000
**Decision:** Fork the entire screening + final-selection cycle to a post-2000-only data subset. Lineage stored separately from the all-years cycle:
- `reports/preliminary_decisions_post2000.qmd` (forked from `preliminary_decisions.qmd`).
- `notebooks/dh_preliminary_diagnostics_post2000.ipynb`, `dh_intermediate_diagnostics_post2000.ipynb`, `dh_refined_diagnostics_post2000.ipynb`.
- `src/idd_tc_mortality/grid/build_refined_specs_post2000.py` (forked from `build_refined_specs.py`).
- All output nodes at `<stage>/20260517*` (preliminary, refined, refined-final).
**Why:** EM-DAT documents incomplete pre-2000 death reporting; reporting completeness is plausibly SDI-correlated, biasing the SDI-mortality relationship — suspected mechanistic cause of the ~30× low-SDI overprediction surfaced in 2026-05-16 predict-stage plots. Refitting on post-2000 (1,693 rows vs ~9k all-years) cuts the dataset roughly in half but removes the bias source at the input layer. Forking (vs editing in place) preserves the all-years cycle as the audit record of the prior decision while we test the post-2000 hypothesis.
**Revisit if:** Post-2000 predict-stage results don't show the low-SDI overprediction improving, OR a future EM-DAT vintage with consistent pre-2000 reporting becomes available.

## 2026-05-17: Asymmetric calibration gate for post-2000 — [0.1, 1.5]
**Decision:** Replace the 2026-05-12 symmetric `full_pred_obs_ratio_oos ∈ [0.9, 1.1]` gate with an **asymmetric** `[0.1, 1.5]` for the post-2000 lineage. Applied uniformly in `preliminary_decisions_post2000.qmd`, `dh_intermediate_diagnostics_post2000.ipynb`, and `dh_refined_diagnostics_post2000.ipynb`. The all-years gate is preserved unchanged in `preliminary_decisions.qmd`.
**Why:** Death distribution is dominated by mega-events (Nargis 2008 = ~138k of ~210k total observed deaths in 2000-2021). A well-specified threshold-aware tail-rate model will *honestly under-predict* those events — they live in the extreme tail and a 22-year window doesn't pin the mean down. Therefore `total_pred / total_obs < 1` is the expected, correct result, not a defect. A ratio of exactly 1.0 implies compensating overprediction of the bulk-of-storms to balance the under-predicted mega-events — itself a failure mode. The asymmetric gate allows under-prediction up to 10× while capping over-prediction at 1.5×. Empirical confirmation: the drop-top-N trimmed-ratio sweep (N ∈ {0, 5, 10, 25}) on the 7,020 [0.1, 1.5]-calibrated configs shows median trim_n=10_oos = 0.79 → trim_n=25_oos = 2.40, indicating that most configs over-predict the bulk-of-storms by 2-2.5× once mega-events are removed — exactly the failure mode the symmetric gate was selecting *for*.
**Revisit if:** Per-storm trimmed analysis (which the post-2000 intermediate notebook now does) shows the asymmetric gate is letting through pathologically bulk-overpredictive configs that no post-hoc filter catches. Or if a different upper-bound value (1.2 / 2.0 / etc.) ends up being the empirical winner on this data.

## 2026-05-17: Bulk family narrowed to scaled_logit only (post-2000 lineage)
**Decision:** For the post-2000 refined grid, drop `beta` from the bulk family set. Keep `scaled_logit` only with exposure_mode ∈ {free, free+weight}. Encoded in `build_refined_specs_post2000.py`. Divergence from the all-years cycle, which kept both.
**Why:** On the post-2000 data, scaled_logit dominates beta on `bulk_mae_rate_oos` by 14% at q=0.70, 25% at q=0.85, and 38% at q=0.95. The all-years cycle had them tied; the post-2000 data resolves the tie clearly. The Kendall τ matrix shows rank stability ≥ 0.62 across thresholds on this metric, so the family decision transfers across thresholds.
**Revisit if:** A future data vintage brings them back into a tie, OR if beta provides meaningful diversity for downstream uncertainty propagation that scaled_logit alone can't deliver.

## 2026-05-17: Tail families {gpd, log_logistic, weibull} with per-family exposures (post-2000)
**Decision:** Refined grid uses three tail families with a per-family exposure dict (rather than the all-years Cartesian across all families × all exposures):
- `gpd`:          [free, weight, free+weight]
- `log_logistic`: [free, weight, free+weight]
- `weibull`:      [free]    ← single exposure mode
The `excluded` exposure is dropped from all tail families. Encoded as `TAIL_FAMILY_EXPOSURES` in `build_refined_specs_post2000.py`. Spec count drops from the all-years 24 to 7 tail (family, exposure) combos.
**Why:** Top-25 calibrated survivors from the intermediate trim+IS-OOS analysis showed (a) lognormal, truncated_normal, gamma all absent, (b) `excluded` exposure absent across all families, (c) weibull only ever appeared with `free` (never with weight or free+weight). The narrowing reflects empirical observation, not a structural fitter constraint — weibull × weight/free+weight could be added back if needed.
**Revisit if:** A future cycle wants to test whether weibull × weight is truly invalid or just lost on this data sample.

## 2026-05-17: Threshold 0.90 dropped from post-2000 refined grid
**Decision:** From the six tested thresholds {0.70, 0.75, 0.80, 0.85, 0.90, 0.95}, drop 0.90 entirely. Final viable set: {0.70, 0.75, 0.80, 0.85, 0.95}. Refined-final selection (the in-flight 96-config grid at bail time) pinned to 0.70 only.
**Why:** In the intermediate-survivors notebook, the joint IS+OOS trimmed-ratio screen (gate + trim_n=5/10 OOS-median in [0.25, 2.0] + trim_n=5/10/25 IS in [0.25, 1.2]) produced 25 survivor configs. 5 of 6 thresholds had representation (0.70: 4, 0.75: 3, 0.80: 6, 0.85: 4, 0.95: 8) — but **0.90: 0**. Tracing the screen: 0.90 had 16 configs entering the final filter (IS trim_n=25 in [0.25, 1.2]) and 0 passing. Interpretation: at threshold=0.90, the bulk subset is wide (90% of storms) and the bulk model fit is biased by the high end; once top-25 mega-events are dropped, the remaining smallest storms reveal the bias clearly. Higher (0.95) and lower (0.85) thresholds don't have the same issue — 0.90 is a quantile-boundary artifact.
**Revisit if:** A future cycle uses a different bulk model or per-stage covariate set that lets 0.90 sustain bulk-of-storms calibration.

## 2026-05-17: Time-series metrics computed but NOT used as hard screens (post-2000)
**Decision:** Three new metrics added to `metrics.py`:
- `cor_ts` / `cor_ts_oos` — Pearson r between Σ pred_deaths and Σ obs_deaths grouped by year.
- `beta_0_ts` / `beta_0_ts_oos` — OLS intercept of obs_y ~ beta_0 + beta_p * pred_y.
- `beta_p_ts` / `beta_p_ts_oos` — OLS slope of same regression.
Wired into `run_evaluate.py` for both IS and OOS, with `np.linalg.lstsq` NaN-on-non-finite guards. Tests in `tests/test_metrics.py`. **However, NOT used as hard screening filters in the post-2000 cycle**; only as horror-spotters.
**Why:** On the post-2000 data, year-to-year variation in deaths is dominated by *which year had a mega-event* (Nargis 2008, Haiyan 2013, Sidr 2007), which is not encoded in the storm-level features (basin, is_island, sdi, wind_speed) the model has. The achievable `cor_ts_oos` ceiling across all configs is ~0.44 — a property of the data/model-family match, not of any specific config's quality. A `cor_ts >= 0.5` filter would disqualify everything; a relative filter is fragile. The metrics still appear as visual horror-spotters in the diagnostic notebooks. Recorded for the audit trail as "computed and intentionally not gated."
**Revisit if:** A future cycle adds storm-level features that DO predict year-to-year variation (e.g., explicit climate state indicators per year), OR if a different dataset has more identifiable year-level signal.

## 2026-05-17: Multi-config half-coupled task partition (`build_half_coupled_multiconfig_tasks.py`)
**Decision:** Added a new evaluate task-builder mode that partitions by (s1_cov × s2_cov × bulk_cov × threshold) — pinning the first three covs and one threshold per task, with the worker's Cartesian over bulk_exposures × tail_specs producing the DH configs. For the post-2000 refined grid this gives 512 tasks (256 (s1,s2,bulk) tuples × 2 thresholds) of ~34 configs each (range 14-224). Total DH configs evaluated: 17,500.
**Why:** The existing `build_half_coupled_tasks.py` emits ONE task per DH config (~17,500 tasks here). At ~20-30s per-task overhead from Python startup + manifest load + parquet load, that's hours of overhead. The new multi-config partition keeps half-coupled validity (tail covs are subsets of the task's pinned bulk_cov; tail family×exposure are from a whitelist) while reducing the overhead amortization. Real wall on free cluster: ~3-5 min vs. potentially hours under single-config.
**Revisit if:** A future grid has a much different structural shape that makes (s1, s2, bulk) partitioning unbalanced. Currently the largest task is 224 configs (when |bulk_cov| = 4) which is fine; if that became thousands of configs the partition would need to split further.

## 2026-05-17: Refined-final explicit decoupled grid — 96 configs
**Decision:** After the refined evaluate (17,500 configs) and TOPSIS chain analysis, the final-selection round is an explicit 96-config grid with FULLY DECOUPLED cov sets per stage. Built via `/tmp/build_refined_final_tasks.py` (one-shot script). Locked dimensions:
- S1: logit × free, cov ∈ {all4, no_is_island}
- S2: logit × free, cov ∈ {all4, no_is_island}  ← independent of S1
- Bulk: scaled_logit × {free, free+weight}, cov ∈ {sdi, basin+sdi, sdi+wind_speed, basin+sdi+wind_speed}
- Tail: cov = empty, (family, exposure) ∈ {(log_logistic, free+weight), (log_logistic, weight), (weibull, free)}
- Threshold: 0.70 only

Total: 2 × 2 × 4 × 1 × 2 × 3 × 1 = 96 configs. Run with `model_predictions` enabled from the start (no `--skip-model-predictions`); ~576 prediction parquets in a flat directory, NFS-comfortable.
**Why:** TOPSIS-ranked top-20 chains from the refined evaluate showed a remarkably narrow structural pattern (top chain = S1 all4, S2 = no_is_island, bulk = sdi, tail = ∅ with rank 1; n=2 with both bulk_exposures). The refined-final round samples around this pattern with deliberate decoupling to test whether the tight TOPSIS preferences hold up under structural permutation.
**Caveat:** "Fully decoupled" means 24 of the 96 configs have S2 covs that are NOT a subset of S1 covs (S1 = no_is_island, S2 = all4 — S2 has MORE covs than S1). Structurally odd for a hurdle model but allowed by the decoupled spec. Bobby's call on 2026-05-18 was to KEEP them: "the point of running all combinations of viable covariate configs."
**Revisit if:** The 24 S2⊄S1 configs end up ranking poorly; then the next iteration could drop them. Or if the rank-1 TOPSIS chain doesn't reproduce, indicating the all-cov-search was overfitting to ranking noise.

## 2026-05-18: Log-logistic added to draw_models.py
**Decision:** `src/idd_tc_mortality/uncertainty/draw_models.py` now supports `tail.family = "log_logistic"`. New `_prepare_stage_log_logistic` kit-builder pulls the joint `(beta, log_k)` covariance from scipy BFGS `raw.hess_inv` (with `_psd_project` cleanup — symmetrize + clamp negative eigenvalues, since BFGS hess_inv can drift slightly non-PSD over iterations). `_tail_mean` returns `exp(eta) + threshold_rate` (the log-logistic median, matching `distributions.log_logistic.predict`); `_tail_draw` samples via inverse-CDF: `alpha * (u/(1-u))^(1/k)` with `u ~ Uniform(0,1)`. Toggle semantics preserved: `draw_coefs` controls β, `draw_scale` controls the shape parameter `k`, joint-MVN when both on, marginalized when one is.
**Why:** Post-2000 refined-final picks all use `log_logistic` tail family (none use `gamma`, the previously-supported tail). The draws machinery would have raised `NotImplementedError` for every viable post-2000 winner without this extension. Original estimate was sandbagged at "half a day with tests"; actual work was ~15 minutes including the smoke test.
**Revisit if:** A post-2000 winner uses `weibull` tail (also a possible refined-grid choice). Same pattern — scipy BFGS, has `raw.hess_inv` and a shape parameter — extension is structurally identical to log_logistic.

## 2026-05-18: Subset predict runs auto-skip aggregators (design — implementation deferred)
**Decision:** When `predict_basin` is invoked with `tc_risk_draws` or `toggles` restricted to a subset (vs `None` / all), the inline `aggregate_basin` call at predict_tc.py:209-214 is skipped and the orchestrator's tier-2/3/4 + postprocess enumerations are bypassed. User reads per-tc-draw parquets directly via `predict_output_path(...)`. When the flags are unrestricted (= all 100 tc_draws, all 16 toggles), the full 4-tier DAG runs as today.
**Why:** The existing `aggregate_basin` hardcodes `DIVIDE_BY = 100` (treats missing tc_draws as zero contributions — correct semantics for a stochastic storm-season rollout). For user-requested subsets, that divisor is wrong; the result would be 1/100× the prediction. Parameterizing the divisor through 3 agg tiers is real plumbing; skipping aggregators is one branch in two files and matches the diagnostic-use mental model (read per-tc parquets directly for face-validity).
**Implementation status:** Designed, not yet coded. See STATUS.md Parking lot for the exact edits per file.
**Revisit if:** A consumer wants aggregated outputs over a tc_risk_draw subset — at that point parameterize `DIVIDE_BY` and add `--divide-by` through the agg tiers.

## 2026-05-19: Per-storm_draw fan-out is the parallelization unit for predict-output cleanup
**Decision:** `scripts/cleanup_au_for_repredict.sh` parallelizes by enumerating the per-storm_draw subtree (~450 entries each) inside an exported bash function (`find_one_sd`) dispatched via `xargs -P WORKERS -n 1`. Not global parallel `find` calls across the whole tree.
**Why:** A global parallel `find` still traverses the full 39K-entry tree per worker — five workers ran for 50 minutes on the first rewrite. The per-sd partition gives each worker a strictly bounded ~450-entry walk and scales linearly with `WORKERS`. Same pattern as the `done_manifest.py` per-sd ThreadPoolExecutor.
**Revisit if:** Predict output topology stops being storm_draw-partitioned (e.g., a future restructure flattens the sd level).

## 2026-05-22: Resource audit — post-2000 pipeline (sacct 2026-05-17 + workflow 579099)
**Decision:** Adopt the following allocation targets for the new-data rerun, based on sacct measurements from the post-2000 final cycle (fit + evaluate 2026-05-17; predict workflow 579099 2026-05-20):

| Template | Current ask | Observed max RSS | Observed max runtime | New target |
|---|---|---|---|---|
| `fit_component` | 1G / 5m | 0.28G | 100s | **512M / 5m** |
| `evaluate_worker_taskfile` | 2G / 30m | 0.58G | 222s (3.7m) | **1G / 5m** |
| `predict_year_bin` | 1G / 10m | 0.85G | 383s (6.4m) | **1G / 10m** (leave as-is) |
| `aggregate_finalize` | 2G / 15m | 2.00G (99.9%) | 221s | **3G / 15m** |

**Why:**
- `fit_component`: 31,590 tasks; mean 237MB / 21s, max 280MB / 100s. Current 1G / 5m is ~4× over on memory.
- `evaluate_worker_taskfile`: 1,787 tasks; mean 302MB / 97s, max 584MB / 222s. Current 2G / 30m is ~7× over on memory and ~8× over on runtime.
- `predict_year_bin`: 5,655 tasks; mean 556MB / 231s, max 851MB / 383s. Current 1G / 10m is appropriately sized; no change.
- `aggregate_finalize`: OOM'd at 1G on first attempt (RSS = 1.00G), barely completed at 2G (RSS = 1.998G = 99.9%). With new data the volume will grow; 2G will OOM again. 3G gives ~50% headroom.

**Bimodal predict_year_bin runtime:** The histogram shows two clear modes (~180s fast, ~255s slow) with a gap at 210-240s. Both modes run throughout the full 80-minute workflow window and span the same job-ID range — they are temporally interleaved, not segregated. This rules out a simple contention explanation (contention would cluster slow tasks early). Likely cause: two task-size classes driven by different year-range TC event counts. Not a resource concern; max runtime (383s) is well within the 10m wall.

**Sacct-not-jobmon-API:** Workflow IDs for the 2026-05-17 fit/evaluate runs were not recoverable from log files (worker stderr only; orchestrator stdout was not captured to disk). Sacct date-window pull (`bcreiner, 2026-05-17–18`) served as the historical record. `workflow_resource_stats` seeding for the ResourcePredictor will need the IDs — retrieve by querying jobmon DB directly or by capturing orchestrator stdout in future runs.

**Revisit if:** A future run shows `aggregate_finalize` memory growing further (3G might need to become 4G), or `evaluate_worker_taskfile` max RSS creeps above 800MB under a larger grid.

## 2026-05-26: (s1, s2, threshold) task partitioning for evaluate
**Decision:** Evaluate orchestrator partitions as cov-matched `(s1_spec_id, s2_spec_id, threshold_quantile)` triples (BUNDLE_SIZE=2 → 180 tasks). `_load_is_groups` matches each IS s1 spec's `covariate_combo` to IS s2 specs with the same combo. Worker reads `s2_id` from `entry[2]` and filters `_s2s` to that specific spec before calling `_evaluate_group`.
**Why:** Old `(s1, threshold)` grouping loaded all cov-matched s2s per task, causing super-linear memory growth with the number of s2 specs. Pinning one s2 per task caps memory at ~0.58 GiB and gives 360-way parallelism. Calibration: BUNDLE_SIZE=2 → median 7 min / 0.57 GiB → well-allocated at 1G/11m.
**Revisit if:** Per-task memory exceeds ~700 MiB (drop BUNDLE_SIZE to 1 or re-probe).

## 2026-05-26: Per-entry atomic checkpointing in evaluate bundle worker
**Decision:** Each `(s1, s2, q)` entry in the bundle loop writes its metric rows atomically to `partials/entries/entry_{s1[:8]}_{s2[:8]}_{q:.4f}.parquet` via `atomic_write_parquet(registry=registry)`. Loop gates on `subtask_skip(entry_path)` at the top of each iteration; on skip, reads the entry parquet back to extend `all_rows` for the terminal bundle write. `AtomicRegistry` wraps the full loop, writing a per-task sidecar to `<output_dir>/.atomic/bundle_{i:05d}.json`.
**Why:** Without per-entry checkpoints, a mid-bundle task death retries from entry 0. Correct pattern for larger BUNDLE_SIZE; at BUNDLE_SIZE=2 the cost is at most one wasted entry on retry.
**Revisit if:** Per-entry NFS writes become a bottleneck at much larger BUNDLE_SIZE.

## 2026-05-26: @with_probe() replaces hand-rolled --probe-n / --no-probe
**Decision:** `run-evaluate-orchestrate` uses `@with_probe()` from `idd_tools.jobmon` instead of hand-rolled click options. Wire: `probe_only = None if no_probe else probe_n`.
**Why:** `idd_tools` already provides this pattern with correct help text and defaults. The hand-rolled version (commit d614bd3) was added one commit before discovering the decorator existed.
**Revisit if:** `idd_tools` changes the `with_probe()` interface.

## 2026-05-19: All dependencies must be declared in pyproject.toml, not just installed
**Decision:** Every runtime import the project makes must appear in `pyproject.toml` `[tool.poetry.dependencies]`. Includes packages that look "infrastructural" (xarray, netCDF4, matplotlib, seaborn, scikit-learn) and the cluster runtime (`jobmon_installer_ihme` pinned to `==10.11.6`, with IHME artifactory as a supplemental poetry source).
**Why:** The 2026-05-18 post-2000 predict workflow lost 4,454 tasks (entire 45K submission silently failed) because xarray was installed in the prior env but never declared. Env rebuild dropped it; orchestrator submitted; every worker failed at import. There was no test catching the missing declaration because all unit tests hit Python-level imports in modules that didn't transitively need xarray.
**Revisit if:** Poetry's resolver evolves to detect undeclared imports at install time (unlikely).

## 2026-06-09: Evaluate redesigned — in-memory re-fit replaces .pkl load/save
**Decision:** Evaluate workers now re-fit every model component in-memory using `fit_one_component(spec, df_subset)` (cached per component+fold via `result_cache`). No separate fit stage, no saved .pkl files, no NFS reads at evaluate time. The fit orchestrator is vestigial (kept pending deletion). Partition: one (s1, s2, threshold) triple per task (`BUNDLE_SIZE=1` → 360 preliminary tasks), each running its full cov-matched bulk×tail cartesian and all OOS folds.
**Why:** Loading 22,854 .pkl files from NFS at 180–360-way concurrency caused an NFS read storm (4.1M stat calls at startup + per-config .pkl reads → TIMEOUT even at 45m/5G). Re-fitting (~0.1s/component) eliminates the shared-resource contention entirely. Probe confirmed 5–6 min/task at 360-way with zero contention.
**Revisit if:** Component fitting grows >10× slower (currently ~0.1s), making re-fit cost exceed NFS read cost even at low concurrency.

## 2026-06-09: Survivor evaluate for intermediate stage
**Decision:** Between preliminary screening and intermediate diagnostics, run a small survivors evaluate (18 tasks, cov-matched survivor subset, WITH model_predictions ON) into `02-evaluate/{VINTAGE}_survivors/`. Driven by `run-evaluate-orchestrate --survivors <json>` where the JSON encodes per-stage screening decisions (family/exposure allow-lists). The 20260608 survivor config is at `scripts/survivors_20260608.json`.
**Why:** The intermediate notebook needs per-config prediction arrays (model_predictions) for drop-top-N calibration and per-storm analysis. The full preliminary run uses `--skip-model-predictions` (avoiding the 162K-file write storm), so the survivors run (~864 configs → ~5K prediction files) generates them cheaply.
**Revisit if:** The full preliminary run is someday small enough to afford model_predictions without storm risk.

## 2026-06-09: AU basin as first-class fitted basin; lon-split removed from predict
**Decision:** With `basins_standard` as the ingest column, AU is a real fitted basin (530 rows in the new data). Removed the IBTrACS lon-split from `predict/data_prep.py`, `predict_tc.py`, `predict_year_bin.py`, and `predict/orchestrate.py`. AU storms are scored with the AU coefficient; no genesis-longitude lookup required.
**Why:** `basins_standard` was already adopted in the 20260522 cycle (AU=441 rows); the lon-split was stale/inconsistent. With AU in the fit data, the lon-split was scoring AU storms with SI/SP coefficients.
**Revisit if:** A future data source removes AU as a standard basin.

## 2026-07-06: Deliverable finalization promoted to `run-finalize-deliverable` (was ad-hoc heredocs)
**Decision:** The three post-processing steps that produced the shipped M1 `direct_deaths.{parquet,nc}` — (a) draw-level A1/A0 blend export, (b) super-region median adjustment, (c) FHS-population rate merge — are now one committed CLI, `src/idd_tc_mortality/predict/finalize_deliverable.py` (`run-finalize-deliverable`). It writes `draw_level_<cell>_ssp_{unadjusted,adjusted}.{parquet,nc}` for a given `--pred-root`/`--mid`, with everything parameterized (cell, scenarios, hierarchy/obs/pop paths, ref-scenarios, obs-years, split-year). The median-adjust logic (`super_region_median_ratios`, `apply_super_region_adjustment`) is factored out and unit-tested on a synthetic 2-super-region frame. Regenerating from the existing 20260608 M1 partials reproduces the shipped file **positionally byte-identically** (8,669,768 rows, total-deaths + SR-31 checksums, all columns exact).
**Why:** The deliverable had been produced by ad-hoc in-session heredocs that existed only in a transcript — not reproducible, and blocking the SDI rerun. Promotion makes steps 2–3 of any rerun copy-paste reproducible and the numbers provably identical to what shipped. Recovered the exact heredocs from the session transcript rather than re-deriving, so the promoted code is behavior-equivalent by construction (then verified).
**Omitted intermediate (not a silent change):** the original chain also merged a country-and-up `build_population` pop/rate between (a) and (b); step (c) then dropped and fully overwrote both columns with the FHS population. That intermediate never affected the shipped bytes, so it is omitted as dead computation. Final rates come entirely from the FHS past+future population (sex_id=3, age_group_id=22), which covers admin-1 (level 4) — `build_population` does not.
**SR-31 OPEN DECISION:** super-region 31 (Central Europe/E. Europe/Central Asia) has observed median deaths = 0 → ratio 0 → its deaths + rates are hard-zeroed. The shipped file used this hard zero, so `--no-sr31-guard` is the default. The older 20260515 notebook instead replaced a 0 ratio with `0.1 × (smallest non-zero ratio)`, exposed as `--sr31-guard`. Whether the guard should become the default is unresolved — vet with the data producer before the SDI rerun ships.
**Revisit if:** the SDI rerun (or a new death model) is finalized — re-decide the SR-31 guard then; or if the blend rollup is wired into the orchestrator so `summary.parquet` (which (b) reads for the predicted-median baseline) is produced as part of the pipeline rather than a manual step.
