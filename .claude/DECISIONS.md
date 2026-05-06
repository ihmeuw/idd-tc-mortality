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
