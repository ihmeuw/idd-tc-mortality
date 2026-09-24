# Project status
Updated: 2026-09-23 16:59

## Goals
Double-hurdle TC mortality pipeline: fit/evaluate/select death models on IBTrACS storm data,
forecast direct deaths on CLIMADA storm frames to 2100 under RCP scenarios, deliver draw-level
deaths to rapidresponse/direct_risk. Four-component model: S1 (P(deaths>=1)),
S2 (P(rate>=threshold | S1=1)), bulk rate, tail rate. Stages: validate → grid → fit → evaluate →
select → uncertainty → predict → figures.

## Orientation
Selection is COMPLETE and the deliverable is SHIPPED; the active work is FIGURE TOOLING for
the shipped model, built on idd-figures.

*The shipped model.* The 20260722 v2000 honest-mean cycle's selection (protocol: DECISIONS
2026-08-04) produced a trio of q=0.95 intercept-tail models — gamma 7e4c6c32, gpd_cens
e7a78b19, log_logistic_trunc 9c8497e7 — with interchangeable post-rake forecasts. gpd_cens
e7a78b19 is the shipped model: `direct_risk/2026_08_20_v2000_{adjusted_mean,unadjusted}_
direct_deaths.{parquet,nc}` + `2026_08_20_v2000_model.json`; cell deaths_c1_s1_o0_b1, 100
paired draws (fit seed = storm_draw), A1/A0 BLEND (admin-1 units for the 10 subnationalized
countries, admin-0 rows elsewhere; 398 locations), super-region mean+guard rake recomputed ON
THE BLEND (ratios stored in model.json), ssp scenarios 2015–2100. It supersedes the A0-only,
misdated `2026_08_26_v2000_*` files (left in place; Bobby to delete / notify downstream). Ship
path: `04-predict/20260806_shipped/_ship_gpd_cens_blend.py` on the output node (not in the repo).

*Figure tooling (2026-09-23).* Figures for the shipped model come from the BLENDED deliverable
only — never from the A0 super-region draw file `04-predict/20260804_meanfwd/c1s1_sr_final_*`.
Code lives in the repo as viz modules with click CLIs (registered in pyproject); outputs land in
the new node `05-figures/working/` (working slot, overwritten each run). Layouts use idd-figures
(`panel_grid`, `map_cell_painter`, explicit inch geometry, no auto-layout; `save_figure` is never
bbox-tight, so legends are grid cells or `loc="outside …"` under constrained layout).
- `run-blend-historical` — historical 1985–2014 predicted on the blend with the shipped rake
  ratios → `05-figures/working/shipped_e7a78b19/historical_blend_{unadjusted,adjusted_mean}.parquet`.
- `run-shipped-figures --tag adjusted_mean|unadjusted` — global 2×2 (linear/log × full/2020+,
  observed dots, historical band), super-region 3×2 grid (SR 31 excluded by default), cumulative
  boxes 2023–2100 with paired RCP8.5−RCP2.6.
- `run-storm-three-panel` + `notebooks/20260923/katrina_three_panel.ipynb` — per-storm RR /
  population density / attributable person-years, and max-intensity / exposure-hours panels.
  Stage-1 grids are land/coastal-only by construction: that IS the intended product.
- `run-basin-map` + `notebooks/20260923/basin_map_knobs.ipynb` — IBTrACS vs tc-risk basin map.
- `notebooks/20260923/dh_stage_vetting_final_gpd_cens.ipynb` — stage vets for e7a78b19 (refit on
  the fly + `StagePlotter.register_results`; the stage-4 gpd_cens density panel now renders).

*Numbers to remember.* Cumulative 2023–2100 global deaths, medians over the 100 draws: raked
274k / 265k / 257k (RCP2.6 / 4.5 / 8.5), unadjusted 1.22M / 1.19M / 1.12M; paired 8.5−2.6
median −17k (IQR −32k..−12k). The "458/448/429k" in older notes are the raked MEANS of the A0
super-region file (means ≫ medians: right-skewed draws). The blend raises unadjusted totals
~50% vs A0 (the model is nonlinear in exposure); the rake anchors raked totals back within a few %.

*Uncertainty story (2026-08-06).* Fit (c1_s1) UI dwarfs climate UI ×25; outcome draws matter at
global scale; the draw-mean sits ~45% above the point forecast (Jensen), so rake ratios are
cell-consistent (c1s1). The three scenario-SDI files are near-identical (Δ≤0.003 by 2100):
scenario spread is storm-frame-driven; the upstream question is open.

## Recent steps
- 2026-09-23: idd-figures[maps], geopandas, rasterio added (git sources pinned to main;
  idd-tools path→git realised in the venv). Built storm_three_panel (+ notebook), StagePlotter
  gpd_cens support + final-model vetting notebook (by-basin S1/S2, stage-4 density),
  blend_historical + shipped_figures (global / SR / cumulative from the blend), basin_map port
  (+ knobs notebook; the notebook's GEOS error was a zero-width box). Reconciled the 458k vs 282k
  confusion (mean vs median of the A0 file). Committed as 10f88ba..76b9dc7.
- 2026-08-20: corrected ship as the A1/A0 blend (`2026_08_20_v2000_*`, 398 locations, rake
  recomputed on the blend, scenario SDI); all-SR grid figures (ours vs M1); SDI diagnostic.
- 2026-08-06: four uncertainty cells run; M1 comparison figures; A0-only ship (superseded).
- 2026-08-04: final selection protocol converged; trio wired, forecast, figured.

## Next steps
1. Bobby: decide the parked dirt — `select/model_selection.py`,
   `notebooks/20260714_v2000_sensitivity/decompose.ipynb`,
   `notebooks/20260722/heavy_tail_simple_and_real.ipynb` (all modified, provenance unknown),
   `notebooks/20260804/` (in progress; decompose.ipynb 9.5 MB), `notebooks/temp/`, `test.md`,
   `.claude/selection_{tie_diagnosis,protocol_proposal}.md` — commit, drop, or gitignore.
2. Bobby: delete / notify downstream about the superseded `direct_risk/2026_08_26_v2000_*` files.
3. Pre-existing failing test `tests/viz/test_stage_plots.py::test_vet_stage_3x3_layout` (expects 9
   axes; DEFAULT_VET_CONFIG defines 8 panels) — fix the test or the config.
4. Decide whether `04-predict/20260804_meanfwd/_*.py` and `20260806_shipped/_*.py` (the only record
   of the shipped code path, on the node) should be copied into the repo.
5. Carried: upstream scenario-SDI question; v1985 clone of the selection protocol; push the
   idd-tools inbox note once that repo's tree is settled.
6. Optional figure polish: SR grid with the historical band; stage-2 vet against people-storm-hours
   (the final S2's only covariate; its by-basin panels are flat by construction).

## Parking lot
- cens per-fold k-stability diagnostic; 16-cell diagnostic table + per-stage Hessian conditioning.
- decompose.ipynb prerequisite: sensitivity runs do not exist for the final models.
- Old figure sets in `20260804_meanfwd/figures` (picks_*, pick_*, global_*) — Bobby may clear.
- Feed fleet-measured per-tier costs into packing; consolidate `SUPER_REGIONS`; absolute-path
  tech debt (two-tier paths.yaml before any public release); delete dead fit stage; fix stale
  `test_run_component.py`; basin random effects; draw-count sufficiency; tighten
  `fit/orchestrate.py` `_SLURM_RESOURCES`.
- stage_plots.py carries 24 pre-existing ruff hits (all `Optional[...]`/import-order style).

## Process / skill goals (near-term)
- Compute both summaries (mean AND median) before calling a discrepancy; plot the actual
  deliverable, not a convenient intermediate.
- Never pgrep a pattern contained in the checking command.
- Size runtime asks to measured deployment conditions; zero-row outputs must log loudly.
- Probe-first sizing as default; match response to scope.
