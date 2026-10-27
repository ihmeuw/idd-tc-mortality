# Project status
Updated: 2026-10-27

## Goals
Double-hurdle TC mortality pipeline: fit/evaluate/select death models on IBTrACS storm data,
forecast direct deaths on CLIMADA storm frames to 2100 under RCP scenarios, deliver draw-level
deaths to rapidresponse/direct_risk. Four-component model: S1 (P(deaths>=1)),
S2 (P(rate>=threshold | S1=1)), bulk rate, tail rate. Stages: validate → grid → fit → evaluate →
select → uncertainty → predict → figures.

## Orientation
Selection is COMPLETE and the 2026-08-20 deliverable is SHIPPED. Active work: the CMIP6 vs
IBTrACS comparability report (v2 published, code committed 2026-10-27) and the re-ship of the
same model on the CLIMADA stage4b_v3 frames (STAGED on the node, awaiting Bobby's go). The
figure tooling (2026-09-23) is in place and unchanged.

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

*Comparability numbers (2026-10-26 v2 report: CLIMADA v3 frames, ensemble-of-4 mean pipeline,
start 1985 unless said).* Levels 1985–2025: all-storm global model/observed 0.98; land-hitting
1.75 (landfall fraction 0.79 vs 0.44, confirmed by IBTrACS dist2land 0.45); Cat 5 over land 3.7×
(31% of model Cat 5 storms are Cat 5 over land vs 5% observed; v2 frames had 60%); basins land
NA 0.92, NI 0.98, AU 1.34, WP 1.41, SI 2.11, EP/SP not comparable. Global land total ρ raw/det:
1-yr 0.15/0.14, 5-yr 0.15/0.18, 10-yr 0.12/0.42 — all inside the phase-randomized chance bands
(±0.28, ±0.57, ±0.77); run-vs-run ceilings 0.20/0.31/0.43 (obs-vs-run medians −0.04/−0.13/−0.22).
Start 1970 shows the mid-1970s step (10-yr det 0.59, p 0.14); starts 1990/2000 give 1-yr raw
0.34/0.36 (p 0.02/0.03), the only global land cells outside the band. All-storm global 5-yr raw
0.58 (p 0.14). Basins land 10-yr raw/det: WP 0.52*/0.82*, SI 0.55/0.81*, NA 0.33/−0.54, AU
−0.52/−0.16, NI −0.67/−0.22 (* outside band). Country cells outside the band 1–3% (the null).
Mortality global adjusted 5-yr: obs~CMIP6 0.43 (p 0.05)/−0.33 det; obs~IBTrACS-pred 0.38/0.53;
IBTrACS-pred~CMIP6 0.43/−0.01; levels CMIP6/obs 0.82, IBTrACS-pred/obs 0.38, CMIP6/IBTrACS-pred
2.12 (unadjusted CMIP6/obs 2.97). Kept at 1985: 7 basins, 5 SR, 11 regions, 54 countries (53
mortality); at 2000: 6 basins, 42 (40).

*Uncertainty story (2026-08-06).* Fit (c1_s1) UI dwarfs climate UI ×25; outcome draws matter at
global scale; the draw-mean sits ~45% above the point forecast (Jensen), so rake ratios are
cell-consistent (c1s1). The three scenario-SDI files are near-identical (Δ≤0.003 by 2100):
scenario spread is storm-frame-driven; the upstream question is open.

*CMIP6 vs IBTrACS comparability report (2026-10-26, v2 published the same day).* Interactive artifact
(https://claude.ai/artifact/1gYmPqc3sTud3A6V76wru4, private, Version 4) answering whether big
IBTrACS years/periods line up with big CMIP6 (tc-risk / CLIMADA) periods; spec in
`.claude/cmip6_ibtracs_comparison_prompt.md` (v2: start years 1970/80/85/90/2000, no blocks, chance
band from 1,000 phase-randomized surrogates, run-vs-run ceiling, Bretherton n_eff), working notes in
`.claude/cmip6_ibtracs_comparison_notes.md`. Code: `viz/cmip6_ibtracs_comparison/` (CLI
`run-cmip6-ibtracs-comparison aggregate|predict|stats|export|sanity`, 49 tests). Outputs in
`05-figures/working/cmip6_ibtracs_comparison/` (counts, mortality/, stats/ incl. ceiling.parquet,
artifact/ 95 files 119 MB); publish mirror `~/.claude/artifacts/idd-tc-mortality/cmip6_ibtracs_comparison/`
(publish in ≤64 MB parts). Model frames are the CLIMADA stage4b_v3 consolidated admin0/admin1
(2026-10-26 / 10-18). The model series are the RAW tc-risk/CLIMADA counts: `tracks_per_year` is
Bobby's calibration (TempestExtremes detections per GCM → IBTrACS-scaled per basin → BayesPoisson
change-point bins → integer per bin, `idd-climate-models tc_risk_functions.get_tracks_per_year`),
constant within a bin by design; no seeds-based renormalisation (decided 2026-10-26). Data facts in
the Methods tab: TempestExtremes has no 1965–69 detections for 7/8 runs (floored to 1 → the
1965–74 bin ~40% low: the mid-1970s step); four NA bins with zero detections carry a placeholder
7 storms/yr; the calibration's IBTrACS basin split (NA 7.8, EP 21.5 /yr) differs from IBTrACS in
the tc-risk boxes (NA 18.5, EP 13.2); the IBTrACS CLIMADA land run has no EP basin and thin 1970s
WP/NI; CLIMADA `storm_id` is unique only within a basin. Upstream list for rapidresponse: notes
§"Upstream issues". Correction logged: I had called the raw counts "not frequency estimates".
Committed 2026-10-27 as 86f546a. Paths: the module derives `TS` from
`predict.blend_historical.DR` (no new cluster-path literal); the spec and notes use `${TS}` /
`${ROOT}` placeholders because the repo is PUBLIC (`gh repo view`, 2026-10-27).

*v3-frames deliverable, STAGED (2026-10-26).* `04-predict/20261026_shipped_v3/_ship_gpd_cens_blend_v3.py`
(sed-copy of the August ship; same model e7a78b19, same blend/rake path) wrote
`2026_10_26_v2000_{adjusted_mean,unadjusted}_direct_deaths.{parquet,nc}` + `_model.json` into
`.../direct_risk_staging/` — NOT copied to rapidresponse `direct_risk/`. Forced input change: the
ssp126/ssp585 SDI files s160v1/s161v1 were deleted upstream; s160v7/s161v7 (finalized 2026-10-19)
used, max |Δ| vs ssp245 ≤ 0.03 by 2100 (a v8 pair appeared the same day). QC vs 2026_08_20
(`qc_v3_vs_0820.log`): 385 locations (the 13 inland units gone), global ssp245 adjusted
7233/5714/3758 vs 7249/5765/3809, rake ratios within 0.99–1.16 (high-income 1.16), Caribbean
unadjusted +40–47%, UK/Ireland/Canada −23–38%. Slurm: 8 min, 8.7 GB (16G ask).

## Recent steps
- 2026-10-27 (am): ruff-cleaned the comparability module (18 cosmetic findings), routed its one
  absolute path through `DR`, scrubbed cluster paths from the two notes files; committed 86f546a
  (module + tests + pyproject) and 4f9624c (tracking); pushed, origin up to date.
- 2026-10-26 (pm): frequency investigation (notes §16:28); v3 CLIMADA frames found; deliverable
  recreated on v3 and staged (not shipped); comparability report v2 rebuilt on v3 + spec v2 and
  republished to the same URL (Versions 2–4). Slurm: climada recount 44 s / 3.0 GB; predict ×8
  ~2 min / 1 GB, IBTrACS 31 s; stats 30 min / 3.4 GB (storm-only 20 min / 2.5 GB); export 100 s.
- 2026-10-26 (am): CMIP6 vs IBTrACS comparability pipeline + artifact v1 built and published. Slurm:
  aggregates 15–30 s; 8 mortality runs ~2 min / 1 GB each; stats 7 min / 1.5 GB. One bug fixed
  mid-way (int8 Categorical codes × 100 overflowed and scrambled runs; regression test added).
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
0. Bobby decides: (a) copy the staged `2026_10_26_v2000_*` deliverable to rapidresponse
   `direct_risk/` (and confirm SDI s160v7/s161v7 vs the newer v8 pair); (b) raise the six upstream
   items in the notes with rapidresponse (EP missing / thin 1970s in the IBTrACS CLIMADA land run;
   TE 1965–69 gap; NA placeholder bins; calibration basin split; bin-constant tracks_per_year for
   awareness).
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
  tech debt: the repo is already PUBLIC (gh, 2026-10-27) and `predict/paths.py`,
  `predict/blend_historical.py`, DECISIONS.md and settings.json carry cluster paths — the two-tier
  paths.yaml migration is overdue; delete dead fit stage; fix stale
  `test_run_component.py`; basin random effects; draw-count sufficiency; tighten
  `fit/orchestrate.py` `_SLURM_RESOURCES`.
- stage_plots.py carries 24 pre-existing ruff hits (all `Optional[...]`/import-order style).

## Process / skill goals (near-term)
- Compute both summaries (mean AND median) before calling a discrepancy; plot the actual
  deliverable, not a convenient intermediate.
- Never pgrep a pattern contained in the checking command.
- Size runtime asks to measured deployment conditions; zero-row outputs must log loudly.
- Probe-first sizing as default; match response to scope.
