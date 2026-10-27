# CMIP6 vs IBTrACS comparability report — working notes
Started: 2026-10-26 12:56. Task prompt: .claude/cmip6_ibtracs_comparison_prompt.md.
Paths: `${TS}` is the rapidresponse tropical-storms share and `${ROOT}` this project's output node (see `viz/cmip6_ibtracs_comparison/constants.py`).

## Findings while orienting (facts, not decisions)
- Observed deaths v1985 (3024 rows, 90 admin0 countries, 1584 storms, 1985–2023, zeros kept) =
  every IBTrACS CLIMADA admin0 row with person_storm_hours > 0 (3285 → 3024 after the
  psh>0 filter, minus 6 rows lacking sdi/is_island). prep_frame() reproduces that universe.
- Basin labels in the observed file = genesis point inside the tc-risk box (basin_map.TCRISK_BOXES)
  with longitude wrapped to -180..180. All 1584 storms match. Reused for every IBTrACS storm.
- IBTrACS CLIMADA land files use IBTrACS basins in their path (no AU) and key storms by the
  sid year (genesis calendar year), not by IBTrACS season.
- usa_wind coverage 1970+: NA/EP 100%, WP 88%, SI 74%, SP 78%, NI 49%. wmo_wind is also partial.
- CLIMADA model frame: storm_id is unique only within a basin (one member/draw/year slice:
  79,094 keys without basin vs 98,307 with). notebooks/20260924/storm_count_by_category.py
  grouped without basin → merged distinct storms (undercount). Quantified in the sanity stage.
- storm_draw_table: fit-draw seeds 1..100 are split unevenly across runs (MRI r1 has 1 draw).
  For the mortality side every run gets all 100 fit draws (averaged), so run-level
  differences are hazard only and the level matches the ship's draw-mean (Jensen) basis.
- Slurm: account proj_rapidresponse, partition all.q. Model frames read in ~2 s (pyarrow).

## Plan
Module: src/idd_tc_mortality/viz/cmip6_ibtracs_comparison/ (constants, storms, mortality,
stats, export, cli). CLI: run-cmip6-ibtracs-comparison {aggregate,predict,stats,export,sanity}.
Work dir: 05-figures/working/cmip6_ibtracs_comparison/ (+ slurm/ for sbatch + logs).
Compute: aggregate parts as Slurm jobs (sized after a first run via sacct); predict = probe
(1 mv, 2 fit draws) → extrapolate → 8 mv jobs + 1 IBTrACS job; stats job; export local.

## Log
- 2026-10-26 13:11 aggregate stages done. tracks 15 s, climada 28 s (Slurm). IBTrACS: 6,067 storms 1970–2025,
  4,838 ≥18 m/s, 410 with no wind in either usa_wind or wmo_wind; land files 2,004 storms, 0 outside boxes.
- 2026-10-26 13:11 sanity (global, 1970–2014, mean annual count per member): tracks Cat3+/4+/5 = 17.56/9.73/2.99
  (identical to 20260924 tc-risk script). CLIMADA land: pipeline 12.36/6.76/1.82 vs script
  11.47/6.48/1.80 (basin-keyed storms +8%/+4%/+1%). IBTrACS all: 21.42/13.98/3.76 vs script
  (season, usa only) 21.09/13.82/3.76. IBTrACS land: 4.91/1.93/0.13 → model land-hitting intense
  storms are ~2.5× (Cat3+) to ~14× (Cat5) observed. sanity.json in the work dir.
- 2026-10-26 13:11 mortality probes (2 fit draws): run = 630K A0 + 548K A1 rows, ~1 s/fit-draw/frame, 1.25 GB;
  IBTrACS 3K+2K rows, read-dominated (23 s), 0.6 GB. Full jobs: 3G/20min per run, 2G/10min IBTrACS.
  IBTrACS blend drops parents [6,11,90,95,102,130,163,165] (no storm rows in Brazil/Ethiopia).
- 2026-10-26 13:25 BUG (mine, fixed): densify used int8 Categorical codes × 100 → overflow for run index ≥ 2;
  the first stats runs were scrambled (MPI showed as zeros). Cast to int64 + regression test; rerun.
- 2026-10-26 13:25 DATA FACT: tc-risk emits a FIXED number of tracks per year per draw within each time bin
  (e.g. CMCC 60/yr for 1965–1974, 98/yr for every bin from 1975). Year-to-year variability in
  model track counts within a bin is zero by construction; only intensities (and landfall sampling)
  vary. 7 of 8 runs have ~40% fewer tracks in the 1965–1974 bins than afterwards (MPI flat 85/yr).
- 2026-10-26 13:25 DATA FACT: IBTrACS CLIMADA land run (ibtracs_stage4b_pafs_admin0): EP basin dir exists only in
  2006 and 2015 (1 storm each) → observed land-hitting EP ≈ 0; NI 0 storms in the 1970s, 9 in the
  1980s; WP 47 storm files in the 1970s vs 175–190/decade after. usa_wind coverage does not explain
  WP/NI (≈1.0); cause is upstream in the CLIMADA IBTrACS processing. Both sides step up in the
  mid-1970s for unrelated reasons → linear detrending leaves a shared step → spurious detrended
  agreement at the global land level (first, scrambled run showed ρ_det 0.47–0.9). Period stays
  1970–2025 per the prompt; flagged in report + methods; 1980 start is the obvious sensitivity.
- 2026-10-26 13:37 Published (private): https://claude.ai/artifact/1gYmPqc3sTud3A6V76wru4 (89 files, 59 MB, from
  ~/.claude/artifacts/idd-tc-mortality/cmip6_ibtracs_comparison/). Final stats job 417 s / rc=0.
  Kept geos: 7 basins, 5 SR, 11 regions, 57 countries (storms) / 53 (mortality).
- 2026-10-26 13:37 Headline (ens-of-4 mean pipeline): global land total ρ raw −0.26 (ma10) / 0.30 (blk10),
  detrended 0.59 / 0.80 — driven by the shared mid-1970s step; all-storm total ma10 0.28 / 0.76;
  Cat5 all-storm raw 0.34 (ma10), detrended −0.66 (trend only). Model land-hitting level 1.7× obs
  (65–71 vs 35–42/yr); obs below the entire 800-draw band in every period. Mortality global
  adjusted ma5: obs~CMIP6 0.43 raw / −0.36 det; obs~IBTrACS-pred 0.38 / 0.53; IBTrACS-pred~CMIP6
  0.44 / −0.03; CMIP6 deaths ≈ 2× model-on-IBTrACS (obs ≈ 0.4× model-on-IBTrACS, dominated by
  1991 Bangladesh and 2008 Nargis spikes).
- Not done / open: palette validator (no node on the host); 1980-start sensitivity; seed-normalised
  tc-risk frequency; EP / 1970s gaps in the IBTrACS CLIMADA land run are upstream (rapidresponse).

## 2026-10-26 16:28 — Spec v2 diff, frequency investigation (a–d), upstream list

### Spec v2 vs what is built (prompt file re-read 2026-10-26 16:08)
Unchanged and already built: data sources, period/splice, ≥18 m/s floor, categories, basin mapping,
per-run/GCM/ensemble/pooled × mean/median series, the six match measures raw + detrended, the
≥10-storm filter + dropped list, the three-way mortality, five geography levels, module + CLI,
Slurm compute, multi-file static artifact, six tabs, methods content, "no reanalysis run".
Changes needed for the rebuild:
1. Start-year dimension (decision 9): {1970, 1980, 1985 default, 1990, 2000}; mortality {1985,
   1990, 2000}. Built: a single period (storms 1970–2025, mortality 1985–2023). Everything in
   `stats.py` (detrend, MA, terciles, ≥10 filter, n_eff, bands) must restart at X; stats table
   gains `start_year`; every tab gets the dropdown; Overview gets a start-year panel.
2. Drop blocks (`blk5/10/15` in `STORM_WINDOWS`/`MORTALITY_WINDOWS`, `block_means`, the Overview
   heatmap columns, series-tab option). Add: (i) chance band — ~1,000 phase-randomized surrogates
   of the observed annual series (from X), smoothed identically, correlated with the model series
   → percentile / p-value + 95% band (Overview, Scorecard); (ii) run-vs-run ceiling — each run vs
   the other 7 at the same window/detrend/start, shown as a spread beside obs-vs-model; (iii)
   n_eff by Bretherton et al. 1999 (N(1−r1·r2)/(1+r1·r2) from the two smoothed series' lag-1
   autocorrelations) instead of years/window.
3. Primary model series = seed-normalised frequency (after agreement; see b); raw track counts
   move to the sanity tab.
Compute note for 2(i): rank the surrogate matrix once per (geo, outcome, window, detrend, start)
and take ρ against every model series as a matrix product; ~650k cells × 1,000 surrogates is
minutes, not hours. Seeds extraction is the expensive new input (see b).

### (a) How tc-risk sets annual frequency — file evidence
- tc-risk generates exactly `tracks_per_year` surviving tracks per year per basin and records how
  many seeds that took: `util/compute.py` lines 288–368 in
  `~/repos/tropical_cyclone_risk` (`n_tracks = namelist.tracks_per_year`; `seeds_per_month
  (year, basin, month)` written to every track file). No frequency variable is computed or written.
- The intended frequency (repo `notebooks/sample_analysis.ipynb`, cells 25–26): survival rate
  Γ = tracks_per_year / seeds_per_year; frequency f = c·Γ with one constant c (seeds/yr) tuned so
  the long-term mean of f equals the observed long-term mean. "c is unknown … a tuning parameter".
- rapidresponse's `tracks_per_year` per (run, scenario, time bin, basin) is written into every
  `config.json` by `~/repos/idd-climate-models/src/idd_climate_models/tc_risk_functions.py`
  `get_tracks_per_year()` → column `{basin}_int` of
  `${TS}/tempestextremes/outputs/cmip6/bayespoisson_time_bins_wide_max_bin_5.csv`.
  That file comes from `notebooks/detect_time_bins.ipynb` + `correcting_detect_time_bins.ipynb`:
  TempestExtremes detections on the GCM's own daily fields → annual count per basin
  (`combined_complete_data.csv`: `raw_storm_count`) → missing years filled with 0 → clipped to
  ≥1 storm (GL ≥7) → × per-model-per-basin multiplier = Σ IBTrACS / Σ clipped TE over the
  historical years (`scaled_basin`; e.g. CMCC WP 2.70, NA 6.77, EP 5.34; MPI WP 5.33) →
  BayesPoisson change-point segmentation synchronized across basins (min 10 / max 30 yr) → bin
  mean → `int(round())` → re-chunked into ≤5-yr compute bins that share the parent bin's integer.
- So `tracks_per_year` IS a frequency estimate — the GCM's own TE count, IBTrACS-level-calibrated
  per basin — but piecewise constant: within a change-point bin every year has the same count
  (CMCC historical: 1965–74 = 57, 1975–85 = 99, 1986–2014 = 86 global). Interannual variability
  is removed by design, not by accident.
- The mid-1970s step: TE has zero detections in 1965–1969 for 7 of 8 runs (only MPI has data
  there; `combined_complete_data.csv` raw counts 0 for every basin). Those years were floored to
  1 storm and averaged into the first bin (1965–1974), dragging it ~40% low (e.g. CMCC WP:
  5×2.70 and 21.6–35.1 → bin mean 15.4 vs 28 for 1970–74 alone). That is the "7/8 runs 40% fewer
  tracks" fact; it is a missing-input artefact, not a GCM feature.
- Placeholder bins: `zero_to_nonzero_changes.csv` — TE found ZERO NA storms in CMCC historical
  1987–2001, CMCC ssp585 2030–41, MPI ssp585 2040–50 and 2062–74; the clip gave them 7 storms/yr
  (multiplier × 1). NA raw TE counts average 0.82/yr with 49% zero years (1975–2014, 8 runs); NI
  1.84 (17% zeros); SP 2.03 (21%). The GCMs barely produce detectable NA storms at this resolution.
- Calibration target: the `ibtracs_count` used for the multipliers (1985–2014 means) is NA 7.8,
  EP 21.5, AU 11.8, NI 5.3, SI 12.2, SP 3.5, WP 26.3 /yr (global 88.4). My IBTrACS counts at
  ≥18 m/s with the tc-risk boxes are NA 18.5, EP 13.2, AU 12.7, NI 5.3, SI 11.4, SP 2.8, WP 25.6
  (global 89.5). Global matches; NA/EP do not (their NA is 0.42× mine, EP 1.6×) — the TE
  `basin_counts` basin definition or threshold differs from tc-risk's boxes. To check upstream.
  This, not frequency, is why the all-storm basin ratios are NA 0.40 and EP 1.79.

### (b) A per-run, per-draw, per-year frequency weight
Weight each track so that a year's weighted count equals the chosen annual frequency F:
  w(run, draw, basin, year) = F(run, draw, basin, year) / tracks_per_year(run, bin, basin).
Because the CLIMADA land frame indexes every tc-risk track (storm_id runs 0..tpy·n_years−1 per
basin-bin; checked for CMCC 1972/1990/2020), the same w applies to: all-track category counts
(Σ w over tracks), the land-hitting frame (Σ w over storm rows at each category / location, after
the one-storm-once rule), and the CMIP6-driven mortality (deaths_row × w before the (storm, draw)
sum in `predict`). The 800 run×draw series survive because w is per draw.
Two candidates for F:
- B, Lin's survival-based frequency: F = c_b · tpy / seeds(run, draw, basin, year) with seeds from
  `seeds_per_month` summed over months. Per-draw Γ has relative SE ≈ 1/√tpy (0.38 for NA's 7
  tracks); the 100-draw mean has ≈ 1/√(100·tpy) (1–4%). c_b is a per-basin constant: ranks within a
  basin do not depend on it at all; it matters for levels (ratio, percentile band) and for
  aggregates that mix basins (global, super-region, cross-basin regions). Natural choices: tune
  c_b so the 1985–2014 mean of F equals the bin-level tpy mean (keeps rapidresponse's level), or
  equals the IBTrACS basin mean (my counts). Cost: read `seeds_per_month` from ~56k track files
  (hist + ssp245 2015–25, 8 runs × 7 basins × 101 files per bin); measured 0.21 s/file → ~3.3 h
  serial, ~4 min each as 56 Slurm tasks. Draw 0 for all bins is already extracted:
  `05-figures/working/cmip6_ibtracs_comparison/seeds_draw0.parquet` (+ `tracks_per_year_config.parquet`).
- A, TE annual counts: F = `scaled_basin(run, basin, year)` from `combined_complete_data.csv` (no
  draw dimension). Cheap, but the raw counts are too small to carry a year signal at basin level
  (NA 0.82/yr, 49% zeros; NI, SP similar) and 1965–69 are missing for 7 runs. Usable at best for
  WP/EP/AU/SI and the global sum; not for NA/NI/SP.
Either way the first bin problem (1965–74) disappears only under B (Γ exists for every year);
under A the 1965–69 years stay empty.

### (c) The shipped forecast pipeline
- `src/idd_tc_mortality/predict/consolidated.py` (docstring, lines 109–114, 144) and the ship
  script `04-predict/20260806_shipped/_ship_gpd_cens_blend.py` (line 103) sum predicted deaths
  over (storm_id, tc_risk_draw) and divide by 100 per (location, year). No weight. So shipped
  deaths per year ∝ tracks_per_year(bin) × per-track deaths: the frequency is the TE-calibrated,
  piecewise-constant one above.
- Time trend: steps at change-points, no within-bin frequency variability. Global tpy by bin,
  ssp245: CMCC 91 → 82 (2033) → 113 (2053) → 102 (2070) → 118 (2080); EC-Earth 93 → 85 (2045)
  → 82 (2072); MPI 82 → 71 (2036) → 77 (2051) → 69 (2081); MRI r1 88 → 78 (2042) → 76 (2072)
  (full table: `tracks_per_year_config.parquet`). Under Lin's frequency (draw 0, c_b tuned to the
  1985–2014 tpy mean) the decade means go the other way for two GCMs: EC-Earth 97 → 129 (2090s),
  MPI flat 80–86, CMCC 86 → 145, MRI r1 94 → 107. So the sign of the forecast frequency trend
  depends on which frequency one believes; the deaths' scenario trends inherit that.
- Level: the global all-storm level is right (model 88.0 vs IBTrACS 89.5 /yr ≥18 m/s, 1985–2025;
  ratio 0.98). The ~2× deaths level (CMIP6-pred vs model-on-IBTrACS) is therefore not a frequency
  calibration error; it is the landfall excess in (d) plus basin allocation (NA 0.40×, EP 1.79×,
  which moves hazard from the US/Caribbean toward Mexico/Central America) and the Cat-5-at-land
  excess. Assessment only; nothing in the deliverable touched.

### (d) Does frequency explain the 1.7× land-hitting and ~14× Cat 5 excess? No.
1985–2025 means per member (model) vs observed (ratios in parentheses):
- All storms ≥18 m/s: 88.0 vs 89.5 (0.98). Cat 3+: 18.4 vs 24.5 (0.75). Cat 5: 3.2 vs 5.2 (0.61).
- Land-hitting ≥18 m/s: 67.7 vs 39.7 (1.70). Landfall fraction 0.77 vs 0.44.
- Independent check: IBTrACS `dist2land` ≤ 0 km gives 40.1 land-hitting storms/yr (fraction 0.45;
  within 100 km: 49.9/yr, 0.56). The CLIMADA IBTrACS land run (39.7/yr) is therefore not the
  problem at the global level; the tc-risk tracks reach land (≥18 m/s over an admin0) far more
  often than observed storms do. Basin landfall fractions model/obs(CLIMADA)/obs(dist2land):
  AU 0.82/0.59/0.51, NA 0.86/0.38/0.47, NI 0.92/0.79/0.77, SI 0.69/0.29/0.24, WP 0.84/0.66/0.64,
  EP 0.63/0.03/0.11 (land run missing EP), SP 0.83/0.14/0.06 (islands; dist2land unreliable).
- Cat 5: the model has fewer Cat 5 storms than observed (0.61×) but 60% of them are Cat 5 over
  land (1.90/yr) vs 5% observed (0.27/yr) → 7.1× for 1985–2025 (the 14× quoted earlier was
  1970–2025). Intensity at landfall, not frequency.
A frequency weight rescales years; it cannot change a landfall fraction or an at-land intensity.
It will change ranks, trends and the 1965–74 step, which is what the report is about.

### Upstream issues to raise with rapidresponse (Bobby raises)
1. IBTrACS CLIMADA land run (`climada/output/ibtracs_stage4b_pafs_admin0/.../historical/{year}/{basin}`):
   EP basin has storm files in only 2 of 56 years (2006, 2015). Observed EP land-hitting ≈ 0.34/yr
   vs 1.46/yr by `dist2land`. The v1985/v2000 training data inherit this (20 EP rows).
2. Same run: no NI storm files in the 1970s (9 in the 1980s); WP 47 files in the 1970s vs 175–190
   per later decade. Not explained by `usa_wind` coverage (WP/NI ≈ 1.0 of my fallback).
3. TempestExtremes: 1965–1969 has zero detections for 7 of 8 runs (MPI has data); the clipped
   zeros were averaged into the 1965–74 change-point bin → `tracks_per_year` ~40% low in 1965–74.
4. TempestExtremes: NA bins with zero detections replaced by 7 storms/yr (CMCC hist 1987–2001;
   CMCC ssp585 2030–41; MPI ssp585 2040–50, 2062–74; `zero_to_nonzero_changes.csv`).
5. Calibration target: the `ibtracs_count` basin split (NA 7.8/yr, EP 21.5/yr, 1985–2014) does not
   match IBTrACS counts in the tc-risk basin boxes at ≥18 m/s (NA 18.5, EP 13.2). Global totals
   agree (88.4 vs 89.5). Likely a basin-mask or threshold difference in TE `basin_counts`.
6. (For awareness, not a bug) `tracks_per_year` is constant within change-point bins, so no model
   storm-count or deaths series carries interannual frequency variability; `seeds_per_month` is
   in every track file and supports Lin's survival-based frequency if wanted.

## 2026-10-26 17:20 — v3 CLIMADA frames: deliverable recreation (staged) and rebuild progress

### New CLIMADA outputs found (Bobby 16:45)
- admin0: `climada/output/stage4b_v3_admin0/_consolidated/storm_exposure_all.parquet` (2026-10-26
  02:51, 36.7M rows, 157 locations, new bool `special_region_flag`, all False).
- admin1: `climada/output/stage4b_v3_admin1/_consolidated/storm_exposure_all.parquet` (2026-10-18
  17:40, 30.7M rows, 199 locations). Same 8 members × 4 scenarios × 1970–2100 × 100 draws as v2.
- vs v2 (historical 1985–2014): dropped Kyrgyzstan/Tajikistan/Uzbekistan and 10 inland admin-1
  units (Iowa, Nebraska, South Dakota, Himachal Pradesh, J&K, AJK, Gilgit-Baltistan, Islamabad,
  Qinghai, Xinjiang); added Burkina Faso, Côte d'Ivoire; West-African row counts ×2.5–3.5;
  Greenland/Bhutan/Lesotho/Eswatini rows −30–40%; land-hitting storms (≥18 m/s) +2.5% per
  member-year (69.7 vs 68.0); exposed population −10%; person-storm-hours −5%; max-wind q99 72→68.
  Same 10 subnationalized parents. IBTrACS per-storm land files NOT rerun (May 28); the new
  `ibtracs_stage3_admin0_admin1_consolidated/paf_ibtracs_*.parquet` (Oct 25) is a PAF summary, not
  per-storm exposure, so the comparison's observed side and the rake reference are unchanged.
- No changelog in the output tree or in Bobby's idd-climate-models clone (last commit 2026-09-23).

### (a) Deliverable recreated on v3 — STAGED, not shipped
- `04-predict/20261026_shipped_v3/_ship_gpd_cens_blend_v3.py` = sed-copy of the August ship script
  (FRAMES → v3, OUT, DR → `direct_risk_staging/`, names `2026_10_26_v2000_*`, meta keys
  `climada_frames`, `supersedes`, `sdi_change`). Slurm 64092623: 8 min 12 s, MaxRSS 8.7 GB
  (ask 4 CPU / 16G / 1 h; first measured run — the August ship ran interactively).
- Input change forced on me: the ssp126/ssp585 scenario-SDI files the August ship read
  (`data/38/future/sdi/future_sdi_s160v1`, `s161v1`) no longer exist. Used the finalized
  successors `s160v7` / `s161v7` (svc_fhs, 2026-10-19, read-only; a v8 pair was being written
  today). Same 484 locations × 1970–2100 through the ship's `bulk_sdi_table`; max |Δ| vs the
  ssp245 file 0.005 (2030), 0.006–0.010 (2060), 0.021–0.031 (2100). Bobby to confirm or pick v8.
- QC vs 2026_08_20 (`_qc_v3_vs_0820.py`, log `qc_v3_vs_0820.log`): 385 locations (398 − the 13
  inland units); global ssp245 adjusted mean 7233 / 5714 / 3758 (2015 / 2050 / 2100) vs
  7249 / 5765 / 3809; unadjusted 28906 / 21566 / 16161 vs 29405 / 22182 / 16868. Rake ratios
  new/old: SR4 (SE Asia…) and 64 (high-income) see the log head above; 103 0.99, 137 1.06,
  158 1.08, 166 1.01, 31 0.80 (ratio 0.0032, near zero either way). Super-region unadjusted
  2040–60: high-income 0.86 (UK 0.66, Ireland 0.62, Canada 0.77, Spain 0.71), North Africa/
  Middle East 0.93, South Asia 0.92, others ≈ 1.0; Caribbean islands +40–47%, Cabo Verde ×2.1,
  El Salvador ×1.39; DPRK adjusted 0.78, Mauritius 0.82, China 0.93.
- Nothing written to rapidresponse `direct_risk/`. Copy = Bobby's call.

### (b) Comparability rebuild (in progress at the time of writing)
- Code: constants (v3 frames, `CLIMADA_VERSION`, starts 1970/80/85/90/2000 and 1985/90/2000,
  windows ma1/5/10 and ma5/10/15, no blocks, 1,000 surrogates, `RAKE_MODEL_JSON` = staged
  2026_10_26 json); stats.py rewritten (start_year dimension; phase-randomized surrogates →
  sur_lo/sur_hi/sur_pct/sur_p; ceiling.parquet with all-pairs / cross-GCM / obs-vs-run spreads;
  Bretherton n_eff; raw bands stored once under start_year 0); export.py (stat_keys arrays,
  per-start overview files, ceiling, rake meta); page.html (start-year control, headline-by-start
  panel, ρ strip, chance/ceiling columns, methods rewritten); `predict --rake-json`. 49/49 tests.
- Runs: climada v3 recount 64091442 (44 s, 3.0 GB); stats --storm-only 64098790; sanity 64098791;
  predict ×9 64099422–64099430 (v3 frames, staged ratios).

## 2026-10-26 18:00 — v2 report published; numbers
- Published to the same URL (https://claude.ai/artifact/1gYmPqc3sTud3A6V76wru4): Versions 2–4
  (three publishes; the 64 MB per-publish cap split the 94 files). Bundle 95 files / 119 MB; largest
  file 6.8 MB. Stale v1 `data/overview.json` removed (per-start `data/overview/{start}.json` now).
- Runs: stats full 64101854 30 min / 3.4 GB (ask 5G/90 min); storm-only 64098790 20 min / 2.5 GB;
  predict ×8 ~2 min / ~1 GB, IBTrACS 31 s / 0.6 GB; export 100 s; screenshots via shoot.sh (port 8766/7).
- Sanity on v3 frames (1970–2014, global per member): CLIMADA land Cat3+/4+/5 11.28/5.24/0.93
  (v2: 12.36/6.76/1.82; the 20260924 no-basin-key method 10.54/5.08/0.93); tracks 17.56/9.73/2.99
  unchanged; IBTrACS land 4.91/1.93/0.13; IBTrACS all 21.42/13.98/3.76.
- Headline numbers: see STATUS "Comparability numbers (v2 report)". Summary: at start 1985 nothing
  at the global land level leaves its chance band (bands ±0.28/±0.57/±0.77 for 1/5/10-yr; n_eff
  41/8.5/6.8); run-vs-run ceilings 0.20/0.31/0.43 with obs-vs-run medians at or below zero;
  starts 1990/2000 give 1-yr raw ρ 0.34/0.36 outside the band (p 0.02/0.03) but detrended p ≥
  0.06; WP land 10-yr 0.52 raw / 0.82 det and SI 0.81 det are outside their bands at 1985.
  Mortality (global adjusted 5-yr, 1985): obs~CMIP6 0.43 raw (p 0.05) / −0.33 det; obs~IBTrACS-pred
  0.38 / 0.53; CMIP6 level 0.82× observed adjusted, 2.12× IBTrACS-pred; from 1990 obs~CMIP6 raw
  0.50 / 0.42 (p 0.00) but detrended −0.13 / −0.52: shared decline, not phasing.
- Not done: palette validator (no node); v8 SDI pair not evaluated; nothing committed.
