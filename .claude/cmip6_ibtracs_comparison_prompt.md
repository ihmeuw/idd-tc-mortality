# Task: CMIP6 (tc-risk/CLIMADA) vs IBTrACS comparability report — interactive artifact

Paths: `${TS}` is the rapidresponse tropical-storms share and `${ROOT}` this project's output node (both defined in `viz/cmip6_ibtracs_comparison/constants.py`).

Build an interactive, multi-tab, data-backed Artifact comparing the CMIP6-driven synthetic
storm record (and the mortality it implies) with the observed IBTrACS record, by GCM and for
the ensemble. The design decisions below were already made with Bobby. Don't reopen them.
Ask only if something is actually blocked.

Read first: `CLAUDE.md` (venv-only: run everything with `.venv/bin/python`, no conda),
`.claude/STATUS.md`, and the two prior scripts this builds on:
`notebooks/20260924/storm_count_by_category.py` (CLIMADA land-affecting counts, model + IBTrACS)
and `notebooks/20260924/tc_risk_storm_count_by_category.py` (all-track counts from tc-risk,
IBTrACS netCDF). Also `src/idd_tc_mortality/predict/blend_historical.py` (`predict_variant`,
hierarchy, A1/A0 blend, shipped rake ratios).

## The question and its framing

Bobby wants to know if "big" IBTrACS years and periods line up with relatively "big" CMIP6
years and periods. All comparisons use **rank correlation, not values**.

The caveat that drives the design: CMIP6 historical runs are free-running coupled models. Their
internal variability (ENSO etc.) is not phased with the real world, so year-to-year
correlations should be about 0. Multi-year agreement mostly reflects shared forced trends,
plus observing-system trends in IBTrACS. Also, overlapping moving averages leave very few
effective samples. Therefore:
- Report every correlation **raw and detrended** (linear detrend of each annual series before
  smoothing), side by side.
- **No non-overlapping blocks.** From 1985, 10-yr blocks leave about 4 points, and rank
  correlation on 4 points can only take a handful of values. Keep the moving averages, and
  make their uncertainty honest with two benchmarks instead:
  - **Chance band:** shuffle the observed series in a way that keeps its smoothness
    (phase-randomized surrogates of the annual series, about 1,000 per series), smooth each
    one identically, and correlate it with the model. Report where the real correlation falls
    in that distribution (a percentile / p-value), and show the 95% band in the Overview and
    Scorecard tabs.
  - **Ceiling (how well one run can predict another):** correlate each CMIP6 run with the
    other runs at the same window. These runs share the climate forcing but not the weather,
    so this is the agreement you'd expect if IBTrACS behaved exactly like one more run. Show
    obs-vs-model next to this run-vs-run spread: obs inside the spread means "as good as can
    be expected"; obs well below it means the model is missing something.
  - Also show the effective sample size, adjusted for autocorrelation (Bretherton et al. 1999).
- Present 1-yr results as the "expected about 0" check.
- No reanalysis-driven (ERA5/AMIP) tc-risk run exists (`tc_risk/output/` has only `cmip6/`).
  Say so in the methods tab as the upgrade path.

## Data

- tc-risk all tracks (already extracted, lifetime max vmax per track):
  `${TS}/figures/tc_risk/track_max_vmax.parquet` (83 MB).
- CLIMADA land-affecting model frame (per storm × admin0, `max_wind_speed`, draws, members):
  `${TS}/climada/output/stage4b_v3_admin0/_consolidated/storm_exposure_all.parquet`
  (937 MB; v3 run, consolidated 2026-10-26; replaces `stage4b_v2`). The admin1 frame is
  `stage4b_v3_admin1/_consolidated/...` (2026-10-18). Countries are admin0, so aggregate A1
  to parents if you use it. The v2 → v3 differences are listed in the notes file.
- IBTrACS land-affecting (CLIMADA, same method as the model side), 1970–2025:
  `${TS}/climada/output/ibtracs_stage4b_pafs_admin0/ibtracs/official/historical/{year}/{basin}/storm_paf/storm_*.parquet`
- IBTrACS all storms: `~/climada/data/IBTrACS.ALL.v04r01.nc` (`usa_wind`, kt → m/s).
  Check `usa_wind` coverage outside US-agency basins, and say what you did about the gaps.
- Members: CMCC-ESM2 r1, EC-Earth3 r1, MPI-ESM1-2-HR r1, MRI-ESM2-0 r1–r5 (8 runs, 4 GCMs),
  100 tc-risk draws each.
- GBD hierarchy: `HIER_PATH` in `blend_historical.py`
  (`fhs_2023_modeling_hierarchy.parquet`). Shapefiles: use idd-figures' maps (they're the right
  GBD shapes). Simplify them heavily for the web (TopoJSON/GeoJSON, target ≤2 MB).
- Observed deaths: `${ROOT}/00-data/20260722_v1985/input.parquet`
  (approved; the v2000 file only covers 2000–2023). Check what row universe it holds
  (all exposed storm-locations vs. a filtered set), and make the CMIP6 side match.
- Shipped mortality model: gpd_cens e7a78b19. Rake ratios from the v3-frames recreation of the
  deliverable, `04-predict/20261026_shipped_v3/direct_risk_staging/2026_10_26_v2000_model.json`
  (staged on the node until Bobby copies it to `direct_risk/`; the August ratios are in
  `direct_risk/2026_08_20_v2000_model.json`). See STATUS for the blend/rake conventions.

## Decisions (final)

1. **Period.** Model = CMIP6 `historical` spliced with **SSP2-4.5 for 2015–2025**. Storm half:
   1970–2025. Mortality half: 1985–2023 (observed range).
2. **"Total storms" floor.** Tropical-storm strength, ≥18 m/s, on both sides.
3. **Categories (m/s).** Cat1 ≥33, Cat2 ≥43, Cat3 ≥50, Cat4 ≥58, Cat5 ≥70; plus 3+ and 4+.
   Outcomes: total storms, total land-hitting storms, Cat1, Cat2, Cat3, Cat4, Cat5, 3+, 4+
   (Cat k = exactly k).
   - Global / basin "all storms": the storm's lifetime max wind (tc-risk `max_vmax`; IBTrACS
     track max).
   - Country: the CLIMADA max wind over that country.
   - **Aggregates (region, super-region, global/basin land-hitting): each storm counts once,
     at its worst category over the units in that aggregate.** So a global land-hitting storm
     counts at the worst it was anywhere on land, and a region's storm counts at the worst it
     was within that region.
   - Map tc-risk basins (AU EP NA NI SI SP WP) to IBTrACS basins/subbasins explicitly, and
     document the mapping (IBTrACS has no AU basin).
4. **Series.**
   - Per run: the annual count as the **mean** over its 100 draws, then separately the
     **median**.
   - Per GCM: the same statistic over its runs (MRI over 5).
   - Ensemble: the same statistic over the 4 GCMs, so MRI isn't over-weighted.
   - Also a pooled ensemble across all 8 runs.
   - Everything is computed twice (mean pipeline, median pipeline) because the answers may
     differ. Keep the full draw distribution for the "observed percentile within model draws"
     measure.
5. **Windows.** Storms: 1, 5, 10-yr moving averages. Mortality: 5, 10, 15-yr moving
   averages. No blocks; see the chance band and ceiling above.
6. **Match measures (rank-based).**
   - Spearman (headline) and Kendall τ.
   - Big-period hit rate: share of IBTrACS top-tercile periods that are also model
     top-tercile.
   - Up/down agreement between consecutive periods.
   - Observed value's percentile within the 800 run × draw values.
   - Secondary, non-rank: model/observed ratio of period totals.
   - Raw and detrended for all of the above.
7. **Sparse places.** Drop locations with fewer than 10 observed land-hitting storms in the
   period. List the dropped ones in the report.
8. **Mortality, three-way.** Observed deaths vs. the shipped model predicted on **IBTrACS**
   storms (1985–2023; pre-2000 is out-of-sample for the v2000 fit, so call that out) vs. the
   shipped model predicted on **CMIP6** storms (historical + SSP2-4.5; `predict_variant`
   takes `scenario=`). This splits hazard error from mortality-model error. Do it for both
   adjusted_mean and unadjusted. Note that the rake is a constant ratio per super-region, so
   within-super-region rank results won't change between the two. Use the same geography
   levels as the storm half.

9. **Start year.** IBTrACS is considered incomplete before 1985, so the whole analysis must
   be subsettable to "year X to present". Start-year options: 1970, 1980, **1985 (default)**,
   1990, 2000; the end stays at 2025 (2023 for mortality).
   - Everything restarts at X: detrending, moving averages, top-tercile cut-offs, the
     ≥10-storm location filter, chance bands, run-vs-run ceilings, and effective n.
   - Precompute every stat for every start year, as a `start_year` dimension in the stats
     table.
   - Mortality starts at 1985 anyway, so only 1985, 1990 and 2000 apply there.

10. **Model series.** The raw tc-risk / CLIMADA counts are the primary model series. The number
   of tracks per year is Bobby's calibration (TempestExtremes detections per GCM, scaled to
   IBTrACS per basin, constant within change-point bins); it is not renormalised by seeds or
   any other frequency. Within a bin the model's year-to-year count variation is zero by
   design, and the report says so. (Decided 2026-10-26 after the frequency investigation.)

Geography levels: global, basin (7), GBD super-region, region, country (admin0).

## Compute

Run heavy work as Slurm jobs (sbatch from the login node; `.venv/bin/python` by absolute
path). Size resources from a probe, not guesses. Put the code in the repo as a proper module
with a click CLI registered in `pyproject.toml`, the same pattern as the other viz modules
(e.g. `src/idd_tc_mortality/viz/cmip6_ibtracs_comparison.py`), with stages
`aggregate` → `stats` → `export`.
- Intermediate parquet goes to
  `${ROOT}/05-figures/working/cmip6_ibtracs_comparison/`
  (atomic writes; zero-row outputs must fail loudly).
- Before building the artifact, sanity-check against the 20260924 figures: global Cat3+/4+/5
  annual means should be close to what those scripts printed.

## Artifact

Load the `artifact-design` and `dataviz` skills first. Multi-file artifact:
- the page, plus JSON data files split per geography level (lazy-loaded; each a few MB; the
  total is well under the limits);
- a precomputed stats table;
- simplified geometry.

Bundle the data as static files. Don't use the artifact database: the data is fixed output.
Publish privately.

Tabs, with shared dropdowns (start year, level → location, outcome, GCM / run / ensemble /
pooled, mean vs median, window, raw vs detrended):
1. **Overview:** heatmap of Spearman by level × outcome × window, raw beside detrended, mean
   and median. Add a small panel showing how the global headline correlations change across
   start years.
2. **Time series:** IBTrACS vs. the selected model series with its draw band, at the chosen
   window.
3. **Maps:** choropleth of a chosen metric for countries / regions / super-regions; a basin
   view.
4. **Scorecard:** sortable table of all metrics for all places, with effective n.
5. **Mortality:** the same controls, three-way series, 5/10/15-yr windows, adjusted vs
   unadjusted.
6. **Methods & caveats:**
   - definitions and the basin mapping;
   - the phase/forced-trend caveat and effective n;
   - the SSP2-4.5 splice;
   - the v1985 observed-deaths choice and the out-of-sample note;
   - dropped locations;
   - no reanalysis-driven run.

Finish by updating `.claude/STATUS.md` (recent steps + numbers to remember). Don't commit
unless Bobby asks. Report back with the artifact link, the headline findings (where agreement
appears, and whether it survives detrending), and anything that surprised you.
