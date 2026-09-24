# Session memory
Updated: 2026-09-23 16:59 (wrap)

## Current task
Session closed by /wrap. Figure tooling for the shipped gpd_cens e7a78b19 model is built,
tested and committed (10f88ba..76b9dc7 + tracking commit). Nothing in flight.

## Context / why
Selection is done (trio, DECISIONS 2026-08-04) and gpd_cens e7a78b19 shipped 2026-08-20 as the
A1/A0 blend (`direct_risk/2026_08_20_v2000_*`). Bobby is making slides/figures of the final model
and wants everything drawn from the shipped blend, on idd-figures, editable via notebooks.

## Where we are
- Deps: idd-figures[maps], geopandas, rasterio in; idd-* sources = git main (idd-tools too).
  Repo CLAUDE.md still says idd-tools is an editable path dep — stale, update it.
- Figures (all under `/mnt/team/idd/pub/idd_tc_mortality/05-figures/working/`):
  `shipped_e7a78b19/` (global_2x2, sr_grid, cumulative_boxes × {adjusted_mean, unadjusted};
  historical_blend_*.parquet), `stage_vetting_e7a78b19/` (s1/s2/bulk/tail default + by-basin +
  bulk_pdf_overlay + tail_pdf), `2005236N23285_{resp_three_panel,exposure_two_panel}.png`,
  `basin_definitions/basin_comparison_global.{png,pdf}`.
- Notebooks (committed, 15b9048): `notebooks/20260923/{katrina_three_panel, dh_stage_vetting_
  final_gpd_cens, basin_map_knobs}.ipynb` — Bobby saved katrina/vetting with outputs (3.3 / 1.7 MB).
- Known pre-existing failure: `tests/viz/test_stage_plots.py::test_vet_stage_3x3_layout` (9 vs 8).
- ERA5 question answered: not used in this repo; climate-data reference 2019–2023; tc_risk forcing 2016–2021.

## Next steps
1. Bobby decides the parked dirt (see ledger) — commit / drop / gitignore.
2. Bobby deletes or flags the superseded `direct_risk/2026_08_26_v2000_*`.
3. Fix `test_vet_stage_3x3_layout`; update repo CLAUDE.md (idd-tools source form).
4. Decide whether to copy the node-only `_*.py` forecast/ship scripts into the repo.
5. Carried: upstream SDI question; v1985 selection clone; idd-tools inbox push.

## Commit groups
Committed this wrap (left the ledger): 10f88ba deps + figure tooling; a01cebf StagePlotter
gpd_cens; 1ae3ce5 DrawModel tail variants; 15b9048 20260923 notebooks; 76b9dc7 20260722 selection
notebooks; tracking files committed in the wrap's final commit (see `git log -1`). Pushed to origin/main.
- **Parked — Bobby review (pre-existing dirt, do NOT commit blind)**:
  `src/idd_tc_mortality/select/model_selection.py`,
  `notebooks/20260714_v2000_sensitivity/decompose.ipynb`,
  `notebooks/20260722/heavy_tail_simple_and_real.ipynb` (modified; +29/−53 lines, provenance
  unknown). Status: parked. How close: Bobby must say what these edits are. Evidence: none.
- **20260804 notebooks**: `notebooks/20260804/` (forecast_comparison.ipynb 16 KB fine;
  decompose.ipynb 9.5 MB with outputs, unmet sensitivity-run prerequisite). Intent: "Add 20260804
  forecast comparison + parameterized decompose notebooks". Status: in-progress. Evidence: none.
- **Selection docs**: `.claude/selection_tie_diagnosis.md` (keep), `.claude/selection_protocol_
  proposal.md` (partly superseded by DECISIONS 2026-08-04). Status: blocked on Bobby (reconcile or drop).
- **Temp / never commit (unless Bobby says)**: `notebooks/temp/2026_09_23_katrina_plots_specific.ipynb`
  (Bobby's CLIMADA scratch, needs mfiking's env), `test.md` (Bobby's gpd_cens math notes at repo root —
  move or drop).

## Resume prompt
idd-tc-mortality: shipped model = gpd_cens e7a78b19, A1/A0 blend, `direct_risk/2026_08_20_v2000_*`.
Figure tooling is committed: CLIs run-blend-historical / run-shipped-figures / run-storm-three-panel /
run-basin-map, notebooks in notebooks/20260923/, outputs in 05-figures/working/. Read STATUS.md
Orientation, then: parked dirt decisions, superseded 08_26 files, the 3x3 test, repo CLAUDE.md
idd-tools wording. Never draw final-model figures from the A0 SR draw file.
