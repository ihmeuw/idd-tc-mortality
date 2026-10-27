# Session memory
Updated: 2026-10-26 18:00

## Current task
DONE this session: (1) frequency investigation a–d (notes §16:28); (2) v3-frames deliverable
recreated and STAGED in 04-predict/20261026_shipped_v3/direct_risk_staging (NOT copied to
rapidresponse direct_risk — Bobby's call; SDI inputs changed to s160v7/s161v7 because v1 files
were deleted); (3) comparability report v2 (v3 frames + spec v2: start years, no blocks,
chance bands, ceilings, Bretherton n_eff) republished to the SAME URL
https://claude.ai/artifact/1gYmPqc3sTud3A6V76wru4 (Version 4). STATUS, notes, spec updated.
Nothing running. Nothing committed.

## Context / why
Raw tracks_per_year counts are Bobby's TE-calibrated frequency (constant within change-point
bins) and stay the primary model series — do NOT propose seeds-based renormalisation again.

## Where we are
- Code: viz/cmip6_ibtracs_comparison/* rewritten (stats/export/page/constants; predict --rake-json);
  49/49 tests. Outputs in 05-figures/working/cmip6_ibtracs_comparison/ (stats/ incl. ceiling.parquet;
  artifact/ 95 files 119 MB). Publish mirror ~/.claude/artifacts/idd-tc-mortality/cmip6_ibtracs_comparison/
  (publish in ≤64 MB parts; last done as 3 publishes).
- Headline numbers: STATUS "Comparability numbers (v2 report)".

## Next steps
1. Bobby: copy staged deliverable to direct_risk? (confirm SDI v7 vs v8); commit module/tests;
   raise the six upstream items (notes §"Upstream issues") with rapidresponse.
2. Optional: palette validator needs node; v8 SDI evaluation.

## Commit groups
- **cmip6-ibtracs-comparison** — Intent: "Add CMIP6 vs IBTrACS comparability pipeline and artifact
  page". Files: src/idd_tc_mortality/viz/cmip6_ibtracs_comparison/*, tests/viz/cmip6_ibtracs_comparison/*,
  pyproject.toml. Status: ready (Bobby review). Evidence: pytest 49/49; Slurm runs rc=0; published.
- **tracking** — .claude/{STATUS.md, memory.md, cmip6_ibtracs_comparison_prompt.md,
  cmip6_ibtracs_comparison_notes.md}. Status: ready.
- **Parked (pre-existing, Bobby review)** — select/model_selection.py, two 2026-07 notebooks,
  notebooks/20260804/, notebooks/temp/, test.md, .claude/selection_*.md (unchanged).
- Node-only (never in repo): 04-predict/20261026_shipped_v3/* (ship copy, sbatch, QC, staging).

## Resume prompt
idd-tc-mortality: comparability report v2 is published (link above); the v3-frames deliverable
is staged on the node, not shipped. Read STATUS Orientation (two 2026-10-26 paragraphs) and the
notes' last three sections. Open decisions are Bobby's: ship copy, commit, upstream items.
