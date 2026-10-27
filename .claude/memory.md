# Session memory
Updated: 2026-10-27 08:59

## Current task
CLOSED (wrap 2026-10-27). The CMIP6 vs IBTrACS comparability report v2 is published
(https://claude.ai/artifact/1gYmPqc3sTud3A6V76wru4, Version 4) and its code is committed
(86f546a) and pushed. The v3-frames re-ship is STAGED on the node
(04-predict/20261026_shipped_v3/direct_risk_staging/2026_10_26_v2000_*), NOT copied to
rapidresponse direct_risk. Nothing running.

## Context / why
Report: do big IBTrACS years/periods line up with big CMIP6 (tc-risk/CLIMADA) periods, by run and
geography. Model series = RAW tc-risk/CLIMADA counts (tracks_per_year is Bobby's TE-calibrated
per-bin frequency; seeds renormalisation rejected 2026-10-26 — do not propose again). Re-ship:
CLIMADA stage4b_v3 frames replaced v2 (13 inland units gone); same model e7a78b19, same ship code.

## Where we are
- Headline (1985 start, global land, ens mean): ρ raw/det 1/5/10-yr 0.15/0.14, 0.15/0.18,
  0.12/0.42, all inside chance bands; full numbers in STATUS "Comparability numbers".
- Forced input change in the re-ship: SDI s160v1/s161v1 deleted upstream → s160v7/s161v7 used
  (v8 pair exists, unevaluated). QC vs 2026_08_20 in 04-predict/20261026_shipped_v3/qc_v3_vs_0820.log.
- Repo is PUBLIC; the module takes paths from predict.blend_historical constants; the notes use
  ${TS}/${ROOT} placeholders. Pre-existing cluster-path literals elsewhere are parked tech debt.

## Next steps
1. Bobby: copy the staged deliverable to direct_risk? confirm SDI v7 vs v8 (rerun is 8 min / 16G if v8).
2. Bobby: raise the six upstream items (notes §"Upstream issues") with rapidresponse.
3. Bobby: parked dirt (STATUS next step 1) — commit, drop, or gitignore.
4. Optional: palette validator (needs node); v8 SDI evaluation.

## Commit groups
- **tracking (wrap 2026-10-27)** — Intent: "Update .claude tracking: 2026-10-27 wrap". Files:
  .claude/{STATUS.md, DECISIONS.md, DEAD_ENDS.md, memory.md}. Status: ready; committed right
  after this write (see git log). Evidence: tracking files only.
- **Parked (pre-existing, Bobby review)** — select/model_selection.py,
  notebooks/20260714_v2000_sensitivity/decompose.ipynb, notebooks/20260722/heavy_tail_simple_and_real.ipynb,
  notebooks/20260804/, notebooks/temp/, test.md, .claude/selection_{tie_diagnosis,protocol_proposal}.md.
  Untouched since before 2026-10-26; provenance unknown; commit/drop/gitignore is Bobby's call.
- Committed this session and gone from the ledger: cmip6-ibtracs-comparison 86f546a; tracking 4f9624c.
- Node-only (never in repo): 04-predict/20261026_shipped_v3/* (ship copy, sbatch, QC, staging).

## Resume prompt
idd-tc-mortality: comparability report v2 published (link above), code committed and pushed
(86f546a); the v3-frames deliverable is staged on the node, not shipped. Read STATUS Orientation
(the lead paragraph plus the two 2026-10-26 paragraphs) and the notes' last three sections. Open
items are Bobby's: direct_risk copy + SDI v7/v8, upstream list, parked dirt.
