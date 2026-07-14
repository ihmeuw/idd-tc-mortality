# Session memory
Updated: 2026-07-14

## Current task
Version-control groundwork for TWO upcoming full pipeline reruns (new input data each; preliminary → intermediate → refined → final diagnostics per cycle). Steps 0–2 + 3.1 of the agreed plan are DONE; next is discussing (a) the vintages to run and (b) the notebook reorganization.

## Context / why
Bobby will rerun the entire pipeline twice with changed input data. Before that: lock the git baseline (everything was uncommitted/unpushed), adopt an explicit vintage contract, and remove the rerun hazards (hardcoded ingest SOURCE_CSV, build_draws stale folds default).

## Where we are
- **Baseline committed and tagged.** 6 commits on main on top of eba4c43: 71afb17 (evaluate/fit/metrics/viz + tests), 3894c64 (rake/postprocess + tests + build_sr_version_files), 719c414 (cycle scripts + 4 live diagnostics notebooks + 20260707 notebook + reports + README), 6f04904 (jobmon 10.12.2 pin), 19c0fc7 (.claude records + vintage-contract DECISIONS entry), 3ab43f9 (step-2 changes). Annotated tag **`cycle-20260608`** on 19c0fc7. All pushed? → see Next steps (push happens at end of this session's work).
- **Vintage contract** documented in DECISIONS.md 2026-07-14 entry: one `<V>`=YYYYMMDD-of-ingest string used verbatim across `00-data/<V>`, `01-preliminary/<V>`, `02-evaluate/<V>_{refit,survivors,refined,final}`, `01-refined/<V>`, `03-draws/<V>/<mid>`, `04-predict/<V>/<mid>_*`; notebooks repoint via global find/replace of the vintage string; cycle closes with tag `cycle-<V>`.
- **Step-2 changes:** ingest `01_prepare_input_data.py` gained `--source-csv`/`--vintage` (write_versioned takes vintage; 3/3 new tests in tests/ingest/); `build_draws.py --folds-path` now required (stale `20260608_final` default removed).
- **Test suite: 708/719 pass.** 11 pre-existing failures: 9 = stale tests/fit/test_run_component.py (documented parking-lot item), +2 NOT previously documented: `tests/step_00_validate/test_validate.py::test_check_basins_empty_noncanonical_when_all_clean` and `tests/viz/test_stage_plots.py::test_vet_stage_3x3_layout`. Not diagnosed (out of scope today).
- **Deliberately NOT committed** (pending notebook discussion): `notebooks/20260515/` (40M, incl. wtf.ipynb + a 10M .bak), `notebooks/20260623/`, `notebooks/archive/`, `notebooks/jobmon_resources/`, `basin_comparison{,_global}.pdf`. `.gitignore` now ignores `*.bak.*`.
- Prior open item unchanged: SR-31 guard + mean-vs-median rake decision for canonical deliverable (5-version comparison shipped 2026-07-13).

## Next steps
1. Push main + `cycle-20260608` tag to origin (if not yet done this session).
2. Discuss the two rerun vintages (source CSVs, naming, sequential vs interleaved).
3. Notebook reorganization discussion ("untenable"): what to do with notebooks/20260515 (40M of forks), 20260623, archive, jobmon_resources; repoint-in-place + tag vs cp into notebooks/<V>/ at cycle close; consolidate preliminary notebook's two config cells; final notebook's hardcoded winner mids + stale prose.
4. Then: ingest cycle-1 data (`--source-csv <new.csv> --vintage <V>` + immediately `02_prepare_is_island.py --vintage <V>`).
5. Carried: SR-31/rake-stat canonical decision; delete dead fit stage; stale test_run_component tests; 2 newly-surfaced test failures (validate, stage_plots).

## Resume prompt
Session 2026-07-14 (idd-tc-mortality, main): Locked the git baseline for two upcoming pipeline reruns. Committed all outstanding work as 6 logical commits, tagged `cycle-20260608` (code+notebook state of the completed cycle), documented the vintage contract in DECISIONS.md (one <V> string across all stage dirs, suffixes _refit/_survivors/_refined/_final kept verbatim so notebook repointing = find/replace). Ingest script now takes --source-csv/--vintage (tests 3/3); build_draws --folds-path is required. Suite 708/719 (9 known-stale run_component + 2 undiagnosed: validate basins test, stage_plots 3x3 test). notebooks/{20260515,20260623,archive,jobmon_resources} and the two basin PDFs intentionally left uncommitted pending the notebook-reorg discussion. Next: pick the two rerun vintages and settle the notebook situation.
