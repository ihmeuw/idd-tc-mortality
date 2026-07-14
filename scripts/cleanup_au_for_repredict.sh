#!/usr/bin/env bash
# scripts/cleanup_au_for_repredict.sh
#
# Pre-stage cleanup for the AU re-predict cycle. The fitted DH models have no
# basin_AU coefficient — training data has zero AU rows (IBTrACS folds the AU
# region into SI and SP at the 135°E meridian). The original predict run
# produced AU outputs against a degenerate basin_AU column.
#
# predict_tc.py's AU branch (see predict/data_prep.py:assign_ibtracs_basin_for_au)
# now reads each AU storm's genesis longitude from the climada track .nc and
# reassigns it to SI (lon<135°E) or SP (lon>=135°E) before passing into the
# model. With that branch in place, the AU outputs and every cross-basin
# aggregate above them are stale and must be removed before re-submitting
# the predict orchestrator.
#
# Five deletion categories (in this order):
#   1. <sd>/<sc>/<yb>/AU/                                       (AU subtrees — per-tc + basin-mean)
#   2. <sd>/<sc>/<yb>/admin_level_exposure_deaths_mean.parquet  (year_bin terminals)
#   3. <sd>/<sc>/admin_level_exposure_deaths_mean.parquet       (scenario terminals)
#   4. <sd>/admin_level_exposure_deaths_mean.parquet            (storm_draw terminals)
#   5. <root>/*.parquet                                         (postprocess outputs)
#
# Why all five and not just AU: aggregate_year_bin fans in on all 7 basins,
# so any year_bin terminal that lingers while AU is gone will be read as
# "done" by the manifest scan, and the orchestrator will enumerate AU
# predict_basin but skip the agg_year_bin task that should re-aggregate it.
# Scenario / storm_draw / postprocess terminals cascade from year_bin and
# must also go.
#
# Critical ordering rule: delete EVERYTHING first, THEN re-run the
# orchestrator. The orchestrator's manifest scan must see a consistent
# "AU and everything downstream is missing" view; partial deletion leaves
# stale aggregates that will never be regenerated.
#
# Parallelization (2026-05-19 rewrite, v2):
#   The v1 "5 finds in parallel" attempt took ~50 min for enumeration alone:
#   each of the 5 global finds traverses the entire ~39K-entry tree, so 5
#   concurrent global finds don't reduce per-find work — they just contend
#   for the NFS metadata server. Deletion never finished.
#
#   v2 fans out across storm_draw subdirs (~87 of them). Each parallel worker
#   walks only its own sd's subtree (~450 entries) and runs the 4 sd-scoped
#   finds in sequence (NFS cache reuse). Postprocess outputs are at the root
#   level — handled by a single cheap find.
#
#   Phase 1 — list sd dirs once (instant).
#   Phase 2 — per-sd parallel enumeration via xargs -P (4 finds per worker);
#             concat per-sd lists into category tmp files; root-level
#             postprocess find runs once.
#   Phase 3 — report counts + samples (cheap, reads tmp files).
#   Phase 4 — if --execute: parallel deletion via xargs -P; rm -rf for AU
#             subtrees, rm -f for the 4 *_mean / postprocess categories.
#
# Path safety: predict output paths are integer- and fixed-string-based
# (storm_draw_<int>/<scenario>/<year_bin>/<basin>/tc_draw_<int>/...), so
# newline-delimited transport between find -> tmp file -> xargs is safe.
#
# Default: dry-run (Phase 1+2+3 only). Pass --execute to add Phase 4.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <storm_draw_dir> [--execute] [--workers N]

Pre-stage cleanup for the AU re-predict cycle. By default lists what would
be deleted; pass --execute to actually delete.

Options:
  --execute      Actually delete (default: dry-run).
  --workers N    Parallel worker count for find + rm. Default: 16.
                 IHME NFS metadata typically saturates 16-32 concurrent ops;
                 higher values can cause contention rather than speedup.
                 Bump to 32 if 16 still feels slow on a less-loaded cluster.

Example:
  $0 /mnt/team/idd/pub/idd_tc_mortality/04-predict/best_topsis_post2000_v2/storm_draws
  $0 /mnt/team/idd/pub/idd_tc_mortality/04-predict/best_topsis_post2000_v2/storm_draws --execute
  $0 /mnt/team/idd/pub/idd_tc_mortality/04-predict/best_topsis_post2000_v2/storm_draws --execute --workers 32
EOF
}

ROOT=""
MODE="dry-run"
WORKERS=16

while (($#)); do
    case "$1" in
        --execute) MODE="execute"; shift ;;
        --workers) WORKERS="${2:?--workers needs a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "ERROR: unknown flag: $1" >&2; usage >&2; exit 2 ;;
        *) ROOT="$1"; shift ;;
    esac
done

if [[ -z "$ROOT" ]]; then
    echo "ERROR: storm_draw_dir is required" >&2
    usage >&2
    exit 1
fi
if [[ ! -d "$ROOT" ]]; then
    echo "ERROR: not a directory: $ROOT" >&2
    exit 1
fi
if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || (( WORKERS < 1 )); then
    echo "ERROR: --workers must be a positive integer (got: $WORKERS)" >&2
    exit 2
fi

echo "Cleanup root : $ROOT"
echo "Mode         : $MODE"
echo "Workers      : $WORKERS"
echo

# ---------------------------------------------------------------------------
# Phase 1 — list storm_draw subdirs (instant, ~87 entries).
# ---------------------------------------------------------------------------
mapfile -t SD_DIRS < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
if (( ${#SD_DIRS[@]} == 0 )); then
    echo "ERROR: no storm_draw subdirs under $ROOT — is this the right path?" >&2
    exit 1
fi
echo "Found ${#SD_DIRS[@]} storm_draw subdirs"
echo

# ---------------------------------------------------------------------------
# Phase 2 — per-sd parallel enumeration. Each worker walks its sd subtree
# once and runs 4 finds against it (NFS cache reuse across the four). The
# root-level postprocess find runs once after the per-sd batch completes.
# ---------------------------------------------------------------------------
TMPDIR="$(mktemp -d -t au_cleanup.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT
mkdir -p \
    "$TMPDIR/parts/au_subtrees" \
    "$TMPDIR/parts/year_bin_means" \
    "$TMPDIR/parts/scenario_means" \
    "$TMPDIR/parts/storm_draw_means"

find_one_sd() {
    local sd="$1"
    local name="${sd##*/}"  # basename — fast, no fork
    # Depth indices are relative to $sd (one level below $ROOT):
    #   <sd>/<sc>/<yb>/AU/                                       -> depth 3
    #   <sd>/<sc>/<yb>/admin_level_exposure_deaths_mean.parquet  -> depth 3
    #   <sd>/<sc>/admin_level_exposure_deaths_mean.parquet       -> depth 2
    #   <sd>/admin_level_exposure_deaths_mean.parquet            -> depth 1
    find "$sd" -mindepth 3 -maxdepth 3 -type d -name 'AU'                                       2>/dev/null >"$TMPDIR/parts/au_subtrees/$name"
    find "$sd" -mindepth 3 -maxdepth 3 -type f -name 'admin_level_exposure_deaths_mean.parquet' 2>/dev/null >"$TMPDIR/parts/year_bin_means/$name"
    find "$sd" -mindepth 2 -maxdepth 2 -type f -name 'admin_level_exposure_deaths_mean.parquet' 2>/dev/null >"$TMPDIR/parts/scenario_means/$name"
    find "$sd" -mindepth 1 -maxdepth 1 -type f -name 'admin_level_exposure_deaths_mean.parquet' 2>/dev/null >"$TMPDIR/parts/storm_draw_means/$name"
}
export -f find_one_sd
export TMPDIR

t0=$(date +%s)
echo "Phase 2: parallel enumeration (workers=$WORKERS) ..."
printf '%s\n' "${SD_DIRS[@]}" | xargs -P "$WORKERS" -n 1 bash -c 'find_one_sd "$1"' _

# Concat per-sd parts into category files. Each sd produced a file (possibly
# empty), so the glob always has matches.
cat "$TMPDIR/parts/au_subtrees/"*      > "$TMPDIR/au_subtrees"
cat "$TMPDIR/parts/year_bin_means/"*   > "$TMPDIR/year_bin_means"
cat "$TMPDIR/parts/scenario_means/"*   > "$TMPDIR/scenario_means"
cat "$TMPDIR/parts/storm_draw_means/"* > "$TMPDIR/storm_draw_means"

# Postprocess outputs live directly under $ROOT — single cheap find, no
# parallelism needed.
find "$ROOT" -mindepth 1 -maxdepth 1 -type f -name '*.parquet' 2>/dev/null >"$TMPDIR/postprocess_outputs"

echo "  Enumeration time: $(($(date +%s) - t0))s"
echo

# ---------------------------------------------------------------------------
# Phase 3 — report counts + samples from tmp files.
# ---------------------------------------------------------------------------
report_from_file() {
    local label="$1"
    local file="$2"
    local n
    n=$(wc -l <"$file")
    printf "%-22s : %d\n" "$label" "$n"
    if (( n > 0 )); then
        awk 'NR<=3 {print "    " $0}' "$file"
        (( n > 3 )) && printf '    ... and %d more\n' "$((n - 3))"
    fi
}

report_from_file "au_subtrees"         "$TMPDIR/au_subtrees"
report_from_file "year_bin_means"      "$TMPDIR/year_bin_means"
report_from_file "scenario_means"      "$TMPDIR/scenario_means"
report_from_file "storm_draw_means"    "$TMPDIR/storm_draw_means"
report_from_file "postprocess_outputs" "$TMPDIR/postprocess_outputs"

echo
if [[ "$MODE" != "execute" ]]; then
    echo "Dry run — no files deleted. Re-run with --execute to apply."
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase 4 — parallel deletion. Each category's tmp file feeds `xargs -P
# WORKERS -n BATCH`, spawning up to WORKERS concurrent rm processes with
# BATCH paths per invocation. -r skips entirely if a category is empty.
#
# au_subtrees use `rm -rf` (each AU dir holds ~100 per-tc parquets); smaller
# BATCH keeps individual rm wall bounded so the worker pool stays balanced.
# *_means and postprocess_outputs are single files: `rm -f` with larger BATCH.
#
# Sequential within phase 4 for observability — total wall is dominated by
# au_subtrees anyway (~95% of unlinks).
# ---------------------------------------------------------------------------
echo "Phase 4: parallel deletion (workers=$WORKERS) ..."
t1=$(date +%s)

t_cat=$(date +%s)
xargs -a "$TMPDIR/au_subtrees" -r -P "$WORKERS" -n 50 rm -rf
echo "  done: au_subtrees         ($(($(date +%s) - t_cat))s)"

t_cat=$(date +%s)
xargs -a "$TMPDIR/year_bin_means" -r -P "$WORKERS" -n 200 rm -f
echo "  done: year_bin_means      ($(($(date +%s) - t_cat))s)"

t_cat=$(date +%s)
xargs -a "$TMPDIR/scenario_means" -r -P "$WORKERS" -n 200 rm -f
echo "  done: scenario_means      ($(($(date +%s) - t_cat))s)"

t_cat=$(date +%s)
xargs -a "$TMPDIR/storm_draw_means" -r -P "$WORKERS" -n 200 rm -f
echo "  done: storm_draw_means    ($(($(date +%s) - t_cat))s)"

t_cat=$(date +%s)
xargs -a "$TMPDIR/postprocess_outputs" -r -P "$WORKERS" -n 200 rm -f
echo "  done: postprocess_outputs ($(($(date +%s) - t_cat))s)"

echo
echo "Phase 4 total: $(($(date +%s) - t1))s"
echo "All five categories cleared. Next: re-run run-predict-orchestrate"
echo "against $ROOT to enumerate the re-do tasks."
