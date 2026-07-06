"""Predict-pipeline done-manifest.

Tracks which terminal outputs already exist on disk so the orchestrator can
prune enumeration — only tasks whose terminal output is missing get
registered with jobmon.

Two write paths:

  - `scan_and_write`  (slow path, step-0 mode):
        walk `<draw_base>/storm_draw_*/...` from scratch, write a fresh
        manifest. Run via the `run-build-done-manifest` CLI or as step 0
        of the orchestrator.

  - `append_done`     (fast path, in-task):
        each tier task calls this once its terminal output has been
        atomically written. Appends one JSONL line. Linux guarantees
        atomic O_APPEND writes under PIPE_BUF (4096 bytes), so concurrent
        appends from many tasks don't tear.

Read path:

  - `read_manifest` parses the JSONL into a per-tier set of key tuples
    suitable for filtering enumerated tasks.

Manifest location: `<draw_base>/.done_manifest.jsonl`.

Line schema (one JSON object per line):
  {"tier": "predict_basin",       "storm_draw": N, "scenario": "...", "year_bin": "...", "basin": "..."}
  {"tier": "aggregate_year_bin",  "storm_draw": N, "scenario": "...", "year_bin": "..."}
  {"tier": "aggregate_scenario",  "storm_draw": N, "scenario": "..."}
  {"tier": "aggregate_storm_draw","storm_draw": N}

Duplicates are tolerated (a tier task that re-runs writes a duplicate line);
`read_manifest` returns sets so they collapse. Step 0 rewrites the manifest
from filesystem truth, compacting any duplicates.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click

from idd_tc_mortality.predict.paths import (
    BASIN_LEVELS,
    MEAN_FILE_NAME,
    SCENARIOS,
    STORM_DRAW_DIR,
)

# Default thread-pool size for the filesystem scan. NFS stats are I/O-bound
# (GIL releases on syscall), so 16 threads parallelize the walk well without
# overwhelming the shared filer.
DEFAULT_SCAN_WORKERS = 16

logger = logging.getLogger(__name__)

MANIFEST_NAME = '.done_manifest.jsonl'

TIER_PREDICT_BASIN      = 'predict_basin'
TIER_AGG_YEAR_BIN       = 'aggregate_year_bin'
TIER_AGG_SCENARIO       = 'aggregate_scenario'
TIER_AGG_STORM_DRAW     = 'aggregate_storm_draw'

TIERS = (
    TIER_PREDICT_BASIN,
    TIER_AGG_YEAR_BIN,
    TIER_AGG_SCENARIO,
    TIER_AGG_STORM_DRAW,
)


def manifest_path(draw_base: Path) -> Path:
    return Path(draw_base) / MANIFEST_NAME


# ---------------------------------------------------------------------------
# Scan path — rebuild manifest from filesystem.
# ---------------------------------------------------------------------------

def _scan_one_storm_draw(sd_dir: Path) -> list[dict]:
    """Scan one storm_draw_N subtree and return its done-records.

    Designed to be the unit of parallelism: each call is a contained NFS walk
    that doesn't share state with other workers. Records the existence of:
      - <sd_dir>/<MEAN_FILE_NAME>                              -> aggregate_storm_draw
      - <sd_dir>/<scenario>/<MEAN_FILE_NAME>                   -> aggregate_scenario
      - <sd_dir>/<scenario>/<year_bin>/<MEAN_FILE_NAME>        -> aggregate_year_bin
      - <sd_dir>/<scenario>/<year_bin>/<basin>/<MEAN_FILE_NAME>-> predict_basin
    """
    records: list[dict] = []
    try:
        sd = int(sd_dir.name.split('_')[-1])
    except ValueError:
        return records

    if (sd_dir / MEAN_FILE_NAME).exists():
        records.append({'tier': TIER_AGG_STORM_DRAW, 'storm_draw': sd})

    for sc_dir in sd_dir.iterdir():
        if not sc_dir.is_dir() or sc_dir.name not in SCENARIOS:
            continue
        sc = sc_dir.name

        if (sc_dir / MEAN_FILE_NAME).exists():
            records.append({
                'tier': TIER_AGG_SCENARIO,
                'storm_draw': sd, 'scenario': sc,
            })

        for yb_dir in sc_dir.iterdir():
            if not yb_dir.is_dir():
                continue
            yb = yb_dir.name

            if (yb_dir / MEAN_FILE_NAME).exists():
                records.append({
                    'tier': TIER_AGG_YEAR_BIN,
                    'storm_draw': sd, 'scenario': sc, 'year_bin': yb,
                })

            for basin_dir in yb_dir.iterdir():
                if not basin_dir.is_dir() or basin_dir.name not in BASIN_LEVELS:
                    continue
                basin = basin_dir.name
                if (basin_dir / MEAN_FILE_NAME).exists():
                    records.append({
                        'tier': TIER_PREDICT_BASIN,
                        'storm_draw': sd, 'scenario': sc,
                        'year_bin': yb, 'basin': basin,
                    })

    return records


def scan_filesystem(draw_base: Path, workers: int = DEFAULT_SCAN_WORKERS) -> list[dict]:
    """Walk the storm_draws tree (parallelized per storm_draw), return all done-records.

    NFS stat calls are I/O-bound; the GIL releases on syscall so threads
    parallelize cleanly. Each worker scans one storm_draw_N subtree.
    """
    draw_base = Path(draw_base)
    sd_dirs = sorted(
        p for p in draw_base.iterdir()
        if p.is_dir() and p.name.startswith('storm_draw_')
    )
    if not sd_dirs:
        return []
    n_workers = max(1, min(workers, len(sd_dirs)))
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for sub in ex.map(_scan_one_storm_draw, sd_dirs):
            records.extend(sub)
    return records


def write_manifest(records: list[dict], path: Path) -> None:
    """Atomic write: dump to .tmp then os.replace.

    Truncates and rewrites — call this only from the scan path, not from
    per-task appends.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, path)


def scan_and_write(draw_base: Path, workers: int = DEFAULT_SCAN_WORKERS) -> tuple[int, dict[str, int]]:
    """Scan the storm_draws tree (threaded) and rewrite the manifest. Returns (total, per_tier_counts)."""
    records = scan_filesystem(draw_base, workers=workers)
    counts: dict[str, int] = {t: 0 for t in TIERS}
    for r in records:
        counts[r['tier']] = counts.get(r['tier'], 0) + 1
    write_manifest(records, manifest_path(draw_base))
    return len(records), counts


# ---------------------------------------------------------------------------
# Append path — incremental updates from individual tasks.
# ---------------------------------------------------------------------------

def append_done(draw_base: Path, tier: str, **keys) -> None:
    """Append one record to the manifest. Concurrent-safe for short lines.

    Linux O_APPEND writes ≤ PIPE_BUF (4096 bytes) are atomic; our line size is
    well under 200 bytes. Multiple tier tasks running in parallel can each
    call this without coordination.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    rec = {'tier': tier, **keys}
    line = json.dumps(rec, sort_keys=False) + '\n'
    if len(line.encode('utf-8')) > 4000:
        logger.warning(
            "Manifest append line is unexpectedly large (%d bytes); atomic-append "
            "guarantee no longer applies.", len(line),
        )
    path = manifest_path(draw_base)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(line)


# ---------------------------------------------------------------------------
# Read path — orchestrator consumes this to prune enumeration.
# ---------------------------------------------------------------------------

def read_manifest(draw_base: Path) -> dict[str, set]:
    """Parse the manifest into per-tier sets of identifying tuples.

    Returns:
        {
            'predict_basin':        {(sd, sc, yb, basin), ...},
            'aggregate_year_bin':   {(sd, sc, yb), ...},
            'aggregate_scenario':   {(sd, sc), ...},
            'aggregate_storm_draw': {sd, ...},
        }

    Returns empty sets if the manifest doesn't exist.
    """
    path = manifest_path(draw_base)
    out: dict[str, set] = {t: set() for t in TIERS}
    if not path.exists():
        return out
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed manifest line %d", line_no)
                continue
            tier = r.get('tier')
            try:
                if tier == TIER_PREDICT_BASIN:
                    out[tier].add((r['storm_draw'], r['scenario'], r['year_bin'], r['basin']))
                elif tier == TIER_AGG_YEAR_BIN:
                    out[tier].add((r['storm_draw'], r['scenario'], r['year_bin']))
                elif tier == TIER_AGG_SCENARIO:
                    out[tier].add((r['storm_draw'], r['scenario']))
                elif tier == TIER_AGG_STORM_DRAW:
                    out[tier].add(r['storm_draw'])
                else:
                    logger.warning("Skipping manifest line with unknown tier %r (line %d)",
                                   tier, line_no)
            except KeyError as e:
                logger.warning("Skipping manifest line %d missing key %s: %s",
                               line_no, e, line)
    return out


# ---------------------------------------------------------------------------
# CLI — `run-build-done-manifest`
# ---------------------------------------------------------------------------

@click.command()
@click.option('--draw-base', type=click.Path(path_type=Path),
              default=str(STORM_DRAW_DIR), show_default=True,
              help='Storm-draw output root.')
@click.option('--workers', type=int, default=DEFAULT_SCAN_WORKERS, show_default=True,
              help='Thread-pool size for the per-storm-draw NFS walk.')
def main(draw_base: Path, workers: int) -> None:
    """Scan the storm_draws tree and write a fresh done-manifest.

    Equivalent to running orchestrate's step 0 standalone. Useful after a
    bulk cleanup (when many done-files have been deleted and the manifest
    is stale) or on first use of the manifest (bootstrap).
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    t0 = time.monotonic()
    logger.info('Scanning %s with %d workers ...', draw_base, workers)
    total, counts = scan_and_write(draw_base, workers=workers)
    dt = time.monotonic() - t0
    logger.info('Wrote %d records to %s in %.1fs', total, manifest_path(draw_base), dt)
    for tier in TIERS:
        logger.info('  %s: %d', tier, counts.get(tier, 0))


if __name__ == '__main__':
    main()
