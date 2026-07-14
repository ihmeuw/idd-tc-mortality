"""
Enumerate (and optionally delete) predict-pipeline outputs invalidated by the
SDI year-boundary fix landed 2026-05-16.

Before the fix, `load_sdi_df` fell through to the future-only branch for
year_bins with year_start in {2020, 2021}, silently dropping rows for those
years from every downstream parquet. Affected outputs need to be deleted and
re-generated.

Affected set, per `storm_draw_table.csv` × `bayespoisson_time_bins_max_bin_5.csv`:
  - Tier 1 (per-tc parquets + basin_means): for every SSP year_bin where
    `year_start <= 2021 AND year_end >= 2022`, all 7 basins.
  - Tier 2 (year_bin_means): the same (sd, sc, yb).
  - Tier 3 (scenario_means): every SSP `scenario_mean` whose year_bin set
    intersects the affected list — in practice every SSP for every finished sd.
  - Tier 4 (storm_draw_means): every finished sd's `storm_draw_mean` (because
    at least one of its SSP scenario_means changed).

Historical-scenario outputs are untouched (their bins all end <= 2014).

Default mode is `--dry-run`: enumerate and print a summary, no deletions.
Pass `--execute` to actually delete. Sample paths are always printed so the
caller can spot-check before running with `--execute`.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

DRAW_BASE          = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/storm_draws')
STORM_DRAW_TABLE   = '/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv'
TIME_BINS          = '/mnt/team/rapidresponse/pub/tropical-storms/tempestextremes/outputs/cmip6/bayespoisson_time_bins_max_bin_5.csv'

SSPS    = ('ssp126', 'ssp245', 'ssp585')
BASINS  = ('AU', 'EP', 'NA', 'NI', 'SI', 'SP', 'WP')
MEAN_FN = 'admin_level_exposure_deaths_mean.parquet'


def find_affected(verbose: bool = False) -> dict[tuple[int, str], list[str]]:
    """Return {(sd, sc): [year_bin_str]} for every finished sd × SSP where the
    bin spans 2020 or 2021."""
    sdt = pd.read_csv(STORM_DRAW_TABLE, keep_default_na=False, na_values=[''])
    tb  = pd.read_csv(TIME_BINS,        keep_default_na=False, na_values=[''])

    finished_sd = sorted({
        int(p.name.split('_')[-1])
        for p in DRAW_BASE.iterdir()
        if p.is_dir() and p.name.startswith('storm_draw_')
        and (p / MEAN_FN).exists()
    })
    if verbose:
        print(f'finished storm_draws on disk: {len(finished_sd)}', file=sys.stderr)

    affected: dict[tuple[int, str], list[str]] = {}
    for sd in finished_sd:
        row = sdt.loc[sdt['storm_draw'] == sd]
        if row.empty:
            continue
        m, v = row['source_id'].iloc[0], row['variant_label'].iloc[0]
        for sc in SSPS:
            bins_mv = (tb[(tb['model'] == m) & (tb['variant'] == v) & (tb['scenario'] == sc)]
                       [['bin_idx', 'start_year', 'end_year']].drop_duplicates())
            bad = bins_mv[(bins_mv['start_year'] <= 2021) & (bins_mv['end_year'] >= 2020)]
            if not bad.empty:
                affected[(sd, sc)] = [
                    f"{int(r.start_year)}-{int(r.end_year)}" for r in bad.itertuples()
                ]
    return affected


def enumerate_paths(affected: dict[tuple[int, str], list[str]]) -> dict[str, list[Path]]:
    """Return {category: [Path]} of files / dirs that the rerun will overwrite.

    Categories:
        per_tc_files       — individual tc-draw parquets (admin_level_exposure_deaths*.parquet)
        basin_means        — admin_level_exposure_deaths_mean.parquet under (sd, sc, yb, basin)
        year_bin_means     — admin_level_exposure_deaths_mean.parquet under (sd, sc, yb)
        scenario_means     — admin_level_exposure_deaths_mean.parquet under (sd, sc)
        storm_draw_means   — admin_level_exposure_deaths_mean.parquet under (sd)
    """
    out: dict[str, list[Path]] = defaultdict(list)

    # Tier 1 outputs (per (sd, sc, yb, basin))
    for (sd, sc), ybs in affected.items():
        for yb in ybs:
            for basin in BASINS:
                bdir = DRAW_BASE / f'storm_draw_{sd}' / sc / yb / basin
                if not bdir.exists():
                    continue
                for f in bdir.iterdir():
                    if not f.is_file() or f.suffix != '.parquet':
                        continue
                    if f.name == MEAN_FN:
                        out['basin_means'].append(f)
                    elif f.name.startswith('admin_level_exposure_deaths'):
                        out['per_tc_files'].append(f)
                    # ignore anything else (no expected extras)

    # Tier 2 (per (sd, sc, yb))
    for (sd, sc), ybs in affected.items():
        for yb in ybs:
            p = DRAW_BASE / f'storm_draw_{sd}' / sc / yb / MEAN_FN
            if p.exists():
                out['year_bin_means'].append(p)

    # Tier 3 (per (sd, sc)) — every SSP scenario_mean for every affected sd.
    distinct_sd_sc = set(affected.keys())
    for (sd, sc) in distinct_sd_sc:
        p = DRAW_BASE / f'storm_draw_{sd}' / sc / MEAN_FN
        if p.exists():
            out['scenario_means'].append(p)

    # Tier 4 (per sd) — every storm_draw_mean of every affected sd.
    distinct_sd = {sd for (sd, _) in affected.keys()}
    for sd in distinct_sd:
        p = DRAW_BASE / f'storm_draw_{sd}' / MEAN_FN
        if p.exists():
            out['storm_draw_means'].append(p)

    return out


def summarize(paths: dict[str, list[Path]]) -> None:
    print('\nDeletion plan:')
    print(f'  {"category":<22} {"count":>10}  example')
    print(f'  {"-" * 22}  {"-" * 9}  {"-" * 60}')
    total = 0
    for cat in ('per_tc_files', 'basin_means', 'year_bin_means',
                'scenario_means', 'storm_draw_means'):
        files = paths.get(cat, [])
        sample = str(files[0]) if files else '(none)'
        print(f'  {cat:<22} {len(files):>10,}  {sample}')
        total += len(files)
    print(f'  {"TOTAL":<22} {total:>10,}')


def execute(paths: dict[str, list[Path]]) -> int:
    n_deleted = 0
    n_failed  = 0
    for cat, files in paths.items():
        for p in files:
            try:
                p.unlink()
                n_deleted += 1
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f'FAILED to delete {p}: {e}', file=sys.stderr)
                n_failed += 1
    print(f'\nDeleted {n_deleted:,} files ({n_failed} failures).')
    return 0 if n_failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split('\n\n')[0])
    ap.add_argument('--execute', action='store_true',
                    help='Actually delete files. Without this flag the script is dry-run only.')
    ap.add_argument('--quiet', action='store_true',
                    help='Skip the progress messages on stderr.')
    args = ap.parse_args(argv)

    affected = find_affected(verbose=not args.quiet)
    n_sd_sc  = len(affected)
    n_sd     = len({sd for (sd, _) in affected})
    n_yb     = sum(len(v) for v in affected.values())

    print(f'\nAffected scope:')
    print(f'  finished sd × SSP : {n_sd_sc}')
    print(f'  distinct sd       : {n_sd}')
    print(f'  affected year_bins: {n_yb}')

    paths = enumerate_paths(affected)
    summarize(paths)

    if args.execute:
        print('\n>>> EXECUTING DELETES <<<')
        return execute(paths)

    print('\n(dry-run; pass --execute to delete)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
