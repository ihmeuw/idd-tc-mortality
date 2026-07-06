"""One-shot cleanup: remove stale cross-basin aggregates before re-running.

The basin-list bug ([paths.py BASIN_LEVELS] was missing AU and had the
non-existent SA) corrupted every aggregate that fans across basins:

  - year_bin_mean: <storm_draw>/<scenario>/<year_bin>/admin_level_exposure_deaths_mean.parquet
  - scenario_mean: <storm_draw>/<scenario>/admin_level_exposure_deaths_mean.parquet
  - storm_draw_mean: <storm_draw>/admin_level_exposure_deaths_mean.parquet

Per-tc-draw parquets and per-basin basin-means for the 6 already-predicted
basins (EP, NA, NI, SI, SP, WP) are NOT removed — they are correct, and the
predict_basin skip path uses them. AU has no output yet (it was never run).
Any SA subfolders that exist (artifact of failed iteration) are removed.

Defaults to dry-run: lists what would be deleted, deletes nothing. Pass
--execute to actually delete.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import click

from idd_tc_mortality.predict.paths import STORM_DRAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _collect_targets(storm_draw_dir: Path) -> dict[str, list[Path]]:
    """Walk the storm_draw output tree and bucket the deletion targets."""
    targets: dict[str, list[Path]] = {
        "storm_draw_mean": [],
        "scenario_mean":   [],
        "year_bin_mean":   [],
        "sa_folders":      [],
    }
    if not storm_draw_dir.exists():
        return targets

    for sd_dir in sorted(p for p in storm_draw_dir.iterdir()
                         if p.is_dir() and p.name.startswith('storm_draw_')):
        sd_mean = sd_dir / 'admin_level_exposure_deaths_mean.parquet'
        if sd_mean.exists():
            targets["storm_draw_mean"].append(sd_mean)

        for sc_dir in (p for p in sd_dir.iterdir() if p.is_dir()):
            sc_mean = sc_dir / 'admin_level_exposure_deaths_mean.parquet'
            if sc_mean.exists():
                targets["scenario_mean"].append(sc_mean)

            for yb_dir in (p for p in sc_dir.iterdir() if p.is_dir()):
                yb_mean = yb_dir / 'admin_level_exposure_deaths_mean.parquet'
                if yb_mean.exists():
                    targets["year_bin_mean"].append(yb_mean)

                sa_dir = yb_dir / 'SA'
                if sa_dir.exists() and sa_dir.is_dir():
                    targets["sa_folders"].append(sa_dir)

    return targets


@click.command()
@click.option('--storm-draw-dir', type=click.Path(path_type=Path),
              default=str(STORM_DRAW_DIR), show_default=True)
@click.option('--execute', is_flag=True,
              help="Actually delete. Default is dry-run (list only).")
def main(storm_draw_dir, execute):
    storm_draw_dir = Path(storm_draw_dir)
    targets = _collect_targets(storm_draw_dir)

    counts = {k: len(v) for k, v in targets.items()}
    logger.info("Deletion targets under %s:", storm_draw_dir)
    for k, n in counts.items():
        logger.info("  %-18s : %d", k, n)
    total = sum(counts.values())
    logger.info("  ------------------")
    logger.info("  TOTAL              : %d", total)

    # Show up to 3 samples per category for sanity-checking.
    for k, paths in targets.items():
        if not paths:
            continue
        logger.info("Sample %s:", k)
        for p in paths[:3]:
            logger.info("  %s", p)
        if len(paths) > 3:
            logger.info("  ... and %d more", len(paths) - 3)

    if not execute:
        logger.info("Dry run — no files deleted. Re-run with --execute to apply.")
        return

    # Actual deletion.
    for k, paths in targets.items():
        for p in paths:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        logger.info("Deleted %d %s entries.", len(paths), k)
    logger.info("Done.")


if __name__ == "__main__":
    main()
