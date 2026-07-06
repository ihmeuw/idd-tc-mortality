"""Per-basin predict task: one (storm_draw, scenario, year_bin, basin).

For one (sd, sc, yb, basin):
  1. Read every admin_level_exposure parquet under that basin folder.
  2. Merge island_cov + SDI, filter to exposed/populated rows.
  3. Apply the 16-cell (c, s, o, b) toggle factorial of the TOPSIS-winner draw
     models for this storm_draw.
  4. Write one parquet per tc_draw with 16 deaths_* columns + pass-through.
  5. Aggregate the per-tc parquets into a single basin-mean parquet (inline).

Each tc_draw and the basin-mean are written via atomic .tmp -> os.replace,
and skipped if the final file is already present. Logs one INFO line per
completed subtask so live progress is visible in the Slurm out-log.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import click
import pandas as pd

from idd_tc_mortality.predict.aggregate_basin import aggregate_basin
from idd_tc_mortality.predict.data_prep import (
    load_island_cov,
    load_sdi_df,
    prep_admin_df,
)
from idd_tc_mortality.predict.models import load_models_for_storm_draw, lookup_model_variant
from idd_tc_mortality.predict.paths import (
    DRAWS_DIR,
    ISLAND_COV_PATH,
    NEW_SDI_PATH,
    OLD_SDI_PATH,
    SEED,
    STAGE4_DIR,
    STORM_DRAW_DIR,
    STORM_DRAW_TABLE_PATH,
    atomic_write_parquet,
    basin_folder,
    basin_mean_path,
    input_admin_path,
    predict_output_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _toggle_slug(c: int, s: int, o: bool, b: bool) -> str:
    return f"c{c}_s{s}_o{int(o)}_b{int(b)}"


def predict_basin(
    *,
    storm_draw: int,
    scenario: str,
    year_bin: str,
    basin: str,
    stage4_dir: Path,
    storm_draw_dir: Path,
    draws_dir: Path,
    storm_draw_table_path: str,
    island_cov_path: str,
    old_sdi_path: str,
    new_sdi_path: str,
    exposure_col: str = 'person_storm_hours',
) -> dict:
    """Run predict for every tc_draw in one basin folder, then write basin-mean.

    Subtask skip: tc_draw outputs and the basin-mean are each skipped if their
    final parquet already exists. The basin-mean is the terminal subtask, so
    if it exists the function returns immediately without re-reading anything.
    """
    t_task = time.monotonic()
    tag = f"sd={storm_draw} {scenario} {year_bin} {basin}"

    mean_path = basin_mean_path(storm_draw_dir, storm_draw, scenario, year_bin, basin)
    if mean_path.exists():
        logger.info("%s: basin-mean already present (%s) — task is a no-op.", tag, mean_path)
        return {"basin_mean_path": str(mean_path), "n_predicted": 0, "n_skipped": 0}

    model, variant = lookup_model_variant(storm_draw, storm_draw_table_path)
    in_basin_dir = stage4_dir / model / variant / scenario / year_bin / basin
    if not in_basin_dir.exists():
        raise FileNotFoundError(
            f"{tag}: basin folder absent in stage4_v2 ({in_basin_dir}). "
            "This task was enumerated from the canonical CSVs + BASIN_LEVELS, so a "
            "missing input folder is a real upstream mismatch, not a benign skip."
        )

    tc_dirs = sorted(
        p for p in in_basin_dir.iterdir()
        if p.is_dir() and p.name.startswith('tc_risk_draw_')
    )
    if not tc_dirs:
        raise FileNotFoundError(
            f"{tag}: no tc_risk_draw_* folders in {in_basin_dir}. "
            "Empty basin folder means climada output is incomplete — fail loudly."
        )

    out_dir = basin_folder(storm_draw_dir, storm_draw, scenario, year_bin, basin)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Recovery from prior partial runs (basin-mean was absent above):
    #   (1) Any leftover .tmp from a killed atomic write is half-written. Drop it.
    #   (2) The latest-mtime per-tc parquet is the only file the previous run
    #       could have killed mid-write (writes are sequential). Re-predicting
    #       it is cheap and avoids reading every file just to verify integrity.
    for tmp in out_dir.glob('*.tmp'):
        logger.info("%s: removing stale tmp %s", tag, tmp.name)
        tmp.unlink()

    existing = [
        p for p in out_dir.glob('admin_level_exposure_deaths*.parquet')
        if p.name != 'admin_level_exposure_deaths_mean.parquet'
    ]
    if existing:
        latest = max(existing, key=lambda p: p.stat().st_mtime)
        logger.info("%s: re-doing latest-mtime per-tc %s (suspect for mid-write kill).",
                    tag, latest.name)
        latest.unlink()

    n_total = len(tc_dirs)
    n_already_done = sum(
        1 for tc_dir in tc_dirs
        if predict_output_path(
            storm_draw_dir, storm_draw, scenario, year_bin, basin,
            int(tc_dir.name.rsplit('_', 1)[1]),
        ).exists()
    )
    logger.info("%s: %d tc_draws total, %d already done, %d to predict.",
                tag, n_total, n_already_done, n_total - n_already_done)

    # Lazy-load the heavy per-task resources only if any tc_draws actually need work.
    # SDI is loaded ONCE for the whole task (all locations, full year range) and
    # merged into each tc_draw's admin_df. Previously this was reloaded per tc_draw
    # and dominated wallclock; loading once cuts ~50s/task on cold cache.
    models_by_cs = None
    island_cov   = None
    sdi_df       = None

    n_predicted = 0
    n_skipped   = n_already_done
    for tc_dir in tc_dirs:
        tc_draw = int(tc_dir.name.rsplit('_', 1)[1])
        out_path = predict_output_path(
            storm_draw_dir, storm_draw, scenario, year_bin, basin, tc_draw,
        )
        if out_path.exists():
            continue

        if models_by_cs is None:
            t_load = time.monotonic()
            models_by_cs = load_models_for_storm_draw(storm_draw, draws_dir)
            island_cov   = load_island_cov(island_cov_path)
            sdi_df       = load_sdi_df(
                year_bin, location_ids=None,
                old_path=old_sdi_path, new_path=new_sdi_path,
            )
            logger.info("%s: per-task resources loaded in %.2fs (models, island_cov, SDI).",
                        tag, time.monotonic() - t_load)

        t_tc = time.monotonic()
        in_path = input_admin_path(
            stage4_dir, model, variant, scenario, year_bin, basin, tc_draw,
        )
        if not in_path.exists():
            raise FileNotFoundError(
                f"{tag} tc={tc_draw}: tc_risk_draw_ folder exists but admin parquet "
                f"is missing ({in_path}). Incomplete climada output — fail loudly."
            )

        admin_df = pd.read_parquet(in_path)
        if admin_df.empty:
            atomic_write_parquet(admin_df.iloc[0:0], out_path)
            n_predicted += 1
            logger.info("%s tc=%d: empty admin -> empty parquet written (%.2fs).",
                        tag, tc_draw, time.monotonic() - t_tc)
            continue

        # AU is a first-class fitted basin (basins_standard ingest), so every
        # basin — including AU — is scored directly with its own coefficient.
        # No IBTrACS lon-split: the folder name is the basin we stamp.
        prepped = prep_admin_df(admin_df, island_cov, sdi_df, basin, exposure_col=exposure_col)
        if prepped.empty:
            empty = pd.DataFrame(columns=[
                'location_id', 'year', 'basin', 'exposed', 'total_population',
            ])
            atomic_write_parquet(empty, out_path)
            n_predicted += 1
            logger.info("%s tc=%d: empty after prep -> empty parquet written (%.2fs).",
                        tag, tc_draw, time.monotonic() - t_tc)
            continue

        for (c, s), model_inst in models_by_cs.items():
            for o in (False, True):
                for b in (False, True):
                    slug = _toggle_slug(c, s, o, b)
                    out  = model_inst.predict(
                        prepped, outcome_draw=o, expected_bernoulli=b, seed=SEED,
                    )
                    prepped[f'deaths_{slug}'] = out['deaths'].values

        deaths_cols = [c for c in prepped.columns if c.startswith('deaths_')]
        save_cols = ['location_id', 'year', 'basin', 'exposed', 'total_population'] + deaths_cols
        atomic_write_parquet(prepped[save_cols], out_path)
        n_predicted += 1
        logger.info("%s tc=%d: predicted + written (%.2fs).",
                    tag, tc_draw, time.monotonic() - t_tc)

    # Basin-mean is the terminal subtask. aggregate_basin uses atomic write and
    # skips if the mean is already present, but we deleted those up front for
    # this workflow so this will always run on first attempt.
    t_mean = time.monotonic()
    mean_path = aggregate_basin(
        storm_draw=storm_draw, scenario=scenario,
        year_bin=year_bin, basin=basin,
        storm_draw_dir=storm_draw_dir,
    )
    logger.info("%s: basin-mean written -> %s (%.2fs). Task wall %.2fs.",
                tag, mean_path, time.monotonic() - t_mean, time.monotonic() - t_task)

    return {
        "basin_mean_path": str(mean_path),
        "n_predicted":     n_predicted,
        "n_skipped":       n_skipped,
    }


@click.command()
@click.option('--storm-draw',         type=int, required=True)
@click.option('--scenario',           type=str, required=True)
@click.option('--year-bin',           type=str, required=True)
@click.option('--basin',              type=str, required=True)
@click.option('--stage4-dir',         type=click.Path(path_type=Path), default=str(STAGE4_DIR), show_default=True)
@click.option('--storm-draw-dir',     type=click.Path(path_type=Path), default=str(STORM_DRAW_DIR), show_default=True)
@click.option('--draws-dir',          type=click.Path(path_type=Path), default=str(DRAWS_DIR), show_default=True)
@click.option('--storm-draw-table',   type=str, default=STORM_DRAW_TABLE_PATH, show_default=True)
@click.option('--island-cov-path',    type=str, default=ISLAND_COV_PATH, show_default=True)
@click.option('--old-sdi-path',       type=str, default=OLD_SDI_PATH, show_default=True)
@click.option('--new-sdi-path',       type=str, default=NEW_SDI_PATH, show_default=True)
@click.option('--exposure-col',       type=str, default='person_storm_hours', show_default=True,
              help="Climada column to use as exposure. Default 'person_storm_hours'. "
                   "Pass 'total_population_exposed' for experimental runs.")
def main(
    storm_draw, scenario, year_bin, basin,
    stage4_dir, storm_draw_dir, draws_dir, storm_draw_table,
    island_cov_path, old_sdi_path, new_sdi_path,
    exposure_col,
):
    predict_basin(
        storm_draw=storm_draw, scenario=scenario, year_bin=year_bin, basin=basin,
        stage4_dir=Path(stage4_dir), storm_draw_dir=Path(storm_draw_dir),
        draws_dir=Path(draws_dir), storm_draw_table_path=storm_draw_table,
        island_cov_path=island_cov_path,
        old_sdi_path=old_sdi_path, new_sdi_path=new_sdi_path,
        exposure_col=exposure_col,
    )


if __name__ == "__main__":
    main()
