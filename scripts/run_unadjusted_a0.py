"""
Run ANY focus model through the consolidated A0 predict + rollup pipeline to
produce UNADJUSTED predictions (all 16 (c,s,o,b) toggle cells, mean/lower/upper
across storm draws). Generalizes run_model1_a0.py: focus/data/folds/out are args.

In-process (refit -> run_a0 -> rollup); no jobmon, no draws-pickle stage.

    python scripts/run_unadjusted_a0.py \\
        --focus-model-json 03-draws/<vintage>/<mid>/focus_model.json \\
        --data-path 00-data/<vintage>/input.parquet \\
        --folds-path 02-evaluate/<vintage>_final/fold_assignments.parquet \\
        --out-path 04-predict/<vintage>/<mid>_a0_unadj/summary.parquet \\
        [--max-storm-draws N]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import click
import pandas as pd

from idd_tc_mortality.predict.consolidated import run_a0
from idd_tc_mortality.predict.postprocess import DEFAULT_HIERARCHY_PATH
from idd_tc_mortality.refit_with_objects import refit_model_with_objects

ROOT = Path("/mnt/team/idd/pub/idd_tc_mortality")
A0_PATH = "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4b_v2/_consolidated/storm_exposure_all.parquet"
STORM_DRAW_TABLE = "/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv"
OLD_SDI = "/mnt/share/forecasting/data/16/past/sdi/past_sdi_s130v66/sdi.nc"
NEW_SDI = "/mnt/share/forecasting/data/32/future/sdi/future_sdi_s130v66/sdi.nc"


@click.command()
@click.option("--focus-model-json", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--data-path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--folds-path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--is-island-path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Vintage is_island.parquet (00-data/<vintage>/is_island.parquet).")
@click.option("--out-path", required=True, type=click.Path(path_type=Path))
@click.option("--max-storm-draws", type=int, default=None,
              help="Cap storm draws (fast test). Default: all 100.")
def main(focus_model_json, data_path, folds_path, is_island_path, out_path, max_storm_draws):
    t0 = time.time()
    focus = json.loads(Path(focus_model_json).read_text())
    data = pd.read_parquet(data_path)
    folds = pd.read_parquet(folds_path)
    print(f"[refit] {focus_model_json} on {len(data):,} rows "
          f"(thr={focus['threshold_quantile']} tail={focus['tail_family']}/{focus['tail_exposure_mode']}) ...")
    refit_out = refit_model_with_objects(focus, data, folds, n_seeds=1, n_folds=2)

    summary = run_a0(
        refit_out=refit_out, focus=focus, data=data,
        consolidated_path=A0_PATH, storm_draw_table_path=STORM_DRAW_TABLE,
        is_island_path=str(is_island_path), old_sdi_path=OLD_SDI, new_sdi_path=NEW_SDI,
        hierarchy_path=DEFAULT_HIERARCHY_PATH,
        max_storm_draws=max_storm_draws,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out_path, index=False)
    print(f"[done] {time.time()-t0:.1f}s -> {out_path} ({len(summary):,} rows, "
          f"{summary['cell'].nunique()} toggle cells)")


if __name__ == "__main__":
    main()
