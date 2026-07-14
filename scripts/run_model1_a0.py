"""
Run model 1 (best topsis) through the consolidated A0 predict + rollup pipeline.

Test:  python scripts/run_model1_a0.py --max-storm-draws 3 --out-path <p>
Full:  python scripts/run_model1_a0.py --out-path <p>
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
IS_ISLAND = str(ROOT / "00-data" / "current" / "is_island.parquet")
OLD_SDI = "/mnt/share/forecasting/data/16/past/sdi/past_sdi_s130v66/sdi.nc"
NEW_SDI = "/mnt/share/forecasting/data/32/future/sdi/future_sdi_s130v66/sdi.nc"
DATA = str(ROOT / "00-data" / "current" / "input.parquet")
FOLDS = str(ROOT / "02-evaluate" / "20260608_final" / "fold_assignments.parquet")
FOCUS = ROOT / "03-draws" / "20260608" / "6004b730b8d8f96cb6d8443378b04f09" / "focus_model.json"


@click.command()
@click.option("--max-storm-draws", type=int, default=None,
              help="Cap storm draws (for a fast test). Default: all.")
@click.option("--out-path", required=True, type=click.Path(path_type=Path))
def main(max_storm_draws, out_path):
    t0 = time.time()
    focus = json.loads(FOCUS.read_text())
    data = pd.read_parquet(DATA)
    folds = pd.read_parquet(FOLDS)
    print(f"[refit] model 1 on {len(data):,} rows ...")
    refit_out = refit_model_with_objects(focus, data, folds, n_seeds=1, n_folds=2)

    summary = run_a0(
        refit_out=refit_out, focus=focus, data=data,
        consolidated_path=A0_PATH, storm_draw_table_path=STORM_DRAW_TABLE,
        is_island_path=IS_ISLAND, old_sdi_path=OLD_SDI, new_sdi_path=NEW_SDI,
        hierarchy_path=DEFAULT_HIERARCHY_PATH,
        max_storm_draws=max_storm_draws,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out_path, index=False)
    print(f"[done] {time.time()-t0:.1f}s -> {out_path} ({len(summary):,} rows)")

    glob = summary[summary["location_id"] == 1]
    if not glob.empty:
        mc = glob[glob["cell"] == "deaths_c0_s0_o0_b1"]
        print("\nGlobal mean cell (deaths_c0_s0_o0_b1), first 8 (experiment,year):")
        print(mc.sort_values(["experiment_id", "year"]).head(8)[
            ["experiment_id", "year", "mean", "lower", "upper"]].to_string(index=False))


if __name__ == "__main__":
    main()
