"""
Run the driver-of-change sensitivities for one death model (default: the v2000
best, 6004b730) on the v89 SDI baseline.

For each mode in {sdi_const, pop_const, both} it:
  1. submits the consolidated predict for the A0 and A1 frames with
     ``--sensitivity <mode>`` (freeze SDI and/or scale exposure from --anchor-year),
  2. builds the A1/A0 blend and writes the draw-level deliverable for the
     1,1,0,1 cell in two flavours: unadjusted and mean-raked (+ its sr31-guard
     variant). The rake ratios recompute to the baseline values because the
     reference window (<= anchor) is untouched by a post-anchor freeze.

Outputs go to an analysis dir (NOT direct_risk):
    <out-root>/<mode>/<mid>_{a0,a1}/            predict partials + summaries
    <out-root>/deliverables/<prefix>_<mode>_*   draw-level files

The unmodified baseline is 04-predict/20260706/<mid>_blend (already delivered).
Run each mode's predict is jobmon; this script blocks per submission.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from idd_tc_mortality.sensitivity import ANCHOR_YEAR

ROOT = Path("/mnt/team/idd/pub/idd_tc_mortality")
A0 = "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4b_v2/_consolidated/storm_exposure_all.parquet"
A1 = "/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage4b_v2_admin1/_consolidated/storm_exposure_all.parquet"
HIER = "/mnt/team/idd/pub/forecast-mbp/01-raw_data/gbd/fhs_2023_modeling_hierarchy.parquet"
# v89 SDI (data/32 past + future) — passed explicitly; the consolidated pipeline ignores paths.py
OLD_SDI = "/mnt/share/forecasting/data/32/past/sdi/past_sdi_s130v89/sdi.nc"
NEW_SDI = "/mnt/share/forecasting/data/32/future/sdi/future_sdi_s130v89/sdi.nc"
POP_PAST = "/mnt/share/forecasting/data/16/past/population/20250603_etl_run_id_417/population.nc"
POP_FUTURE = "/mnt/share/forecasting/data/32/future/population/future_population_s130v41/summary/summary.nc"


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _predict_cmd(*, cells_file, focus_model, cons, out_dir, data_path, folds_path,
                 is_island_path, mode, anchor_year, memory, runtime, mid) -> list[str]:
    return [
        "run-predict-consolidated",
        "--cells-file", cells_file, "--focus-model", focus_model,
        "--consolidated-path", cons, "--out-dir", str(out_dir),
        "--data-path", data_path, "--folds-path", folds_path,
        "--hierarchy-path", HIER, "--is-island-path", is_island_path,
        "--old-sdi-path", OLD_SDI, "--new-sdi-path", NEW_SDI,
        "--sensitivity", mode, "--anchor-year", str(anchor_year),
        "--pop-past-path", POP_PAST, "--pop-future-path", POP_FUTURE,
        "--no-probe", "--memory", memory, "--runtime", runtime,
        "--workflow-name", f"predict-sens-{mode}-{mid[:8]}",
    ]


@click.command()
@click.option("--mid", default="6004b730b8d8f96cb6d8443378b04f09", show_default=True)
@click.option("--modes", default="sdi_const,pop_const,both", show_default=True,
              help="Comma list of sensitivity modes to run.")
@click.option("--out-root", default=str(ROOT / "04-predict" / "20260721_v2000_sensitivity"),
              show_default=True, help="Analysis output root (NOT direct_risk).")
@click.option("--cells-file", default=str(ROOT / "04-predict" / "20260714" / "predict_cells_5sd.json"),
              show_default=True)
@click.option("--focus-model", default=None,
              help="Default: 03-draws/20260608/<mid>/focus_model.json")
@click.option("--data-path", default=str(ROOT / "00-data" / "20260608" / "input.parquet"), show_default=True)
@click.option("--folds-path", default=str(ROOT / "02-evaluate" / "20260608_final" / "fold_assignments.parquet"),
              show_default=True)
@click.option("--is-island-path", default=str(ROOT / "00-data" / "20260608" / "is_island.parquet"),
              show_default=True)
@click.option("--anchor-year", default=ANCHOR_YEAR, show_default=True, type=int)
@click.option("--memory", default="2G", show_default=True)
@click.option("--runtime", default="12m", show_default=True)
@click.option("--prefix", default="2026_07_21_v2000", show_default=True,
              help="Deliverable filename prefix (mode is appended).")
@click.option("--skip-predict", is_flag=True, default=False,
              help="Skip the predict submissions and only (re)build deliverables from existing partials.")
def main(mid, modes, out_root, cells_file, focus_model, data_path, folds_path,
         is_island_path, anchor_year, memory, runtime, prefix, skip_predict):
    focus_model = focus_model or str(ROOT / "03-draws" / "20260608" / mid / "focus_model.json")
    out_root = Path(out_root)
    deliverables = out_root / "deliverables"
    deliverables.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in modes.split(",") if m.strip()]

    # 1) fire ALL predict workflows (modes x {A0,A1}) CONCURRENTLY, then wait.
    #    Each is an independent jobmon workflow (~20 tasks); the cluster has headroom,
    #    so there is no reason to serialize them.
    if not skip_predict:
        procs = []
        for mode in modes:
            mode_dir = out_root / mode
            for frame, cons in (("a0", A0), ("a1", A1)):
                cmd = _predict_cmd(
                    cells_file=cells_file, focus_model=focus_model, cons=cons,
                    out_dir=mode_dir / f"{mid}_{frame}", data_path=data_path,
                    folds_path=folds_path, is_island_path=is_island_path, mode=mode,
                    anchor_year=anchor_year, memory=memory, runtime=runtime, mid=mid)
                print("[launch]", mode, frame, flush=True)
                procs.append((mode, frame, subprocess.Popen(cmd)))
        failures = []
        for mode, frame, p in procs:
            if p.wait() != 0:
                failures.append(f"{mode}/{frame}")
        if failures:
            raise RuntimeError(f"predict failed for: {', '.join(failures)}")

    # 2) blend + deliver each mode (fast, local) once its A0+A1 are done
    for mode in modes:
        mode_dir = out_root / mode
        _run([
            sys.executable, str(Path(__file__).parent / "build_sr_version_files.py"),
            "--pred-root", str(mode_dir), "--mid", mid,
            "--prefix", f"{prefix}_{mode}",
            "--a0-dir", str(mode_dir / f"{mid}_a0"),
            "--a1-dir", str(mode_dir / f"{mid}_a1"),
            "--out-dir", str(deliverables), "--stats", "mean",
        ])
        print(f"[done] mode {mode} -> {deliverables}/{prefix}_{mode}_*", flush=True)

    print("ALL SENSITIVITY MODES DONE", flush=True)


if __name__ == "__main__":
    main()
