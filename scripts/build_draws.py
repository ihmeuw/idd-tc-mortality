"""
build_draws.py

Build the four (draw_coefs, draw_scale) draw-model pickles for ONE chosen DH
model, for use by the prediction pipeline.

Generalizes the old one-off ``build_topsis_draws.py``: the model spec is read
from a focus_model JSON file instead of being hardcoded, so the same script
builds draws for any selected winner (best-topsis, 2nd-topsis, no-tail-cov, ...).

What it does:
  1. Refit the 4 components on the full fit data, retaining inference objects
     (refit_model_with_objects). OOS is run with n_seeds=1/n_folds=2 and
     discarded -- only the IS fit feeds the draw machinery.
  2. For each (draw_coefs, draw_scale) in {0,1}^2, build n_draws draw models
     and save draws_c{c}_s{s}.pkl alongside a metadata json.
  3. Copy the focus_model into the output dir for provenance.

The focus_model JSON must hold the 13 spec fields:
    threshold_quantile,
    {s1,s2,bulk,tail}_family, {s1,s2,bulk,tail}_exposure_mode,
    {s1,s2,bulk,tail}_cov   (each a JSON string of {axis: bool})

Run (project env):
    python scripts/build_draws.py \\
        --focus-model-json <focus_model.json> \\
        --output-dir /mnt/team/idd/pub/idd_tc_mortality/03-draws/<vintage>/<mid>
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import pandas as pd

from idd_tc_mortality.refit_with_objects import refit_model_with_objects
from idd_tc_mortality.uncertainty import build_draw_models, save_draw_models

ROOT = Path("/mnt/team/idd/pub/idd_tc_mortality")
DEFAULT_DATA_PATH  = ROOT / "00-data" / "current" / "input.parquet"
DEFAULT_FOLDS_PATH = ROOT / "02-evaluate" / "20260608_final" / "fold_assignments.parquet"

CONFIGS: list[tuple[int, int]] = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (draw_coefs, draw_scale)


@click.command()
@click.option("--focus-model-json", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="JSON file with the 13 spec fields (see module docstring).")
@click.option("--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path),
              help="Where to write draws_c{c}_s{s}.pkl + metadata + focus_model.json.")
@click.option("--data-path", default=str(DEFAULT_DATA_PATH), show_default=True,
              help="Fit data (input.parquet) to refit on.")
@click.option("--folds-path", default=str(DEFAULT_FOLDS_PATH), show_default=True,
              help="fold_assignments.parquet (must align row-for-row with --data-path; "
                   "only seed_0 is touched and the OOS result is discarded).")
@click.option("--n-draws", default=100, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
def main(focus_model_json, output_dir, data_path, folds_path, n_draws, seed):
    focus = json.loads(Path(focus_model_json).read_text())
    print(f"[focus] {focus_model_json}")
    print(f"        threshold={focus['threshold_quantile']}  "
          f"tail={focus['tail_family']}/{focus['tail_exposure_mode']}")

    data = pd.read_parquet(data_path)
    folds = pd.read_parquet(folds_path)
    print(f"[load] data {len(data):,} rows ({data_path})")
    print(f"[load] folds {len(folds):,} rows, cols={list(folds.columns)}")
    if len(folds) != len(data):
        raise SystemExit(
            f"ERROR: folds rows ({len(folds):,}) != data rows ({len(data):,}); "
            "fold file is not aligned with this data vintage."
        )

    t0 = time.time()
    print("[refit] refit_model_with_objects (IS-focused, n_seeds=1, n_folds=2) ...")
    refit_out = refit_model_with_objects(
        focus_model=focus, data=data, fold_assignments=folds, n_seeds=1, n_folds=2,
    )
    for stage in ("s1", "s2", "bulk", "tail"):
        if refit_out["is"][stage].get("failed"):
            raise SystemExit(
                f"ERROR: IS stage {stage!r} failed: "
                f"{refit_out['is'][stage]['metrics'].get('__failed__')}"
            )
    threshold_rate = refit_out["is"]["combined"]["threshold_rate"]
    print(f"        all 4 IS stages converged in {time.time() - t0:.1f}s "
          f"(threshold_rate={threshold_rate:.6g})")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "focus_model.json").write_text(json.dumps(focus, indent=2))

    for c, s in CONFIGS:
        tag = f"c{c}_s{s}"
        t_cfg = time.time()
        print(f"[build] {tag}: n_draws={n_draws} draw_coefs={bool(c)} draw_scale={bool(s)} seed={seed}")
        models = build_draw_models(
            refit_out, focus, data,
            n_draws=n_draws, draw_coefs=bool(c), draw_scale=bool(s), seed=seed,
        )
        draws_path = out / f"draws_{tag}.pkl"
        save_draw_models(models, draws_path)
        (out / f"metadata_{tag}.json").write_text(json.dumps({
            "n_draws":        n_draws,
            "seed":           seed,
            "draw_coefs":     bool(c),
            "draw_scale":     bool(s),
            "threshold_rate": threshold_rate,
            "data_path":      str(data_path),
            "folds_path":     str(folds_path),
            "elapsed_seconds": round(time.time() - t_cfg, 2),
        }, indent=2))
        print(f"        wrote {draws_path} ({draws_path.stat().st_size:,} bytes)")

    print(f"[done] {time.time() - t0:.1f}s total -> {out}")


if __name__ == "__main__":
    main()
