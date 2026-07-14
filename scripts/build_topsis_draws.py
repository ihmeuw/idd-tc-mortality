"""
One-off: build 100 draw models for the half-coupled TOPSIS winner for each of
the four (draw_coefs, draw_scale) configurations and save them side-by-side.

The TOPSIS winner observed on the post-gate pool of the 20260513 half-coupled run:
    threshold_quantile = 0.70
    S1   logit / free  cov = {basin, is_island, sdi, wind_speed}
    S2   logit / free  cov = {is_island, sdi, wind_speed}
    bulk scaled_logit / free  cov = {is_island, sdi}
    tail gamma / free+weight  cov = {is_island, sdi}

Outputs per config (c, s) ∈ {0, 1}²:
    draws_c{c}_s{s}.pkl
    metadata_c{c}_s{s}.json
Plus a shared focus_model.json.

Run with the project conda env:

    python scripts/build_topsis_draws.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from idd_tc_mortality.refit_with_objects import refit_model_with_objects
from idd_tc_mortality.uncertainty import build_draw_models, save_draw_models


ROOT = Path("/mnt/team/idd/pub/idd_tc_mortality")
DATA_PATH = ROOT / "00-data" / "20260506" / "input.parquet"
FIT_DIR   = ROOT / "01-preliminary" / "20260507"
FOLDS_PATH = FIT_DIR / "fold_assignments.parquet"

OUTPUT_DIR = ROOT / "03-draws" / "20260514" / "topsis_winner_v1"
FOCUS_PATH = OUTPUT_DIR / "focus_model.json"

CONFIGS = [
    {"draw_coefs": False, "draw_scale": False},
    {"draw_coefs": False, "draw_scale": True},
    {"draw_coefs": True,  "draw_scale": False},
    {"draw_coefs": True,  "draw_scale": True},
]
N_DRAWS = 100
SEED    = 42


def _cov(active_keys: list[str]) -> str:
    full = ("basin", "is_island", "sdi", "wind_speed")
    return json.dumps({k: (k in active_keys) for k in full})


FOCUS_MODEL = {
    "threshold_quantile":  0.70,
    "s1_family":           "logit",
    "s1_exposure_mode":    "free",
    "s2_family":           "logit",
    "s2_exposure_mode":    "free",
    "bulk_family":         "scaled_logit",
    "bulk_exposure_mode":  "free",
    "tail_family":         "gamma",
    "tail_exposure_mode":  "free+weight",
    "s1_cov":              _cov(["basin", "is_island", "sdi", "wind_speed"]),
    "s2_cov":              _cov(["is_island", "sdi", "wind_speed"]),
    "bulk_cov":            _cov(["is_island", "sdi"]),
    "tail_cov":            _cov(["is_island", "sdi"]),
}


def main() -> None:
    t0 = time.time()
    print(f"[load] data : {DATA_PATH}")
    data = pd.read_parquet(DATA_PATH)
    print(f"       rows : {len(data):,}")
    print(f"[load] folds: {FOLDS_PATH}")
    folds = pd.read_parquet(FOLDS_PATH)
    print(f"       cols : {list(folds.columns)}")

    # IS-only matters for the draw machinery; minimise OOS work via n_seeds=1, n_folds=2.
    # The OOS path runs regardless but its result is unused here.
    print("[refit] running refit_model_with_objects (IS-focused, n_seeds=1, n_folds=2)...")
    refit_out = refit_model_with_objects(
        focus_model=FOCUS_MODEL,
        data=data,
        fold_assignments=folds,
        n_seeds=1,
        n_folds=2,
    )
    print(f"       elapsed: {time.time() - t0:.1f}s")

    for stage in ("s1", "s2", "bulk", "tail"):
        entry = refit_out["is"][stage]
        if entry.get("failed"):
            raise RuntimeError(
                f"Stage {stage!r} failed: {entry['metrics'].get('__failed__')}"
            )
    print(f"       IS stages: all converged ({list(refit_out['is']['combined'].keys())})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FOCUS_PATH.write_text(json.dumps(FOCUS_MODEL, indent=2))
    print(f"[save]  {FOCUS_PATH}")

    for cfg in CONFIGS:
        c, s = int(cfg["draw_coefs"]), int(cfg["draw_scale"])
        tag  = f"c{c}_s{s}"
        draws_path = OUTPUT_DIR / f"draws_{tag}.pkl"
        meta_path  = OUTPUT_DIR / f"metadata_{tag}.json"
        t_cfg = time.time()
        print(f"[build] {tag}: n_draws={N_DRAWS}, draw_coefs={bool(c)}, draw_scale={bool(s)}, seed={SEED}")
        models = build_draw_models(
            refit_out, FOCUS_MODEL, data,
            n_draws=N_DRAWS, draw_coefs=bool(c), draw_scale=bool(s), seed=SEED,
        )
        save_draw_models(models, draws_path)
        meta_path.write_text(json.dumps({
            "n_draws":         N_DRAWS,
            "seed":            SEED,
            "draw_coefs":      bool(c),
            "draw_scale":      bool(s),
            "threshold_rate":  refit_out["is"]["combined"]["threshold_rate"],
            "data_path":       str(DATA_PATH),
            "fit_dir":         str(FIT_DIR),
            "elapsed_seconds": round(time.time() - t_cfg, 2),
        }, indent=2))
        print(f"[save]  {draws_path}  ({draws_path.stat().st_size:,} bytes)")
        print(f"[save]  {meta_path}")

    print(f"[done] total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
