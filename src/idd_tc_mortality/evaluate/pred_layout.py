"""Resolve model_predictions files across the flat and task-sharded layouts.

Two layouts exist:

- **flat** (runs before 2026-07-31): every ``{mid}_{fold}_predictions.parquet``
  sits directly in ``model_predictions/``. The 20260722_v1985_final run
  (~307K files) is flat and stays that way.
- **task-sharded**: ``model_predictions/task_XXXXX/`` — one subdirectory per
  evaluate task. Introduced after the 2026-07-31 incident: hundreds of
  concurrent workers creating small files in ONE giant directory serialize on
  NFS directory-metadata updates, capping aggregate throughput fleet-wide
  regardless of task count.

Readers should use :func:`find_model_predictions` and never construct the
flat path directly.
"""

from __future__ import annotations

from pathlib import Path


def find_model_predictions(model_pred_dir: str | Path, filename: str) -> Path:
    """Return the path of ``filename`` under ``model_pred_dir``, either layout.

    Checks the flat location first, then the ``task_*/`` shards.

    Raises:
        FileNotFoundError: if the file is in neither location.
    """
    base = Path(model_pred_dir)
    flat = base / filename
    if flat.exists():
        return flat
    hits = sorted(base.glob(f"task_*/{filename}"))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"{filename} not found in {base} (flat layout) or {base}/task_*/ (sharded layout)"
    )


def load_model_predictions(model_pred_dir: str | Path, mid: str, fold_tag: str,
                           columns: list[str] | None = None):
    """Load one config's predictions for one fold_tag, any layout/format.

    Formats: per-fold files (``{mid}_{fold_tag}_predictions.parquet``, runs
    before 2026-07-31) and batched (``{mid}_predictions.parquet`` with all
    fold_tags stacked — introduced to cut NFS creates 6x). Both may live flat
    or task-sharded; :func:`find_model_predictions` resolves that.
    """
    import pandas as pd

    try:
        path = find_model_predictions(model_pred_dir,
                                      f"{mid}_{fold_tag}_predictions.parquet")
        return pd.read_parquet(path, columns=columns)
    except FileNotFoundError:
        path = find_model_predictions(model_pred_dir,
                                      f"{mid}_predictions.parquet")
        read_cols = None if columns is None else sorted(set(columns) | {"fold_tag"})
        df = pd.read_parquet(path, columns=read_cols)
        df = df[df["fold_tag"] == fold_tag]
        if df.empty:
            raise FileNotFoundError(
                f"fold_tag {fold_tag!r} absent from batched {path}")
        return df if columns is None else df[columns]
