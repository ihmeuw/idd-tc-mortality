"""Checked parquet / JSON writes for the comparison pipeline (atomic, loud on empty)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def write_parquet_checked(df: pd.DataFrame, path: Path, dtypes: dict | None = None) -> Path:
    """Atomic parquet write with explicit dtypes; refuses zero-row frames.

    Writes ``<path>.tmp`` beside the target, validates row count + column names from
    the file metadata (never reads the data back), then renames over the target.
    """
    path = Path(path)
    if len(df) == 0:
        raise RuntimeError(f"refusing to write zero rows to {path}")
    if dtypes:
        df = df.astype(dtypes)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    md = pq.read_metadata(tmp)
    if md.num_rows != len(df) or pq.read_schema(tmp).names != list(df.columns):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"parquet validation failed for {path}: {md.num_rows} rows vs {len(df)}")
    os.replace(tmp, path)
    return path


def write_json(obj, path: Path) -> Path:
    """Atomic compact JSON write (NaN → null)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)
    return path
