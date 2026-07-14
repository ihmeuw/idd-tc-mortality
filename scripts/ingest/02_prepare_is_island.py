"""
02_prepare_is_island.py

One-off ingest script. Pulls the GBD island_bin covariate and writes a static
(location_id, is_island) lookup into the 00-data versioned node, alongside
input.parquet, for use as a prediction covariate at both A0 (national) and A1
(subnational) resolution.

>>> ENV: requires `db_queries`, which lives only in the GBD_shared_functions
    conda env -- NOT the project env. Run it from that env (see below).

What this does:
  1. Pull covariate_id=2608 (island_bin) at release_id=16 via db_queries.
     (release 16 matches the past-SDI release, so the location set is consistent.)
  2. Collapse to one (location_id, is_island) row per location -- island_bin is
     invariant over year/age/sex, so the year replication is dropped. Asserts a
     single distinct value per location (and binary 0/1) so a stray time-varying
     or non-binary value fails loudly rather than being silently deduped away.
  3. Cast to explicit dtypes (location_id int64, is_island int8).
  4. Atomic-write to 00-data/<vintage>/is_island.parquet.

Vintage handling (deliberately different from 01_prepare_input_data.py):
  is_island is model-bound -- it attaches to an EXISTING fit-data vintage, it
  does not create a new one. So <vintage> defaults to the current model vintage
  (resolved from the `current` symlink), the target dir must already exist, and
  the `current` symlink is NOT moved.

Run (from the GBD_shared_functions env):
    conda run -n GBD_shared_functions python scripts/ingest/02_prepare_is_island.py
    conda run -n GBD_shared_functions python scripts/ingest/02_prepare_is_island.py --dry-run
    conda run -n GBD_shared_functions python scripts/ingest/02_prepare_is_island.py --vintage 20260608
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_NODE = Path("/mnt/team/idd/pub/idd_tc_mortality/00-data")

COVARIATE_ID = 2608   # island_bin
RELEASE_ID   = 16     # matches the past-SDI release (data/16)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_is_island(island_cov: pd.DataFrame) -> pd.DataFrame:
    """Collapse the GBD covariate frame to a static (location_id, is_island) lookup.

    island_bin is invariant over year/age/sex, so we keep one row per location.
    Before deduping, assert exactly one distinct value per location and that
    every value is binary -- so a time-varying or non-0/1 value surfaces loudly
    instead of being silently dropped by drop_duplicates.
    """
    df = island_cov.rename(columns={"mean_value": "is_island"})[["location_id", "is_island"]]

    multi = df.groupby("location_id")["is_island"].nunique()
    conflicting = multi[multi > 1]
    if len(conflicting):
        raise ValueError(
            f"{len(conflicting)} location_ids have >1 distinct is_island value "
            f"across year/age/sex (first few: {conflicting.index[:5].tolist()}). "
            "island_bin is supposed to be invariant -- investigate before saving."
        )

    values = set(df["is_island"].dropna().unique())
    if not values <= {0.0, 1.0}:
        raise ValueError(f"is_island has non-binary values: {sorted(values)}")

    return (
        df.drop_duplicates("location_id")
          .astype({"location_id": "int64", "is_island": "int8"})
          .sort_values("location_id")
          .reset_index(drop=True)
    )


def resolve_vintage(output_node: Path, vintage: str | None) -> Path:
    """Return the existing dated dir to write into.

    vintage=None resolves the `current` symlink. The dir must already exist --
    is_island does not create a new vintage, and this function never moves
    `current`.
    """
    if vintage is None:
        current = output_node / "current"
        if not current.is_symlink():
            raise FileNotFoundError(
                f"No `current` symlink at {current}; pass --vintage explicitly."
            )
        dated_dir = current.resolve()
    else:
        dated_dir = output_node / vintage

    if not dated_dir.is_dir():
        raise FileNotFoundError(
            f"Vintage dir does not exist: {dated_dir}. is_island must attach to an "
            "existing fit-data vintage -- create the vintage (or fix --vintage) first."
        )
    return dated_dir


def write_atomic(df: pd.DataFrame, out_path: Path) -> None:
    """Atomic .tmp -> os.replace write, with metadata row-count validation.

    Note: unlike 01_prepare_input_data.write_versioned, this does NOT touch the
    `current` symlink -- the covariate attaches to an existing vintage.
    """
    fd, tmp = tempfile.mkstemp(dir=out_path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        meta = pq.read_metadata(tmp)
        if meta.num_rows != len(df):
            raise RuntimeError(
                f"Parquet metadata row count {meta.num_rows} != expected {len(df)}"
            )
        if out_path.exists():
            out_path.unlink()
        os.replace(tmp, out_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(vintage: str | None = None, dry_run: bool = False) -> None:
    # db_queries exists only in the GBD_shared_functions env; import here (not at
    # module top) so the module stays importable elsewhere (e.g. build_is_island).
    from db_queries import get_covariate_estimates

    print(f"Pulling covariate_id={COVARIATE_ID} (island_bin) release_id={RELEASE_ID} ...")
    raw = get_covariate_estimates(covariate_id=COVARIATE_ID, release_id=RELEASE_ID)
    print(f"  Raw rows: {len(raw):,}")

    is_island = build_is_island(raw)
    print(
        f"  Collapsed to {len(is_island):,} locations "
        f"({int(is_island['is_island'].sum()):,} flagged island)."
    )

    dated_dir = resolve_vintage(OUTPUT_NODE, vintage)
    out_path = dated_dir / "is_island.parquet"
    print(f"  Target: {out_path}")

    if dry_run:
        print("\n[dry-run] No file written.")
        return

    write_atomic(is_island, out_path)
    print(f"\nWrote {len(is_island):,} rows to {out_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vintage",
        default=None,
        help=(
            "00-data vintage dir to write into (YYYYMMDD). Default: resolve the "
            "`current` symlink. The dir must already exist; `current` is not moved."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pull and process without writing any output.",
    )
    args = parser.parse_args()
    try:
        main(vintage=args.vintage, dry_run=args.dry_run)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
