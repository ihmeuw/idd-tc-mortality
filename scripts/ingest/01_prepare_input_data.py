"""
01_prepare_input_data.py

One-off ingest script. Run once per data vintage to produce input.parquet
in the 00-data versioned output node.

What this does:
  1. Read the raw ibtracs + deaths CSV using the exact same settings as
     the old idd-climate-models pipeline (keep_default_na=False, NA-safe).
  2. Apply filters:
       year <= 2023            (exclude incomplete 2024 data)
       exposed_population >= 1 (drop zero/missing exposure)
       level filter (see --level-filter)
  3. Recode missing basin values (NaN from empty strings in CSV) to 'NA'.
  4. Rename columns to match the new pipeline's schema:
       total_deaths       -> deaths
       exposed_population -> exposed
       max_wind_speed     -> wind_speed
  5. Select and cast to explicit dtypes; drop all other columns.
  6. Write atomically to:
       /mnt/team/idd/pub/idd_tc_mortality/00-data/YYYYMMDD/input.parquet
     and update the `current` symlink.

Level filter options (--level-filter):
  all       — use all admin levels without filtering (default)
  level3    — keep only level-3 (national) rows
  aggregate — keep only level-4/5 rows; aggregate up to level-3 using
              path_to_top_parent to identify the parent level-3 location.
              deaths and exposed are summed; wind_speed and sdi are
              exposure-weighted means; other fields take the first value.

Source data:
    /mnt/team/rapidresponse/pub/tropical-storms/data/ibtracs_deaths/
        combined_ibtracs_with_deaths_deduplicated_with_sdi_updated_island.csv
    (4.9 MB, last modified 2025-02-10, maintained by bedalton)

Run:
    conda run -n idd-tc-mortality python scripts/ingest/01_prepare_input_data.py
    conda run -n idd-tc-mortality python scripts/ingest/01_prepare_input_data.py --dry-run
    conda run -n idd-tc-mortality python scripts/ingest/01_prepare_input_data.py --level-filter level3
    conda run -n idd-tc-mortality python scripts/ingest/01_prepare_input_data.py --level-filter aggregate
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths — hardcoded to the specific vintage being ingested
# ---------------------------------------------------------------------------

SOURCE_CSV = Path(
    "/mnt/team/rapidresponse/pub/tropical-storms/data/ibtracs_deaths/"
    "ibtracs_stage4b_pafs_admin0_with_deaths_sdi__island_20260417.csv"
)

OUTPUT_NODE = Path("/mnt/team/idd/pub/idd_tc_mortality/00-data")

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

FILTER_MAX_YEAR    = 2023  # exclude 2024+ (incomplete data)
FILTER_MIN_EXPOSED = 1     # drop rows with zero or missing exposure

# ---------------------------------------------------------------------------
# Column config
# ---------------------------------------------------------------------------

RENAMES = {
    "total_deaths":       "deaths",
    "exposed_population": "exposed",
    "max_wind_speed":     "wind_speed",
    "basins":             "basin",
}

# Columns to keep (after renaming) with target dtypes.
# Identifiers are kept for tracing; model covariates are the first six.
KEEP_COLS: dict[str, str] = {
    # Model inputs
    "deaths":       "int32",
    "exposed":      "float64",
    "wind_speed":   "float32",
    "sdi":          "float32",
    "basin":        "str",
    "is_island":    "int8",
    # Identifiers / metadata
    "storm_id":          "str",
    "location_id":       "int32",
    "year":              "int16",
    "location_name":     "str",
    "path_to_top_parent":"str",
    "super_region_name": "str",
    "region_name":       "str",
}


# ---------------------------------------------------------------------------
# Aggregation helper (used by level_filter='aggregate')
# ---------------------------------------------------------------------------

def _aggregate_subnationals(df_sub: pd.DataFrame) -> pd.DataFrame:
    """Aggregate level-4/5 rows up to level-3 using exposure weighting.

    Groups by (storm_id, level3_id). Within each group:
    - total_deaths, exposed_population: summed
    - max_wind_speed, sdi: exposure-weighted means
    - basin, is_island, year, location_name, super_region_name, region_name: first value
    - location_id: level-3 ID extracted from path_to_top_parent[3]
    - path_to_top_parent: first 4 path elements (the level-3 path)

    All rows are assumed to have exposed_population >= 1 (pre-filtered).
    """
    df_sub = df_sub.copy()
    parts = df_sub["path_to_top_parent"].str.split(",")
    df_sub["_level3_id"] = parts.str[3].astype(int)
    df_sub["_level3_path"] = parts.str[:4].str.join(",")

    agg_rows = []
    for (storm_id, level3_id), grp in df_sub.groupby(["storm_id", "_level3_id"]):
        exp = grp["exposed_population"].values
        total_exp = float(exp.sum())
        w = exp / total_exp
        agg_rows.append({
            "total_deaths":       int(grp["total_deaths"].sum()),
            "exposed_population": total_exp,
            "max_wind_speed":     float((grp["max_wind_speed"].values * w).sum()),
            "sdi":                float((grp["sdi"].values * w).sum()),
            "basin":              grp["basin"].iloc[0],
            "is_island":          grp["is_island"].iloc[0],
            "storm_id":           storm_id,
            "location_id":        level3_id,
            "year":               grp["year"].iloc[0],
            "location_name":      grp["location_name"].iloc[0],
            "path_to_top_parent": grp["_level3_path"].iloc[0],
            "super_region_name":  grp["super_region_name"].iloc[0],
            "region_name":        grp["region_name"].iloc[0],
        })
    return pd.DataFrame(agg_rows)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame, level_filter: str = "all") -> pd.DataFrame:
    """Apply filters, recode, rename, cast, and select columns.

    Parameters
    ----------
    level_filter : {'all', 'level3', 'aggregate'}
        'all'       — use all admin levels (default)
        'level3'    — keep only level-3 (national) rows
        'aggregate' — keep only level-4/5 rows, aggregate to level-3
    """
    # --- Year and exposure filters (always applied) ---
    mask = (
        (df["year"] <= FILTER_MAX_YEAR)
        & (df["exposed_population"] >= FILTER_MIN_EXPOSED)
    )
    df = df.loc[mask].copy()
    print(f"  After year/exposure filters: {len(df):,} rows")

    # --- Level filter ---
    if level_filter == "level3":
        df = df.loc[df["level"] == 3].copy()
        print(f"  Level filter (level3): {len(df):,} rows")
    elif level_filter == "aggregate":
        df_sub = df.loc[df["level"].isin([4, 5])].copy()
        print(f"  Level filter (aggregate): {len(df_sub):,} level-4/5 source rows")
        df = _aggregate_subnationals(df_sub)
        print(f"  After aggregation to level-3: {len(df):,} rows")
    else:  # 'all'
        print(f"  Level filter (all): {len(df):,} rows")

    # --- Rename columns (must precede basin recode) ---
    df = df.rename(columns=RENAMES)

    # --- TEMPORARY: take first basin for multi-basin rows (e.g. "EP,NA" -> "EP") ---
    # Source data sometimes lists two basins for storms that crossed basin boundaries.
    # The upstream data team will fix this at source; remove this once that is done.
    multi_basin = df["basin"].str.contains(",", na=False)
    if multi_basin.any():
        df.loc[multi_basin, "basin"] = df.loc[multi_basin, "basin"].str.split(",").str[0]
        print(f"  TEMPORARY: took first basin for {multi_basin.sum():,} multi-basin rows")

    # --- Recode basin: NaN (empty string in CSV) -> 'NA' ---
    # The CSV stores North Atlantic as empty string ''; with na_values=['']
    # these arrive as NaN. We recode to the canonical 'NA' code.
    n_missing_basin = df["basin"].isna().sum()
    if n_missing_basin:
        df["basin"] = df["basin"].fillna("NA")
        print(f"  Recoded {n_missing_basin:,} missing basin values -> 'NA'")
    else:
        print("  No missing basin values.")

    # --- Select and cast ---
    df = df[list(KEEP_COLS.keys())].copy()
    for col, dtype in KEEP_COLS.items():
        if dtype == "str":
            df[col] = df[col].astype(str)
        else:
            df[col] = df[col].astype(dtype)

    return df.reset_index(drop=True)


def write_versioned(df: pd.DataFrame, output_node: Path) -> Path:
    """Atomic write to dated subdirectory; update current symlink."""
    dated_dir = output_node / date.today().strftime("%Y%m%d")
    dated_dir.mkdir(parents=True, exist_ok=True)

    out_path = dated_dir / "input.parquet"

    fd, tmp = tempfile.mkstemp(dir=dated_dir, suffix=".parquet.tmp")
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

    # Update current symlink (relative so the directory is portable)
    current_link = output_node / "current"
    if current_link.is_symlink():
        current_link.unlink()
    current_link.symlink_to(dated_dir.name)

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False, level_filter: str = "all") -> None:
    if not SOURCE_CSV.exists():
        print(f"ERROR: source CSV not found: {SOURCE_CSV}", file=sys.stderr)
        sys.exit(1)

    print(f"Source: {SOURCE_CSV}")
    print(f"Output node: {OUTPUT_NODE}")
    print(f"Level filter: {level_filter}")
    print()

    # Read with NA-safe settings (identical to old idd-climate-models data.py)
    print("Reading CSV...")
    df_raw = pd.read_csv(
        SOURCE_CSV,
        keep_default_na=False,   # critical: prevents 'NA' (North Atlantic) -> NaN
        na_values=[""],           # only blank cells become NaN
        dtype={"basin": str},     # critical: force basin to str before any NA inference
    )
    print(f"  Raw rows: {len(df_raw):,}  columns: {len(df_raw.columns)}")

    print("Applying filters and preparing...")
    df_out = prepare(df_raw, level_filter=level_filter)
    print(f"  Output rows: {len(df_out):,}  columns: {len(df_out.columns)}")

    print()
    print("Basin distribution after prep:")
    print(df_out["basin"].value_counts().to_string())
    print()
    print("Year range:", df_out["year"].min(), "-", df_out["year"].max())
    print("Dtypes:")
    for col, dtype in df_out.dtypes.items():
        print(f"  {col:<25} {dtype}")

    if dry_run:
        print("\n[dry-run] No files written.")
        return

    print("\nWriting parquet...")
    out_path = write_versioned(df_out, OUTPUT_NODE)
    print(f"  Wrote {len(df_out):,} rows to {out_path}")
    print(f"  current -> {(OUTPUT_NODE / 'current').resolve().name}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Read and process without writing any output."
    )
    parser.add_argument(
        "--level-filter",
        choices=["all", "level3", "aggregate"],
        default="all",
        help=(
            "How to handle IHME admin levels. "
            "'all': use all levels (default). "
            "'level3': national-level rows only. "
            "'aggregate': aggregate level-4/5 rows to level-3 using exposure weighting."
        ),
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, level_filter=args.level_filter)
