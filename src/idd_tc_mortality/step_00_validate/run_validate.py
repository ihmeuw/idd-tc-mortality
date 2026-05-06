"""
CLI entry point for step_00_validate.

Usage:
    run-validate --input <path> --output-dir <dir>

Reads input parquet, prints a basin frequency table, optionally recodes
noncanonical basins to 'NA', and writes cleaned parquet to a versioned
output node under output-dir.

Output structure:
    <output-dir>/YYYYMMDD/validated.parquet
    <output-dir>/current  →  YYYYMMDD/           (symlink, updated on write)
"""

from __future__ import annotations

import logging
import os
import pathlib
from datetime import date

import click
import pandas as pd
import pyarrow.parquet as pq_meta

from idd_tc_mortality.constants import BASIN_LEVELS
from idd_tc_mortality.step_00_validate.validate import check_basins, recode_noncanonical_basins

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _print_basin_table(result: dict[str, dict[str, int]]) -> None:
    """Print a two-section frequency table: canonical then noncanonical."""
    canonical = result["canonical"]
    noncanonical = result["noncanonical"]

    click.echo("\nBasin validation results")
    click.echo("=" * 30)

    click.echo("Canonical basins:")
    if canonical:
        for basin in BASIN_LEVELS:
            if basin in canonical:
                click.echo(f"  {basin:<8} {canonical[basin]:>6}")
    else:
        click.echo("  (none)")

    click.echo("Noncanonical basins:")
    if noncanonical:
        for val, count in sorted(noncanonical.items()):
            click.echo(f"  {val!r:<12} {count:>6}")
    else:
        click.echo("  (none)")
    click.echo()


def _write_versioned(df: pd.DataFrame, output_dir: pathlib.Path) -> pathlib.Path:
    """Write df to a dated subdirectory and update the current symlink.

    Atomic write: temp file → validate row count → rename to final.
    """
    dated_dir = output_dir / date.today().strftime("%Y%m%d")
    dated_dir.mkdir(parents=True, exist_ok=True)

    out_path = dated_dir / "validated.parquet"
    tmp_path = dated_dir / "validated.parquet.tmp"

    df.to_parquet(tmp_path, index=False)

    # Metadata-only validation — never read full file back
    meta = pq_meta.read_metadata(tmp_path)
    if meta.num_rows != len(df):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Parquet metadata row count {meta.num_rows} != expected {len(df)}. "
            "Aborting write."
        )

    if out_path.exists():
        out_path.unlink()
    tmp_path.rename(out_path)

    # Update current symlink (relative, so the directory is portable)
    current_link = output_dir / "current"
    if current_link.is_symlink():
        current_link.unlink()
    current_link.symlink_to(dated_dir.name)

    return out_path


@click.command()
@click.option(
    "--input", "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to input parquet file.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Base output directory. Cleaned parquet written to YYYYMMDD/ subdirectory.",
)
def main(input_path: str, output_dir: str) -> None:
    """Validate and optionally recode basin values in a TC mortality dataset."""
    df = pd.read_parquet(input_path)
    result = check_basins(df)

    _print_basin_table(result)

    total = sum(result["canonical"].values()) + sum(result["noncanonical"].values())
    n_noncanonical = sum(result["noncanonical"].values())

    if not result["noncanonical"]:
        click.echo("All basin values are canonical. No changes needed.")
        return

    click.echo(
        f"{n_noncanonical} of {total} rows have noncanonical basin values."
    )

    basin_list = "/".join(BASIN_LEVELS)
    if click.confirm(
        f"Set all non-[{basin_list}] values to 'NA'?",
        default=False,
    ):
        df_clean = recode_noncanonical_basins(df)
        out_path = _write_versioned(df_clean, pathlib.Path(output_dir))
        click.echo(f"Wrote {len(df_clean):,} rows to {out_path}")
        click.echo(
            f"Summary: {n_noncanonical} noncanonical basin values replaced with 'NA'."
        )
    else:
        click.echo("Exiting without writing.")
