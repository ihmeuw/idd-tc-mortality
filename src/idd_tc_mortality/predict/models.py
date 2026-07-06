"""Load the four (c, s) draw models for a given storm_draw."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from idd_tc_mortality.uncertainty import load_draw_models


def load_models_for_storm_draw(storm_draw: int, draws_dir: Path) -> dict:
    """Return {(c, s): DrawModel} for c, s in {0, 1} at the storm_draw - 1 index
    of each of the four pickles in `draws_dir`."""
    draw_idx = storm_draw - 1
    return {
        (c, s): load_draw_models(draws_dir / f'draws_c{c}_s{s}.pkl')[draw_idx]
        for c in (0, 1) for s in (0, 1)
    }


def lookup_model_variant(
    storm_draw: int, storm_draw_table_path: str,
) -> tuple[str, str]:
    """Look up (model, variant) for a storm_draw from the registry CSV."""
    tbl = pd.read_csv(storm_draw_table_path)
    row = tbl.loc[tbl['storm_draw'] == storm_draw]
    if row.empty:
        raise ValueError(
            f"storm_draw={storm_draw} not found in {storm_draw_table_path}"
        )
    return str(row['source_id'].values[0]), str(row['variant_label'].values[0])
