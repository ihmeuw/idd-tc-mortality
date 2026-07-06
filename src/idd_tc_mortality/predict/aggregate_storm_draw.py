"""Storm-draw aggregation: concat scenario means for one storm_draw.

Different scenarios are alternative futures — never combined arithmetically.
Adds a `scenario` column so the stacked frame preserves which alternative
each row belongs to.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from idd_tc_mortality.predict.done_manifest import (
    TIER_AGG_STORM_DRAW,
    append_done,
)
from idd_tc_mortality.predict.paths import (
    SCENARIOS,
    STORM_DRAW_DIR,
    atomic_write_parquet,
    scenario_mean_path,
    storm_draw_mean_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def aggregate_storm_draw(
    *,
    storm_draw: int,
    storm_draw_dir: Path,
) -> Path:
    out_path = storm_draw_mean_path(storm_draw_dir, storm_draw)
    if out_path.exists():
        logger.info("storm_draw-mean already present at %s — skipping.", out_path)
        return out_path

    missing = [
        scenario for scenario in SCENARIOS
        if not scenario_mean_path(storm_draw_dir, storm_draw, scenario).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"sd={storm_draw}: scenario-mean missing for {missing}. "
            "Upstream aggregate_scenario task(s) reported success but did not "
            "produce the expected output — fail loudly."
        )

    frames = []
    for scenario in SCENARIOS:
        p = scenario_mean_path(storm_draw_dir, storm_draw, scenario)
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df = df.copy()
        df['scenario'] = scenario
        frames.append(df)
    logger.info("sd=%d: %d/%d scenario means present, %d non-empty.",
                storm_draw, len(SCENARIOS), len(SCENARIOS), len(frames))

    if not frames:
        atomic_write_parquet(
            pd.DataFrame(columns=['scenario', 'location_id', 'year', 'exposed_mean', 'total_population']),
            out_path,
        )
        logger.warning("All scenario means empty for storm_draw=%d — wrote empty.", storm_draw)
        append_done(storm_draw_dir, TIER_AGG_STORM_DRAW, storm_draw=storm_draw)
        return out_path

    combined = pd.concat(frames, axis=0, ignore_index=True)
    cols = ['scenario'] + [c for c in combined.columns if c != 'scenario']
    atomic_write_parquet(combined[cols], out_path)
    append_done(storm_draw_dir, TIER_AGG_STORM_DRAW, storm_draw=storm_draw)
    return out_path


@click.command()
@click.option('--storm-draw',     type=int, required=True)
@click.option('--storm-draw-dir', type=click.Path(path_type=Path), default=str(STORM_DRAW_DIR), show_default=True)
def main(storm_draw, storm_draw_dir):
    out_path = aggregate_storm_draw(
        storm_draw=storm_draw,
        storm_draw_dir=Path(storm_draw_dir),
    )
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
