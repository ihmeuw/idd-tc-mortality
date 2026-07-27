"""Driver-of-change sensitivity analyses for the consolidated predict pipeline.

Each sensitivity freezes one driver from an anchor year (default 2023) onward and
re-predicts, so the change since the anchor can be attributed to SDI vs population:

- ``sdi_const``: hold SDI at its anchor-year value for all years >= anchor.
- ``pop_const``: hold population constant by scaling exposure (person_storm_hours)
  by ``pop(anchor) / pop(year)`` for all years >= anchor.
- ``both``: apply both.

The transforms (``frame_adjust``) run inside the predict worker, on the prepped
per-storm-draw frame, right before prediction. ``decompose`` reads the delivered
results back for the comparison notebook.
"""

from idd_tc_mortality.sensitivity.frame_adjust import (
    ANCHOR_YEAR,
    SENSITIVITY_MODES,
    apply_sensitivity,
    freeze_sdi,
    population_ratio,
    scale_exposure,
)

__all__ = [
    "ANCHOR_YEAR",
    "SENSITIVITY_MODES",
    "apply_sensitivity",
    "freeze_sdi",
    "population_ratio",
    "scale_exposure",
]
