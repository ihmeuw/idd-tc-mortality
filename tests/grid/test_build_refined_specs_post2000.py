import numpy as np
import pytest

from idd_tc_mortality.constants import QUANTILE_LEVELS
from idd_tc_mortality.grid.build_refined_specs_post2000 import (
    build_specs,
    snap_to_quantile_levels,
)


def test_snap_maps_exact_cli_floats_to_canonical_keys():
    # 0.80 and 0.90 are np.linspace artifacts (0.7999..., 0.8999...);
    # CLI-exact floats must resolve to those canonical values.
    snapped = snap_to_quantile_levels([0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    assert snapped == [float(q) for q in QUANTILE_LEVELS]
    assert 0.8 not in snapped
    assert any(abs(q - 0.8) < 1e-9 for q in snapped)


def test_snap_rejects_non_grid_threshold():
    with pytest.raises(ValueError, match="0.65"):
        snap_to_quantile_levels([0.65])


def test_build_specs_carries_snapped_thresholds():
    thresholds = snap_to_quantile_levels([0.80])
    specs = build_specs(thresholds)
    qs = {s["threshold_quantile"] for s in specs if s["component"] != "s1"}
    assert qs == {float(QUANTILE_LEVELS[np.argmin(np.abs(QUANTILE_LEVELS - 0.8))])}
