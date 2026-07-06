"""Tests for the deliverable-finalization median adjustment.

Covers, on a synthetic 2-super-region hierarchy:
  - super_region_median_ratios: obs_median / pred_median per super-region, with
    a 0-observed super-region -> ratio 0 by default, and the --sr31-guard
    replacement (0 -> 0.1 x smallest non-zero ratio).
  - apply_super_region_adjustment: each location scaled by ITS super-region's
    ratio, unmapped super-regions left unscaled, and Global recomputed as the
    sum of the adjusted super-regions (hierarchy stays consistent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.predict.finalize_deliverable import (
    apply_super_region_adjustment,
    super_region_median_ratios,
)

# Synthetic FHS-shaped hierarchy: Global(1) over super-regions A(100) and B(200),
# each with one country child (101, 201). super_region_id of a super-region is
# itself; Global has none.
HIERARCHY = pd.DataFrame(
    {
        "location_id":     [1,     100, 101, 200, 201],
        "level":           [0,     1,   3,   1,   3],
        "super_region_id": [np.nan, 100, 100, 200, 200],
    }
)


def test_super_region_median_ratios_and_guard():
    cell = "deaths_c1_s1_o0_b1"
    ref = ["historical"]
    obs_years = (2000, 2001)

    # Observed super-region deaths: A median = median(10, 30) = 20 ; B median = 0.
    obs = pd.DataFrame(
        {
            "location_id": [100, 100, 200, 200],
            "year":        [2000, 2001, 2000, 2001],
            "deaths":      [10.0, 30.0, 0.0, 0.0],
        }
    )
    # Predicted (blend summary mean) over ref x obs_years: A median = 50, B = 10.
    summary = pd.DataFrame(
        {
            "location_id":   [100, 100, 200, 200],
            "cell":          [cell] * 4,
            "experiment_id": ["historical"] * 4,
            "year":          [2000, 2001, 2000, 2001],
            "mean":          [40.0, 60.0, 5.0, 15.0],
        }
    )

    ratio = super_region_median_ratios(summary, obs, HIERARCHY, cell, ref, obs_years)
    assert ratio.loc[100] == pytest.approx(20.0 / 50.0)   # 0.4
    assert ratio.loc[200] == pytest.approx(0.0)           # 0 observed -> hard zero

    guarded = super_region_median_ratios(
        summary, obs, HIERARCHY, cell, ref, obs_years, sr31_guard=True
    )
    assert guarded.loc[100] == pytest.approx(0.4)          # non-zero unchanged
    assert guarded.loc[200] == pytest.approx(0.1 * 0.4)    # 0 -> 0.1 x smallest non-zero


def test_apply_super_region_adjustment_scales_and_rerolls_global():
    # One (storm_draw, scenario, year) cell; Global's raw value is deliberately
    # inconsistent to prove it gets recomputed from the adjusted super-regions.
    draws = pd.DataFrame(
        {
            "storm_draw":   [1, 1, 1, 1, 1],
            "ssp_scenario": ["ssp245"] * 5,
            "year_id":      [2050] * 5,
            "location_id":  [1, 100, 101, 200, 201],
            "deaths":       [999.0, 50.0, 50.0, 30.0, 30.0],
        }
    )
    ratios = pd.Series({100: 0.4, 200: 0.0})

    out = apply_super_region_adjustment(draws, ratios, HIERARCHY).set_index("location_id")["deaths"]

    assert out.loc[100] == pytest.approx(50.0 * 0.4)   # super-region A scaled
    assert out.loc[101] == pytest.approx(50.0 * 0.4)   # its country scaled by same ratio
    assert out.loc[200] == pytest.approx(0.0)          # super-region B hard-zeroed
    assert out.loc[201] == pytest.approx(0.0)          # its country too
    # Global = sum of adjusted super-regions (20 + 0), NOT its raw 999.
    assert out.loc[1] == pytest.approx(20.0)


def test_apply_super_region_adjustment_leaves_unmapped_unscaled():
    # A super-region with no ratio entry must pass through unchanged (x1).
    draws = pd.DataFrame(
        {
            "storm_draw":   [1, 1, 1],
            "ssp_scenario": ["ssp245"] * 3,
            "year_id":      [2050] * 3,
            "location_id":  [1, 100, 200],
            "deaths":       [0.0, 50.0, 30.0],
        }
    )
    ratios = pd.Series({100: 0.4})   # 200 absent

    out = apply_super_region_adjustment(draws, ratios, HIERARCHY).set_index("location_id")["deaths"]
    assert out.loc[100] == pytest.approx(20.0)   # scaled
    assert out.loc[200] == pytest.approx(30.0)   # unmapped -> unscaled
    # Global = sum of the super-regions listed in ratios only (loc 100) -> 20.
    assert out.loc[1] == pytest.approx(20.0)
