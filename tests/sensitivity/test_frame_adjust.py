"""Unit tests for the sensitivity frame transforms."""

import numpy as np
import pandas as pd
import pytest

from idd_tc_mortality.sensitivity.frame_adjust import (
    apply_sensitivity,
    freeze_sdi,
    scale_exposure,
)

ANCHOR = 2023


def _frame():
    # two locations, years spanning the anchor, with distinct sdi/exposed per year
    rows = []
    for loc in (10, 20):
        for yr in (2015, 2023, 2024, 2050):
            rows.append({"location_id": loc, "year": yr,
                         "sdi": 0.5 + 0.001 * (yr - 2015) + 0.01 * (loc == 20),
                         "exposed": 100.0 + (yr - 2015) + loc,
                         "total_population": 1000.0})
    return pd.DataFrame(rows)


def _sdi_table():
    rows = []
    for loc in (10, 20):
        for yr in range(2015, 2051):
            rows.append({"location_id": loc, "year": yr,
                         "sdi": 0.5 + 0.001 * (yr - 2015) + 0.01 * (loc == 20)})
    return pd.DataFrame(rows)


def test_freeze_sdi_holds_anchor_and_leaves_past():
    f = _frame()
    out = freeze_sdi(f, _sdi_table(), anchor_year=ANCHOR)
    # pre-anchor untouched
    pre = out[out.year < ANCHOR]
    assert (pre["sdi"].values == f[f.year < ANCHOR]["sdi"].values).all()
    # year >= anchor equals that location's anchor-year sdi
    for loc in (10, 20):
        anchor_sdi = _sdi_table().query("location_id == @loc and year == @ANCHOR")["sdi"].iloc[0]
        got = out[(out.location_id == loc) & (out.year >= ANCHOR)]["sdi"].unique()
        assert got.size == 1 and np.isclose(got[0], anchor_sdi)


def test_scale_exposure_matches_ratio_and_noop_at_anchor():
    f = _frame()
    # ratio 1.0 at anchor, 0.5 at 2024, 0.25 at 2050 for both locations
    ratio = {}
    for loc in (10, 20):
        ratio[(loc, 2023)] = 1.0
        ratio[(loc, 2024)] = 0.5
        ratio[(loc, 2050)] = 0.25
    out = scale_exposure(f, ratio, anchor_year=ANCHOR)
    # pre-anchor untouched
    assert (out[out.year < ANCHOR]["exposed"].values == f[f.year < ANCHOR]["exposed"].values).all()
    # anchor is a no-op (ratio 1.0)
    a_in = f[f.year == ANCHOR].set_index("location_id")["exposed"]
    a_out = out[out.year == ANCHOR].set_index("location_id")["exposed"]
    assert np.allclose(a_in.values, a_out.values)
    # 2024 halved, 2050 quartered
    for loc in (10, 20):
        e_in = f[(f.location_id == loc) & (f.year == 2024)]["exposed"].iloc[0]
        e_out = out[(out.location_id == loc) & (out.year == 2024)]["exposed"].iloc[0]
        assert np.isclose(e_out, 0.5 * e_in)
        e_in50 = f[(f.location_id == loc) & (f.year == 2050)]["exposed"].iloc[0]
        e_out50 = out[(out.location_id == loc) & (out.year == 2050)]["exposed"].iloc[0]
        assert np.isclose(e_out50, 0.25 * e_in50)


def test_scale_exposure_missing_ratio_left_unchanged():
    f = _frame()
    out = scale_exposure(f, {}, anchor_year=ANCHOR)  # no ratios at all
    # every row unchanged (missing -> 1.0)
    assert np.allclose(out["exposed"].values, f["exposed"].values)


def test_both_composes_sdi_and_pop():
    f = _frame()
    ratio = {(loc, yr): 0.5 for loc in (10, 20) for yr in (2024, 2050)}
    ratio.update({(loc, 2023): 1.0 for loc in (10, 20)})
    out = apply_sensitivity(f, "both", sdi_table=_sdi_table(), pop_ratio=ratio, anchor_year=ANCHOR)
    # sdi frozen
    got = out[(out.location_id == 10) & (out.year == 2050)]["sdi"].iloc[0]
    anchor_sdi = _sdi_table().query("location_id == 10 and year == @ANCHOR")["sdi"].iloc[0]
    assert np.isclose(got, anchor_sdi)
    # exposure scaled
    e_in = f[(f.location_id == 10) & (f.year == 2050)]["exposed"].iloc[0]
    e_out = out[(out.location_id == 10) & (out.year == 2050)]["exposed"].iloc[0]
    assert np.isclose(e_out, 0.5 * e_in)


def test_none_is_identity():
    f = _frame()
    out = apply_sensitivity(f, "none")
    pd.testing.assert_frame_equal(out, f)


def test_apply_sensitivity_requires_inputs():
    f = _frame()
    with pytest.raises(ValueError):
        apply_sensitivity(f, "sdi_const")  # no sdi_table
    with pytest.raises(ValueError):
        apply_sensitivity(f, "pop_const")  # no pop_ratio
    with pytest.raises(ValueError):
        apply_sensitivity(f, "bogus")
