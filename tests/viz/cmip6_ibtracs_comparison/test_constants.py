"""Basin assignment, categorisation and outcome sums."""

import numpy as np
import pandas as pd

from idd_tc_mortality.viz.cmip6_ibtracs_comparison import constants as c


def test_wrap_lon():
    assert list(c.wrap_lon([183.7, -176.3, 180.0, -180.0, 10.0])) == [-176.3, -176.3, -180.0, -180.0, 10.0]


def test_assign_tcrisk_basin_boxes_and_wrap():
    lon = [-60, -140, 140, 70, 60, 150, 183.7, -30, 110, np.nan]
    lat = [20, 20, 20, 15, -20, -20, -10, -20, 20, 5]
    got = list(c.assign_tcrisk_basin(lon, lat))
    #                      NA     EP    WP    NI    SI    AU    SP(wrapped) SA→none  WP  nan
    assert got == ["NA", "EP", "WP", "NI", "SI", "AU", "SP", c.NO_BASIN, "WP", c.NO_BASIN]


def test_assign_tcrisk_basin_edges_half_open():
    # lon=100 is the SI/AU and NI/WP boundary; lat=0 is the hemisphere boundary
    assert list(c.assign_tcrisk_basin([100.0, 99.9, 100.0], [-10.0, -10.0, 0.0])) == ["AU", "SI", "WP"]


def test_categorize_codes():
    wind = [10, 18, 32.9, 33, 43, 50, 58, 70, 99, np.nan]
    assert list(c.categorize(wind)) == [0, 1, 1, 2, 3, 4, 5, 6, 6, 0]


def test_outcome_counts_sums():
    df = pd.DataFrame({f"n{k}": [k] for k in range(1, 7)})
    oc = c.outcome_counts(df).iloc[0]
    assert oc["total"] == 21 and oc["cat1"] == 2 and oc["cat5"] == 6
    assert oc["cat3plus"] == 4 + 5 + 6 and oc["cat4plus"] == 5 + 6


def test_gcm_of_and_members():
    assert c.gcm_of("MRI-ESM2-0_r3i1p1f1") == "MRI-ESM2-0"
    assert len(c.RUNS) == 8 and len(c.GCMS) == 4
