import json

from idd_tools.jobmon import inflate_cells

from idd_tc_mortality.cache import component_id
from idd_tc_mortality.evaluate.build_final_cells import parse_pack_costs
from idd_tc_mortality.evaluate.build_final_cost_probe import (
    build_probe,
    fit_tier_costs,
    tier_sizes,
)
from idd_tc_mortality.grid.build_final_specs_tailvariant import build_specs


def _manifest(tmp_path, vintage="v2000"):
    specs = build_specs(vintage)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({component_id(s): s for s in specs}))
    (tmp_path / "fold_assignments.parquet").write_bytes(b"")  # copied verbatim
    return p


def test_build_probe_two_sizes_per_tier_family_pure(tmp_path):
    p = _manifest(tmp_path)
    doc = build_probe(str(p), "tailvariant-v2000", tmp_path / "probe")
    tasks = doc["tasks"]
    # v2000: 5 families x 2 thresholds = 10 tiers x 2 sizes
    assert len(tasks) == 20
    for t in tasks:
        cells = inflate_cells(t["task_args"])
        assert len({c["tier"] for c in cells}) == 1   # family-pure
    # gpd_cens@0.95 probes use the small sizes
    sizes = {}
    for t in tasks:
        cells = inflate_cells(t["task_args"])
        sizes.setdefault(cells[0]["tier"], []).append(len(cells))
    assert sorted(sizes["gpd_cens@0.95"]) == [8, 16]
    assert sorted(sizes["gamma@0.70"]) == [128, 256]


def test_fit_tier_costs_two_point_slope():
    doc = {"tasks": [
        {"task_args": {"cells": [{"tier": "gamma@0.70"}] * 100}},
        {"task_args": {"cells": [{"tier": "gamma@0.70"}] * 200}},
        {"task_args": {"cells": [{"tier": "gpd_cens@0.95"}] * 8}},
        {"task_args": {"cells": [{"tier": "gpd_cens@0.95"}] * 16}},
    ]}
    elapsed = {0: 60 + 100 * 1.5, 1: 60 + 200 * 1.5,
               2: 60 + 8 * 45.0, 3: 60 + 16 * 45.0}
    marginals, startup = fit_tier_costs(doc, elapsed)
    assert abs(marginals["gamma@0.70"] - 1.5) < 1e-6
    assert abs(marginals["gpd_cens@0.95"] - 45.0) < 1e-6
    assert abs(startup - 60) < 1e-6


def test_parse_pack_costs_roundtrip():
    costs = parse_pack_costs("gamma@0.70:1.5, gpd_cens@0.95:45")
    assert costs == {"gamma@0.70": 1.5, "gpd_cens@0.95": 45.0}


def test_tier_sizes_defaults():
    s = tier_sizes(["gamma@0.70", "gpd_cens@0.95"])
    assert s["gamma@0.70"] == [128, 256]
    assert s["gpd_cens@0.95"] == [8, 16]
