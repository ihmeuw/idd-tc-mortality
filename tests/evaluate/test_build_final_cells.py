"""Unit tests for the final-grid cell enumeration + partition."""

import json

from idd_tools.jobmon import inflate_cells

from idd_tc_mortality.cache import component_id
from idd_tc_mortality.evaluate.build_final_cells import (
    build_final_cells_manifest,
    enumerate_final_cells,
)
from idd_tc_mortality.grid.build_final_specs import build_specs


def _write_manifest(tmp_path):
    """Manifest built exactly as run-evaluate-orchestrate --manifest-only does."""
    specs = build_specs()
    manifest = {component_id(s): s for s in specs}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p


def test_enumerate_final_cells_count(tmp_path):
    p = _write_manifest(tmp_path)
    cells, s2id_to_ncov, n_skipped = enumerate_final_cells(str(p))
    assert len(cells) == 4608
    assert n_skipped == 0
    # every cell's four spec_ids resolve against the manifest
    manifest = json.loads(p.read_text())
    for c in cells:
        for axis in ("s1_spec_id", "s2_spec_id", "bulk_spec_id", "tail_spec_id"):
            assert c[axis] in manifest
    # all 4,608 are distinct full configs
    assert len({tuple(sorted(c.items())) for c in cells}) == 4608
    # s2 tiers: the two s2 cov sets have 3 and 4 covariates on
    assert set(s2id_to_ncov.values()) == {3, 4}


def test_partition_is_twelve_tasks_covering_4608(tmp_path):
    p = _write_manifest(tmp_path)
    out = tmp_path / "cells_manifest.json"
    doc = build_final_cells_manifest(str(p), str(out))
    tasks = doc["tasks"]
    assert len(tasks) == 12                      # 2 s1 × (2 s2 cov × 3 thresholds)
    total = sum(len(inflate_cells(t["task_args"])) for t in tasks)
    assert total == 4608
    assert {t["task_features"]["s2_n_cov"] for t in tasks} == {3, 4}
    # one (s1, s2) group per task → 384 cells each
    assert all(len(inflate_cells(t["task_args"])) == 384 for t in tasks)
    assert out.exists()


def _write_tailvariant_manifest(tmp_path, vintage):
    from idd_tc_mortality.grid.build_final_specs_tailvariant import build_specs as tv_specs
    specs = tv_specs(vintage)
    manifest = {component_id(s): s for s in specs}
    p = tmp_path / f"manifest_{vintage}.json"
    p.write_text(json.dumps(manifest))
    return p


def test_tailvariant_grids_enumerate_and_pack(tmp_path):
    from idd_tc_mortality.grid.build_final_specs_tailvariant import n_configs

    for vintage in ("v1985", "v2000"):
        p = _write_tailvariant_manifest(tmp_path, vintage)
        out = tmp_path / f"cells_{vintage}.json"
        doc = build_final_cells_manifest(
            str(p), str(out),
            grid=f"tailvariant-{vintage}",
            pack_target_s=600.0,
        )
        total = sum(len(inflate_cells(t["task_args"])) for t in doc["tasks"])
        assert total == n_configs(vintage)     # nothing dropped or duplicated
        assert len(doc["tasks"]) < total       # packing produced multi-cell tasks
