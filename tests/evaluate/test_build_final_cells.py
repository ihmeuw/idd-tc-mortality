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
    assert len(cells) == 256
    assert n_skipped == 0
    # every cell's four spec_ids resolve against the manifest
    manifest = json.loads(p.read_text())
    for c in cells:
        for axis in ("s1_spec_id", "s2_spec_id", "bulk_spec_id", "tail_spec_id"):
            assert c[axis] in manifest
    # all 256 are distinct full configs
    assert len({tuple(sorted(c.items())) for c in cells}) == 256
    # s2 tiers: the two s2 cov sets have 3 and 4 covariates on
    assert set(s2id_to_ncov.values()) == {3, 4}


def test_partition_is_four_tasks_covering_256(tmp_path):
    p = _write_manifest(tmp_path)
    out = tmp_path / "cells_manifest.json"
    doc = build_final_cells_manifest(str(p), str(out))
    tasks = doc["tasks"]
    assert len(tasks) == 4                       # 2 s1 × 2 s2
    total = sum(len(inflate_cells(t["task_args"])) for t in tasks)
    assert total == 256
    assert {t["task_features"]["s2_n_cov"] for t in tasks} == {3, 4}
    assert out.exists()
