"""Tests for build_refined_cells — the structure-C (s1 free, s2 ⊇ bulk ⊇ tail)
cell enumeration and partition.

The manifest fixture is built from the real refined spec list
(build_refined_specs_post2000.build_specs) keyed by component_id, so the
cells' spec_ids resolve exactly as they do in production.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idd_tc_mortality.cache import component_id
from idd_tc_mortality.grid.build_refined_specs_post2000 import build_specs
from idd_tc_mortality.evaluate.build_refined_cells import (
    _cov_n_on,
    build_refined_cells_manifest,
    enumerate_refined_cells,
)

THRESHOLDS = [0.70, 0.85]

# Expected structure-C totals on the 4-axis / {0.70,0.85} refined grid.
EXPECTED_TOTAL_CELLS = 114_688
EXPECTED_N_TASKS = 512
# {|s2_cov|: (n_tasks, cells_per_task)} — cells_per_task = 14 × 3^|s2_cov|.
EXPECTED_TIERS = {0: (32, 14), 1: (128, 42), 2: (192, 126), 3: (128, 378), 4: (32, 1134)}


def _on_axes(cov: dict[str, bool]) -> frozenset[str]:
    return frozenset(axis for axis, on in cov.items() if on)


@pytest.fixture
def manifest_path(tmp_path: Path) -> str:
    """Write a manifest.json built from the real refined IS spec list."""
    specs = build_specs(THRESHOLDS)
    manifest = {component_id(s): s for s in specs}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return str(p)


@pytest.fixture
def manifest_lookup(manifest_path: str) -> dict[str, dict]:
    """component_id -> spec, for resolving cells back to their covariate combos."""
    with open(manifest_path) as f:
        return json.load(f)


def test_cov_n_on():
    assert _cov_n_on({"a": True, "b": False, "c": True}) == 2
    assert _cov_n_on({"a": False, "b": False}) == 0


def test_enumerate_count_and_no_skips(manifest_path):
    cells, s2id_to_ncov, n_skipped = enumerate_refined_cells(manifest_path)
    assert len(cells) == EXPECTED_TOTAL_CELLS
    assert n_skipped == 0
    # Every s2 spec_id maps to a |s2_cov| in 0..4.
    assert set(s2id_to_ncov.values()) <= {0, 1, 2, 3, 4}


def test_every_cell_has_exactly_the_axis_keys(manifest_path):
    cells, _, _ = enumerate_refined_cells(manifest_path)
    expected = {"s1_spec_id", "s2_spec_id", "bulk_spec_id", "tail_spec_id"}
    # build_hierarchical_cellset requires this; assert it directly.
    assert all(set(c.keys()) == expected for c in cells)


def test_nesting_invariant_tail_subset_bulk_subset_s2(manifest_path, manifest_lookup):
    """Every emitted cell must satisfy tail_cov ⊆ bulk_cov ⊆ s2_cov."""
    cells, _, _ = enumerate_refined_cells(manifest_path)
    for c in cells:
        s2_on   = _on_axes(manifest_lookup[c["s2_spec_id"]]["covariate_combo"])
        bulk_on = _on_axes(manifest_lookup[c["bulk_spec_id"]]["covariate_combo"])
        tail_on = _on_axes(manifest_lookup[c["tail_spec_id"]]["covariate_combo"])
        assert bulk_on <= s2_on, f"bulk {bulk_on} ⊄ s2 {s2_on}"
        assert tail_on <= bulk_on, f"tail {tail_on} ⊄ bulk {bulk_on}"


def test_s1_is_independent_of_s2(manifest_path, manifest_lookup):
    """s1_cov must NOT be constrained to nest with s2_cov: there must exist a
    cell where s1 has an axis s2 lacks AND s2 has an axis s1 lacks (neither is
    a subset of the other) — proving s1 is free, not part of the chain."""
    cells, _, _ = enumerate_refined_cells(manifest_path)
    found_incomparable = False
    for c in cells:
        s1_on = _on_axes(manifest_lookup[c["s1_spec_id"]]["covariate_combo"])
        s2_on = _on_axes(manifest_lookup[c["s2_spec_id"]]["covariate_combo"])
        if not (s1_on <= s2_on) and not (s2_on <= s1_on):
            found_incomparable = True
            break
    assert found_incomparable, "s1_cov appears nested with s2_cov — s1 is not free"


def test_all_16_s1_covs_appear_with_a_fixed_s2(manifest_path, manifest_lookup):
    """For a fixed s2_cov, all 16 s1_covs should appear — full independence."""
    cells, _, _ = enumerate_refined_cells(manifest_path)
    # Pick the empty s2_cov; collect the distinct s1_covs paired with it.
    target_s2 = None
    s1_covs_with_target = set()
    for c in cells:
        s2_on = _on_axes(manifest_lookup[c["s2_spec_id"]]["covariate_combo"])
        if s2_on == frozenset():
            target_s2 = c["s2_spec_id"]
        if c["s2_spec_id"] == target_s2 or s2_on == frozenset():
            s1_covs_with_target.add(
                _on_axes(manifest_lookup[c["s1_spec_id"]]["covariate_combo"])
            )
    assert len(s1_covs_with_target) == 16


def test_partition_task_count_and_tiers(manifest_path, tmp_path):
    out = tmp_path / "cells_manifest.json"
    build_refined_cells_manifest(manifest_path, out, workflow_name="test")
    doc = json.loads(out.read_text())
    tasks = doc["tasks"]
    assert len(tasks) == EXPECTED_N_TASKS

    total_cells = sum(len(t["task_args"]["cells"]) for t in tasks)
    assert total_cells == EXPECTED_TOTAL_CELLS

    # Per-|s2_cov| tier: task count + cells/task.
    tier_task_counts: dict[int, int] = {}
    for t in tasks:
        k = t["task_features"]["s2_n_cov"]
        tier_task_counts[k] = tier_task_counts.get(k, 0) + 1
        assert len(t["task_args"]["cells"]) == EXPECTED_TIERS[k][1]
    assert tier_task_counts == {k: v[0] for k, v in EXPECTED_TIERS.items()}


def test_every_task_groups_one_s1_s2_pair(manifest_path, tmp_path):
    """rectangular_partition(fix=[s1_spec_id, s2_spec_id]) → every cell in a
    task shares the same (s1_spec_id, s2_spec_id)."""
    out = tmp_path / "cells_manifest.json"
    build_refined_cells_manifest(manifest_path, out, workflow_name="test")
    doc = json.loads(out.read_text())
    for t in doc["tasks"]:
        cells = t["task_args"]["cells"]
        s1_ids = {c["s1_spec_id"] for c in cells}
        s2_ids = {c["s2_spec_id"] for c in cells}
        assert len(s1_ids) == 1 and len(s2_ids) == 1


def test_max_per_task_splits_fat_groups(manifest_path, tmp_path):
    """With max_per_task=100, the 1,134-cell |s2_cov|=4 groups must split into
    multiple tasks, none exceeding the cap; total cells unchanged."""
    out = tmp_path / "cells_manifest.json"
    build_refined_cells_manifest(manifest_path, out, workflow_name="test", max_per_task=100)
    doc = json.loads(out.read_text())
    tasks = doc["tasks"]
    assert all(len(t["task_args"]["cells"]) <= 100 for t in tasks)
    assert sum(len(t["task_args"]["cells"]) for t in tasks) == EXPECTED_TOTAL_CELLS
    assert len(tasks) > EXPECTED_N_TASKS  # fat groups were chunked
