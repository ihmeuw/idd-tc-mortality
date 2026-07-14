"""Tests for build_evaluate_tasks partition modes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idd_tc_mortality.evaluate.build_evaluate_tasks import (
    _build_one_per_s1_s2_threshold,
    _build_one_per_s1_s2_threshold_coupled,
    _build_one_per_s1_threshold_coupled,
    _cov_key,
    _load_is_specs_by_component,
    _MODES,
)


def _make_manifest(tmp_path: Path) -> Path:
    """Synthetic manifest: 2 S1 (no threshold), 3 S2 / 4 bulk / 5 tail per
    threshold across 2 thresholds. Plus a handful of OOS specs that must be
    filtered out by the IS-only loader."""
    manifest = {}

    for i in range(2):
        cid = f"s1_{i:02d}"
        manifest[cid] = {"component": "s1", "fold_tag": "is"}

    for q_idx, q in enumerate([0.70, 0.80]):
        for j in range(3):
            cid = f"s2_q{q_idx}_{j:02d}"
            manifest[cid] = {"component": "s2", "fold_tag": "is", "threshold_quantile": q}
        for j in range(4):
            cid = f"bulk_q{q_idx}_{j:02d}"
            manifest[cid] = {"component": "bulk", "fold_tag": "is", "threshold_quantile": q}
        for j in range(5):
            cid = f"tail_q{q_idx}_{j:02d}"
            manifest[cid] = {"component": "tail", "fold_tag": "is", "threshold_quantile": q}

    manifest["s1_oos_should_be_excluded"] = {"component": "s1", "fold_tag": "s0_f0"}
    manifest["s2_oos_should_be_excluded"] = {
        "component": "s2", "fold_tag": "s0_f0", "threshold_quantile": 0.70,
    }

    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p


def test_one_per_s1_s2_threshold_task_count(tmp_path):
    by_component = _load_is_specs_by_component(str(_make_manifest(tmp_path)))
    tasks = _build_one_per_s1_s2_threshold(by_component)
    # 2 S1 × (3 S2 × 2 thresholds) = 12 tasks
    assert len(tasks) == 2 * (3 * 2)


def test_one_per_s1_s2_threshold_task_shape(tmp_path):
    by_component = _load_is_specs_by_component(str(_make_manifest(tmp_path)))
    tasks = _build_one_per_s1_s2_threshold(by_component)
    for t in tasks:
        assert len(t["s1_spec_ids"]) == 1
        assert len(t["s2_spec_ids"]) == 1
        # All 4 bulk and 5 tail specs at THIS threshold belong to the task.
        assert len(t["bulk_spec_ids"]) == 4
        assert len(t["tail_spec_ids"]) == 5
        # Bulk/tail thresholds match the task's threshold.
        q = t["threshold_quantile"]
        for bcid in t["bulk_spec_ids"]:
            assert f"_q{0 if q == 0.70 else 1}_" in bcid
        for tcid in t["tail_spec_ids"]:
            assert f"_q{0 if q == 0.70 else 1}_" in tcid


def test_one_per_s1_s2_threshold_task_indices_dense(tmp_path):
    by_component = _load_is_specs_by_component(str(_make_manifest(tmp_path)))
    tasks = _build_one_per_s1_s2_threshold(by_component)
    assert [t["task_index"] for t in tasks] == list(range(len(tasks)))


def test_one_per_s1_s2_threshold_excludes_oos_specs(tmp_path):
    by_component = _load_is_specs_by_component(str(_make_manifest(tmp_path)))
    tasks = _build_one_per_s1_s2_threshold(by_component)
    all_s1_ids = {x for t in tasks for x in t["s1_spec_ids"]}
    all_s2_ids = {x for t in tasks for x in t["s2_spec_ids"]}
    assert "s1_oos_should_be_excluded" not in all_s1_ids
    assert "s2_oos_should_be_excluded" not in all_s2_ids


def test_one_per_s1_s2_threshold_registered_in_modes():
    assert "one_per_s1_s2_threshold" in _MODES
    assert _MODES["one_per_s1_s2_threshold"] is _build_one_per_s1_s2_threshold


def test_one_per_s1_s2_threshold_s2_without_threshold_skipped(tmp_path):
    # An S2 spec missing threshold_quantile must not produce a task.
    manifest = {
        "s1_00":   {"component": "s1", "fold_tag": "is"},
        "s2_good": {"component": "s2", "fold_tag": "is", "threshold_quantile": 0.70},
        "s2_bad":  {"component": "s2", "fold_tag": "is"},
        "bulk_00": {"component": "bulk", "fold_tag": "is", "threshold_quantile": 0.70},
        "tail_00": {"component": "tail", "fold_tag": "is", "threshold_quantile": 0.70},
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    by_component = _load_is_specs_by_component(str(p))
    tasks = _build_one_per_s1_s2_threshold(by_component)
    assert len(tasks) == 1
    assert tasks[0]["s2_spec_ids"] == ["s2_good"]


# ---------------------------------------------------------------------------
# Coupled partitions
# ---------------------------------------------------------------------------

def _make_coupled_manifest(tmp_path: Path) -> Path:
    """Two cov_keys: covA = {basin: True, sdi: False}, covB = {basin: False, sdi: True}.
    One S1 spec per cov, then 2 S2 / 3 bulk / 4 tail specs per (cov, threshold)
    across 2 thresholds."""
    manifest = {}
    covs = {
        "A": {"basin": True,  "sdi": False},
        "B": {"basin": False, "sdi": True},
    }
    for cov_name, cov in covs.items():
        manifest[f"s1_{cov_name}"] = {
            "component": "s1", "fold_tag": "is", "covariate_combo": cov,
        }
        for q_idx, q in enumerate([0.70, 0.80]):
            for j in range(2):
                manifest[f"s2_{cov_name}_q{q_idx}_{j}"] = {
                    "component": "s2", "fold_tag": "is",
                    "threshold_quantile": q, "covariate_combo": cov,
                }
            for j in range(3):
                manifest[f"bulk_{cov_name}_q{q_idx}_{j}"] = {
                    "component": "bulk", "fold_tag": "is",
                    "threshold_quantile": q, "covariate_combo": cov,
                }
            for j in range(4):
                manifest[f"tail_{cov_name}_q{q_idx}_{j}"] = {
                    "component": "tail", "fold_tag": "is",
                    "threshold_quantile": q, "covariate_combo": cov,
                }
    p = tmp_path / "manifest_coupled.json"
    p.write_text(json.dumps(manifest))
    return p


def test_cov_key_stable_under_dict_ordering():
    a = {"basin": True, "sdi": False, "wind_speed": True, "is_island": False}
    b = {"wind_speed": True, "is_island": False, "basin": True, "sdi": False}
    assert _cov_key({"covariate_combo": a}) == _cov_key({"covariate_combo": b})


def test_one_per_s1_threshold_coupled_task_count(tmp_path):
    by_component = _load_is_specs_by_component(str(_make_coupled_manifest(tmp_path)))
    tasks = _build_one_per_s1_threshold_coupled(by_component)
    # 2 S1 covs × 2 thresholds = 4 tasks
    assert len(tasks) == 4


def test_one_per_s1_threshold_coupled_filters_by_cov(tmp_path):
    by_component = _load_is_specs_by_component(str(_make_coupled_manifest(tmp_path)))
    tasks = _build_one_per_s1_threshold_coupled(by_component)
    for t in tasks:
        # Every spec_id in the task carries the S1's cov letter (A or B).
        s1_cov_letter = t["s1_spec_ids"][0].split("_")[1]  # "s1_A" -> "A"
        for cid in t["s2_spec_ids"] + t["bulk_spec_ids"] + t["tail_spec_ids"]:
            assert f"_{s1_cov_letter}_" in cid


def test_one_per_s1_s2_threshold_coupled_task_count(tmp_path):
    by_component = _load_is_specs_by_component(str(_make_coupled_manifest(tmp_path)))
    tasks = _build_one_per_s1_s2_threshold_coupled(by_component)
    # 2 S1 covs × 2 S2 per cov per threshold × 2 thresholds = 8 tasks
    assert len(tasks) == 2 * 2 * 2


def test_one_per_s1_s2_threshold_coupled_filters_by_cov(tmp_path):
    by_component = _load_is_specs_by_component(str(_make_coupled_manifest(tmp_path)))
    tasks = _build_one_per_s1_s2_threshold_coupled(by_component)
    for t in tasks:
        s1_cov_letter = t["s1_spec_ids"][0].split("_")[1]
        # S2 in the same task must share cov; bulks/tails must share cov.
        assert f"_{s1_cov_letter}_" in t["s2_spec_ids"][0]
        for cid in t["bulk_spec_ids"] + t["tail_spec_ids"]:
            assert f"_{s1_cov_letter}_" in cid
    # Per-task config count = 3 bulks × 4 tails = 12 (uniform).
    for t in tasks:
        assert len(t["bulk_spec_ids"]) == 3
        assert len(t["tail_spec_ids"]) == 4


def test_one_per_s1_s2_threshold_coupled_no_mismatched_cov_pairs(tmp_path):
    # An S2 from covA must not be paired with an S1 from covB.
    by_component = _load_is_specs_by_component(str(_make_coupled_manifest(tmp_path)))
    tasks = _build_one_per_s1_s2_threshold_coupled(by_component)
    for t in tasks:
        s1_letter = t["s1_spec_ids"][0].split("_")[1]
        s2_letter = t["s2_spec_ids"][0].split("_")[1]
        assert s1_letter == s2_letter


def test_coupled_modes_registered_in_modes():
    assert _MODES["one_per_s1_threshold_coupled"]    is _build_one_per_s1_threshold_coupled
    assert _MODES["one_per_s1_s2_threshold_coupled"] is _build_one_per_s1_s2_threshold_coupled
