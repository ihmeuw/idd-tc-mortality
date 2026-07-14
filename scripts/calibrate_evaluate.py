"""
Calibration probe for evaluate bundling.

Submits 3 tasks in parallel with n=1, 2, 4 groups each under generous resources.
After completion prints sacct RSS and runtime for each task, then computes:
    BUNDLE_SIZE = ceil((300 - a) / b)
    memory      = 2 × max(MaxRSS)
    runtime     = a + b × BUNDLE_SIZE  (with 1.5× safety)
where t(n) = a + b×n is fit by least-squares to the 3 (n, elapsed) points.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANIFEST_PATH   = "/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/01-preliminary/20260522/manifest.json"
RESULTS_DIR     = "/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/01-preliminary/20260522"
DATA_PATH       = "/mnt/team/idd/pub/idd_tc_mortality/00-data/20260522/input.parquet"
OUTPUT_DIR      = "/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/02-evaluate/20260522"
FOLD_ASSIGNMENTS = "/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/01-preliminary/20260522/fold_assignments.parquet"

CALIB_SIZES = [1, 2, 4]

# ---------------------------------------------------------------------------
# Load groups (same logic as orchestrator)
# ---------------------------------------------------------------------------
with open(MANIFEST_PATH) as f:
    _manifest = json.load(f)

s1_spec_ids: set[str] = set()
thresholds: set[float] = set()
for cid, spec in _manifest.items():
    if spec.get("fold_tag", "is") != "is":
        continue
    if spec["component"] == "s1":
        s1_spec_ids.add(cid)
    q = spec.get("threshold_quantile")
    if q is not None:
        thresholds.add(q)

groups = [(s1_id, q) for s1_id in sorted(s1_spec_ids) for q in sorted(thresholds)]
print(f"Total groups: {len(groups)}")

# ---------------------------------------------------------------------------
# Write bundle files
# ---------------------------------------------------------------------------
calib_dir = Path(OUTPUT_DIR) / ".calibration"
calib_dir.mkdir(parents=True, exist_ok=True)

bundle_paths: list[Path] = []
for i, n in enumerate(CALIB_SIZES):
    chunk = groups[:n]
    bp = calib_dir / f"calib_{i:02d}_n{n}.json"
    bp.write_text(json.dumps([[s, q] for s, q in chunk]))
    bundle_paths.append(bp)
    print(f"  bundle {i}: {n} group(s) → {bp.name}")

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
from idd_tools.jobmon import (
    Task, TaskManifest, TaskTemplateSpec,
    collect_workflow, submit_with_manifest,
)

manifest = TaskManifest(
    workflow_name="evaluate-calibration-20260522",
    tasks=[
        Task(
            index=i,
            task_id=f"calib_n{n}",
            task_template="evaluate_worker_bundle",
            task_args={
                "specs_path":            MANIFEST_PATH,
                "results_dir":           RESULTS_DIR,
                "data_path":             DATA_PATH,
                "output_dir":            str(calib_dir),
                "fold_assignments_path": FOLD_ASSIGNMENTS,
                "bundle_file":           str(bundle_paths[i]),
                "bundle_index":          i,
            },
            task_features={},
            depends_on=[],
        )
        for i, n in enumerate(CALIB_SIZES)
    ],
)

templates = {
    "evaluate_worker_bundle": TaskTemplateSpec(
        command_template=(
            "run-evaluate"
            " --specs-path {specs_path}"
            " --results-dir {results_dir}"
            " --data-path {data_path}"
            " --output-dir {output_dir}"
            " --fold-assignments {fold_assignments_path}"
            " --bundle-file {bundle_file}"
            " --bundle-index {bundle_index}"
        ),
        node_args=["bundle_index"],
        task_args=["specs_path", "results_dir", "data_path", "output_dir",
                   "fold_assignments_path", "bundle_file"],
        op_args=[],
    ),
}

print("\nSubmitting 3 calibration tasks (4G / 1h)...")
result = submit_with_manifest(
    manifest,
    output_dir=calib_dir,
    templates=templates,
    resources={"memory": "4G", "runtime": "1h"},
    tool_name="idd-tc-mortality",
    cluster_name="slurm",
    project="proj_rapidresponse",
    queue="all.q",
    log_method=print,
)

print(f"\nWorkflow {result.workflow_id} finished with status {result.status!r}")

# ---------------------------------------------------------------------------
# Sacct + fit
# ---------------------------------------------------------------------------
import numpy as np

df = collect_workflow(result.workflow_id)
print("\nSacct results:")
print(df[["job_name", "elapsed_seconds", "max_rss_gib"]].to_string(index=False))

ns   = np.array(CALIB_SIZES, dtype=float)
rts  = np.array([df.loc[df["job_name"].str.endswith(f"n{n}"), "elapsed_seconds"].iloc[0]
                 for n in CALIB_SIZES], dtype=float)
rsss = np.array([df.loc[df["job_name"].str.endswith(f"n{n}"), "max_rss_gib"].iloc[0]
                 for n in CALIB_SIZES], dtype=float)

# Least-squares fit of t(n) = a + b*n
A = np.column_stack([np.ones_like(ns), ns])
a, b = np.linalg.lstsq(A, rts, rcond=None)[0]
print(f"\nFit: t(n) = {a:.1f} + {b:.1f}×n  (load={a:.0f}s, marginal={b:.1f}s/group)")

bundle_size = math.ceil((300 - a) / b) if b > 0 else 999
max_rss     = float(rsss.max())
rt_raw      = a + b * bundle_size
runtime_s   = rt_raw * 1.5

print(f"\nRecommended settings:")
print(f"  BUNDLE_SIZE = {bundle_size}")
print(f"  memory      = {2 * max_rss:.2f} GiB  → set to {math.ceil(2 * max_rss)}G")
print(f"  runtime     = {runtime_s:.0f}s ({runtime_s/60:.1f} min) → set to {math.ceil(runtime_s/60)}m")
