"""
Deterministic component ID hashing and result persistence.

A component is identified by a spec dict describing the combination of
choices that produced it (threshold quantile, covariates, model family,
component type). The spec is hashed to a stable MD5 hex string so that
repeated runs with identical inputs resolve to the same files without
any central registry.

File layout under output_dir:
    {component_id}.pkl   — pickled FitResult (exact object roundtrip)
    {component_id}.json  — human-readable metadata for inspection

JSON writes are atomic (write to temp, rename). Pkl writes raise on
collision unless overwrite=True is explicitly passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

from idd_tc_mortality.distributions.base import FitResult


def model_id(
    s1_spec: dict[str, Any],
    s2_spec: dict[str, Any],
    bulk_spec: dict[str, Any],
    tail_spec: dict[str, Any],
) -> str:
    """Return a deterministic MD5 hex string for a four-component assembled model.

    The ID is derived from all four component spec dicts so that any difference
    in component, family, covariates, threshold, or fold_tag produces a distinct
    model ID.
    """
    canonical = json.dumps(
        {"s1": s1_spec, "s2": s2_spec, "bulk": bulk_spec, "tail": tail_spec},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.md5(canonical.encode()).hexdigest()


def component_id(spec: dict[str, Any]) -> str:
    """Return a deterministic MD5 hex string for a component spec dict.

    Key order in spec does not affect the result — the dict is sorted
    recursively before serialization.
    """
    canonical = json.dumps(spec, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(canonical.encode()).hexdigest()


def save_result(
    result: FitResult,
    spec: dict[str, Any],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save a FitResult to output_dir.

    Writes two files:
        {component_id}.pkl  — full FitResult via pickle
        {component_id}.json — human-readable metadata (atomic write)

    Raises
    ------
    FileExistsError
        If the pkl file already exists and overwrite=False.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cid = component_id(spec)
    pkl_path = output_dir / f"{cid}.pkl"
    json_path = output_dir / f"{cid}.json"

    if pkl_path.exists() and not overwrite:
        raise FileExistsError(
            f"Result already exists at {pkl_path}. "
            "Pass overwrite=True to replace it."
        )

    # Write pkl
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Write json atomically
    metadata = {
        "component_id": cid,
        "spec": spec,
        "family": result.family,
        "converged": result.converged,
        "n_obs": int(len(result.fitted_values)),
        "param_names": result.param_names,
        "meta": result.meta,
    }
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        os.replace(tmp_path, json_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_result(component_id_str: str, output_dir: str | Path) -> FitResult:
    """Load and return a FitResult from output_dir.

    Raises
    ------
    FileNotFoundError
        If no pkl file exists for this component_id.
    """
    pkl_path = Path(output_dir) / f"{component_id_str}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"No cached result found for component_id={component_id_str!r} "
            f"in {output_dir}. "
            "Run the fit first or check that output_dir is correct."
        )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def result_exists(component_id_str: str, output_dir: str | Path) -> bool:
    """Return True if a pkl file exists for this component_id, else False."""
    return (Path(output_dir) / f"{component_id_str}.pkl").exists()
