"""
Tests for fit/run_component.py.

Covers:
  - Main command runs end-to-end for a valid spec and saves result.
  - Skips if result exists and --overwrite not passed.
  - Overwrites if --overwrite passed.
  - Raises UsageError if spec_id not in manifest.
  - Raises UsageError if manifest key does not match the spec hash.
  - Result written to output_dir is loadable via cache.load_result.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from idd_tc_mortality.cache import component_id, load_result, result_exists
from idd_tc_mortality.fit.run_component import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMBO = {"wind_speed": True, "sdi": False, "basin": False, "is_island": False}

_S1_SPEC = {
    "component":          "s1",
    "covariate_combo":    _COMBO,
    "threshold_quantile": None,
    "family":             None,
}

_BULK_SPEC = {
    "component":          "bulk",
    "covariate_combo":    _COMBO,
    "threshold_quantile": 0.75,
    "family":             "gamma",
}


def _make_df(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    exposed = rng.uniform(5_000, 200_000, n)
    wind = rng.normal(0, 1, n)
    death_rate = np.exp(-4.0 + 0.3 * wind) * rng.gamma(3.0, 1.0 / 3.0, n)
    deaths = np.maximum(np.round(death_rate * exposed), 0.0)
    # Ensure some non-zero deaths for meaningful fits.
    deaths[:20] = np.maximum(deaths[:20], 1.0)
    return pd.DataFrame({
        "deaths":     deaths,
        "exposed":    exposed,
        "wind_speed": wind,
        "sdi":        rng.uniform(0.3, 0.7, n),
        "basin":      "NA",
        "is_island":  0.0,
    })


def _write_manifest(path, specs: list[dict]) -> dict[str, dict]:
    manifest = {component_id(s): s for s in specs}
    with open(path, "w") as f:
        json.dump(manifest, f)
    return manifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df(tmp_path_factory):
    df = _make_df()
    p = tmp_path_factory.mktemp("data") / "data.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture()
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# End-to-end: successful fit
# ---------------------------------------------------------------------------

def test_run_s1_writes_result(runner, sample_df, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_S1_SPEC])
    cid = component_id(_S1_SPEC)

    result = runner.invoke(main, [
        "--spec-id",    cid,
        "--manifest",   str(manifest_path),
        "--data-path",  str(sample_df),
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code == 0, result.output
    assert result_exists(cid, tmp_path)


def test_run_bulk_gamma_writes_result(runner, sample_df, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_BULK_SPEC])
    cid = component_id(_BULK_SPEC)

    result = runner.invoke(main, [
        "--spec-id",    cid,
        "--manifest",   str(manifest_path),
        "--data-path",  str(sample_df),
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code == 0, result.output
    assert result_exists(cid, tmp_path)


def test_loaded_result_has_correct_family(runner, sample_df, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_S1_SPEC])
    cid = component_id(_S1_SPEC)

    runner.invoke(main, [
        "--spec-id",    cid,
        "--manifest",   str(manifest_path),
        "--data-path",  str(sample_df),
        "--output-dir", str(tmp_path),
    ])

    fit = load_result(cid, tmp_path)
    assert fit.family == "s1"


# ---------------------------------------------------------------------------
# Skip if already exists (no --overwrite)
# ---------------------------------------------------------------------------

def test_skip_if_exists(runner, sample_df, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_S1_SPEC])
    cid = component_id(_S1_SPEC)

    invoke_args = [
        "--spec-id",    cid,
        "--manifest",   str(manifest_path),
        "--data-path",  str(sample_df),
        "--output-dir", str(tmp_path),
    ]

    # First run: fits and saves.
    r1 = runner.invoke(main, invoke_args)
    assert r1.exit_code == 0, r1.output

    # Second run: should skip (result exists, no --overwrite).
    r2 = runner.invoke(main, invoke_args)
    assert r2.exit_code == 0, r2.output
    assert "skipping" in r2.output.lower()


# ---------------------------------------------------------------------------
# Overwrite
# ---------------------------------------------------------------------------

def test_overwrite_replaces_result(runner, sample_df, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_S1_SPEC])
    cid = component_id(_S1_SPEC)

    invoke_args = [
        "--spec-id",    cid,
        "--manifest",   str(manifest_path),
        "--data-path",  str(sample_df),
        "--output-dir", str(tmp_path),
    ]

    r1 = runner.invoke(main, invoke_args)
    assert r1.exit_code == 0, r1.output

    mtime1 = (tmp_path / f"{cid}.pkl").stat().st_mtime_ns

    r2 = runner.invoke(main, invoke_args + ["--overwrite"])
    assert r2.exit_code == 0, r2.output

    mtime2 = (tmp_path / f"{cid}.pkl").stat().st_mtime_ns
    assert mtime2 >= mtime1  # file was re-written


# ---------------------------------------------------------------------------
# Error: spec_id not in manifest
# ---------------------------------------------------------------------------

def test_missing_spec_id_errors(runner, sample_df, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_S1_SPEC])

    result = runner.invoke(main, [
        "--spec-id",    "a" * 32,
        "--manifest",   str(manifest_path),
        "--data-path",  str(sample_df),
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Error: manifest key does not match spec hash
# ---------------------------------------------------------------------------

def test_corrupt_manifest_key_errors(runner, sample_df, tmp_path):
    # Write a manifest where the key doesn't match the spec hash.
    corrupt = {"a" * 32: _S1_SPEC}
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(corrupt, f)

    result = runner.invoke(main, [
        "--spec-id",    "a" * 32,
        "--manifest",   str(manifest_path),
        "--data-path",  str(sample_df),
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Graceful non-convergence: fit raises → sentinel saved, exit 0
# ---------------------------------------------------------------------------

def test_fit_exception_saves_sentinel(runner, sample_df, tmp_path, monkeypatch):
    """When fit_one_component raises, a sentinel result is saved and the CLI exits 0."""
    from idd_tc_mortality.fit import run_component as rc_mod

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_S1_SPEC])
    cid = component_id(_S1_SPEC)

    monkeypatch.setattr(rc_mod, "fit_one_component", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated failure")))

    result = runner.invoke(main, [
        "--spec-id",    cid,
        "--manifest",   str(manifest_path),
        "--data-path",  str(sample_df),
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code == 0, result.output
    assert result_exists(cid, tmp_path)


def test_fit_exception_sentinel_has_converged_false(runner, sample_df, tmp_path, monkeypatch):
    """Sentinel result from a failed fit has converged=False."""
    from idd_tc_mortality.fit import run_component as rc_mod

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_S1_SPEC])
    cid = component_id(_S1_SPEC)

    monkeypatch.setattr(rc_mod, "fit_one_component", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("oops")))

    runner.invoke(main, [
        "--spec-id",    cid,
        "--manifest",   str(manifest_path),
        "--data-path",  str(sample_df),
        "--output-dir", str(tmp_path),
    ])

    fit = load_result(cid, tmp_path)
    assert fit.converged is False


def test_fit_exception_sentinel_has_error_meta(runner, sample_df, tmp_path, monkeypatch):
    """Sentinel result carries error message and type in meta."""
    from idd_tc_mortality.fit import run_component as rc_mod

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_S1_SPEC])
    cid = component_id(_S1_SPEC)

    monkeypatch.setattr(rc_mod, "fit_one_component", lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad input")))

    runner.invoke(main, [
        "--spec-id",    cid,
        "--manifest",   str(manifest_path),
        "--data-path",  str(sample_df),
        "--output-dir", str(tmp_path),
    ])

    fit = load_result(cid, tmp_path)
    assert fit.meta.get("error") == "bad input"
    assert fit.meta.get("error_type") == "ValueError"


def test_fit_exception_sentinel_has_empty_params(runner, sample_df, tmp_path, monkeypatch):
    """Sentinel result has empty params and fitted_values arrays."""
    from idd_tc_mortality.fit import run_component as rc_mod

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_S1_SPEC])
    cid = component_id(_S1_SPEC)

    monkeypatch.setattr(rc_mod, "fit_one_component", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("crash")))

    runner.invoke(main, [
        "--spec-id",    cid,
        "--manifest",   str(manifest_path),
        "--data-path",  str(sample_df),
        "--output-dir", str(tmp_path),
    ])

    fit = load_result(cid, tmp_path)
    assert len(fit.params) == 0
    assert len(fit.fitted_values) == 0
    assert fit.param_names == []
