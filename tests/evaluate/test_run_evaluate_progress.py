import math

from idd_tc_mortality.evaluate.run_evaluate import _cells_progress_line


def test_progress_line_midway():
    line = _cells_progress_line(64, 5184, elapsed_s=32.0)
    assert line == "Cells progress: 64/5184 (1%), elapsed 32s, 2.00 cells/s, eta 2560s"


def test_progress_line_final_cell():
    line = _cells_progress_line(100, 100, elapsed_s=50.0)
    assert "100/100 (100%)" in line
    assert "eta 0s" in line


def test_progress_line_zero_elapsed_no_crash():
    line = _cells_progress_line(1, 10, elapsed_s=0.0)
    assert "0.00 cells/s" in line
    assert "eta nan" in line
    assert math.isnan(float("nan"))  # documents the sentinel choice


def test_nfs_safe_replace_swallows_phantom_enoent(tmp_path, monkeypatch):
    import os
    from idd_tc_mortality.evaluate.run_evaluate import _nfs_safe_replace

    dest = tmp_path / "out.parquet"
    dest.write_text("done")          # server already applied the rename

    def phantom(src, dst):
        raise FileNotFoundError(src)

    monkeypatch.setattr(os, "replace", phantom)
    _nfs_safe_replace(str(tmp_path / "gone.parquet.tmp"), dest)   # must not raise


def test_nfs_safe_replace_reraises_real_enoent(tmp_path, monkeypatch):
    import os

    import pytest
    from idd_tc_mortality.evaluate.run_evaluate import _nfs_safe_replace

    def phantom(src, dst):
        raise FileNotFoundError(src)

    monkeypatch.setattr(os, "replace", phantom)
    with pytest.raises(FileNotFoundError):
        # destination absent -> the rename genuinely failed -> re-raise
        _nfs_safe_replace(str(tmp_path / "gone.tmp"), tmp_path / "never_written.parquet")


def test_load_cell_chunks_resume_math(tmp_path):
    import pandas as pd

    from idd_tc_mortality.evaluate.run_evaluate import _chunk_path, _load_cell_chunks

    entries = tmp_path / "entries"
    entries.mkdir()
    pd.DataFrame([{"m": 1.0}, {"m": 2.0}]).to_parquet(_chunk_path(entries, 7, 0))
    pd.DataFrame([{"m": 3.0}]).to_parquet(_chunk_path(entries, 7, 1))
    # chunk 3 exists but chunk 2 missing -> only contiguous chunks count
    pd.DataFrame([{"m": 9.0}]).to_parquet(_chunk_path(entries, 7, 3))

    rows, start = _load_cell_chunks(entries, 7, chunk_size=256)
    assert start == 512                      # 2 contiguous chunks x 256
    assert [r["m"] for r in rows] == [1.0, 2.0, 3.0]

    rows0, start0 = _load_cell_chunks(entries, 99, chunk_size=256)
    assert (rows0, start0) == ([], 0)


def test_flush_model_predictions_one_file_per_config(tmp_path):
    import pandas as pd

    from idd_tc_mortality.evaluate.run_evaluate import _flush_model_predictions

    buf = {"mid1": [pd.DataFrame({"x": [1], "fold_tag": ["insample"]}),
                    pd.DataFrame({"x": [2], "fold_tag": ["oos_seed0"]})]}
    _flush_model_predictions(buf, tmp_path)
    assert buf == {}
    files = list(tmp_path.glob("*.parquet"))
    assert [f.name for f in files] == ["mid1_predictions.parquet"]
    assert len(pd.read_parquet(files[0])) == 2
