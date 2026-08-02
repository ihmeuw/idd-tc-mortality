import pytest

from idd_tc_mortality.evaluate.pred_layout import find_model_predictions


def test_finds_flat_layout(tmp_path):
    (tmp_path / "abc_insample_predictions.parquet").touch()
    p = find_model_predictions(tmp_path, "abc_insample_predictions.parquet")
    assert p == tmp_path / "abc_insample_predictions.parquet"


def test_finds_sharded_layout(tmp_path):
    shard = tmp_path / "task_00042"
    shard.mkdir()
    (shard / "abc_oos_seed1_predictions.parquet").touch()
    p = find_model_predictions(tmp_path, "abc_oos_seed1_predictions.parquet")
    assert p == shard / "abc_oos_seed1_predictions.parquet"


def test_flat_wins_over_shard(tmp_path):
    (tmp_path / "f.parquet").touch()
    shard = tmp_path / "task_00001"
    shard.mkdir()
    (shard / "f.parquet").touch()
    assert find_model_predictions(tmp_path, "f.parquet") == tmp_path / "f.parquet"


def test_missing_raises_with_both_locations(tmp_path):
    with pytest.raises(FileNotFoundError, match="flat layout.*sharded layout"):
        find_model_predictions(tmp_path, "nope.parquet")


def test_load_batched_format(tmp_path):
    import pandas as pd

    from idd_tc_mortality.evaluate.pred_layout import load_model_predictions

    df = pd.DataFrame({
        "predicted_rate": [1.0, 2.0, 3.0, 4.0],
        "exposed": [10.0] * 4,
        "fold_tag": ["insample", "insample", "oos_seed0", "oos_seed0"],
    })
    df.to_parquet(tmp_path / "abc_predictions.parquet")
    out = load_model_predictions(tmp_path, "abc", "oos_seed0")
    assert list(out["predicted_rate"]) == [3.0, 4.0]
    out2 = load_model_predictions(tmp_path, "abc", "insample",
                                  columns=["predicted_rate", "exposed"])
    assert list(out2.columns) == ["predicted_rate", "exposed"]
    assert len(out2) == 2


def test_load_prefers_per_fold_file(tmp_path):
    import pandas as pd

    from idd_tc_mortality.evaluate.pred_layout import load_model_predictions

    pd.DataFrame({"predicted_rate": [9.0], "fold_tag": ["insample"]}).to_parquet(
        tmp_path / "abc_insample_predictions.parquet")
    pd.DataFrame({"predicted_rate": [1.0], "fold_tag": ["insample"]}).to_parquet(
        tmp_path / "abc_predictions.parquet")
    out = load_model_predictions(tmp_path, "abc", "insample")
    assert list(out["predicted_rate"]) == [9.0]
