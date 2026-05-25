from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_metrics_jsonl_appends_one_row_per_call(tmp_run_dir: Path) -> None:
    from hf_train.output import MetricsJsonlWriter
    w = MetricsJsonlWriter(tmp_run_dir / "metrics.jsonl")
    w.write({"step": 50, "phase": "train", "loss": 2.5})
    w.write({"step": 100, "phase": "train", "loss": 2.1})
    w.write({"step": 100, "phase": "eval", "eval_loss": 2.0})
    lines = (tmp_run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"step": 50, "phase": "train", "loss": 2.5}
    assert json.loads(lines[2]) == {"step": 100, "phase": "eval", "eval_loss": 2.0}


def test_metrics_jsonl_minimal_schema_no_null_keys(tmp_run_dir: Path) -> None:
    """Rows include only keys that apply — no nulls for absent metrics."""
    from hf_train.output import MetricsJsonlWriter
    w = MetricsJsonlWriter(tmp_run_dir / "metrics.jsonl")
    w.write({"step": 50, "phase": "train", "loss": 2.5})
    row = json.loads((tmp_run_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert "eval_loss" not in row
    assert "eval_runtime" not in row


def test_write_results_completed(tmp_run_dir: Path) -> None:
    from hf_train.output import write_results_json
    write_results_json(
        run_dir=tmp_run_dir,
        status="completed",
        run_name="test_run",
        task="causal_lm",
        stopped_by="max_steps",
        total_steps=100,
        total_epochs=1.0,
        total_wall_seconds=42.0,
        best_checkpoint=str(tmp_run_dir / "checkpoints" / "checkpoint-100"),
        best_method="latest",
        best_metric=None,
        final_metric={"name": "eval_loss", "value": 1.5},
    )
    data = json.loads((tmp_run_dir / "results.json").read_text())
    assert data["status"] == "completed"
    assert data["stopped_by"] == "max_steps"
    assert data["best_method"] == "latest"


def test_write_results_error(tmp_run_dir: Path) -> None:
    from hf_train.output import write_results_json
    write_results_json(
        run_dir=tmp_run_dir,
        status="error",
        run_name=None,
        task="causal_lm",
        stopped_by="error",
        error_type="data_load",
        error_message="dataset not found",
    )
    data = json.loads((tmp_run_dir / "results.json").read_text())
    assert data["status"] == "error"
    assert data["error_type"] == "data_load"
    assert data["error_message"] == "dataset not found"


def test_write_config_resolved_yaml(tmp_run_dir: Path) -> None:
    """write_config_resolved writes the YAML and returns its path."""
    from hf_train.output import write_config_resolved
    import yaml
    config_dict = {"task": "causal_lm", "training": {"max_steps": 100}}
    path = write_config_resolved(tmp_run_dir, config_dict)
    assert path == tmp_run_dir / "config.resolved.yaml"
    assert path.exists()
    loaded = yaml.safe_load(path.read_text())
    assert loaded == config_dict


def test_prepare_run_dir_creates_if_missing(tmp_path: Path) -> None:
    from hf_train.output import prepare_run_dir
    target = tmp_path / "does_not_exist"
    prepare_run_dir(target, overwrite=False)
    assert target.is_dir()


def test_prepare_run_dir_empty_existing_ok(tmp_path: Path) -> None:
    from hf_train.output import prepare_run_dir
    target = tmp_path / "empty_dir"
    target.mkdir()
    prepare_run_dir(target, overwrite=False)  # empty is fine


def test_prepare_run_dir_non_empty_no_overwrite_fails(tmp_path: Path) -> None:
    from hf_train.output import prepare_run_dir, RunDirNotEmpty
    target = tmp_path / "non_empty"
    target.mkdir()
    (target / "something.txt").write_text("x")
    with pytest.raises(RunDirNotEmpty):
        prepare_run_dir(target, overwrite=False)


def test_prepare_run_dir_non_empty_overwrite_ok(tmp_path: Path) -> None:
    from hf_train.output import prepare_run_dir
    target = tmp_path / "non_empty"
    target.mkdir()
    (target / "something.txt").write_text("x")
    prepare_run_dir(target, overwrite=True)  # warns, does not raise
    # existing file is preserved (we don't wipe)
    assert (target / "something.txt").exists()


def test_compute_best_from_trainer_state_uses_best_model_checkpoint(tmp_run_dir: Path) -> None:
    from hf_train.output import compute_best
    state = {
        "best_model_checkpoint": str(tmp_run_dir / "checkpoints" / "checkpoint-500"),
        "log_history": [
            {"step": 100, "loss": 1.0, "epoch": 0.5},
            {"step": 500, "eval_loss": 0.8, "epoch": 1.0},
            {"step": 1000, "eval_loss": 0.9, "epoch": 1.5},
        ],
    }
    best_ckpt, best_method, best_metric = compute_best(state, metric_name="eval_loss", mode="min")
    assert best_ckpt == str(tmp_run_dir / "checkpoints" / "checkpoint-500")
    assert best_method == "early_stopping"
    assert best_metric == {"name": "eval_loss", "value": 0.8, "mode": "min"}


def test_compute_best_fallback_to_latest(tmp_run_dir: Path) -> None:
    """No best_model_checkpoint and no tracked metric -> 'latest' method, no best_metric."""
    from hf_train.output import compute_best
    state = {
        "log_history": [{"step": 100, "loss": 1.0}],
        "global_step": 100,
    }
    best_ckpt, best_method, best_metric = compute_best(state, metric_name=None, mode="min")
    assert best_method == "latest"
    assert best_metric is None
