from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock


def test_metrics_jsonl_callback_appends_train_row(tmp_run_dir: Path) -> None:
    from hf_train.train.callbacks import MetricsJsonlCallback
    cb = MetricsJsonlCallback(tmp_run_dir / "metrics.jsonl")
    args = MagicMock()
    state = MagicMock(global_step=50, epoch=0.5)
    control = MagicMock()
    cb.on_log(args, state, control, logs={"loss": 2.1, "learning_rate": 1e-4})
    lines = (tmp_run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["step"] == 50
    assert row["epoch"] == 0.5
    assert row["phase"] == "train"
    assert row["loss"] == 2.1
    assert row["learning_rate"] == 1e-4


def test_metrics_jsonl_callback_appends_eval_row(tmp_run_dir: Path) -> None:
    from hf_train.train.callbacks import MetricsJsonlCallback
    cb = MetricsJsonlCallback(tmp_run_dir / "metrics.jsonl")
    args = MagicMock()
    state = MagicMock(global_step=100, epoch=1.0)
    control = MagicMock()
    cb.on_log(args, state, control, logs={"eval_loss": 1.8, "eval_runtime": 4.0})
    row = json.loads((tmp_run_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert row["phase"] == "eval"
    assert row["eval_loss"] == 1.8


def test_walltime_watchdog_triggers_stop_when_elapsed_exceeds_budget() -> None:
    from hf_train.train.callbacks import WalltimeWatchdog
    cb = WalltimeWatchdog(max_train_seconds=0)  # immediately expired
    args = MagicMock()
    state = MagicMock()
    control = MagicMock(should_save=False, should_training_stop=False)
    cb.on_train_begin(args, state, control)
    cb.on_step_end(args, state, control)
    assert control.should_save is True
    assert control.should_training_stop is True


def test_walltime_watchdog_no_trigger_when_unset() -> None:
    from hf_train.train.callbacks import WalltimeWatchdog
    cb = WalltimeWatchdog(max_train_seconds=None)
    args = MagicMock()
    state = MagicMock()
    control = MagicMock(should_save=False, should_training_stop=False)
    cb.on_train_begin(args, state, control)
    cb.on_step_end(args, state, control)
    assert control.should_save is False
    assert control.should_training_stop is False
