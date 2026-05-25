"""HF TrainerCallback subclasses for metrics streaming + walltime safety."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Union

from transformers import TrainerCallback

from hf_train.output import MetricsJsonlWriter

# Keys HF Trainer emits in logs whose presence indicates an eval row vs train row.
_EVAL_KEY_HINTS = ("eval_loss", "eval_runtime", "eval_samples_per_second")


class MetricsJsonlCallback(TrainerCallback):
    """Appends one JSON row per log event to metrics.jsonl with phase=train|eval."""

    def __init__(self, jsonl_path: Union[str, Path]) -> None:
        self._writer = MetricsJsonlWriter(jsonl_path)

    def on_log(self, args, state, control, logs: Optional[dict[str, Any]] = None, **kwargs):
        if not logs:
            return
        phase = "eval" if any(k in logs for k in _EVAL_KEY_HINTS) else "train"
        row: dict[str, Any] = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "phase": phase,
        }
        # Drop None epoch from the minimal schema
        if row["epoch"] is None:
            del row["epoch"]
        row.update({k: v for k, v in logs.items() if v is not None})
        self._writer.write(row)


class WalltimeWatchdog(TrainerCallback):
    """Triggers a clean save+stop when elapsed wall-clock exceeds max_train_seconds."""

    def __init__(self, max_train_seconds: Optional[int]) -> None:
        self.max_train_seconds = max_train_seconds
        self._start_time: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self.max_train_seconds is None or self._start_time is None:
            return
        elapsed = time.monotonic() - self._start_time
        if elapsed >= self.max_train_seconds:
            control.should_save = True
            control.should_training_stop = True
