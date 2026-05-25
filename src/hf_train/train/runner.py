"""Orchestrates the full training run.

The runner is thin: it picks a handler, picks a backend, builds a Trainer,
and runs it. All real work happens in handlers / backend / Trainer / callbacks.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from hf_train.backend import Backend, get_backend
from hf_train.config import Config
from hf_train.output import (
    compute_best,
    prepare_run_dir,
    write_config_resolved,
    write_results_json,
)
from hf_train.tasks import get_handler
from hf_train.train.callbacks import MetricsJsonlCallback, WalltimeWatchdog

logger = logging.getLogger(__name__)


class DataLoadError(RuntimeError):
    pass


class ModelLoadError(RuntimeError):
    pass


def run(config: Config) -> int:
    """Execute the run described by config. Returns the exit code.

    All errors are caught and surfaced via results.json + structured exit codes.
    """
    run_dir = prepare_run_dir(config.output.output_dir, overwrite=config.output.overwrite)
    write_config_resolved(run_dir, config.model_dump(mode="json"))

    # File-based logging (separate from stdout/stderr captured by ensemble_launcher).
    # We explicitly set both the handler level AND the root logger level so train.log
    # captures INFO from our package regardless of how the caller configured logging.
    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    file_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    if root_logger.level == logging.WARNING or root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    backend: Optional[Backend] = None
    try:
        backend = get_backend(config.training.backend)
        backend.setup_env()
        logger.info("Backend selected: %s (%s)", backend.device, backend.capabilities())
    except RuntimeError as e:
        write_results_json(
            run_dir=run_dir, status="error", run_name=config.output.run_name,
            task=config.task, stopped_by="error",
            error_type="backend_init", error_message=str(e),
        )
        return 2

    start = time.monotonic()
    try:
        handler = get_handler(config.task)
        handler.validate_config(config)

        try:
            preprocessor = handler.prepare_preprocessor(
                config.model.name, trust_remote_code=config.model.trust_remote_code,
            )
            train_ds = handler.prepare_dataset(
                config.data, preprocessor, split=config.data.train_split,
            )
            eval_ds = (
                handler.prepare_dataset(config.data, preprocessor, split=config.data.eval_split)
                if config.data.eval_split else None
            )
        except Exception as e:
            raise DataLoadError(str(e)) from e

        try:
            model = handler.prepare_model(config.model, num_labels=config.model.num_labels)
        except Exception as e:
            raise ModelLoadError(str(e)) from e

        # Backend-specific optimization (ipex.optimize on XPU; no-op elsewhere).
        # HF Trainer reconstructs the optimizer internally, so we pass model only.
        if backend.device == "xpu":
            model, _ = backend.optimize(model, None, dtype=config.training.precision)

        args = _build_training_arguments(config, run_dir)
        callbacks: list[Any] = [
            MetricsJsonlCallback(run_dir / "metrics.jsonl"),
            WalltimeWatchdog(config.training.max_train_seconds),
        ]
        if config.training.early_stopping is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config.training.early_stopping.patience,
                    early_stopping_threshold=config.training.early_stopping.threshold,
                )
            )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=handler.make_collator(preprocessor),
            compute_metrics=handler.make_compute_metrics(),
            callbacks=callbacks,
        )

        train_output = trainer.train()
        elapsed = time.monotonic() - start

        stopped_by = _classify_stopped_by(config, train_output, callbacks)
        state_dict = {
            "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
            "log_history": getattr(trainer.state, "log_history", []),
            "global_step": getattr(trainer.state, "global_step", 0),
        }
        metric_name = (
            config.training.early_stopping.metric if config.training.early_stopping else None
        )
        mode = config.training.early_stopping.mode if config.training.early_stopping else "min"
        best_ckpt, best_method, best_metric = compute_best(state_dict, metric_name, mode)

        final_metric = None
        if state_dict["log_history"]:
            last = state_dict["log_history"][-1]
            if "eval_loss" in last:
                final_metric = {"name": "eval_loss", "value": last["eval_loss"]}

        write_results_json(
            run_dir=run_dir,
            status="completed",
            run_name=config.output.run_name,
            task=config.task,
            stopped_by=stopped_by,
            total_steps=state_dict["global_step"],
            total_epochs=getattr(trainer.state, "epoch", None),
            total_wall_seconds=elapsed,
            best_checkpoint=best_ckpt,
            best_method=best_method,
            best_metric=best_metric,
            final_metric=final_metric,
            config_path=str(run_dir / "config.resolved.yaml"),
        )
        return 0

    except DataLoadError as e:
        logger.exception("Data load failed")
        write_results_json(
            run_dir=run_dir, status="error", run_name=config.output.run_name,
            task=config.task, stopped_by="error",
            error_type="data_load", error_message=str(e),
        )
        return 3
    except ModelLoadError as e:
        logger.exception("Model load failed")
        write_results_json(
            run_dir=run_dir, status="error", run_name=config.output.run_name,
            task=config.task, stopped_by="error",
            error_type="model_load", error_message=str(e),
        )
        return 4
    except KeyboardInterrupt:
        logger.warning("Interrupted by signal")
        write_results_json(
            run_dir=run_dir, status="error", run_name=config.output.run_name,
            task=config.task, stopped_by="error",
            error_type="interrupted", error_message="SIGINT/SIGTERM received",
        )
        return 130
    except Exception as e:
        logger.exception("Training crashed")
        write_results_json(
            run_dir=run_dir, status="error", run_name=config.output.run_name,
            task=config.task, stopped_by="error",
            error_type="training", error_message=f"{type(e).__name__}: {e}",
        )
        return 5


def _build_training_arguments(config: Config, run_dir: Path) -> TrainingArguments:
    t = config.training
    kwargs: dict[str, Any] = {
        "output_dir": str(run_dir / "checkpoints"),
        "overwrite_output_dir": True,
        "learning_rate": t.learning_rate,
        "weight_decay": t.weight_decay,
        "warmup_ratio": t.warmup_ratio,
        "lr_scheduler_type": t.lr_scheduler_type,
        "per_device_train_batch_size": t.per_device_train_batch_size,
        "per_device_eval_batch_size": t.per_device_eval_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "gradient_checkpointing": t.gradient_checkpointing,
        "logging_steps": t.logging_steps,
        "save_strategy": t.save_strategy,
        "save_steps": t.save_steps or 500,
        "save_total_limit": t.save_total_limit,
        "eval_strategy": t.eval_strategy,
        "eval_steps": t.eval_steps or 500,
        "seed": t.seed,
        "bf16": t.precision == "bf16",
        "fp16": t.precision == "fp16",
        "torch_compile": t.torch_compile,
        "load_best_model_at_end": t.early_stopping is not None,
        "metric_for_best_model": (
            t.early_stopping.metric if t.early_stopping else None
        ),
        "greater_is_better": (
            t.early_stopping.mode == "max" if t.early_stopping else None
        ),
        "report_to": [],  # default to no reporters; user opts in via trainer_kwargs
    }
    if t.max_steps is not None:
        kwargs["max_steps"] = t.max_steps
    else:
        kwargs["num_train_epochs"] = t.max_epochs

    # User escape hatch — overrides anything above
    kwargs.update(t.trainer_kwargs)

    return TrainingArguments(**kwargs)


def _classify_stopped_by(config: Config, train_output: Any, callbacks: list[Any]) -> str:
    """Determine which stopping criterion fired."""
    for cb in callbacks:
        if isinstance(cb, WalltimeWatchdog) and cb.fired:
            return "max_train_seconds"
    if config.training.early_stopping is not None:
        # EarlyStopping sets should_training_stop=True when triggered.
        # If we got here cleanly under max_steps/max_epochs, we can't perfectly
        # distinguish — prefer early_stopping if it's configured AND we stopped
        # before reaching max.
        if config.training.max_steps and train_output.global_step < config.training.max_steps:
            return "early_stopping"
        if config.training.max_epochs and (
            train_output.metrics.get("epoch", 0) < config.training.max_epochs
        ):
            return "early_stopping"
    return "max_steps" if config.training.max_steps else "max_epochs"
