"""Output-directory writers: metrics.jsonl, results.json, config.resolved.yaml."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class RunDirNotEmpty(RuntimeError):
    """Raised when output.output_dir is non-empty and overwrite=False."""


def prepare_run_dir(run_dir: Union[str, Path], overwrite: bool) -> Path:
    """Ensure run_dir exists. Reject non-empty unless overwrite=True (then warn)."""
    p = Path(run_dir)
    if p.exists():
        if any(p.iterdir()) and not overwrite:
            raise RunDirNotEmpty(
                f"output.output_dir is non-empty: {p}. "
                "Set output.overwrite=true to write into it anyway."
            )
        if any(p.iterdir()) and overwrite:
            logger.warning("output_dir is non-empty; proceeding because overwrite=true: %s", p)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


def write_config_resolved(run_dir: Union[str, Path], config_dict: dict[str, Any]) -> Path:
    """Write the resolved config to run_dir/config.resolved.yaml. Returns the path."""
    path = Path(run_dir) / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(config_dict, sort_keys=False))
    return path


class MetricsJsonlWriter:
    """Appends one JSON object per line. Minimal schema: only the keys passed in."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str))
            f.write("\n")


def compute_best(
    trainer_state: dict[str, Any],
    metric_name: Optional[str],
    mode: str,
) -> tuple[Optional[str], str, Optional[dict[str, Any]]]:
    """Resolve best_checkpoint / best_method / best_metric from a trainer_state.json dict.

    Returns:
        (best_checkpoint, best_method, best_metric_dict_or_None)
    """
    if trainer_state.get("best_model_checkpoint"):
        best_value: Optional[float] = None
        if metric_name:
            history = trainer_state.get("log_history", [])
            values = [e[metric_name] for e in history if metric_name in e]
            if values:
                best_value = min(values) if mode == "min" else max(values)
        best_metric = (
            {"name": metric_name, "value": best_value, "mode": mode}
            if metric_name and best_value is not None
            else None
        )
        return trainer_state["best_model_checkpoint"], "early_stopping", best_metric

    # Fallback: latest by step
    history = trainer_state.get("log_history", [])
    if history:
        # We don't have a checkpoint path to claim. Return None for path.
        return None, "latest", None
    return None, "latest", None


def write_results_json(
    run_dir: Union[str, Path],
    status: str,
    run_name: Optional[str],
    task: str,
    stopped_by: str,
    total_steps: Optional[int] = None,
    total_epochs: Optional[float] = None,
    total_wall_seconds: Optional[float] = None,
    best_checkpoint: Optional[str] = None,
    best_method: Optional[str] = None,
    best_metric: Optional[dict[str, Any]] = None,
    final_metric: Optional[dict[str, Any]] = None,
    config_path: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Path:
    """Write run_dir/results.json. Minimal schema; nulls omitted."""
    payload: dict[str, Any] = {
        "status": status,
        "task": task,
        "stopped_by": stopped_by,
    }
    if run_name is not None:
        payload["run_name"] = run_name
    if total_steps is not None:
        payload["total_steps"] = total_steps
    if total_epochs is not None:
        payload["total_epochs"] = total_epochs
    if total_wall_seconds is not None:
        payload["total_wall_seconds"] = total_wall_seconds
    if best_checkpoint is not None:
        payload["best_checkpoint"] = best_checkpoint
    if best_method is not None:
        payload["best_method"] = best_method
    if best_metric is not None:
        payload["best_metric"] = best_metric
    if final_metric is not None:
        payload["final_metric"] = final_metric
    if config_path is not None:
        payload["config_path"] = config_path
    if error_type is not None:
        payload["error_type"] = error_type
    if error_message is not None:
        payload["error_message"] = error_message

    path = Path(run_dir) / "results.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
