"""Config loading + resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from hf_train.config.defaults import apply_task_defaults
from hf_train.config.schema import (
    Config,
    DataConfig,
    EarlyStoppingConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)

__all__ = [
    "Config",
    "DataConfig",
    "EarlyStoppingConfig",
    "ModelConfig",
    "OutputConfig",
    "TrainingConfig",
    "load_config",
    "dump_config",
]


def load_config(path: Union[str, Path]) -> Config:
    """Load YAML config from `path`, apply task defaults, validate."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) or {}
    raw = apply_task_defaults(raw)
    return Config.model_validate(raw)


def dump_config(cfg: Config, path: Union[str, Path]) -> None:
    """Write the resolved config to `path` as YAML."""
    p = Path(path)
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
