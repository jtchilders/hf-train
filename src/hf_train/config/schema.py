"""Pydantic v2 schema for the YAML config."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


TaskName = Literal["causal_lm", "image_classification", "semantic_segmentation"]


class ModelConfig(BaseModel):
    source: Literal["hub", "from_config"] = "hub"
    name: str
    revision: str = "main"
    trust_remote_code: bool = False
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    num_labels: Optional[int] = None
    id2label: Optional[dict[int, str]] = None


class DataConfig(BaseModel):
    source: Literal["hub"] = "hub"
    name: str
    config: Optional[str] = None
    train_split: str = "train"
    # Set explicitly per-dataset (e.g. "validation", "test", or null to disable
    # eval). Defaulting to a specific split name silently fails for datasets
    # that don't have it; making this explicit forces the user to confirm what
    # the dataset actually exposes.
    eval_split: Optional[str] = None
    text_column: Optional[str] = None
    image_column: Optional[str] = None
    label_column: Optional[str] = None
    mask_column: Optional[str] = None
    max_seq_length: Optional[int] = None
    image_size: Optional[int] = None


class EarlyStoppingConfig(BaseModel):
    patience: int = 3
    metric: str = "eval_loss"
    mode: Literal["min", "max"] = "min"
    threshold: float = 0.0


class TrainingConfig(BaseModel):
    backend: Literal["auto", "xpu", "cuda", "cpu"] = "auto"
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    torch_compile: bool = False
    learning_rate: float = 5.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    max_epochs: Optional[float] = None
    max_steps: Optional[int] = None
    max_train_seconds: Optional[int] = None
    early_stopping: Optional[EarlyStoppingConfig] = None
    eval_strategy: Literal["no", "epoch", "steps"] = "epoch"
    eval_steps: Optional[int] = None
    logging_steps: int = 50
    save_strategy: Literal["no", "epoch", "steps"] = "epoch"
    save_steps: Optional[int] = None
    save_total_limit: int = 3
    seed: int = 42
    trainer_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_stopping_and_cadence(self) -> "TrainingConfig":
        if (self.max_epochs is None) == (self.max_steps is None):
            raise ValueError(
                "Exactly one of training.max_epochs or training.max_steps must be set"
            )
        if self.eval_strategy == "steps" and self.eval_steps is None:
            raise ValueError("training.eval_steps required when eval_strategy='steps'")
        if self.save_strategy == "steps" and self.save_steps is None:
            raise ValueError("training.save_steps required when save_strategy='steps'")
        return self


class OutputConfig(BaseModel):
    output_dir: str
    overwrite: bool = False
    run_name: Optional[str] = None
    notes: Optional[str] = None


class Config(BaseModel):
    task: TaskName
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    output: OutputConfig

    @model_validator(mode="after")
    def _validate_task_dependent(self) -> "Config":
        if self.task == "causal_lm" and not self.data.text_column:
            raise ValueError("data.text_column is required for task=causal_lm")
        if self.task in ("image_classification", "semantic_segmentation"):
            if self.model.num_labels is None:
                raise ValueError(f"model.num_labels is required for task={self.task}")
            if not self.data.image_column:
                raise ValueError(f"data.image_column is required for task={self.task}")
        if self.task == "image_classification" and not self.data.label_column:
            raise ValueError("data.label_column is required for task=image_classification")
        if self.task == "semantic_segmentation" and not self.data.mask_column:
            raise ValueError("data.mask_column is required for task=semantic_segmentation")
        # If the user asked for eval but didn't set eval_split, that's a contradiction
        # we'd rather catch here than at dataset-load time.
        if self.training.eval_strategy != "no" and self.data.eval_split is None:
            raise ValueError(
                "data.eval_split must be set when training.eval_strategy != 'no'. "
                "Set data.eval_split to the dataset's eval split name (e.g. "
                "'validation' or 'test'), or set training.eval_strategy='no' to "
                "disable eval."
            )
        return self
