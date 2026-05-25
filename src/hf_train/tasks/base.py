"""Abstract TaskHandler interface — implemented per HF task type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class TaskHandler(ABC):
    """One concrete handler per HF task type. Registered in tasks/__init__.py.

    The runner does not know about specific tasks; it only invokes this interface.
    Adding a new task means writing one tasks/<name>.py and adding one HANDLERS entry.
    """

    task_name: str          # e.g. "causal_lm"
    auto_model_class: type  # e.g. AutoModelForCausalLM

    @abstractmethod
    def prepare_preprocessor(self, model_name: str, trust_remote_code: bool = False) -> Any:
        """Return the tokenizer or image_processor appropriate for the task."""

    @abstractmethod
    def prepare_model(self, model_cfg: Any, num_labels: Optional[int] = None) -> Any:
        """Load (hub) or build (from_config) the model."""

    @abstractmethod
    def prepare_dataset(self, data_cfg: Any, preprocessor: Any, split: str) -> Any:
        """Load + preprocess a dataset split. Returns HF Dataset ready for the collator."""

    @abstractmethod
    def make_collator(self, preprocessor: Any) -> Callable:
        """Per-batch collation logic."""

    def make_compute_metrics(self) -> Optional[Callable]:
        """Optional. Returns fn(EvalPrediction) -> dict, or None for eval_loss only."""
        return None

    def validate_config(self, config: Any) -> None:
        """Optional per-task validation beyond schema-level checks."""
        pass
