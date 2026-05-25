"""Task handler registry."""

from __future__ import annotations

from hf_train.tasks.base import TaskHandler

__all__ = ["TaskHandler", "HANDLERS", "get_handler"]

# Populated lazily to avoid importing heavy ML libraries when the package
# is just being introspected (e.g., `hf-train --help`).
HANDLERS: dict[str, type[TaskHandler]] = {}


def _register_handlers() -> None:
    """Import all concrete handlers and populate HANDLERS."""
    global HANDLERS
    if HANDLERS:
        return
    from hf_train.tasks.causal_lm import CausalLMHandler
    from hf_train.tasks.image_classification import ImageClassificationHandler
    from hf_train.tasks.semantic_segmentation import SemanticSegmentationHandler

    HANDLERS = {
        "causal_lm": CausalLMHandler,
        "image_classification": ImageClassificationHandler,
        "semantic_segmentation": SemanticSegmentationHandler,
    }


def get_handler(task_name: str) -> TaskHandler:
    """Instantiate the handler for the given task. Raises if unknown."""
    _register_handlers()
    if task_name not in HANDLERS:
        raise ValueError(
            f"Unknown task: {task_name!r}. Available: {sorted(HANDLERS)}"
        )
    return HANDLERS[task_name]()
