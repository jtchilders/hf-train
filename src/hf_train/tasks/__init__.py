"""Task handler registry."""

from __future__ import annotations

from hf_train.tasks.base import TaskHandler

__all__ = ["TaskHandler", "HANDLERS", "get_handler"]

# Populated lazily to avoid importing heavy ML libraries when the package
# is just being introspected (e.g., `hf-train --help`).
HANDLERS: dict[str, type[TaskHandler]] = {}


_HANDLER_MODULES: dict[str, tuple[str, str]] = {
    "causal_lm": ("hf_train.tasks.causal_lm", "CausalLMHandler"),
    "image_classification": (
        "hf_train.tasks.image_classification",
        "ImageClassificationHandler",
    ),
    "semantic_segmentation": (
        "hf_train.tasks.semantic_segmentation",
        "SemanticSegmentationHandler",
    ),
}


def _register_handlers() -> None:
    """Import all concrete handlers and populate HANDLERS.

    Missing handler modules are skipped silently so that partial installations
    (or in-progress task implementations) don't break unrelated tasks.
    """
    global HANDLERS
    if HANDLERS:
        return
    import importlib

    for name, (module_path, class_name) in _HANDLER_MODULES.items():
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            continue
        HANDLERS[name] = getattr(module, class_name)


def get_handler(task_name: str) -> TaskHandler:
    """Instantiate the handler for the given task. Raises if unknown."""
    _register_handlers()
    if task_name not in HANDLERS:
        raise ValueError(
            f"Unknown task: {task_name!r}. Available: {sorted(HANDLERS)}"
        )
    return HANDLERS[task_name]()
