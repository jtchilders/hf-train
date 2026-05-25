"""Per-task default fillers, applied after the task discriminator is known."""

from __future__ import annotations

from typing import Any


_TASK_DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
    "causal_lm": {
        "data": {"max_seq_length": 1024},
    },
    "image_classification": {
        "data": {"image_size": 224},
    },
    "semantic_segmentation": {
        "data": {"image_size": 224},
    },
}


def apply_task_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge task-specific defaults into a raw (pre-Pydantic) config dict.

    User-set values always win; defaults only fill absent keys.
    """
    task = raw.get("task")
    if task not in _TASK_DEFAULTS:
        return raw
    for section_name, section_defaults in _TASK_DEFAULTS[task].items():
        section = raw.setdefault(section_name, {})
        for k, v in section_defaults.items():
            section.setdefault(k, v)
    return raw
