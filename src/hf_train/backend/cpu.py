"""CPU backend — fallback for tests and dev without GPU."""

from __future__ import annotations

from typing import Any

from hf_train.backend.base import Backend


class CpuBackend(Backend):
    device = "cpu"

    def setup_env(self) -> None:
        pass

    def optimize(self, model: Any, optimizer: Any, dtype: str) -> tuple[Any, Any]:
        return model, optimizer

    def capabilities(self) -> dict[str, Any]:
        return {"device": "cpu", "supported_dtypes": ["fp32"]}
