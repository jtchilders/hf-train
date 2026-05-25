"""CUDA backend — commodity NVIDIA GPU. No-op optimize (no analog of ipex.optimize)."""

from __future__ import annotations

from typing import Any

from hf_train.backend.base import Backend


class CudaBackend(Backend):
    device = "cuda"

    def setup_env(self) -> None:
        pass

    def optimize(self, model: Any, optimizer: Any, dtype: str) -> tuple[Any, Any]:
        return model, optimizer

    def capabilities(self) -> dict[str, Any]:
        return {"device": "cuda", "supported_dtypes": ["fp32", "bf16", "fp16"]}
