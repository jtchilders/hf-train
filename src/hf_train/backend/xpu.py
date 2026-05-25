"""XPU backend — Aurora Intel GPU + Intel Extension for PyTorch (IPEX).

IPEX is imported lazily so this module can be imported on non-Aurora machines
without the IPEX wheel installed (e.g., during CI on CPU-only dev boxes).
"""

from __future__ import annotations

import os
from typing import Any

from hf_train.backend.base import Backend


class XpuBackend(Backend):
    device = "xpu"

    def __init__(self) -> None:
        try:
            import intel_extension_for_pytorch  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "intel-extension-for-pytorch is required for the XPU backend. "
                "On Aurora: `module load frameworks`. Elsewhere: pick a different backend."
            ) from e

    def setup_env(self) -> None:
        os.environ.setdefault("ACCELERATE_USE_XPU", "true")

    def optimize(self, model: Any, optimizer: Any, dtype: str) -> tuple[Any, Any]:
        import intel_extension_for_pytorch as ipex
        import torch

        dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        # ipex.optimize returns `(model, optimizer)` when an optimizer was
        # passed in, else just `model`. HF Trainer reconstructs the optimizer
        # itself, so we usually pass optimizer=None and have to handle both.
        result = ipex.optimize(model, optimizer=optimizer, dtype=torch_dtype)
        if isinstance(result, tuple):
            return result
        return result, optimizer

    def capabilities(self) -> dict[str, Any]:
        return {"device": "xpu", "supported_dtypes": ["fp32", "bf16"]}
