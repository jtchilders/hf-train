"""Backend factory + auto-detection."""

from __future__ import annotations

from typing import Literal

from hf_train.backend.base import Backend
from hf_train.backend.cpu import CpuBackend
from hf_train.backend.cuda import CudaBackend
from hf_train.backend.xpu import XpuBackend

__all__ = ["Backend", "CpuBackend", "CudaBackend", "XpuBackend", "auto_detect", "get_backend"]

BackendName = Literal["auto", "xpu", "cuda", "cpu"]


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _xpu_available() -> bool:
    try:
        import torch
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


def auto_detect() -> Backend:
    """Return the best available backend: xpu > cuda > cpu."""
    if _xpu_available():
        return XpuBackend()
    if _cuda_available():
        return CudaBackend()
    return CpuBackend()


def get_backend(name: BackendName) -> Backend:
    """Return a backend instance by name. 'auto' delegates to auto_detect()."""
    if name == "auto":
        return auto_detect()
    if name == "xpu":
        return XpuBackend()
    if name == "cuda":
        return CudaBackend()
    if name == "cpu":
        return CpuBackend()
    raise ValueError(f"Unknown backend: {name}")
