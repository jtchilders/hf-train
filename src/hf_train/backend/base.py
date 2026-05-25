"""Abstract backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Abstracts device + IPEX-style optimization across XPU/CUDA/CPU."""

    device: str  # "xpu" | "cuda" | "cpu"

    @abstractmethod
    def setup_env(self) -> None:
        """Set env vars (e.g., ACCELERATE_USE_XPU) before HF Trainer initializes."""

    @abstractmethod
    def optimize(self, model: Any, optimizer: Any, dtype: str) -> tuple[Any, Any]:
        """Apply backend-specific optimization (ipex.optimize on XPU; no-op elsewhere)."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Backend capabilities (supported dtypes, etc.) for logging."""
