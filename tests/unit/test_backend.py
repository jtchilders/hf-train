from __future__ import annotations

from unittest.mock import patch

import pytest


def test_cpu_backend_device_string() -> None:
    from hf_train.backend import get_backend
    b = get_backend("cpu")
    assert b.device == "cpu"


def test_cpu_backend_optimize_is_noop() -> None:
    from hf_train.backend import get_backend
    b = get_backend("cpu")
    model, optimizer = object(), object()
    new_model, new_opt = b.optimize(model, optimizer, dtype="fp32")
    assert new_model is model
    assert new_opt is optimizer


def test_cuda_backend_device_string() -> None:
    from hf_train.backend import get_backend
    b = get_backend("cuda")
    assert b.device == "cuda"


def test_cuda_backend_optimize_is_noop() -> None:
    from hf_train.backend import get_backend
    b = get_backend("cuda")
    model, optimizer = object(), object()
    new_model, new_opt = b.optimize(model, optimizer, dtype="fp32")
    assert new_model is model
    assert new_opt is optimizer


def test_xpu_backend_lazy_import_fails_clean_without_ipex() -> None:
    """XPU backend must NOT import IPEX at module import time. Instantiating
    when IPEX is missing should raise a clear, actionable error (not ImportError
    bubbling from deep inside)."""
    from hf_train.backend import get_backend
    # Skip if IPEX actually IS installed (don't false-fail on Aurora dev)
    try:
        import intel_extension_for_pytorch  # noqa: F401
        pytest.skip("IPEX is installed; cannot test the missing-IPEX path here")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="intel-extension-for-pytorch"):
        get_backend("xpu")


def test_auto_detect_falls_back_to_cpu_when_nothing_available() -> None:
    """With both XPU and CUDA mocked unavailable, auto_detect returns cpu."""
    from hf_train.backend import auto_detect
    with patch("hf_train.backend._cuda_available", return_value=False), \
         patch("hf_train.backend._xpu_available", return_value=False):
        b = auto_detect()
        assert b.device == "cpu"


def test_auto_detect_picks_cuda_when_available_no_xpu() -> None:
    from hf_train.backend import auto_detect
    with patch("hf_train.backend._cuda_available", return_value=True), \
         patch("hf_train.backend._xpu_available", return_value=False):
        b = auto_detect()
        assert b.device == "cuda"


def test_auto_detect_prefers_xpu_when_both_available() -> None:
    """If both somehow report available, prefer XPU (we're an Aurora-first tool)."""
    from hf_train.backend import auto_detect
    with patch("hf_train.backend._cuda_available", return_value=True), \
         patch("hf_train.backend._xpu_available", return_value=True), \
         patch("hf_train.backend.XpuBackend.__init__", return_value=None):
        # XpuBackend's __init__ is patched to skip the IPEX import for this test
        from hf_train.backend.xpu import XpuBackend
        b = auto_detect()
        assert isinstance(b, XpuBackend)
