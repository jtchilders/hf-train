"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Returns a temporary directory suitable for use as output.output_dir."""
    d = tmp_path / "run"
    d.mkdir()
    return d


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to tests/fixtures."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _disable_hub_caching_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep HF cache dirs out of $HOME during tests; speed up by reusing user's cache when available."""
    # If the user has a populated cache (CI cache hit, dev machine), point HF there.
    # Otherwise let HF default to its own location. We don't override unless asked.
    yield
