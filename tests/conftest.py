"""pytest 全局夹具。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def logdir(tmp_path: Path) -> Iterator[Path]:
    """隔离的会话目录(永不动真实 ~/.local/share)。"""
    d = tmp_path / "logs" / "2026-0928-abcd01"
    d.mkdir(parents=True)
    yield d
