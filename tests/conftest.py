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


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[Path]:
    """隔离的工作区(shell/read/save 在这里操作)。"""
    d = tmp_path / "workspace"
    d.mkdir()
    yield d


@pytest.fixture(autouse=True)
def _reset_agent_state(workspace: Path, logdir: Path) -> Iterator[None]:
    """每个测试前后清干净全局可变状态(mock 脚本、工具注册表、ContextVar)。

    仿 gptme conftest 的"teardown 卫生"哲学:agent 是有状态程序,
    测试之间的状态泄漏是最难查的一类 bug。
    """
    from my_agent_alpha import context
    from my_agent_alpha.llm import mock as llm_mock
    from my_agent_alpha.tools import clear_tools

    llm_mock.clear_scripted()
    clear_tools()
    with context.agent_context(workspace, logdir):
        yield
    llm_mock.clear_scripted()
    clear_tools()
