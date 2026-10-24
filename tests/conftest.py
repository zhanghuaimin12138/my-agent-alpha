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


@pytest.fixture
def clean_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """隔离的用户层数据目录(长期记忆测试用,别污染真实 ~/.local/share)。

    返回 tmp 根:测试在它下面自建 ws/、logs/ 等子目录。
    (XDG_DATA_HOME 其实在下面的 autouse 夹具里也会设,这里显式再设一遍,
    让单独使用本夹具的测试不依赖 autouse 的存在。)
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_agent_state(
    workspace: Path, logdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """每个测试前后清干净全局可变状态(mock 脚本、工具注册表、ContextVar)。

    仿 gptme conftest 的"teardown 卫生"哲学:agent 是有状态程序,
    测试之间的状态泄漏是最难查的一类 bug。
    数据目录(XDG_DATA_HOME)也全局隔离:用户层的真实长期记忆
    (~/.local/share/...)会经 render_memory_index 注进所有会话,
    2026-10-07 真机存了第一条用户层记忆后,没隔离的 chat 测试全挂了。
    """
    from my_agent_alpha import context
    from my_agent_alpha.confirm import clear_confirm_hooks
    from my_agent_alpha.hooks import clear_hooks
    from my_agent_alpha.llm import mock as llm_mock
    from my_agent_alpha.tools import clear_tools
    from my_agent_alpha.tools import todo as todo_mod

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    # 默认关掉会话自动命名:命名钩子的额外模型调用会消耗 mock 脚本序列,
    # 搅乱无关测试;需要测命名时在测试里设回 "1"(见 test_naming.py)
    monkeypatch.setenv("ALPHA_AUTO_NAME", "0")

    llm_mock.clear_scripted()
    clear_tools()
    clear_confirm_hooks()
    clear_hooks()
    todo_mod._reset_todos()  # todo 的清单/门控计数是模块级状态,不随注册表清空
    context.clear_current_model()  # 模型也是 ContextVar:不清会泄漏给后面的测试
    with context.agent_context(workspace, logdir):
        yield
    llm_mock.clear_scripted()
    clear_tools()
