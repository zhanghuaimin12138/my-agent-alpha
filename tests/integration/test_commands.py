"""命令框架的端到端测试:命令经 chat() 主循环装配后的行为。

框架机制(注册/别名/分发)的单元测试在 tests/unit/commands/test_commands.py。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager
from my_agent_alpha.chat import chat
from my_agent_alpha.llm.mock import set_scripted_replies


def test_e2e_exit_command_via_chat(logdir: Path, workspace: Path) -> None:
    """chat() 装配了内置命令:/exit 结束会话(更新自旧版 test_exit_command)。"""
    with pytest.raises(SystemExit):
        chat(
            ["/exit"],
            logdir=logdir,
            workspace=workspace,
            model="mock/echo",
            non_interactive=True,
        )
    manager = LogManager.load(logdir)
    assert not any(m.role == "user" for m in manager.messages)  # 命令不进日志


def test_e2e_unknown_command_treated_as_text(logdir: Path, workspace: Path) -> None:
    set_scripted_replies(["收到"])
    chat(
        ["/神秘命令"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )
    manager = LogManager.load(logdir)
    assert any(m.role == "user" and m.content == "/神秘命令" for m in manager.messages)
