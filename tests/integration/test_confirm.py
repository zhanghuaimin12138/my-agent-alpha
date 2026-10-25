"""确认链的集成与端到端:钩子挂上 execute_msg 后的行为、chat() 里的拒绝流程。"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.confirm import (
    ConfirmationResult,
    clear_confirm_hooks,
    register_confirm_hook,
)
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.message import DECLINED_CONTENT
from my_agent_alpha.tools import execute_msg
from my_agent_alpha.tools.base import ToolUse, parse_tooluses
from my_agent_alpha.tools.shell_safety import guardrail_hook, readonly_hook


def _shell_use(command: str) -> ToolUse:
    """从一条 markdown 围栏造出 shell 调用(测试辅助)。"""
    uses = parse_tooluses(f"```shell\n{command}\n```")
    assert uses and uses[0].tool.name == "shell"
    return uses[0]


# ---- 执行集成(execute_msg + 确认链)----


def test_dangerous_command_skipped_not_executed(workspace: Path) -> None:
    clear_confirm_hooks()
    register_confirm_hook("guardrail", guardrail_hook, priority=200)
    results = execute_msg(Message("assistant", "```shell\nrm -rf /\n```"))
    assert len(results) == 1
    assert results[0].content.startswith(DECLINED_CONTENT)
    assert "安全护栏" in results[0].content


def test_readonly_executes_without_prompt(workspace: Path) -> None:
    """只读命令:注册了"一律问"的钩子也不该被拦(readonly 层先放行)。"""
    clear_confirm_hooks()
    register_confirm_hook("ask_all", lambda u: ConfirmationResult("skip"), priority=0)
    register_confirm_hook("readonly", readonly_hook, priority=10)
    results = execute_msg(Message("assistant", "```shell\necho auto-pass\n```"))
    assert "auto-pass" in results[0].content


def test_read_tool_never_confirms(workspace: Path) -> None:
    """只读工具(read)完全绕过确认链:哪怕链上说全都拒绝。"""
    clear_confirm_hooks()
    register_confirm_hook("ask_all", lambda u: ConfirmationResult("skip"), priority=100)
    (workspace / "r.txt").write_text("内容\n", encoding="utf-8")
    results = execute_msg(Message("assistant", "```read\nr.txt\n```"))
    assert "内容" in results[0].content  # 照常执行


def test_edit_flow_replaces_command(workspace: Path) -> None:
    """钩子返回 edit:命令被改写后执行(原始命令不执行)。"""
    clear_confirm_hooks()
    register_confirm_hook(
        "editor", lambda u: ConfirmationResult("edit", edited_content="echo EDITED")
    )
    results = execute_msg(Message("assistant", "```shell\necho ORIGINAL\n```"))
    assert "EDITED" in results[0].content
    assert "ORIGINAL" not in results[0].content


def test_edit_flow_native_kwargs(workspace: Path) -> None:
    clear_confirm_hooks()
    register_confirm_hook(
        "editor", lambda u: ConfirmationResult("edit", edited_content="echo NATIVE-EDIT")
    )
    results = execute_msg(Message("assistant", '@shell(c1): {"command": "echo orig"}'))
    assert "NATIVE-EDIT" in results[0].content


def test_save_confirmed_in_autonomous_mode(workspace: Path) -> None:
    """非交互(链空):save 自动放行 —— 已有行为的回归锚。"""
    clear_confirm_hooks()
    results = execute_msg(Message("assistant", "```save s.txt\n内容\n```"))
    assert "已创建" in results[0].content


# ---- 端到端 ----


def _fake_input(answers: list[str]):
    it = iter(answers)

    def _input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return _input


def test_e2e_user_declines_save(
    logdir: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """交互模式:用户对 save 答 n → 文件不创建、回合结束、会话活着。"""
    set_scripted_replies(["保存:\n```save nope.txt\n不该写\n```", "好的"])
    # 确认问 n;随后外层取用户输入遇到 EOF 退出
    monkeypatch.setattr("builtins.input", _fake_input(["n"]))
    chat(
        ["保存一下"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=False,
    )
    manager = LogManager.load(logdir)
    assert any(
        m.role == "system" and m.content.startswith(DECLINED_CONTENT) for m in manager.messages
    )
    assert not (workspace / "nope.txt").exists()


def test_e2e_guardrail_blocks_even_non_interactive(logdir: Path, workspace: Path) -> None:
    """安全护栏不分模式:非交互(相当于全放行)也拦灾难命令,回合停止。"""
    set_scripted_replies(["清理:\n```shell\nrm -rf /\n```", "好的"])
    chat(
        ["清理一下"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )
    manager = LogManager.load(logdir)
    assert any(
        m.role == "system" and m.content.startswith(DECLINED_CONTENT) and "安全护栏" in m.content
        for m in manager.messages
    )
    # 回合被拒绝终止:第二轮"好的"没被消费(脚本应还剩一条)
    from my_agent_alpha.llm.mock import _scripted

    assert len(_scripted) == 1
