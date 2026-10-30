"""阶段 4 确认链测试:安全分级、钩子链、终端询问、拒绝/改写流程。"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.confirm import (
    ConfirmationResult,
    clear_confirm_hooks,
    cli_confirm_hook,
    get_confirmation,
    is_auto_confirm,
    register_confirm_hook,
    set_auto_confirm,
)
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.message import DECLINED_CONTENT
from my_agent_alpha.tools import execute_msg
from my_agent_alpha.tools.base import ToolUse, parse_tooluses
from my_agent_alpha.tools.shell_safety import (
    denied_reason,
    guardrail_hook,
    is_readonly_command,
    readonly_hook,
)


def _shell_use(command: str) -> ToolUse:
    """从一条 markdown 围栏造出 shell 调用(测试辅助)。"""
    uses = parse_tooluses(f"```shell\n{command}\n```")
    assert uses and uses[0].tool.name == "shell"
    return uses[0]


# ---- 命令安全分级 ----


def test_readonly_commands_pass() -> None:
    for cmd in [
        "ls -la",
        "ls",
        "cat a.txt",
        "ls | wc -l",
        "git status",
        "git log --oneline",
        "find . -name x",
        "echo hi",
    ]:
        assert is_readonly_command(cmd), cmd


def test_writing_commands_dont_pass() -> None:
    for cmd in [
        "rm file.txt",  # rm 不在只读名单
        "cat a > b",  # 重定向写文件
        "echo $(rm -rf x)",  # 命令替换里藏东西
        "find . -delete",  # find 带删除标志
        "git push",  # git 写子命令
        "python3 evil.py",  # 能执行任意代码的绝不放行
        "touch x",  # 会创建文件
        "ls; rm x",  # 混在管道/分号里的一段不安全
    ]:
        assert not is_readonly_command(cmd), cmd


def test_denied_patterns() -> None:
    assert denied_reason("rm -rf /") is not None
    assert denied_reason("rm -rf ~") is not None
    assert denied_reason("mkfs.ext4 /dev/sdb") is not None
    assert denied_reason("dd if=x of=/dev/sda") is not None
    assert denied_reason("shutdown now") is not None
    # 正常命令不误拦
    assert denied_reason("rm -rf ./build") is None  # 目录内清理:拦到"询问"层,不硬拒
    assert denied_reason("rm file.txt") is None
    assert denied_reason("chmod 777 file.txt") is None


# ---- 两个钩子 ----


def test_guardrail_hook() -> None:
    assert guardrail_hook(_shell_use("rm -rf /")) is not None
    assert guardrail_hook(_shell_use("rm -rf /")).action == "skip"
    assert guardrail_hook(_shell_use("ls")) is None  # 不管普通命令
    save_use = parse_tooluses("```save a.txt\n内容\n```")[0]
    assert guardrail_hook(save_use) is None  # 只管 shell


def test_readonly_hook() -> None:
    assert readonly_hook(_shell_use("ls -la")).action == "confirm"
    assert readonly_hook(_shell_use("rm file.txt")) is None  # 非只读:交给下一层


# ---- 钩子链机制 ----


def test_no_hooks_auto_confirm() -> None:
    """链上没人管 → 自动放行(自主模式:测试/脚本/-y)。"""
    clear_confirm_hooks()
    assert get_confirmation(_shell_use("rm file.txt")).action == "confirm"


def test_priority_order() -> None:
    """高优先级先答;答了就不问下一家;返回 None 才往下传。"""
    clear_confirm_hooks()
    register_confirm_hook("low", lambda u: ConfirmationResult("skip"), priority=0)
    register_confirm_hook("high", lambda u: None, priority=100)  # 高优先级但不管
    assert get_confirmation(_shell_use("ls")).action == "skip"


def test_auto_confirm_state() -> None:
    clear_confirm_hooks()
    register_confirm_hook("always_ask", lambda u: ConfirmationResult("skip"))
    set_auto_confirm(True)
    assert get_confirmation(_shell_use("ls")).action == "confirm"  # 放行状态压过一切
    assert is_auto_confirm()
    set_auto_confirm(False)
    assert get_confirmation(_shell_use("ls")).action == "skip"


# ---- 执行集成 ----


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


# ---- 终端询问钩子 ----


def _fake_input(answers: list[str]):
    it = iter(answers)

    def _input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return _input


def test_cli_hook_yes(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_confirm_hooks()
    monkeypatch.setattr("builtins.input", _fake_input(["y"]))
    assert cli_confirm_hook(_shell_use("rm file.txt")).action == "confirm"


def test_cli_hook_no(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_confirm_hooks()
    monkeypatch.setattr("builtins.input", _fake_input(["n"]))
    result = cli_confirm_hook(_shell_use("rm file.txt"))
    assert result.action == "skip"


def test_cli_hook_edit_then_a(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_confirm_hooks()
    monkeypatch.setattr("builtins.input", _fake_input(["e", "echo FIXED", "x"]))
    result = cli_confirm_hook(_shell_use("rm file.txt"))
    assert result.action == "edit"
    assert result.edited_content == "echo FIXED"
    # 选 a 之后:本会话不再问(钩子退位返回 None,由链尾放行)
    monkeypatch.setattr("builtins.input", _fake_input(["a"]))
    assert cli_confirm_hook(_shell_use("rm file.txt")).action == "confirm"
    assert cli_confirm_hook(_shell_use("rm other.txt")) is None
    from my_agent_alpha.confirm import get_confirmation

    assert get_confirmation(_shell_use("rm other.txt")).action == "confirm"


def test_cli_hook_eof_means_skip(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_confirm_hooks()
    monkeypatch.setattr("builtins.input", _fake_input([]))  # 立即 EOF
    assert cli_confirm_hook(_shell_use("rm file.txt")).action == "skip"


# ---- 端到端 ----


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
