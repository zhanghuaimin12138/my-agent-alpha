"""审查修复的 chat 级回归(阶段 1 对抗式审查确认的发现)。

走完整会话循环的部分收在这里;模块级的审查回归已按归属拆进
unit/ 各文件(save/read/shell/llm/prompts/base)。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager
from my_agent_alpha.chat import chat
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.message import INTERRUPT_CONTENT, LLM_REQUEST_FAILED_PREFIX


def _fake_input(seq: list[str]):
    """把字符串序列变成 input():耗尽后抛 EOFError(模拟 Ctrl-D 退出)。"""
    it = iter(seq)

    def _input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return _input


# ---- 审查 #1:中断不丢已完成的步骤(端到端)----


def test_interrupted_turn_persists_assistant_and_stops(
    logdir: Path, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """端到端:工具被中断 → assistant 消息与已完成结果落盘,turn 停止。"""

    def interrupting_tool(code, args, kwargs):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt()

    from my_agent_alpha import tools as tools_pkg
    from my_agent_alpha.tools.base import ToolSpec

    culprit = ToolSpec(name="culprit", desc="", block_types=("culprit",), execute=interrupting_tool)
    base = tools_pkg.get_tools()
    tools_pkg._tools_var.set((*base, culprit))
    set_scripted_replies(["执行:\n```culprit\nx\n```"])
    try:
        chat(
            ["触发中断"],
            logdir=logdir,
            workspace=workspace,
            model="mock/echo",
            non_interactive=True,
        )
    finally:
        tools_pkg._tools_var.set(base)

    manager = LogManager.load(logdir)
    roles = [m.role for m in manager.messages]
    # assistant 消息(含调用)与中断标记都必须在日志里 —— 不得被静默吞掉
    assert "assistant" in roles
    assert any(
        m.role == "system" and m.content.startswith(INTERRUPT_CONTENT) for m in manager.messages
    )
    # turn 停了:没有第二步生成(中断后不再调用 LLM)
    assistant_count = roles.count("assistant")
    assert assistant_count == 1


# ---- 审查 #2:provider 异常不杀交互 REPL ----


def test_provider_error_interactive_returns_to_prompt(
    logdir: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", _fake_input(["你好", "再见"]))
    set_scripted_replies(["好的"])

    # 模拟瞬时 provider 故障:第一次调用抛错,第二次正常
    from my_agent_alpha import chat as chat_mod

    real_reply = chat_mod.reply
    flaky_state = {"failed": False}

    def flaky_reply(messages, model, **kwargs):  # type: ignore[no-untyped-def]
        if not flaky_state["failed"]:
            flaky_state["failed"] = True
            raise RuntimeError("模拟网络错误")
        return real_reply(messages, "mock/echo", **kwargs)

    monkeypatch.setattr(chat_mod, "reply", flaky_reply)

    manager = chat(
        [],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=False,
    )
    # 第二轮 "再见" 能正常拿到回复:REPL 活着
    assert any(m.content == "好的" for m in manager.messages)
    assert any(
        m.role == "system" and m.content.startswith(LLM_REQUEST_FAILED_PREFIX)
        for m in manager.messages
    )


def test_provider_error_non_interactive_raises(logdir: Path, workspace: Path) -> None:
    with pytest.raises(RuntimeError, match="nosuchprov"):
        chat(
            ["你好"],
            logdir=logdir,
            workspace=workspace,
            model="nosuchprov/model-x",
            non_interactive=True,
        )


# ---- 审查 #3:空输入循环而非递归 ----


def test_blank_input_loop_not_recursion(
    logdir: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", _fake_input([""] * 3000 + ["有效输入"]))
    set_scripted_replies(["收到"])
    manager = chat([], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=False)
    assert any(m.role == "user" and m.content == "有效输入" for m in manager.messages)


def test_interactive_eof_exits_and_ctrlc_at_prompt_continues(
    logdir: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_input(prompt: str = "") -> str:
        calls.append(prompt)
        if len(calls) == 1:
            raise KeyboardInterrupt  # 提示符处的 Ctrl-C:提示后继续
        if len(calls) == 2:
            return "一条消息"
        raise EOFError  # 然后退出

    monkeypatch.setattr("builtins.input", fake_input)
    set_scripted_replies(["回复"])
    manager = chat([], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=False)
    assert any(m.role == "user" for m in manager.messages)
    assert len(calls) == 3  # Ctrl-C 一次、消息一次、EOF 一次
