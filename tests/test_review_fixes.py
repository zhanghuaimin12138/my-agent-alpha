"""审查修复的回归测试(阶段 1 对抗式审查确认的 18 条发现)。"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.llm import reply
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.message import INTERRUPT_CONTENT, LLM_REQUEST_FAILED_PREFIX
from my_agent_alpha.tools import execute_msg

# ---- 审查 #1:中断不丢已完成的步骤 ----


def test_interrupt_preserves_partial_results(logdir: Path, workspace: Path) -> None:
    """两个工具调用,第一个完成、第二个被 Ctrl-C:全部要有日志记录。"""

    def interrupting_tool(code, args, kwargs):  # type: ignore[no-untyped-def]
        yield Message("system", "第一个工具的结果")
        raise KeyboardInterrupt()

    from my_agent_alpha import tools as tools_pkg
    from my_agent_alpha.tools.base import ToolSpec

    culprit = ToolSpec(name="culprit", desc="", block_types=("culprit",), execute=interrupting_tool)
    base = tools_pkg.get_tools()
    tools_pkg._tools_var.set((*base, culprit))
    try:
        assistant = Message(
            "assistant",
            "```shell\necho done-first\n```\n```culprit\nx\n```\n```shell\necho never\n```",
        )
        results = execute_msg(assistant)
    finally:
        tools_pkg._tools_var.set(base)

    contents = [m.content for m in results]
    # 第一个工具(shell)的结果保留
    assert any("done-first" in c for c in contents)
    # 被中断的调用与未执行的调用都有标记
    assert any(c.startswith(INTERRUPT_CONTENT) and "culprit" in c for c in contents)
    assert any("因中断未执行" in c and "never" not in c for c in contents)


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

    def flaky_reply(messages, model):  # type: ignore[no-untyped-def]
        if not flaky_state["failed"]:
            flaky_state["failed"] = True
            raise RuntimeError("模拟网络错误")
        return real_reply(messages, "mock/echo")

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


def _fake_input(seq: list[str]):
    """把字符串序列变成 input():耗尽后抛 EOFError(模拟 Ctrl-D 退出)。"""
    it = iter(seq)

    def _input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return _input


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


# ---- 审查 #11/#12:模型名解析 ----


def test_empty_model_resolves_to_mock() -> None:
    from my_agent_alpha.llm import resolve_model

    meta = resolve_model("")
    assert meta.provider == "mock"
    assert reply([Message("user", "x")], "").content == "Echo: x"


def test_openrouter_bare_errors_clearly() -> None:
    from my_agent_alpha.llm import resolve_model

    with pytest.raises(ValueError, match="显式指定"):
        resolve_model("openrouter")


# ---- 审查 #7:占位符只拒"整个文件是占位符" ----


def test_placeholder_substring_allowed(workspace: Path) -> None:
    msg = Message("assistant", "```save ellipsis.txt\n正常内容包含 ... 也可以\n```")
    results = execute_msg(msg)
    assert "已创建" in results[0].content
    assert (workspace / "ellipsis.txt").exists()


def test_placeholder_whole_file_rejected(workspace: Path) -> None:
    msg = Message("assistant", "```save stub.txt\n# ...\n```")
    assert "占位符" in execute_msg(msg)[0].content


# ---- 审查 #17:save 覆盖 / append 建文件 ----


def test_save_overwrite_branch(workspace: Path) -> None:
    target = workspace / "f.txt"
    target.write_text("旧内容", encoding="utf-8")
    msg = Message("assistant", "```save f.txt\n新内容\n```")
    assert "已覆盖" in execute_msg(msg)[0].content
    # save 原样写围栏内内容,不擅自补换行
    assert target.read_text(encoding="utf-8") == "新内容"


def test_append_creates_missing_file(workspace: Path) -> None:
    msg = Message("assistant", "```append new.md\n第一行\n```")
    execute_msg(msg)
    assert (workspace / "new.md").read_text(encoding="utf-8") == "第一行\n"


# ---- 审查 #18:read 支持 fence 头路径 ----


def test_read_head_arg(workspace: Path) -> None:
    (workspace / "head.txt").write_text("内容\n", encoding="utf-8")
    msg = Message("assistant", "```read head.txt\n```")
    assert "内容" in execute_msg(msg)[0].content


# ---- 审查 #14:系统提示词内容 ----


def test_prompt_contains_tool_docs_and_time() -> None:
    from my_agent_alpha.prompts import get_prompt
    from my_agent_alpha.tools import get_tools

    msgs = get_prompt(get_tools())
    assert len(msgs) == 2
    static, dynamic = msgs
    assert static.pinned and static.hide
    assert "工具调用协议" in static.content
    assert "## shell" in static.content
    assert "## save" in static.content
    assert "当前时间" in dynamic.content
    assert "工具调用协议" not in dynamic.content


# ---- 审查 #5/#6/#15:shell 的安全姿态 ----


def test_shell_non_utf8_output_does_not_crash(workspace: Path) -> None:
    msg = Message("assistant", '```shell\nprintf "\\xff\\xfe\\xff"\n```')
    results = execute_msg(msg)
    assert "ReturnCode: 0" in results[0].content  # 解码容错,不炸


def test_shell_timeout_branch(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from my_agent_alpha.tools import shell as shell_mod

    monkeypatch.setattr(shell_mod, "_TIMEOUT", 1)
    msg = Message("assistant", "```shell\nsleep 5\n```")
    results = execute_msg(msg)
    assert "超时" in results[0].content


def test_shell_truncation_saves_full_output(
    workspace: Path, logdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from my_agent_alpha.tools import shell as shell_mod

    monkeypatch.setattr(shell_mod, "_MAX_OUTPUT_CHARS", 200)
    msg = Message("assistant", "```shell\nseq 1 500\n```")
    results = execute_msg(msg)
    assert "已截断" in results[0].content
    # 全量输出真的存了盘,且内容完整
    files = list((logdir / "tool-outputs" / "shell").glob("*.txt"))
    assert len(files) == 1
    assert "500" in files[0].read_text(encoding="utf-8")


def test_shell_full_output_files_never_collide(workspace: Path, logdir: Path) -> None:
    for _ in range(3):
        execute_msg(Message("assistant", "```shell\necho x\n```"))
    files = list((logdir / "tool-outputs" / "shell").glob("*.txt"))
    assert len(files) == 3  # 微秒级文件名,同一秒内也不互相覆盖


# ---- 审查 #8:CRLF 归一化 ----


def test_crlf_fence_parsed_clean() -> None:
    from my_agent_alpha.tools.base import ToolUse

    content = "执行:\r\n```shell\r\nls\r\n```\r\n"
    uses = list(ToolUse.iter_from_content(content))
    assert len(uses) == 1
    assert uses[0].content == "ls"  # 不带 \r
