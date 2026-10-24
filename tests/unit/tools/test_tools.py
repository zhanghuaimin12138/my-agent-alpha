"""tools/__init__.py 测试:注册表、白名单裁剪、execute_msg 的执行语义。

单个工具自己的行为在 unit/tools/test_shell.py、test_read.py、test_save.py 等;
解析细节在 unit/tools/test_base.py。
"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha import Message
from my_agent_alpha.message import INTERRUPT_CONTENT
from my_agent_alpha.tools import (
    execute_msg,
    get_tool,
    get_tool_for_langtag,
    get_tools,
    restrict_tools,
)
from my_agent_alpha.tools.base import ToolSpec

# ---- 注册表 ----


def test_registry_has_builtin_tools() -> None:
    names = {t.name for t in get_tools()}
    assert {"shell", "read", "save", "append"} <= names


def test_restrict_tools_whitelist() -> None:
    """白名单裁剪(受限子代理的地基):只留指定工具,没找到的名字如实报告。

    注册表是 markdown 围栏和原生通道共用的查找入口,裁剪后
    ```shell 围栏和 @shell 调用一起失效 —— 不存在"白名单漏了一条通道"。
    """
    missing = restrict_tools(["read", "grep", "memory", "不存在的工具"])
    assert missing == ("不存在的工具",)  # 想留但注册表里没有的,返回给调用方告警
    assert get_tool("shell") is None  # shell 被裁掉
    assert get_tool_for_langtag("shell") is None  # 围栏 ```shell 不再被识别
    assert get_tool("memory") is not None  # 白名单内的还在


def test_whitelist_gates_late_added_tools() -> None:
    """白名单要管住"晚到的"工具:MCP 在 chat() 里才连上、才 add_tools,
    而 --tools 在那之前就生效了 —— 不设闸的话受限子代理会把远端工具漏进来。
    """
    from my_agent_alpha.tools import add_tools

    restrict_tools(["read", "memory"])
    outsider = ToolSpec(name="mcp.evil", desc="远端工具", execute=None)
    insider = ToolSpec(name="memory", desc="同名占位", execute=None)
    add_tools((outsider, insider))
    assert get_tool("mcp.evil") is None  # 白名单外的进不来
    assert get_tool("memory") is not None  # 白名单内的照常追加


def test_langtag_lookup() -> None:
    assert get_tool_for_langtag("shell") is not None
    assert get_tool_for_langtag("bash") is not None
    assert get_tool_for_langtag("save") is not None
    assert get_tool_for_langtag("python") is None  # 未注册的 langtag 不是工具


# ---- execute_msg 的执行语义 ----


def test_think_glued_fence_still_executes(workspace: Path) -> None:
    """端到端回归:围栏粘在 </think> 后面同一行,工具照样要执行出结果。"""
    msg = Message("assistant", "<think>先打个招呼。</think>```shell\necho hi\n```")
    results = execute_msg(msg)
    assert len(results) == 1
    assert "hi" in results[0].content


def test_execute_error_becomes_system_message() -> None:
    def boom(code, args, kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("炸了")

    bad_tool = ToolSpec(name="bad", desc="", block_types=("bad",), execute=boom)
    from my_agent_alpha import tools as tools_pkg

    original = tools_pkg._tools_var.get()
    tools_pkg._tools_var.set((*tuple(list(original or ())), bad_tool))
    try:
        msg = Message("assistant", "```bad\nx\n```")
        results = execute_msg(msg)
        assert len(results) == 1
        assert results[0].role == "system"
        assert "执行失败" in results[0].content
        assert "炸了" in results[0].content
    finally:
        tools_pkg._tools_var.set(original)


# ---- 原生通道的参数传递(kwargs)----


def test_shell_via_native_kwargs(workspace: Path) -> None:
    msg = Message("assistant", '@shell(call_9): {"command": "echo native-shell"}')
    results = execute_msg(msg)
    assert "native-shell" in results[0].content
    assert results[0].call_id == "call_9"  # 结果盖上了调用编号


def test_save_via_native_kwargs(workspace: Path) -> None:
    msg = Message("assistant", '@save(call_1): {"path": "n.txt", "content": "原生内容"}')
    results = execute_msg(msg)
    assert "已创建" in results[0].content
    assert (workspace / "n.txt").read_text(encoding="utf-8") == "原生内容"


def test_read_via_native_kwargs(workspace: Path) -> None:
    (workspace / "r.txt").write_text("内容\n", encoding="utf-8")
    msg = Message("assistant", '@read(call_1): {"path": "r.txt"}')
    assert "内容" in execute_msg(msg)[0].content


def test_shell_missing_command_reports_error(workspace: Path) -> None:
    msg = Message("assistant", "@shell(call_1): {}")
    results = execute_msg(msg)
    assert "没有收到要执行的命令" in results[0].content


# ---- 审查 #1:中断不丢已完成的步骤 ----


def test_interrupt_preserves_partial_results(logdir: Path, workspace: Path) -> None:
    """两个工具调用,第一个完成、第二个被 Ctrl-C:全部要有日志记录。"""

    def interrupting_tool(code, args, kwargs):  # type: ignore[no-untyped-def]
        yield Message("system", "第一个工具的结果")
        raise KeyboardInterrupt()

    from my_agent_alpha import tools as tools_pkg

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
