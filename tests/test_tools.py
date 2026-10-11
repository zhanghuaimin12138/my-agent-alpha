"""工具系统测试:ToolUse 解析、执行、内置工具真实执行。"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha import Message
from my_agent_alpha.tools import (
    execute_msg,
    get_tool,
    get_tool_for_langtag,
    get_tools,
    restrict_tools,
)
from my_agent_alpha.tools.base import ToolSpec, ToolUse


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


def test_iter_from_content_maps_fences(workspace: Path) -> None:
    content = "我来执行:\n```shell\necho hi\n```\n再读文件:\n```read src/a.py\na.py\n```"
    uses = list(ToolUse.iter_from_content(content))
    assert [u.tool.name for u in uses] == ["shell", "read"]
    assert uses[0].content == "echo hi"
    assert uses[1].args == ["src/a.py"]
    assert uses[1].content == "a.py"


def test_unknown_langtag_ignored() -> None:
    content = "示例:\n```python\nprint('说明代码')\n```"
    assert list(ToolUse.iter_from_content(content)) == []


def test_unterminated_fence_not_executed() -> None:
    content = "```shell\necho 半截"
    assert list(ToolUse.iter_from_content(content)) == []


# ---- 思考区剥离在工具层的回归(kimi 粘连格式曾静默丢失调用)----


def test_native_call_inside_think_ignored() -> None:
    """思考区里的 @原生调用行是示例/草稿,不解析(修复前会被当成真调用执行)。"""
    content = '<think>比如 @save(call_x): {"path": "a.txt", "content": "x"}</think>我想想'
    assert list(ToolUse.iter_from_content(content)) == []


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


# ---- 内置工具:真实执行 ----


def test_shell_executes_and_reports_rc(workspace: Path) -> None:
    msg = Message("assistant", "```shell\necho hello-spine\n```")
    results = execute_msg(msg)
    assert len(results) == 1
    assert "hello-spine" in results[0].content
    assert "ReturnCode: 0" in results[0].content


def test_shell_failure_reported(workspace: Path) -> None:
    msg = Message("assistant", "```shell\nexit 3\n```")
    results = execute_msg(msg)
    assert "ReturnCode: 3" in results[0].content


def test_shell_runs_in_workspace(workspace: Path) -> None:
    (workspace / "marker.txt").write_text("x", encoding="utf-8")
    msg = Message("assistant", "```shell\nls marker.txt\n```")
    assert "marker.txt" in execute_msg(msg)[0].content


def test_read_numbers_lines(workspace: Path) -> None:
    target = workspace / "a.txt"
    target.write_text("第一行\n第二行\n", encoding="utf-8")
    msg = Message("assistant", "```read\na.txt\n```")
    out = execute_msg(msg)[0].content
    assert "a.txt" in out
    assert "1\t第一行" in out
    assert "2\t第二行" in out


def test_read_missing_file_is_message_not_crash(workspace: Path) -> None:
    msg = Message("assistant", "```read\n不存在.txt\n```")
    out = execute_msg(msg)[0].content
    assert "不存在" in out


def test_save_writes_file(workspace: Path) -> None:
    msg = Message("assistant", "```save out/hello.py\nprint('hi')\n```")
    results = execute_msg(msg)
    assert "已创建" in results[0].content
    assert (workspace / "out" / "hello.py").read_text(encoding="utf-8") == "print('hi')"


def test_save_rejects_placeholder(workspace: Path) -> None:
    msg = Message("assistant", "```save stub.py\n# ...\n```")
    results = execute_msg(msg)
    assert "占位符" in results[0].content
    assert not (workspace / "stub.py").exists()


def test_save_without_path_reports_error(workspace: Path) -> None:
    msg = Message("assistant", "```save\n没路径\n```")
    assert "路径" in execute_msg(msg)[0].content


def test_append_appends_with_newline(workspace: Path) -> None:
    target = workspace / "log.md"
    target.write_text("- 第一条", encoding="utf-8")
    msg = Message("assistant", "```append log.md\n- 第二条\n```")
    execute_msg(msg)
    assert target.read_text(encoding="utf-8") == "- 第一条\n- 第二条\n"


def test_tool_prompt_renders() -> None:
    shell = get_tool("shell")
    assert shell is not None
    doc = shell.get_tool_prompt()
    assert "## shell" in doc
    assert "```shell" in doc
