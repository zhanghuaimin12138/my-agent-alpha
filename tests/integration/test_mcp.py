"""MCP 客户端测试:连接真实子进程服务器,列工具、调用、超时、转 ToolSpec。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from my_agent_alpha import Message
from my_agent_alpha.config import McpServer
from my_agent_alpha.tools import execute_msg


@pytest.fixture(scope="module")
def echo_server_path() -> Path:
    """测试用的 MCP 服务器脚本(内容见本目录 mcp_echo_server.py,SDK 2.x API)。"""
    return Path(__file__).parent / "mcp_echo_server.py"


def test_mcp_connect_list_call(echo_server_path: Path) -> None:
    from my_agent_alpha.mcp_client import SyncMCPClient

    server = McpServer(name="t", command=sys.executable, args=[str(echo_server_path)])
    client = SyncMCPClient(server)
    try:
        names = [t.name for t in client.list_tools()]
        assert "echo" in names and "add" in names
        assert client.call("echo", {"text": "你好"}) == "echo: 你好"
        assert client.call("add", {"a": 2, "b": 3}) == "5"
    finally:
        client.close()


def test_mcp_timeout_readable_and_not_wedged(echo_server_path: Path) -> None:
    """超时两件事(2026-10-04 真实会话踩坑,见 docs/踩坑记录.md 条目 2):

    1. 报错是人话:TimeoutError 的 str() 是空串,必须翻译成可读的 RuntimeError,
       否则模型只能看到"执行失败: "后面一片空白;
    2. 超时在协程内取消单次调用,同一连接不被一次挂起堵死。
    """
    from my_agent_alpha.mcp_client import SyncMCPClient

    server = McpServer(name="t", command=sys.executable, args=[str(echo_server_path)], timeout=0.5)
    client = SyncMCPClient(server)
    try:
        # slow 工具睡 3 秒,客户端 0.5 秒就该报超时(而不是等满 3 秒或永远挂着)
        with pytest.raises(RuntimeError, match="超时"):
            client.call("slow", {"seconds": 3})
        # 关键回归:超时的调用已被取消,队列没被堵死,下一个调用正常返回
        assert client.call("echo", {"text": "还活着"}) == "echo: 还活着"
    finally:
        client.close()


def test_mcp_tools_become_first_class_specs(echo_server_path: Path) -> None:
    """远端工具 → 本地 ToolSpec(名字带服务器前缀、参数表换算、可执行)。"""
    from my_agent_alpha.mcp_client import create_mcp_tools

    specs, clients = create_mcp_tools(
        (McpServer(name="t", command=sys.executable, args=[str(echo_server_path)]),)
    )
    try:
        names = {s.name for s in specs}
        assert names == {"t.echo", "t.add", "t.slow"}
        echo_spec = next(s for s in specs if s.name == "t.echo")
        assert "Echo" in echo_spec.desc
        assert [p.name for p in echo_spec.parameters] == ["text"]
        # 手动把远端工具塞进注册表再解析(parse 要查注册表)
        from my_agent_alpha import tools as tools_pkg

        tools_pkg.add_tools(specs)
        results = execute_msg(Message("assistant", '@t.echo(c1): {"text": "hi"}'))
        assert results[0].content == "echo: hi"
        assert results[0].call_id == "c1"
        # markdown 围栏通道也必须能调(block_types 全名 + 围栏内容是 JSON 参数体)
        results2 = execute_msg(Message("assistant", '```t.echo\n{"text": "hi"}\n```'))
        assert results2[0].content == "echo: hi"
    finally:
        for c in clients:
            c.close()


def test_mcp_spec_has_markdown_instructions(echo_server_path: Path) -> None:
    """MCP 工具的说明书要自带 JSON 参数体说明和可照抄的示例。

    markdown 通道下参数表对模型不可见(只有原生通道转 JSON Schema 时用),
    不把参数名写进提示词,模型只能瞎猜 —— 真实会话里 kimi 连猜两次:
    裸文本、{"q": ...}(正确键名是 query),每错一次浪费一轮交互。
    """
    from my_agent_alpha.mcp_client import create_mcp_tools

    specs, clients = create_mcp_tools(
        (McpServer(name="t", command=sys.executable, args=[str(echo_server_path)]),)
    )
    try:
        echo_spec = next(s for s in specs if s.name == "t.echo")
        # 说明里讲清格式约定,并列出参数名和必填性
        assert "JSON 参数体" in echo_spec.instructions
        assert "text" in echo_spec.instructions and "必填" in echo_spec.instructions
        # 示例是可直接照抄的围栏,键名 = 参数名
        assert "```t.echo" in echo_spec.examples
        assert '"text"' in echo_spec.examples
        # 渲染进系统提示词后依然完整(get_tool_prompt 是提示词的拼装入口)
        rendered = echo_spec.get_tool_prompt()
        assert "JSON 参数体" in rendered and "```t.echo" in rendered
    finally:
        for c in clients:
            c.close()


def test_mcp_bad_server_skipped() -> None:
    """连不上的服务器:告警跳过,不拖垮 create_mcp_tools。"""
    from my_agent_alpha.mcp_client import create_mcp_tools

    bad = McpServer(
        name="bad",
        command=sys.executable,
        args=["-c", "import sys; sys.exit(1)"],  # 启动即退出
    )
    specs, clients = create_mcp_tools((bad,))
    assert specs == () and clients == []
