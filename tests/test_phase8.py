"""阶段 8 测试:长期记忆、subagent、MCP。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.config import Config, McpServer
from my_agent_alpha.context import set_current_model
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.memory_store import (
    list_memories,
    memory_dirs,
    render_memory_index,
    save_memory,
)
from my_agent_alpha.tools import execute_msg

# ============ 长期记忆 ============


@pytest.fixture
def clean_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把用户层数据目录也指到 tmp(不然测试会污染真实的 ~/.local/share)。"""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    return tmp_path


def test_save_and_list(clean_dirs: Path) -> None:
    ws = clean_dirs / "ws"
    ws.mkdir()
    save_memory("部署流程", "服务器发布步骤\n1. ssh …", ws, project_level=True)
    entries = list_memories(ws)
    assert [e.name for e in entries] == ["部署流程"]
    assert entries[0].title == "服务器发布步骤"
    # 索引文件生成了,且有指针
    index = (ws / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "[部署流程](部署流程.md)" in index


def test_two_layers_shadowing(clean_dirs: Path) -> None:
    """同名记忆:项目层盖住用户层(gptme 的 nearest-first)。"""
    ws = clean_dirs / "ws"
    ws.mkdir()
    save_memory("偏好", "用户层的偏好", ws, project_level=False)
    save_memory("偏好", "项目层的偏好", ws, project_level=True)
    entries = list_memories(ws)
    assert len(entries) == 1  # 同名只剩一个
    assert "项目层" in entries[0].path.read_text(encoding="utf-8")
    # 两层的目录都真实存在文件,只是索引取近层
    assert len(memory_dirs(ws)) == 2


def test_render_index_for_prompt(clean_dirs: Path) -> None:
    ws = clean_dirs / "ws"
    ws.mkdir()
    save_memory("部署流程", "服务器发布步骤\n1. ssh …", ws, project_level=True)
    text = render_memory_index(ws)
    assert "长期记忆" in text and "部署流程" in text
    # 链接必须带路径,模型才能直接 read(项目层 → memory/xxx.md)
    assert "](memory/部署流程.md)" in text
    # 空目录 → 空文本(会话就不注入)
    assert render_memory_index(clean_dirs / "nope") == ""


def test_memory_tool_writes_project_layer(clean_dirs: Path) -> None:
    msg = Message("assistant", "```memory 测试记忆\n这是标题\n这是正文\n```")
    results = execute_msg(msg)  # 无确认钩子 = 自动放行
    assert "已记住" in results[0].content
    # 写到了当前工作区(测试夹具设过 workspace)的 memory/ 目录
    from my_agent_alpha.context import get_workspace

    assert (get_workspace() / "memory" / "测试记忆.md").exists()


def test_e2e_memory_injected_into_session(
    clean_dirs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPHA_LOGS_HOME", str(clean_dirs / "logs"))
    ws = clean_dirs / "ws"
    ws.mkdir()
    save_memory("旧约定", "回答要简短", ws, project_level=True)
    set_scripted_replies(["好的"])
    chat(
        ["hi"],
        logdir=clean_dirs / "logs" / "c1",
        workspace=ws,
        model="mock/echo",
        non_interactive=True,
    )
    manager = LogManager.load(clean_dirs / "logs" / "c1")
    # 精确匹配注入的记忆索引(系统提示词里也有"长期记忆"字样,别撞上)
    seeded = [m for m in manager.messages if m.content.startswith("# 长期记忆")]
    assert seeded and "旧约定" in seeded[0].content
    assert seeded[0].pinned and seeded[0].hide  # 提示词类消息


# ============ subagent ============


def test_subagent_tool_spawns_and_returns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """子代理用 mock 模型离线跑:派活 → 子进程真跑 → 结论回传。"""
    monkeypatch.setenv("ALPHA_LOGS_HOME", str(tmp_path / "logs"))
    set_current_model("mock/echo")  # 子代理继承父会话的模型
    from my_agent_alpha import context

    (tmp_path / "ws").mkdir()
    with context.agent_context(tmp_path / "ws", tmp_path / "logs" / "parent"):
        msg = Message("assistant", "```subagent\n帮我说声你好\n```")
        results = execute_msg(msg)
    texts = [m.content for m in results]
    assert any("已派出子代理" in t for t in texts)
    final = texts[-1]
    assert "子代理" in final and "Echo" in final  # 子代理的回复回来了
    # 子代理的会话日志真的在磁盘上
    sub_dirs = [d for d in (tmp_path / "logs").iterdir() if d.name.startswith("subagent-")]
    assert len(sub_dirs) == 1


def test_subagent_native_kwargs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHA_LOGS_HOME", str(tmp_path / "logs"))
    set_current_model("mock/echo")
    from my_agent_alpha import context

    (tmp_path / "ws").mkdir()
    with context.agent_context(tmp_path / "ws", tmp_path / "logs" / "p"):
        msg = Message("assistant", '@subagent(s1): {"prompt": "算一下 1+1"}')
        results = execute_msg(msg)
    assert any("Echo" in m.content for m in results)


def test_e2e_parent_dispatches_subagent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """端到端:父代理(剧本)决定派子代理,真实子进程跑完回传,父收尾。"""
    monkeypatch.setenv("ALPHA_LOGS_HOME", str(tmp_path / "logs"))
    ws = tmp_path / "ws"
    ws.mkdir()
    set_scripted_replies(
        [
            "这活儿独立,我派个 subagent:\n```subagent\n报数:一到三\n```",
            "子代理已答复,任务完成。",
        ]
    )
    chat(
        ["派活"],
        logdir=tmp_path / "logs" / "parent",
        workspace=ws,
        model="mock/echo",
        non_interactive=True,
    )
    manager = LogManager.load(tmp_path / "logs" / "parent")
    contents = [m.content for m in manager.messages]
    assert any("✅ 子代理" in c and "Echo" in c for c in contents)
    assert any("任务完成" in c for c in contents)


# ============ MCP ============


@pytest.fixture(scope="module")
def echo_server_path() -> Path:
    """测试用的 MCP 服务器脚本(内容见 tests/mcp_echo_server.py,SDK 2.x API)。"""
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


def test_mcp_tools_become_first_class_specs(echo_server_path: Path) -> None:
    """远端工具 → 本地 ToolSpec(名字带服务器前缀、参数表换算、可执行)。"""
    from my_agent_alpha.mcp_client import create_mcp_tools

    specs, clients = create_mcp_tools(
        (McpServer(name="t", command=sys.executable, args=[str(echo_server_path)]),)
    )
    try:
        names = {s.name for s in specs}
        assert names == {"t.echo", "t.add"}
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


def test_config_parses_mcp_servers(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.toml").write_text(
        'model = "glm/glm-5.3"\n'
        "[[mcp.servers]]\n"
        'name = "weather"\n'
        'command = "python"\n'
        'args = ["-m", "weather_server"]\n',
        encoding="utf-8",
    )
    config = Config.load(workspace=tmp_path / "ws", user_config_dir=cfg_dir)
    assert len(config.mcp_servers) == 1
    assert config.mcp_servers[0].name == "weather"
    assert config.mcp_servers[0].args == ("-m", "weather_server")
