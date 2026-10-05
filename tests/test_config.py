"""阶段 5 分层配置测试:用户层 → 项目层 → 环境变量 → 命令行。"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha.config import (
    Config,
    effective_max_steps,
    effective_model,
    effective_tool_format,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_no_config_files_gives_defaults(tmp_path: Path) -> None:
    config = Config.load(workspace=tmp_path / "ws", user_config_dir=tmp_path / "cfg")
    assert config.model is None
    assert config.tool_format is None
    assert config.max_steps is None


def test_user_layer_loaded(tmp_path: Path) -> None:
    _write(tmp_path / "cfg" / "config.toml", 'model = "glm/glm-5.3"\nmax_steps = 30\n')
    config = Config.load(workspace=tmp_path / "ws", user_config_dir=tmp_path / "cfg")
    assert config.model == "glm/glm-5.3"
    assert config.max_steps == 30
    assert config.tool_format is None


def test_project_layer_overrides_user(tmp_path: Path) -> None:
    _write(tmp_path / "cfg" / "config.toml", 'model = "glm/glm-5.3"\nmax_steps = 30\n')
    ws = tmp_path / "ws"
    _write(ws / "alpha.toml", 'model = "deepseek/deepseek-chat"\n')
    config = Config.load(workspace=ws, user_config_dir=tmp_path / "cfg")
    assert config.model == "deepseek/deepseek-chat"  # 项目覆盖用户
    assert config.max_steps == 30  # 项目没配的键,用户层的值保留


def test_mcp_servers_parsed_with_timeout(tmp_path: Path) -> None:
    """[[mcp.servers]] 条目能解析出 name/command/args/env,timeout 可选、默认 120。"""
    ws = tmp_path / "ws"
    _write(
        ws / "alpha.toml",
        "\n".join(
            [
                "[[mcp.servers]]",
                'name = "websearch"',
                'command = "npx"',
                'args = ["-y", "some-server"]',
                "timeout = 180",
                "",
                "[mcp.servers.env]",
                'USE_PROXY = "true"',
            ]
        )
        + "\n",
    )
    config = Config.load(workspace=ws, user_config_dir=tmp_path / "cfg")
    assert len(config.mcp_servers) == 1
    server = config.mcp_servers[0]
    assert (server.name, server.command, server.args) == ("websearch", "npx", ("-y", "some-server"))
    assert server.timeout == 180.0
    assert server.env == {"USE_PROXY": "true"}
    # 不写 timeout → 默认 120 秒
    _write(ws / "alpha.toml", '[[mcp.servers]]\nname = "t"\ncommand = "echo"\n')
    config2 = Config.load(workspace=ws, user_config_dir=tmp_path / "cfg")
    assert config2.mcp_servers[0].timeout == 120.0


def test_broken_toml_ignored(tmp_path: Path) -> None:
    _write(tmp_path / "cfg" / "config.toml", "这不是合法 toml [[[\n")
    _write(tmp_path / "ws" / "alpha.toml", 'model = "glm/glm-5.3"\n')
    config = Config.load(workspace=tmp_path / "ws", user_config_dir=tmp_path / "cfg")
    assert config.model == "glm/glm-5.3"  # 坏文件被忽略,好文件照常生效


def test_effective_model_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 本机 .bashrc 可能设了 ALPHA_MODEL(真实环境),先摘掉保证测试自洽
    monkeypatch.delenv("ALPHA_MODEL", raising=False)
    config = Config(model="glm/glm-5.3")
    assert effective_model(None, config) == "glm/glm-5.3"  # 配置兜底
    assert effective_model("deepseek/chat", config) == "deepseek/chat"  # 命令行最大
    monkeypatch.setenv("ALPHA_MODEL", "moonshot/kimi")
    assert effective_model(None, config) == "moonshot/kimi"  # 环境变量 > 配置
    assert effective_model("deepseek/chat", config) == "deepseek/chat"  # 命令行仍最大


def test_effective_tool_format_and_max_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert effective_tool_format(None, Config()) == "markdown"  # 默认
    assert effective_tool_format("tool", Config()) == "tool"
    assert effective_tool_format(None, Config(tool_format="tool")) == "tool"
    assert effective_max_steps(Config(max_steps=7), env_default=50) == 7
    assert effective_max_steps(Config(), env_default=50) == 50
