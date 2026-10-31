"""分层配置:用户配置 → 项目配置 → 环境变量,层层覆盖。

三处配置(优先级从低到高):

1. 用户配置:~/.config/my_agent_alpha/config.toml(全局默认);
2. 项目配置:工作区(workspace)/alpha.toml (每个项目自己的偏好);
3. 环境变量:ALPHA_MODEL / ALPHA_TOOL_FORMAT(临时覆盖,最高)。

示例文件内容(alpha.toml 或 config.toml):

    model = "glm/glm-5.3"
    tool_format = "tool"
    max_steps = 30

原则:文件坏了(格式错误)不崩溃 —— 告警并忽略该文件;
没配的键回落到代码默认值。
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_CONFIG_FILENAME = "alpha.toml"


@dataclass(frozen=True)
class McpServer:
    """一个 MCP 服务器的启动配置单(stdio 模式:拉起一个子进程,用它的工具)。

    对应 alpha.toml 里的:
        [[mcp.servers]]
        name = "weather"
        command = "python"
        args = ["-m", "weather_server"]
        timeout = 180  # 可选:这个服务器的工具调用超时秒数(默认 120)
    """

    name: str  # 起的名字
    command: str  # 用什么程序启动
    args: tuple[str, ...] = ()  # 启动时传递的参数
    env: dict[str, str] | None = None  # 启动时给他的环境变量
    # 单次工具调用的超时秒数。搜索/渲染网页类工具天然偏慢,默认给足 120 秒;
    # 可以按服务器调(快服务器调小,慢工具调大)
    timeout: float = 120.0


@dataclass(frozen=True)
class Config:
    """读取并合并好的配置。值为 None 表示"这一层没配",用代码默认值。"""

    model: str | None = None
    tool_format: str | None = None
    max_steps: int | None = None
    mcp_servers: tuple[McpServer, ...] = ()

    @classmethod
    def load(cls, workspace: Path, user_config_dir: Path | None = None) -> Config:
        """读用户配置和项目配置,合并(项目覆盖用户)。

        user_config_dir 参数是给测试注入临时目录用的;不传就是默认位置。
        """
        from my_agent_alpha.dirs import get_config_dir  # 用户级配置的目录，~/.config/my_agent_alpha

        base_dir = user_config_dir if user_config_dir is not None else get_config_dir()
        merged: dict[str, object] = {}
        merged.update(_read_toml(base_dir / "config.toml"))
        merged.update(_read_toml(workspace / PROJECT_CONFIG_FILENAME))
        return cls(
            model=_as_str(merged.get("model")),
            tool_format=_as_str(merged.get("tool_format")),
            max_steps=_as_int(merged.get("max_steps")),
            mcp_servers=_parse_mcp_servers(merged.get("mcp")),
        )


def _parse_mcp_servers(raw: object) -> tuple[McpServer, ...]:
    """解析 [mcp] 表下的 servers 列表;格式不对的条目跳过并告警。"""
    if not isinstance(raw, dict):
        return ()
    servers_raw = raw.get("servers")
    if not isinstance(servers_raw, list):
        return ()
    servers: list[McpServer] = []
    for item in servers_raw:
        if not isinstance(item, dict) or "name" not in item or "command" not in item:
            logger.warning("忽略不完整的 mcp.servers 条目(需要 name 和 command):%r", item)
            continue
        env = item.get("env") if isinstance(item.get("env"), dict) else None
        raw_timeout = item.get("timeout", 120.0)
        timeout = float(raw_timeout) if isinstance(raw_timeout, (int, float)) else 120.0
        servers.append(
            McpServer(
                name=str(item["name"]),
                command=str(item["command"]),
                args=tuple(str(a) for a in item.get("args", [])),
                env={str(k): str(v) for k, v in env.items()} if env else None,
                timeout=timeout,
            )
        )
    return tuple(servers)


def _read_toml(path: Path) -> dict[str, object]:
    """读一个 TOML 文件;不存在返回空,格式坏了告警并当空(不许配置拖垮程序)。"""
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        logger.warning("配置文件读取失败,已忽略(%s): %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def effective_model(cli_value: str | None, config: Config) -> str:
    """决定最终用哪个模型:命令行 > 环境变量 > 项目配置 > 用户配置 > 默认。

    示例:effective_model(None, config) → "glm/glm-5.3"(来自配置)
         effective_model("deepseek/chat", config) → "deepseek/chat"(命令行最大)
    """
    if cli_value:
        return cli_value
    env_value = os.environ.get("ALPHA_MODEL")
    if env_value:
        return env_value
    return config.model or "mock/echo"


def effective_tool_format(cli_value: str | None, config: Config) -> str:
    """决定工具调用通道,优先级同 effective_model。默认 tool(原生 function calling)。"""
    if cli_value:
        return cli_value
    env_value = os.environ.get("ALPHA_TOOL_FORMAT")
    if env_value:
        return env_value
    return config.tool_format or "tool"


def effective_max_steps(config: Config, env_default: int) -> int:
    """决定单回合步数上限:配置文件 > 环境变量默认值(启动时的 ALPHA_MAX_STEPS)。"""
    return config.max_steps if config.max_steps is not None else env_default
