"""路径计算:程序的数据应该放在哪。

Linux 有个惯例(叫 XDG 规范):用户数据放 ~/.local/share/<程序名>/,
配置放 ~/.config/<程序名>/,而不是在家目录里到处建点开头的文件夹。
我们遵守这个惯例;支持 Windows 时再换 platformdirs 库处理。
"""

from __future__ import annotations

import os
from pathlib import Path

# 环境变量统一用 ALPHA_ 前缀(对应 gptme 的 GPTME_ 前缀)
_LOGS_HOME_ENV = "ALPHA_LOGS_HOME"


def get_data_dir() -> Path:
    """返回程序的数据目录(不存在则创建)。

    默认 ~/.local/share/my_agent_alpha;
    如果系统设置了 XDG_DATA_HOME 环境变量,则用它替代 ~/.local/share。
    """
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    data_dir = base / "my_agent_alpha"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_logs_dir() -> Path:
    """返回所有会话日志的根目录(不存在则创建)。

    默认是 数据目录/logs,即 ~/.local/share/my_agent_alpha/logs;
    设置环境变量 ALPHA_LOGS_HOME 可以改到别处(测试时用来和真实数据隔离)。
    """
    override = os.environ.get(_LOGS_HOME_ENV)
    logs_dir = Path(override).expanduser() if override else get_data_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_config_dir() -> Path:
    """返回配置目录(不存在则创建):~/.config/my_agent_alpha。

    遵守 XDG_CONFIG_HOME;用户级 config.toml 放这里。
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    config_dir = base / "my_agent_alpha"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir
