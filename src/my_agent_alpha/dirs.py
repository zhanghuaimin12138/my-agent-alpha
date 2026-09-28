"""XDG 风格的路径解析(零依赖版)。

阶段 0 只面向 Linux/macOS(手动解析 XDG 环境变量);
将来需要支持 Windows 时再引入 platformdirs(gptme 的选择)。
"""

from __future__ import annotations

import os
from pathlib import Path

# 环境变量统一用 ALPHA_ 前缀(对应 gptme 的 GPTME_ 前缀)
_LOGS_HOME_ENV = "ALPHA_LOGS_HOME"


def get_data_dir() -> Path:
    """数据目录:<XDG_DATA_HOME 或 ~/.local/share>/my_agent_alpha(不存在则创建)。"""
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    data_dir = base / "my_agent_alpha"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_logs_dir() -> Path:
    """会话日志根目录,可用 ALPHA_LOGS_HOME 覆盖(测试与多环境隔离用)。"""
    override = os.environ.get(_LOGS_HOME_ENV)
    logs_dir = Path(override).expanduser() if override else get_data_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir
