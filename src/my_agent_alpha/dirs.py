"""路径计算:程序的数据应该放在哪。

Linux 有个惯例(叫 XDG 规范):用户数据放 ~/.local/share/<程序名>/,
配置放 ~/.config/<程序名>/,而不是在家目录里到处建点开头的文件夹。
我们遵守这个惯例;支持 Windows 时再换 platformdirs 库处理。
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

"""
程序的数据目录：
XDG_DATA_HOME/my-agent-alpha
~/.local/share/my-agent-alpha (XDG_DATA_HOME环境变量未设置)

程序配置目录：
XDG_CONFIG_HOME/my-agent-alpha
~/.config/my-agent-alpha (XDG_CONFIG_HOME环境变量未设置)

"""


# 环境变量统一用 ALPHA_ 前缀(对应 gptme 的 GPTME_ 前缀)
_LOGS_HOME_ENV = "ALPHA_LOGS_HOME"  # 设置了即替代 数据目录/projects

# 数据/配置目录名(连字符,2026-10-26 日志架构重构定稿)
_APP_DIR = "my-agent-alpha"


def get_data_dir() -> Path:
    """返回程序的数据目录(不存在则创建)。

    默认返回 ~/.local/share/my-agent-alpha;
    如果系统设置了 XDG_DATA_HOME 环境变量,则用它替代 ~/.local/share。
    """
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    data_dir = base / _APP_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def workspace_key(workspace: Path) -> str:
    """工作区桶名:可读前缀(给人看)+ 路径指纹(保证唯一)。
    workspace->绝对路径->日志目录名称（~/.local/share/my-agent-alpha/projects/<得到这个名称>/logs/会话id/)

    三步:
    1. 归一 —— expanduser+resolve 把 . 、.. 、结尾斜杠、~ 、符号链接
       统一翻译成唯一的绝对真实路径:同一个目录不管怎么写进来,键都相同;
    2. 姓名段 —— 真实路径里非字母数字换成 -(连续合并),只给人看,
       不承担唯一性;
    3. 指纹 —— 真实路径 SHA-256 的前 8 位十六进制(约 43 亿种组合,
       个人量级撞不上)。孪生命名(如 my_agent_alpha 与 my-agent-alpha)
       的姓名段相同,靠指纹必然分桶 —— Claude Code 无此机制,实测会串记忆。

    示例:/home/yby/projects/my_agent_alpha(workspace)
        → -home-yby-projects-my-agent-alpha-d15c78a0
    """
    real = str(workspace.expanduser().resolve())
    readable = re.sub(r"[^A-Za-z0-9]+", "-", real)
    digest = hashlib.sha256(real.encode("utf-8")).hexdigest()[:8]
    return f"{readable}-{digest}"


def get_session_root(workspace: Path) -> Path:
    """返回某个工作区的会话日志根目录(不存在则创建)。

    默认 数据目录/projects/<工作区键>/logs;
    设置环境变量 ALPHA_LOGS_HOME 时,用它替代 projects/ 这一层
    (测试时用来和真实数据隔离)。

    示例:~/.local/share/my-agent-alpha/projects/-home-yby-projects-…-d15c78a0/logs
    """
    override = os.environ.get(_LOGS_HOME_ENV)
    base = Path(override).expanduser() if override else get_data_dir() / "projects"
    root = base / workspace_key(workspace) / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_config_dir() -> Path:
    """返回配置目录(不存在则创建):~/.config/my-agent-alpha。

    遵守 XDG_CONFIG_HOME;用户级 config.toml 放这里。
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    config_dir = base / _APP_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir
