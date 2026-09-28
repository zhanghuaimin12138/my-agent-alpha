"""my_agent_alpha —— 仿造 gptme 的个人 AI agent(学习型重实现)。

架构蓝图见 docs/gptme-study.md;开发记录见 docs/devlog/。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from my_agent_alpha.logmanager import (
    Log,
    LogManager,
    generate_conversation_id,
)
from my_agent_alpha.message import Message, Role, utcnow

__version__ = "0.1.0"

__all__ = [
    "Log",
    "LogManager",
    "Message",
    "Role",
    "__version__",
    "generate_conversation_id",
    "main",
    "utcnow",
]

_DEFAULT_MODEL = os.environ.get("ALPHA_MODEL", "mock/echo")


def main(argv: list[str] | None = None) -> None:
    """控制台入口:解析参数 → 装配会话 → 进入对话循环。"""
    parser = argparse.ArgumentParser(
        prog="my-agent-alpha",
        description="个人 AI agent(仿 gptme 的学习型重实现)",
    )
    parser.add_argument("prompts", nargs="*", help="开场提示词;缺省进入交互模式")
    parser.add_argument(
        "-m", "--model", default=_DEFAULT_MODEL, help="模型名,如 deepseek/deepseek-chat"
    )
    parser.add_argument("-n", "--non-interactive", action="store_true", help="跑完队列提示词即退出")
    parser.add_argument("--name", help="会话名(logdir 目录名);缺省自动生成")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="agent 的工作目录")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    from my_agent_alpha.chat import chat
    from my_agent_alpha.dirs import get_logs_dir

    session_name = args.name or generate_conversation_id()
    logdir = get_logs_dir() / session_name
    non_interactive = args.non_interactive or bool(args.prompts)

    try:
        chat(
            args.prompts,
            logdir=logdir,
            workspace=args.workspace,
            model=args.model,
            non_interactive=non_interactive,
        )
    except KeyboardInterrupt:
        sys.exit(130)
