"""my_agent_alpha —— 仿造 gptme 的个人 AI agent(学习型重实现)。

架构蓝图见 docs/gptme-study.md;开发记录见 docs/devlog/。
"""

from __future__ import annotations

import argparse
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

# 模型/通道的最终取值在 main() 里按 命令行 > 环境变量 > 配置文件 解析(见 config.py)


def main(argv: list[str] | None = None) -> None:
    """控制台入口:解析参数 → 装配会话 → 进入对话循环。

    主干只留流程编排,三块细节各收进一个嵌套函数:
    build_parser(命令行参数)、
    load_settings(配置与生效值)、
    make_logdir(会话目录)。
    """

    def build_parser() -> argparse.ArgumentParser:
        """定义全部命令行参数。"""
        parser = argparse.ArgumentParser(
            prog="my-agent-alpha",
            description="个人 AI agent",
        )
        parser.add_argument("prompts", nargs="*", help="开场提示词;缺省进入交互模式")
        parser.add_argument(
            "-m", "--model", default=None, help="模型名,如 deepseek/deepseek-chat(默认取配置)"
        )
        parser.add_argument(
            "-n", "--non-interactive", action="store_true", help="跑完队列提示词即退出"
        )
        parser.add_argument("--name", help="会话名(logdir 目录名);缺省自动生成")
        parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="agent 的工作目录")
        parser.add_argument(
            "--tool-format",
            choices=["markdown", "tool"],
            default=None,
            help="工具调用通道:markdown(围栏代码块)或 tool(原生 function calling)",
        )
        parser.add_argument(
            "-y", "--yes", action="store_true", help="全部自动确认(安全护栏仍然生效)"
        )
        parser.add_argument(
            "--tools",
            default=None,
            help="工具白名单(逗号分隔,如 read,grep,memory);缺省 = 全部内置工具",
        )
        parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
        return parser

    def load_settings(args: argparse.Namespace) -> tuple[str, str]:
        """初始化配置(最大步数+工具解析模式+生效模型)。

        (命令行 > 环境变量 > 配置文件(项目>用户))、同一批 effective_* 函数、同时解析
        同时使用,所以收在同一个函数里;max_steps 同族,也在这里落位
        (改的是 chat 模块的全局上限,属于副作用)。
        """
        from my_agent_alpha import chat as chat_mod
        from my_agent_alpha.config import (
            Config,
            effective_max_steps,
            effective_model,
            effective_tool_format,
        )

        # 加载用户配置和项目配置（项目覆盖用户）
        config = Config.load(args.workspace)

        # 激活配置
        model = effective_model(args.model, config)  # 激活模型
        tool_format = effective_tool_format(args.tool_format, config)
        chat_mod.MAX_STEPS = effective_max_steps(config, chat_mod.MAX_STEPS)
        return model, tool_format

    def make_logdir(args: argparse.Namespace) -> Path:
        """获取持久化日志目录(当前工作区->日志所在目录)+处理命名参数(--name)。

        目录形如(见 dirs.py):
        ~/.local/share/my-agent-alpha/projects/-home-yby-projects-…-d15c78a0/logs/会话名
        """
        from my_agent_alpha.dirs import get_session_root
        from my_agent_alpha.hooks.naming import write_session_title

        session_name = args.name or generate_conversation_id()
        logdir = get_session_root(args.workspace) / session_name
        if args.name:
            # 会话命名
            write_session_title(logdir, args.name)
        return logdir

    args = build_parser().parse_args(argv)

    from my_agent_alpha.chat import chat
    from my_agent_alpha.tools import restrict_tools

    prompts = list(args.prompts)
    if prompts == ["-"]:  # "-" = 从标准输入读提示词(给 subagent 用,防超长参数)
        prompts = [sys.stdin.read()]

    if args.tools:
        # 白名单要先于一切工具使用生效;没匹配上的名字如实告警
        # (多半是拼写错,或该工具来自未连接的 MCP 服务器)
        missing = restrict_tools([t.strip() for t in args.tools.split(",") if t.strip()])
        if missing:
            print(f"(警告: 白名单里的这些工具不存在,已忽略: {', '.join(missing)})")

    model, tool_format = load_settings(args)
    logdir = make_logdir(args)
    non_interactive = args.non_interactive or bool(args.prompts)

    try:
        chat(
            prompts,
            logdir=logdir,
            workspace=args.workspace,
            model=model,
            non_interactive=non_interactive,
            tool_format=tool_format,
            yes=args.yes,
        )
    except KeyboardInterrupt:
        sys.exit(130)
