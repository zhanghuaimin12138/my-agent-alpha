"""my_agent_alpha —— 仿造 gptme 的个人 AI agent(学习型重实现)。

架构蓝图见 docs/gptme-study.md;开发记录见 docs/devlog/。
"""

from my_agent_alpha.logmanager import Log, LogManager, generate_conversation_id
from my_agent_alpha.message import Message, Role, utcnow

__version__ = "0.1.0"

__all__ = [
    "Log",
    "LogManager",
    "Message",
    "Role",
    "__version__",
    "generate_conversation_id",
    "utcnow",
]


def main() -> None:
    """控制台入口(阶段 0 占位:打印版本;阶段 1 换成 CLI 对话循环)。"""
    print(f"my-agent-alpha {__version__}")
