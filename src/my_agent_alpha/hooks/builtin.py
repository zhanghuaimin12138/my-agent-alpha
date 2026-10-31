"""内置钩子:开箱即用的那几个。

演示钩子系统的价值:这两个功能都不需要改 chat.py 一行代码。
"""

from __future__ import annotations

from collections.abc import Iterator

from my_agent_alpha.hooks import HookType, register_hook
from my_agent_alpha.logmanager import LogManager
from my_agent_alpha.message import Message


def usage_hook(message: Message, **_ctx: object) -> Iterator[Message]:
    """generation.post:模型回复后,把 token 用量记成一条消息。

    产出的消息会进日志 —— 于是也进了模型上下文(gptme 的"token 感知":
    模型看得见自己花了多少,长对话时多少会收敛一点)。

    示例产出:[用量] 输入 123 / 输出 456 tokens
    """
    usage = (message.metadata or {}).get("usage") if message.metadata else None
    if usage:
        yield Message(
            "system",
            f"[用量] 输入 {usage.get('input', '?')} / 输出 {usage.get('output', '?')} tokens",
        )


def session_info_hook(manager: LogManager, **_ctx: object) -> Iterator[Message]:
    """session.start:会话开始时报告会话目录(纯打印,不产消息)。

    注意:钩子也可以是普通函数(不一定是生成器),只要返回一个消息迭代器;
    这里返回"空的",表示本钩子只打印、不产出。
    """
    print(f"(会话目录:{manager.logdir})")
    return iter(())


def register_builtin_hooks() -> None:
    """注册全部内置钩子(幂等:同名重注册自动替换)。"""
    register_hook("builtin.usage", HookType.GENERATION_POST, usage_hook, priority=0)
    register_hook("builtin.session_info", HookType.SESSION_START, session_info_hook, priority=0)

    # 会话自动命名(turn.post);todo 工具的门控与重放;上下文自动压缩(turn.post)
    from my_agent_alpha.hooks.autocompact import register_autocompact_hook
    from my_agent_alpha.hooks.naming import register_naming_hook
    from my_agent_alpha.hooks.todo import register_todo_hooks

    register_naming_hook()
    register_todo_hooks()
    register_autocompact_hook()
