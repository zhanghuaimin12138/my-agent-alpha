"""斜杠命令框架:/ 开头的用户输入走这里,不进对话。

用法(命令处理器的写法):

    @command("greet", "hi")           # 注册 /greet,别名 /hi
    def greet_cmd(ctx: CommandContext) -> None:
        print(f"你好,{ctx.args}")

处理器拿到 CommandContext(参数 + 会话管理器 + 模型名),
直接 print 输出即可;需要留痕时可往 ctx.manager 里 append 消息。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from my_agent_alpha.logmanager import LogManager

# 命令处理器:拿到上下文,做任何事(通常打印)
CommandHandler = Callable[["CommandContext"], None]
# 函数变量，传入上下文(有manager和model)，返回none的函数


@dataclass(frozen=True)
class CommandContext:
    """传给命令处理器的全部环境。

    示例:输入 "/model glm-5.3" →
        ctx.args = ["glm-5.3"], ctx.full = "glm-5.3"
    """

    args: list[str]  # 按空白切开的参数
    full: str  # 命令名之后的完整原文(参数里要保留空格时用它)
    manager: LogManager  # 会话管理器(可查消息、可追加)
    model: str  # 当前模型名


# 注册表:命令名/别名 → 处理器
# (记录别名到同一个处理器的映射;注销时两边都要清)
_commands: dict[str, CommandHandler] = {}
_owned_aliases: dict[str, list[str]] = {}  # 命令名 → 它的别名(注销时用)


def command(name: str, *aliases: str) -> Callable[[CommandHandler], CommandHandler]:
    """装饰器:注册一个斜杠命令。

    示例:
        @command("help", "?")
        def help_cmd(ctx): ...
        → /help 和 /? 都会调它

        help_cmd就是一个commandHandler
    """

    def decorator(fn: CommandHandler) -> CommandHandler:  # 函数还没传，在@command（）下面传
        # name，aliases，fn写进注册表的逻辑
        _commands[name] = fn
        for alias in _owned_aliases.get(name, []):
            _commands.pop(alias, None)  # 换掉旧别名
        _owned_aliases[name] = list(aliases)
        for alias in aliases:
            _commands[alias] = fn
        return fn

    return decorator


def unregister_command(name: str) -> None:
    """按名字注销一个命令(连别名一起)。"""
    for alias in _owned_aliases.pop(name, []):
        _commands.pop(alias, None)
    _commands.pop(name, None)


def get_commands() -> dict[str, CommandHandler]:
    """全部已注册命令(名字 → 处理器),供 /help 展示。"""
    return dict(_commands)


def execute_cmd(content: str, manager: LogManager, model: str) -> bool:
    """处理一条用户输入:是已注册命令就执行,返回 True;否则返回 False
    (调用方应把它当普通对话内容处理)。

    命令输入不会进日志(阶段 1 以来的行为,保持)。
    执行命令的函数（在此有传入上下文的逻辑）
    """
    if not content.startswith("/"):
        return False
    name, _, rest = content.lstrip("/").partition(" ")
    handler = _commands.get(name.strip().lower())
    if handler is None:
        return False
    ctx = CommandContext(
        args=rest.split(),
        full=rest.strip(),
        manager=manager,
        model=model,
    )
    handler(ctx)
    return True
