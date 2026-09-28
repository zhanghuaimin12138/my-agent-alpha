"""工具注册表与执行入口。

阶段 1:显式内置工具列表(shell/read/save/append);模块扫描发现、
允许列表、MCP 接入是后续阶段的事。
"""

from __future__ import annotations

from contextvars import ContextVar

from my_agent_alpha.message import INTERRUPT_CONTENT, Message
from my_agent_alpha.tools import read as read_mod
from my_agent_alpha.tools import save as save_mod
from my_agent_alpha.tools import shell as shell_mod
from my_agent_alpha.tools.base import ToolSpec, ToolUse

_tools_var: ContextVar[tuple[ToolSpec, ...] | None] = ContextVar("alpha_tools", default=None)

# 内置工具模块(每个模块暴露模块级 tool / 多个 tool)
_BUILTIN_MODULES = (shell_mod, read_mod, save_mod)


def _collect_builtins() -> tuple[ToolSpec, ...]:
    specs: list[ToolSpec] = []
    for mod in _BUILTIN_MODULES:
        for attr in vars(mod).values():
            if isinstance(attr, ToolSpec) and all(attr.name != s.name for s in specs):
                specs.append(attr)
    return tuple(sorted(specs, key=lambda t: (t._sort_key, t.name)))


def init_tools() -> None:
    """初始化工具注册表(幂等;每个执行上下文各自持有一份)。"""
    if _tools_var.get() is None:
        _tools_var.set(_collect_builtins())


def clear_tools() -> None:
    _tools_var.set(None)


def get_tools() -> tuple[ToolSpec, ...]:
    init_tools()
    tools = _tools_var.get()
    assert tools is not None  # init_tools 已保证
    return tools


def get_tool(name: str) -> ToolSpec | None:
    return next((t for t in get_tools() if t.name == name), None)


def get_tool_for_langtag(lang: str) -> ToolSpec | None:
    """markdown fence 语言标记 → 工具(如 "shell"、"save")。"""
    return next((t for t in get_tools() if lang in t.block_types), None)


def execute_msg(msg: Message) -> list[Message]:
    """执行 assistant 消息里的所有工具调用,返回结果消息列表。

    - 结果按调用顺序产出;每个工具的失败转为 system 错误消息,不中断其他调用;
    - 中断(Ctrl-C)不丢弃已完成的调用:已产出的结果保留,被中断的调用
      以"未执行"消息收尾(仿 gptme execute_msg 的 KeyboardInterrupt 配对语义)——
      assistant 消息与已执行副作用因此永远有日志记录。
    """
    results: list[Message] = []
    tooluses = list(ToolUse.iter_from_content(msg.content))
    for i, tooluse in enumerate(tooluses):
        if not tooluse.is_runnable:
            continue
        try:
            results.extend(tooluse.execute())
        except KeyboardInterrupt:
            results.append(
                Message("system", f"{INTERRUPT_CONTENT} 工具 {tooluse.tool.name} 被中断,未执行")
            )
            for u in tooluses[i + 1 :]:
                if u.is_runnable:
                    results.append(Message("system", f"工具 {u.tool.name} 因中断未执行"))
            return results
    return results
