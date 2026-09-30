"""工具注册表(现在有哪些工具)和批量执行入口(把模型要的工具都跑一遍)。

阶段 1:内置工具写死在 _BUILTIN_MODULES(shell/read/save 三个模块);
自动发现新工具、按需开关、接入 MCP 外部工具都是后续阶段的事。
"""

from __future__ import annotations

from contextvars import ContextVar

from my_agent_alpha.message import INTERRUPT_CONTENT, Message
from my_agent_alpha.tools import read as read_mod
from my_agent_alpha.tools import save as save_mod
from my_agent_alpha.tools import shell as shell_mod
from my_agent_alpha.tools.base import ToolSpec, ToolUse

# 已加载的工具列表存在 ContextVar 里(而不是全局变量):
# 将来多线程跑多个会话时,每个线程可以有自己的工具列表,互不干扰
_tools_var: ContextVar[tuple[ToolSpec, ...] | None] = ContextVar("alpha_tools", default=None)

# 内置工具模块(每个模块暴露模块级 tool / 多个 tool)
_BUILTIN_MODULES = (shell_mod, read_mod, save_mod)


def _collect_builtins() -> tuple[ToolSpec, ...]:
    """把三个tool里的 ToolSpec 都收集起来(按名字去重、排序)。"""
    specs: list[ToolSpec] = []
    for mod in _BUILTIN_MODULES:
        for attr in vars(mod).values():  # 遍历模块里所有对象
            if isinstance(attr, ToolSpec) and all(attr.name != s.name for s in specs):
                specs.append(attr)
    return tuple(sorted(specs, key=lambda t: (t._sort_key, t.name)))


def init_tools() -> None:
    """把内置工具装进注册表(已装过就什么都不做)。"""
    if _tools_var.get() is None:
        _tools_var.set(_collect_builtins())


def clear_tools() -> None:
    """清空注册表(测试之间重置状态用)。"""
    _tools_var.set(None)


def get_tools() -> tuple[ToolSpec, ...]:
    """返回当前全部工具(必要时先自动初始化)。"""
    init_tools()
    tools = _tools_var.get()
    assert tools is not None  # init_tools 已保证
    return tools


def get_tool(name: str) -> ToolSpec | None:
    """按工具名查找,找不到返回 None。"""
    return next((t for t in get_tools() if t.name == name), None)


def get_tool_for_langtag(lang: str) -> ToolSpec | None:
    """按代码块的语言标记查找工具:"shell"→shell 工具、"save"→save 工具。

    是 ToolUse.iter_from_content 的查找入口(识别模型写的代码块时用)。
    """
    return next((t for t in get_tools() if lang in t.block_types), None)


def execute_msg(msg: Message) -> list[Message]:
    """把一条模型回复里要的所有工具调用都执行一遍,返回结果消息列表。

    示例:模型回复里有两个调用(shell、read),返回列表就是
    [shell 的结果, read 的结果],顺序和调用顺序一致。

    - 某个工具出错:错误变成一条 system 消息放结果里,
      不影响后面的调用继续执行(模型看到错误会自己想办法);
    - 用户按 Ctrl-C:已经执行完的结果保留,被中断的和还没轮到的调用
      各生成一条"未执行"说明 —— 保证"执行过什么"在日志里永远有记录。
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
