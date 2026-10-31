"""工具注册表(现在有哪些工具)和批量执行入口(把模型要的工具都跑一遍)。

阶段 1:内置工具写死在 _BUILTIN_MODULES(shell/read/save 三个模块);
自动发现新工具、按需开关、接入 MCP 外部工具都是后续阶段的事。
"""

from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import replace

from my_agent_alpha.message import INTERRUPT_CONTENT, Message
from my_agent_alpha.tools import edit as edit_mod
from my_agent_alpha.tools import grep as grep_mod
from my_agent_alpha.tools import memory as memory_mod
from my_agent_alpha.tools import read as read_mod
from my_agent_alpha.tools import save as save_mod
from my_agent_alpha.tools import shell as shell_mod
from my_agent_alpha.tools import subagent as subagent_mod
from my_agent_alpha.tools import todo as todo_mod
from my_agent_alpha.tools.base import ToolSpec, ToolUse

# 已加载的工具列表存在 ContextVar 里(而不是全局变量):
# 将来多线程跑多个会话时,每个线程可以有自己的工具列表,互不干扰
_tools_var: ContextVar[tuple[ToolSpec, ...] | None] = ContextVar("alpha_tools", default=None)

# 生效中的工具白名单(None = 没有限制)。restrict_tools 设置;
# 当前进程使用的工具就是_tools_var
# _allowed_var!=none: _tools_var = _allowed_var ∩ 实际装进注册表的工具(来自mcp,等等)
# _allowed_var==none: _tools_var = 实际装进注册表的工具

# add_tools 追加(MCP 连上后)时也要过这道闸 —— 否则"晚到的"工具
# 会绕过白名单混进受限子代理(--tools 在 chat 连 MCP 之前就生效了)
_allowed_var: ContextVar[frozenset[str] | None] = ContextVar("alpha_tools_allowed", default=None)

# 内置工具模块(每个模块暴露模块级 tool / 多个 tool)
_BUILTIN_MODULES = (
    shell_mod,
    read_mod,
    save_mod,
    memory_mod,
    subagent_mod,
    edit_mod,
    todo_mod,
    grep_mod,
)


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
    """清空注册表和白名单(测试之间重置状态用)。"""
    _tools_var.set(None)
    _allowed_var.set(None)


def add_tools(specs: tuple[ToolSpec, ...]) -> None:
    """往注册表追加工具(按名字去重;白名单生效时只收白名单内的)。

    用途:阶段 8 的 MCP —— 连上外部服务器后,把远端工具包成本地
    ToolSpec 追加进来(先 init_tools 保证内置表就位)。
    """
    init_tools()
    allowed = _allowed_var.get()
    if allowed is not None:
        specs = tuple(t for t in specs if t.name in allowed)  # 晚到的也过闸
    current = _tools_var.get() or ()
    existing = {t.name for t in current}
    merged = current + tuple(t for t in specs if t.name not in existing)
    _tools_var.set(merged)


def get_tools() -> tuple[ToolSpec, ...]:
    """返回当前全部工具(必要时先自动初始化)。"""
    init_tools()
    tools = _tools_var.get()
    assert tools is not None  # init_tools 已保证
    return tools


def get_tool(name: str) -> ToolSpec | None:
    """按工具名查找,找不到返回 None。"""
    return next((t for t in get_tools() if t.name == name), None)


def restrict_tools(allowed: Sequence[str]) -> tuple[str, ...]:
    """根据传入参数allowed裁剪当前的_tools_var和_allowed_var,并且返回allowed有但是tools_var没有的工具名称。

    给"派受限子代理"用:如记忆整理代理只给 read/grep/edit/memory,
    不给 shell —— 危险面收窄到白名单内(Claude Code 的 dream 代理同款
    约束:写只走 Edit/Write,Bash 只读)。

    示例:restrict_tools(["read", "grep", "memory"]) 之后,
    get_tool("shell") 返回 None,markdown 围栏 ```shell 也不再被识别。
    """
    init_tools()
    current = _tools_var.get() or ()
    names = {t.name for t in current}
    # 根据allowed裁剪当前的tools_var
    _tools_var.set(tuple(t for t in current if t.name in set(allowed)))
    # 设置_allowed_var
    _allowed_var.set(frozenset(allowed))  # 之后 add_tools(MCP)也要过这道闸
    return tuple(a for a in allowed if a not in names)  # 想留但注册表里没有的


def get_tool_for_langtag(lang: str) -> ToolSpec | None:
    """按代码块的语言标记查找工具:"shell"→shell 工具、"save"→save 工具。

    是 ToolUse.iter_from_content 的查找入口(识别模型写的代码块时用)。
    """
    return next((t for t in get_tools() if lang in t.block_types), None)


def execute_msg(msg: Message) -> list[Message]:
    """把一条模型回复里要的所有工具调用都执行一遍,返回结果消息列表。

    这边必须返回消息列表,因为:
    1.要处理中断
    2.一个工具执行有多个msg的情况,直接yield出去会导致不知道哪条msg是该工具的最后一条

    示例:模型回复里有两个调用(shell、read),返回列表就是
    [shell 的结果, read 的结果],顺序和调用顺序一致。

    - 某个工具出错:错误变成一条 system 消息放结果里,
      不影响后面的调用继续执行(模型看到错误会自己想办法);
    - 用户按 Ctrl-C:已经执行完的结果保留,被中断的和还没轮到的调用
      各生成一条"未执行"说明 —— 保证"执行过什么"在日志里永远有记录;
    - 原生调用的结果:最后一条消息盖上 call_id(接口要求每个调用
      恰好配一个带编号的结果;盖在最后一条是因为有的工具产出多条消息,
      编号只能盖一个)。
    """
    results: list[Message] = []
    tooluses = list(ToolUse.iter_from_content(msg.content))  # 从一条assitant消息解析出list[tooluse]

    for i, tooluse in enumerate(tooluses):  # 遍历list[tooluse]
        if not tooluse.is_runnable:
            continue
        try:
            # 得到一个tooluse的全部执行结果，因为可能yield多条消息，
            # 这边要用list收集起来，以便在最后一条msg上盖上call_id的编号
            executed = list(tooluse.execute())
        except KeyboardInterrupt:
            results.append(
                Message("system", f"{INTERRUPT_CONTENT} 工具 {tooluse.tool.name} 被中断,未执行")
            )
            for u in tooluses[i + 1 :]:
                if u.is_runnable:
                    results.append(Message("system", f"工具 {u.tool.name} 因中断未执行"))
            return results
        if tooluse.call_id and executed:
            # 如果这个tooluse有call_id并且有执行结果，那就把call_id写在最后一条msg上
            # 原生工具调用有一条铁律:一个 call_id ←必须恰好配→ 一条 tool 角色的结果消息

            # 把 call_id 盖在最后一条结果上(消息不可变,用 replace 造新的)
            executed[-1] = replace(executed[-1], call_id=tooluse.call_id)
        results.extend(executed)

    return results
