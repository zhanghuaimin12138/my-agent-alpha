"""通用钩子系统:在对话生命周期的固定节点上,让外部代码插入行为。

背景(仿 gptme 支柱三):agent 的核心循环不该为每个新功能改代码 ——
"每次生成后记一笔用量"、"会话开始时打招呼"、"每轮结束提醒待办"……
这些横切功能全部做成**钩子**:注册到某个生命周期点上,循环走到那里就调用。

两种钩子(并存,各有分工):

1. 本模块的**消息钩子**:钩子是生成器函数,产出(yield)若干条 Message,
   循环负责把消息追加进日志 —— 于是它们自然进入模型上下文、进入终端显示;
   (产出 0 条也完全可以:纯打日志/纯通知的钩子)

2. confirm.py 的**确认链**(阶段 4):它要的是"放行/拒绝/改写"这种
   **返回值**而不是消息,契约不同,所以保持独立(gptme 也是这么分的:
   TOOL_CONFIRM 是唯一返回值语义的例外)。

生命周期点(触发时机见 chat.py 中的调用处):

    session.start   会话装配完成、系统提示词就位后
    session.end     会话即将结束
    turn.pre        收到一条用户输入、开始处理前
    turn.post       一个回合处理完毕
    step.pre        中层循环每次调用模型前
    generation.post 模型回复生成完毕(带 usage 用量)
    loop.continue   外层循环每圈结束(可用来排"下一步该做什么"的接力消息:
                    产出的 user 消息会被回灌成下一轮输入,system 消息只落日志;
                    回灌的 user 消息不要以 / 开头,否则会被当成斜杠命令)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from my_agent_alpha.message import Message


class HookType(StrEnum):
    """生命周期点。值就是显示用的名字,如 "session.start"。"""

    SESSION_START = "session.start"
    SESSION_END = "session.end"
    TURN_PRE = "turn.pre"
    TURN_POST = "turn.post"
    STEP_PRE = "step.pre"
    GENERATION_POST = "generation.post"
    LOOP_CONTINUE = "loop.continue"


class StopPropagation:
    """钩子产出它来制止后续(低优先级的)钩子继续跑。"""


# 消息钩子:调用时拿到上下文关键字参数,产出若干条 Message(或什么都不产出)
HookFunc = Callable[..., Iterator[Message]]


@dataclass(frozen=True)
class Hook:
    """一个已注册的钩子。priority 大的先跑;同名重注册 = 替换。"""

    name: str
    hook_type: HookType
    fn: HookFunc
    priority: int = 0


# 注册表:生命周期点 → 钩子列表(按优先级从高到低排)
# (单会话阶段的简单实现;将来多会话时升级为 ContextVar,像工具注册表那样)
_registry: dict[HookType, list[Hook]] = {}


def register_hook(name: str, hook_type: HookType, fn: HookFunc, priority: int = 0) -> None:
    """注册一个钩子;同名再注册会替换旧的(方便测试和热更新)。"""
    hooks = _registry.setdefault(hook_type, [])
    _registry[hook_type] = [h for h in hooks if h.name != name]
    _registry[hook_type].append(Hook(name, hook_type, fn, priority))
    _registry[hook_type].sort(key=lambda h: -h.priority)


def unregister_hook(name: str) -> None:
    """按名字注销一个钩子(不存在则什么都不做)。"""
    for hook_type in _registry:
        _registry[hook_type] = [h for h in _registry[hook_type] if h.name != name]


def clear_hooks() -> None:
    """清空全部钩子(测试之间重置用)。"""
    _registry.clear()


def get_hooks(hook_type: HookType | None = None) -> list[Hook]:
    """列出钩子(全部或某个生命周期点的),供 /hooks 命令展示。"""
    if hook_type is None:
        return [h for hooks in _registry.values() for h in hooks]
    return list(_registry.get(hook_type, []))


def trigger_hook(hook_type: HookType, **ctx: object) -> Iterator[Message]:
    """触发一个生命周期点:按优先级逐个运行钩子,产出它们产出的消息。

    - 钩子抛异常:记一条警告,继续跑后面的钩子(一个坏钩子不许拖垮会话);
    - 钩子产出 StopPropagation:停止后续钩子;
    - ctx 原样传给钩子(如 manager=..., message=...),钩子按需取用。

    示例:
        for msg in trigger_hook(HookType.GENERATION_POST, message=m):
            manager.append(msg)
    """
    for hook in get_hooks(hook_type):
        try:
            for item in hook.fn(**ctx):
                if isinstance(item, StopPropagation):
                    return
                yield item
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("钩子 %s 出错(已跳过): %s", hook.name, e)
