"""确认链:工具执行前问一句"可以吗?" —— 可插拔的审批关卡。

核心思想(仿 gptme 的 TOOL_CONFIRM,支柱三的第一个实战):
"要不要确认"不是一个写死在工具里的 if,而是一条**按优先级排序的钩子链** ——
每个钩子看一眼这次调用,给出三种答复之一:

- 返回 ConfirmationResult → 用它的结论(confirm 放行 / skip 拒绝 / edit 改写);
- 返回 None → "我不管",交给下一个钩子;
- 链走完没人管 → 自动放行(自主模式:脚本、测试、-y 都走这里)。

示例(阶段 4 实际注册的三层):
    优先级 200  安全护栏   危险命令 → 直接 skip(不分模式,永远在岗)
    优先级 10  只读放行   ls/grep 等只读命令 → confirm(不烦用户)
    优先级 0   终端询问   其余 → 打印预览问用户 y/n/e/a(仅交互模式注册)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from my_agent_alpha.tools.base import ToolUse

# 一条确认钩子:看一眼调用,给出结论(ConfirmationResult)或返回 None 表示不管
# 钩子函数定义:传入一个ToolUse，得到是否能放行状态
ConfirmHook = Callable[[ToolUse], "ConfirmationResult | None"]


@dataclass(frozen=True)
class ConfirmationResult:
    """一次确认的结论。

    示例:ConfirmationResult("confirm")               → 放行
         ConfirmationResult("skip", message="太危险")  → 拒绝执行
         ConfirmationResult("edit", edited_content="ls -la")  → 改成这个再执行
    """

    action: str  # "confirm"(放行)/ "skip"(拒绝)/ "edit"(改写后再执行)
    edited_content: str | None = None  # action="edit" 时:改写后的内容
    message: str = ""  # 给模型看的说明(为什么拒绝等)


# 钩子注册表:模块级列表 [(优先级, 名字, 函数)],高优先级先问
# (阶段 5 会升级成通用 hook 系统;现在先保持一条专用链,简单)
_hooks: list[tuple[int, str, ConfirmHook]] = []
# 会话级"不再询问"开关(用户在终端选了 a 之后打开)
_auto_confirm = False


def register_confirm_hook(name: str, fn: ConfirmHook, priority: int = 0) -> None:
    """注册一条确认钩子(同名重复注册 = 替换)。优先级大的先被问到。"""
    global _hooks
    _hooks = [(p, n, f) for p, n, f in _hooks if n != name]  # 去重
    _hooks.append((priority, name, fn))
    _hooks.sort(key=lambda t: -t[0])  # 优先级从高到低


def clear_confirm_hooks() -> None:
    """清空钩子和放行状态(测试之间重置、会话开始时重建)。"""
    global _hooks, _auto_confirm
    _hooks = []
    _auto_confirm = False


def set_auto_confirm(value: bool) -> None:
    """打开/关闭"本会话不再询问"(终端里选 a 时打开)。"""
    global _auto_confirm
    _auto_confirm = value


def is_auto_confirm() -> bool:
    """当前是否处于"本会话不再询问"状态。"""
    return _auto_confirm


# 传入工具 ---试图走遍所有钩子--- 得到放行状态
def get_confirmation(tooluse: ToolUse) -> ConfirmationResult:
    """走一遍钩子链,决定这次调用怎么办。

    链上没人答复 → 自动放行(自主模式:非交互、测试、-y)。
    """
    if _auto_confirm:
        return ConfirmationResult("confirm")
    for _priority, _name, fn in _hooks:
        result = fn(tooluse)
        if result is not None:
            return result
    return ConfirmationResult("confirm")
