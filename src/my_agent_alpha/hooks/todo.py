"""todo 的两个钩子:完成门控(拦截偷懒收尾)+ 会话恢复(重放重建清单)。

清单本身(状态与增删改查)在 tools/todo.py;本模块只放挂在生命周期上
的两段逻辑,它们读写那边的状态:

1. 完成门控(loop.continue):回合结束而没有工具调用、清单上还有未完成项时,
   自动注入一条 user 消息把模型拉回来继续干;连续 MAX_GATE_WARNINGS 轮
   仍不动手就熔断(产出报告,不再催);
2. 会话恢复(session.start):扫日志里的 todo 调用重放一遍,重建清单状态。

注册入口是 register_todo_hooks(hooks/builtin.py 启动时调用)。
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Sequence

from my_agent_alpha.hooks import HookType, register_hook
from my_agent_alpha.logmanager import LogManager
from my_agent_alpha.message import (
    DECLINED_CONTENT,
    INTERRUPT_CONTENT,
    LLM_REQUEST_FAILED_PREFIX,
    Message,
)
from my_agent_alpha.tools.base import ToolUse
from my_agent_alpha.tools.todo import (
    command_text,
    format_item,
    has_incomplete_todos,
    incomplete_items,
    parse_and_apply,
    reset_todos,
)

# 门控消息的统一前缀(终端识别 + 测试定位)
GATE_PREFIX = "[todo 门控]"

# 连续多少轮"提醒后仍无工具调用"就熔断(环境变量可调,便于测试)
MAX_GATE_WARNINGS = int(os.environ.get("ALPHA_MAX_TODO_GATES", "2"))

# 用量统计消息的前缀:判断回合怎么收尾时要跳过它(它只是随附的统计)
_USAGE_PREFIX = "[用量]"

# 当前这段"催促序列"已连续几轮没见到工具调用(真干活就归零,熔断后也归零)
_gate_warnings = 0


def reset_gate_warnings() -> None:
    """门控计数归零(conftest 测试隔离用,和 tools.todo.reset_todos 配对)。"""
    global _gate_warnings
    _gate_warnings = 0


def _turn_ran_tools(messages: Sequence[Message]) -> bool:
    """从日志尾部往回看:本回合(最近一条 user 之后)有没有真执行过工具调用。

    遇到带可运行调用的 assistant 就算"干过活"(收尾那轮没有调用不影响);
    遇到 user 就是回合边界,不用再往前看。
    """
    for msg in reversed(messages):
        if msg.role == "user":
            return False
        if msg.role == "assistant" and any(
            tu.is_runnable for tu in ToolUse.iter_from_content(msg.content)
        ):
            return True
    return False


def _tail_is_terminal_marker(messages: Sequence[Message]) -> bool:
    """日志尾部是不是"用户已表态/请求失败"式的收尾标记。

    中断、用户拒绝执行、LLM 请求失败都属于"不该再催"的结束方式;
    [用量] 之类的随附 system 消息先跳过,找到回合最后一条实质消息再判断。
    """
    for msg in reversed(messages):
        if msg.role == "system" and msg.content.startswith(_USAGE_PREFIX):
            continue
        return msg.role == "system" and msg.content.startswith(
            (INTERRUPT_CONTENT, DECLINED_CONTENT, LLM_REQUEST_FAILED_PREFIX)
        )
    return False


def _gate_message() -> str:
    """拼门控注入的 user 消息:未完成清单 + 怎么解除门控。"""
    lines = [f"{GATE_PREFIX} 你的待办清单还有未完成事项,请继续执行,不要现在结束:"]
    for item in incomplete_items():
        lines.append(format_item(item))
    lines.append(
        "继续工作直到全部完成;每完成一项就用 todo update <id> state=completed 标记;"
        "某项确实无法完成或不再需要时,用 todo remove <id> 移除并说明原因。"
        "(若用户刚刚给出了新指示,优先响应指示;之后再回到这份待办。)"
    )
    return "\n".join(lines)


def todo_gate_hook(manager: LogManager, **_ctx: object) -> Iterator[Message]:
    """loop.continue 门控:想收尾但清单未清 → 注入 user 消息拉回来。

    产出 user 消息会被外层循环回灌成下一轮输入(见 chat._run_chat_loop);
    连续 MAX_GATE_WARNINGS 轮提醒仍无工具调用 → 产出 system 熔断报告
    (不回灌,循环自然结束)。
    """
    global _gate_warnings
    msgs = manager.messages
    if _turn_ran_tools(msgs):
        _gate_warnings = 0  # 这回合真干过活:催促额度重新计,从这轮重新数
    if not has_incomplete_todos():
        _gate_warnings = 0  # 清单干净:计数归零,下个任务从头计
        return
    if _tail_is_terminal_marker(msgs):
        return  # 中断/拒绝/失败收的尾:用户已表态,不催
    if _gate_warnings >= MAX_GATE_WARNINGS:
        _gate_warnings = 0  # 熔断后额度归零:用户下次输入是新的意图,给重新计数的机会
        yield Message(
            "system",
            f"{GATE_PREFIX} 已连续 {MAX_GATE_WARNINGS} 轮提醒仍未推进待办"
            "(期间没有任何工具调用),停止催促,本会话回到正常节奏。",
        )
        return
    _gate_warnings += 1
    yield Message("user", _gate_message())


# ---------------- 会话恢复(session.start 重放钩子)----------------


def replay_todo_hook(manager: LogManager, **_ctx: object) -> Iterator[Message]:
    """session.start 重放:扫日志里的 todo 调用,重建清单状态。

    读完整历史(master_messages)而不是压缩后的视图:todo 的 add/update
    可能早已被压缩掉,用视图重放会漏掉那些状态变更,重启后清单就错了。
    只重演会改状态的操作(add/update/remove/clear,read 跳过);
    坏命令静默跳过 —— 状态停在最后一个有效操作上(对照 gptme 的 _replay)。
    先 reset_todos 再重放:同进程第二次 chat() 时不会叠加出重复条目。
    """
    reset_todos()
    for msg in manager.master_messages:
        if msg.role != "assistant":
            continue
        for tu in ToolUse.iter_from_content(msg.content):
            if tu.tool.name != "todo":
                continue
            command = command_text(tu.content, tu.args, tu.kwargs)
            if command.split()[:1] == ["read"]:
                continue  # 只读查询不改状态,跳过
            # 重放遇到意外(坏命令等):保持当前状态,继续扫后面的
            with contextlib.suppress(Exception):
                parse_and_apply(command)
    return iter(())  # 重放只重建内存状态,不往日志里产消息


def register_todo_hooks() -> None:
    """注册 todo 的两个钩子(hooks/builtin.py 启动时调用)。"""
    register_hook("todo.gate", HookType.LOOP_CONTINUE, todo_gate_hook, priority=0)
    register_hook("todo.replay", HookType.SESSION_START, replay_todo_hook, priority=10)
