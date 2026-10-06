"""todo 工具:内存任务清单 + 完成门控(第二轮 R2)。

治什么:多步任务做到一半,模型一句"我完成了"就溜 —— 没有任何机制
逼它把说过要做的事做完。本模块两部分配合:

1. 任务清单工具:模型自己维护的待办事项(增删改查),存在进程内存里;
2. 完成门控(loop.continue 钩子):回合结束而没有工具调用、清单上
   还有未完成项时,自动注入一条 user 消息把模型拉回来继续干;
   连续 MAX_GATE_WARNINGS 轮仍不动手就熔断(产出报告,不再催)。

状态不写单独的文件:todo 的每次调用都是普通消息,已经记在
conversation.jsonl 里;恢复会话时 session.start 钩子把日志里的
todo 调用重放一遍,状态就精确重建了(对照 gptme 的重放设计)。

用法(围栏体就是一行子命令,原生通道给 {"command": "..."}):

    ```todo
    add 修复登录页的空指针
    ```

    ```todo
    update 1 state=in_progress
    ```

子命令文法:
- add <text>                新增待办(text 是 add 后面的全部文本)
- update <id> key=value     修改:key 是 state 或 text,可组合
- remove <id>               删除某条
- clear                     清空清单,编号从 #1 重新开始
- read                      查看当前清单
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

from my_agent_alpha.hooks import HookType, register_hook
from my_agent_alpha.logmanager import LogManager
from my_agent_alpha.message import (
    DECLINED_CONTENT,
    INTERRUPT_CONTENT,
    LLM_REQUEST_FAILED_PREFIX,
    Message,
)
from my_agent_alpha.tools.base import Parameter, ToolSpec, ToolUse

# 状态只有三种(gptme 还有 paused,但研究报告里它的语义连原文都讲不清,砍掉)
TodoState = Literal["pending", "in_progress", "completed"]

# 给人/模型看的状态中文名
_STATE_LABELS: dict[str, str] = {
    "pending": "待办",
    "in_progress": "进行中",
    "completed": "已完成",
}

# 门控消息的统一前缀(终端识别 + 测试定位,对齐 NUDGE_PREFIX 的做法)
GATE_PREFIX = "[todo 门控]"

# 连续多少轮"提醒后仍无工具调用"就熔断(环境变量可调,便于测试)
MAX_GATE_WARNINGS = int(os.environ.get("ALPHA_MAX_TODO_GATES", "2"))

# 用量统计消息的前缀:判断回合怎么收尾时要跳过它(它只是随附的统计)
_USAGE_PREFIX = "[用量]"

# 子命令一览(报错时原样展示给模型,让它自己纠正)
_USAGE = (
    "add <text> | update <id> state=pending|in_progress|completed"
    " | update <id> text=<新描述> | remove <id> | clear | read"
)


@dataclass
class TodoItem:
    """清单里的一条待办。"""

    id: int
    text: str
    state: TodoState = "pending"


# ---- 模块级状态(进程内存,对照 gptme 的 _current_todos)----
# _todos: 编号 → 条目;_next_id: 下一个要发的编号(clear 后回到 1,
# 这是"重放日志能精确重建状态"的前提);_gate_warnings: 门控已连续催了几轮
_todos: dict[int, TodoItem] = {}
_next_id: int = 1
_gate_warnings: int = 0


def _reset_todos() -> None:
    """清空清单和门控计数(conftest 测试隔离、重放前归零都用它)。"""
    global _next_id, _gate_warnings
    _todos.clear()
    _next_id = 1
    _gate_warnings = 0


def render_todos() -> str:
    """把清单渲染成紧凑的行式文本(read 的输出,也是门控消息的正文)。

    示例:
        todo 清单(未完成 2/3):
        #1 [进行中] 实现 todo 工具
        #2 [待办] 写门控测试
        #3 [已完成] 调研 gptme 实现
    """
    if not _todos:
        return "todo 清单为空。"
    incomplete = sum(1 for t in _todos.values() if t.state != "completed")
    lines = [f"todo 清单(未完成 {incomplete}/{len(_todos)}):"]
    for item in sorted(_todos.values(), key=lambda t: t.id):
        lines.append(f"#{item.id} [{_STATE_LABELS[item.state]}] {item.text}")
    return "\n".join(lines)


def has_incomplete_todos() -> bool:
    """清单上是否还有未完成项(完成门控的判定条件)。"""
    return any(t.state != "completed" for t in _todos.values())


def _apply_add(text: str) -> str:
    """新增一条待办,编号自动递增;返回给模型看的回执。"""
    global _next_id
    if not text.strip():
        return f"错误: add 后面要写上要做的事。\n文法:{_USAGE}"
    item = TodoItem(id=_next_id, text=text.strip())
    _todos[item.id] = item
    _next_id += 1
    return f"#{item.id} 已添加:[待办] {item.text}"


def _parse_updates(tokens: list[str]) -> dict[str, str] | str:
    """解析 update 的 key=value 参数;格式不对时返回错误说明。

    宽容处理:text 的值可以带空格 —— 后面跟着的不含 = 的词
    续接到上一个 text 的值上(如 "update 1 text=修复 登录页")。
    解析：
    1.state=pending|in_progress|completed"
    2.text=<新描述>
    """
    changes: dict[str, str] = {}
    last_key: str | None = None
    for tok in tokens:
        if "=" in tok:
            key, value = tok.split("=", 1)
            if key not in ("state", "text"):
                return f"错误: update 只支持 state= / text=,不认识 {key}=。\n文法:{_USAGE}"
            changes[key] = value
            last_key = key
        elif last_key == "text":
            changes["text"] += " " + tok  # text 值里的空格被切开了,拼回去
        else:
            return f"错误: update 的参数要写成 key=value 形式。\n文法:{_USAGE}"
    return changes


def _apply_update(rest: str) -> str:
    """修改某条待办的 state/text;rest 是 id 和 key=value 们。
    
    1.update |<id> state=pending|in_progress|completed"
    2.update |<id> text=<新描述>
    """
    tokens = rest.split()
    if not tokens or not tokens[0].isdigit():
        return f"错误: update 要先给编号,如 update 1 state=completed。\n文法:{_USAGE}"
    item_id = int(tokens[0])
    item = _todos.get(item_id)
    if item is None:
        known = "、".join(f"#{i}" for i in sorted(_todos)) or "(清单是空的)"
        return f"错误: 清单里没有 #{item_id}(现有:{known})"
    changes = _parse_updates(tokens[1:])
    if isinstance(changes, str):
        return changes
    if not changes:
        return f"错误: update 没有要改的内容。\n文法:{_USAGE}"
    if "state" in changes:
        if changes["state"] not in _STATE_LABELS:
            legal = "、".join(_STATE_LABELS)
            return f"错误: state 只能是 {legal},收到的是 {changes['state']}。"
        item.state = changes["state"]  # type: ignore[assignment]
    if "text" in changes:
        if not changes["text"].strip():
            return "错误: text 的值不能为空。"
        item.text = changes["text"].strip()
    return f"#{item.id} 已更新:[{_STATE_LABELS[item.state]}] {item.text}"


def _apply_remove(rest: str) -> str:
    """删除某条待办。
    
    remove <id>
    """
    tokens = rest.split()
    if not tokens or not tokens[0].isdigit():
        return f"错误: remove 要给编号,如 remove 1。\n文法:{_USAGE}"
    item_id = int(tokens[0])
    item = _todos.pop(item_id, None)
    if item is None:
        return f"错误: 清单里没有 #{item_id}"
    return f"#{item.id} 已删除:{item.text}"


def _apply_clear() -> str:
    """清空清单并把编号归 1(重放日志时 clear 之前的历史也自然作废)。"""
    global _next_id
    count = len(_todos)
    _todos.clear()
    _next_id = 1
    return f"已清空 {count} 项,编号从 #1 重新开始。"


def _parse_and_apply(command: str) -> str:
    """解析一行子命令并应用到清单;返回回执或错误说明(不抛异常)。

    示例:"add 修 bug" → 新增一条;"update 1 state=completed" → 标记完成。
    """
    command = command.strip()
    if not command:
        return f"错误: todo 的命令是空的。\n文法:{_USAGE}"
    sub, _, rest = command.partition(" ")
    if sub == "add":
        return _apply_add(rest)
    if sub == "update":
        return _apply_update(rest)
    if sub == "remove":
        return _apply_remove(rest)
    if sub == "clear":
        return _apply_clear()
    if sub == "read":
        return render_todos()
    return f"错误: 未知子命令 {sub}。\n文法:{_USAGE}"


def _command_text(code: str | None, args: list[str] | None, kwargs: dict[str, str] | None) -> str:
    """从两种通道的参数里取出子命令文本。

    原生通道:kwargs["command"];markdown 通道:围栏体 code;
    模型偶尔把命令写在围栏头上(```todo add xxx)→ 退回用 args 拼接。
    """
    if kwargs:
        return kwargs.get("command", "")
    return (code or "").strip() or " ".join(args or [])


def execute_todo(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> Iterator[Message]:
    """todo 工具的执行入口:取子命令 → 应用 → 包成 system 消息返回。"""
    command = _command_text(code, args, kwargs)
    yield Message("system", _parse_and_apply(command))


tool = ToolSpec(
    name="todo",
    desc=(
        "维护本次会话的任务清单:多步任务开工前先建清单,每完成一项就标记,"
        "收尾前清单应当清空(未清空会被系统拦下来要求继续)。"
    ),
    instructions=(
        "多步任务(约 3 步以上)开工前先用 add 逐条建清单;\n"
        "开始做某项时 update <id> state=in_progress,完成一项立刻 update <id> state=completed;\n"
        "确实不做的用 remove <id> 移除(顺带说明原因);全部完成后可 read 确认清零。\n"
        f"子命令(一次一行):{_USAGE}"
    ),
    examples=(
        "```todo\nadd 分析需求\n```\n"
        "```todo\nadd 实现功能\n```\n"
        "```todo\nupdate 1 state=in_progress\n```\n"
        "```todo\nupdate 1 state=completed\n```"
    ),
    block_types=("todo",),
    parameters=(
        Parameter(
            "command",
            "string",
            "子命令,如 'add 修复登录' / 'update 1 state=completed' / 'read'",
        ),
    ),
    read_only=True,  # 只动进程内存,不碰文件系统 → 免确认链
    execute=execute_todo,
)


# ---------------- 完成门控(loop.continue 钩子)----------------


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
    for item in sorted(_todos.values(), key=lambda t: t.id):
        if item.state != "completed":
            lines.append(f"#{item.id} [{_STATE_LABELS[item.state]}] {item.text}")
    lines.append(
        "继续工作直到全部完成;每完成一项就用 todo update <id> state=completed 标记;"
        "某项确实无法完成或不再需要时,用 todo remove <id> 移除并说明原因。"
        "(若用户刚刚给出了新指示,优先响应指示;之后再回到这份待办。)"
    )
    return "\n".join(lines)


def todo_gate_hook(manager: LogManager, **_ctx: object) -> Iterator[Message]:
    """loop.continue 门控:想收尾但清单未清 → 注入 user 消息拉回来。

    产出 user 消息会被外层循环回灌成下一轮输入(见 _run_chat_loop);
    连续 MAX_GATE_WARNINGS 轮提醒仍无工具调用 → 产出 system 熔断报告
    (不回灌,循环自然结束)。
    """
    global _gate_warnings
    msgs = manager.messages
    if _turn_ran_tools(msgs):
        _gate_warnings = 0  # 这回合真干过活:催促额度重新计(对齐 NUDGE 的重置哲学)
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

    只重演会改状态的操作(add/update/remove/clear,read 跳过);
    坏命令静默跳过 —— 状态停在最后一个有效操作上(对照 gptme 的 _replay)。
    先 _reset_todos 再重放:同进程第二次 chat() 时不会叠加出重复条目。
    """
    _reset_todos()
    for msg in manager.messages:
        if msg.role != "assistant":
            continue
        for tu in ToolUse.iter_from_content(msg.content):
            if tu.tool.name != "todo":
                continue
            command = _command_text(tu.content, tu.args, tu.kwargs)
            if command.split()[:1] == ["read"]:
                continue  # 只读查询不改状态,跳过
            # 重放遇到意外(坏命令等):保持当前状态,继续扫后面的
            with contextlib.suppress(Exception):
                _parse_and_apply(command)
    return iter(())  # 重放只重建内存状态,不往日志里产消息


def register_todo_hooks() -> None:
    """注册 todo 的两个钩子(hooks/builtin.py 启动时调用)。"""
    register_hook("todo.gate", HookType.LOOP_CONTINUE, todo_gate_hook, priority=0)
    register_hook("todo.replay", HookType.SESSION_START, replay_todo_hook, priority=10)
