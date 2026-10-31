"""todo 工具:模型自己维护的内存任务清单(增删改查)。

治什么:多步任务做到一半,模型一句"我完成了"就溜 —— 没有任何机制
逼它把说过要做的事做完。这个工具让模型把待办一条条记下来;
"没做完不许收尾"的拦截(完成门控)和"恢复会话时重建清单"(重放)
是两个钩子,实现在 hooks/todo.py —— 清单状态存放在本模块,
钩子只是读写它,两边分工见那边模块开头的说明。

状态不写单独的文件:todo 的每次调用都是普通消息,已经记在
conversation.jsonl 里;恢复会话时 session.start 钩子把日志里的
todo 调用重放一遍,状态就精确重建了。

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

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from my_agent_alpha.message import Message
from my_agent_alpha.tools.base import Parameter, ToolSpec

# 状态只有三种(gptme 还有 paused,但研究报告里它的语义连原文都讲不清,砍掉)
TodoState = Literal["pending", "in_progress", "completed"]

# 给人/模型看的状态中文名
_STATE_LABELS: dict[str, str] = {
    "pending": "待办",
    "in_progress": "进行中",
    "completed": "已完成",
}

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
# 这是"重放日志能精确重建状态"的前提)
_todos: dict[int, TodoItem] = {}
_next_id: int = 1


def reset_todos() -> None:
    """清空清单、编号归 1(conftest 测试隔离、重放钩子开工前归零都用它)。"""
    global _next_id
    _todos.clear()
    _next_id = 1


def format_item(item: TodoItem) -> str:
    """把一条待办渲染成一行 "#1 [进行中] 文本"(read 输出与门控消息共用)。"""
    return f"#{item.id} [{_STATE_LABELS[item.state]}] {item.text}"


def incomplete_items() -> list[TodoItem]:
    """未完成的待办,按编号排好序(完成门控逐条列出它们时用)。"""
    return sorted((t for t in _todos.values() if t.state != "completed"), key=lambda t: t.id)


def render_todos() -> str:
    """把清单渲染成紧凑的行式文本(read 的输出)。

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
        lines.append(format_item(item))
    return "\n".join(lines)


def has_incomplete_todos() -> bool:
    """清单上是否还有未完成项(完成门控的判定条件)。"""
    return bool(incomplete_items())


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
    解析:
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


def parse_and_apply(command: str) -> str:
    """解析一行子命令并应用到清单;返回回执或错误说明(不抛异常)。

    公开给重放钩子用(hooks/todo.py 按日志里的调用逐条重演)。
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


def command_text(code: str | None, args: list[str] | None, kwargs: dict[str, str] | None) -> str:
    """从两种通道的参数里取出子命令文本(重放钩子解析日志调用时也用它)。

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
    command = command_text(code, args, kwargs)
    yield Message("system", parse_and_apply(command))


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
