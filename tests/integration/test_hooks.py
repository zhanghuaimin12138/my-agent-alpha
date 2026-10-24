"""钩子在完整会话里的接线:生命周期触发(usage、turn.pre、session.end)。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.hooks import HookType, register_hook, unregister_hook
from my_agent_alpha.llm.mock import set_scripted_replies


def test_e2e_usage_hook_in_log(logdir: Path, workspace: Path) -> None:
    """内置 usage 钩子:每次模型回复后日志里多一条 [用量](默认开启)。"""
    set_scripted_replies(["回复一"])
    chat(["问"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)
    manager = LogManager.load(logdir)
    assert any(m.content.startswith("[用量]") for m in manager.messages)


def test_e2e_custom_turn_pre_hook(logdir: Path, workspace: Path) -> None:
    """自定义钩子:turn.pre 产出的消息进入日志(会话开始前注册)。"""
    set_scripted_replies(["收到"])

    def reminder(**ctx: object) -> Iterator[Message]:
        yield Message("system", "[提醒] 自定义钩子跑过 turn.pre")

    register_hook("test.reminder", HookType.TURN_PRE, reminder)
    try:
        chat(["问"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)
    finally:
        unregister_hook("test.reminder")
    manager = LogManager.load(logdir)
    assert any("[提醒] 自定义钩子跑过" in m.content for m in manager.messages)
    # 顺序:提醒消息在用户消息之后、模型回复之前
    contents = [m.content for m in manager.messages]
    assert contents.index("[提醒] 自定义钩子跑过 turn.pre") < next(
        i for i, c in enumerate(contents) if c == "收到"
    )


def test_e2e_session_end_hook_runs_on_exit(logdir: Path, workspace: Path) -> None:
    """session.end:非交互正常退出也要触发(finally 保证)。"""
    set_scripted_replies(["完"])
    seen: list[str] = []

    def bye(manager, **ctx: object) -> Iterator[Message]:
        seen.append(f"结束,共 {len(manager)} 条消息")  # type: ignore[arg-type]
        return iter(())

    register_hook("test.bye", HookType.SESSION_END, bye)
    try:
        chat(["问"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)
    finally:
        unregister_hook("test.bye")
    assert seen and "结束" in seen[0]
