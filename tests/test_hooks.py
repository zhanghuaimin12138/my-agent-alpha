"""阶段 5 钩子系统测试:注册/触发/优先级/异常隔离/生命周期接线。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.hooks import (
    HookType,
    StopPropagation,
    clear_hooks,
    get_hooks,
    register_hook,
    trigger_hook,
    unregister_hook,
)
from my_agent_alpha.llm.mock import set_scripted_replies


def _msgs(*texts: str) -> Iterator[Message]:
    """造一个产出若干条消息的钩子函数体。"""
    for t in texts:
        yield Message("system", t)


# ---- 注册与触发 ----


def test_register_and_trigger_yields_messages() -> None:
    clear_hooks()
    register_hook("a", HookType.TURN_PRE, lambda **c: _msgs("甲", "乙"))
    out = list(trigger_hook(HookType.TURN_PRE))
    assert [m.content for m in out] == ["甲", "乙"]


def test_priority_order_high_first() -> None:
    clear_hooks()
    register_hook("low", HookType.TURN_PRE, lambda **c: _msgs("低"), priority=0)
    register_hook("high", HookType.TURN_PRE, lambda **c: _msgs("高"), priority=100)
    out = list(trigger_hook(HookType.TURN_PRE))
    assert [m.content for m in out] == ["高", "低"]


def test_same_name_reregister_replaces() -> None:
    clear_hooks()
    register_hook("x", HookType.TURN_PRE, lambda **c: _msgs("旧"))
    register_hook("x", HookType.TURN_PRE, lambda **c: _msgs("新"))
    assert [m.content for m in trigger_hook(HookType.TURN_PRE)] == ["新"]
    assert len(get_hooks(HookType.TURN_PRE)) == 1


def test_unregister_and_clear() -> None:
    clear_hooks()
    register_hook("x", HookType.TURN_PRE, lambda **c: _msgs("1"))
    unregister_hook("x")
    assert list(trigger_hook(HookType.TURN_PRE)) == []
    register_hook("y", HookType.TURN_PRE, lambda **c: _msgs("2"))
    clear_hooks()
    assert get_hooks() == []


def test_context_passed_to_hook() -> None:
    clear_hooks()
    seen: dict[str, object] = {}

    def spy(**ctx: object) -> Iterator[Message]:
        seen.update(ctx)
        return iter(())

    register_hook("spy", HookType.STEP_PRE, spy)
    list(trigger_hook(HookType.STEP_PRE, model="glm/glm-5.3"))
    assert seen["model"] == "glm/glm-5.3"


def test_stop_propagation() -> None:
    """先产消息再喊停:后面的钩子不再运行。"""
    clear_hooks()
    register_hook(
        "stopper",
        HookType.TURN_PRE,
        lambda **c: iter([Message("system", "先出这条"), StopPropagation()]),  # type: ignore[list-item]
        priority=100,
    )
    register_hook("later", HookType.TURN_PRE, lambda **c: _msgs("不该出现"), priority=0)
    out = list(trigger_hook(HookType.TURN_PRE))
    assert [m.content for m in out] == ["先出这条"]


def test_broken_hook_does_not_kill_others() -> None:
    clear_hooks()

    def boom(**ctx: object) -> Iterator[Message]:
        raise RuntimeError("钩子自己炸了")

    register_hook("boom", HookType.TURN_PRE, boom, priority=100)
    register_hook("ok", HookType.TURN_PRE, lambda **c: _msgs("好的"))
    out = list(trigger_hook(HookType.TURN_PRE))
    assert [m.content for m in out] == ["好的"]  # 坏钩子被隔离


# ---- 生命周期接线(端到端) ----


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
