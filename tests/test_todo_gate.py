"""todo 完成门控的端到端测试:拦截注入、额度重置、熔断、恢复重放。

剧本全部用 set_scripted_replies 导演(照 test_tool_nudge.py 的套路):
模型先建清单、再"光说不做",看门控是否把它拉回来;以及重放钩子
能否从日志重建清单状态。收尾话术统一用"任务完成,无需工具。"
—— 已验证不会触发 NUDGE(空回复提醒),保证测的只是 todo 门控。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.hooks import todo
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.tools import todo as todo_tool

# 模型收尾话术(不触发 NUDGE 的"正常最终回答")
_DONE = "任务完成,无需工具。"


def _gate_users(logdir: Path) -> list[Message]:
    """从日志里挑出所有门控注入的 user 消息。"""
    return [
        m
        for m in LogManager.load(logdir).messages
        if m.role == "user" and m.content.startswith(todo.GATE_PREFIX)
    ]


def _gate_systems(logdir: Path) -> list[Message]:
    """从日志里挑出所有门控的 system 消息(熔断报告)。"""
    return [
        m
        for m in LogManager.load(logdir).messages
        if m.role == "system" and m.content.startswith(todo.GATE_PREFIX)
    ]


def test_gate_nudges_then_model_finishes(logdir: Path, workspace: Path) -> None:
    """主剧本:建 2 项 → 空谈收尾被拦 1 次 → 干完 → 正常结束。"""
    set_scripted_replies(
        [
            "```todo\nadd 做事甲\n```",
            "```todo\nadd 做事乙\n```",
            _DONE,  # 空谈收尾:清单未清 → 门控注入
            "```todo\nupdate 1 state=completed\n```",
            "```todo\nupdate 2 state=completed\n```",
            _DONE,  # 这次清单已清:不再拦
        ]
    )
    chat(["做两件事"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)

    users = _gate_users(logdir)
    assert len(users) == 1  # 恰好注入一次
    assert users[0].content.count("做事甲") == 1  # 注入消息在日志里只出现一次(不重复落盘)
    assert "做事乙" in users[0].content  # 注入内容列出未完成项
    assert _gate_systems(logdir) == []  # 没熔断
    assert not todo.has_incomplete_todos()  # 真的干完了


def test_gate_budget_resets_after_execution(
    logdir: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """执行过工具调用后催促额度归零(对齐 NUDGE 的额度重置哲学)。

    上限调到 1:空谈→注入→执行→空谈→再注入(第二次注入就是重置生效的
    证明,没有重置的话这里已经熔断了)→ 空谈 → 熔断。
    """
    monkeypatch.setattr(todo, "MAX_GATE_WARNINGS", 1)
    set_scripted_replies(
        [
            "```todo\nadd 甲\n```",
            _DONE,  # 空谈 → 注入 #1
            "```todo\nupdate 1 state=in_progress\n```",  # 真执行 → 额度重置
            _DONE,  # 空谈 → 注入 #2(重置的证明)
            _DONE,  # 空谈 → 熔断
        ]
    )
    chat(["做件事"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)

    assert len(_gate_users(logdir)) == 2  # 第二次注入 = 重置生效
    assert len(_gate_systems(logdir)) == 1  # 最终熔断一次
    assert "停止催促" in _gate_systems(logdir)[0].content


def test_gate_breaker_after_two_warnings(logdir: Path, workspace: Path) -> None:
    """默认上限 2:建清单后连续 3 次空谈 → 恰好 2 次注入 + 1 次熔断。"""
    set_scripted_replies(
        [
            "```todo\nadd 甲\n```",
            _DONE,  # 空谈 → 注入 #1
            _DONE,  # 空谈 → 注入 #2
            _DONE,  # 空谈 → 熔断(system 不回灌,循环结束)
        ]
    )
    chat(["做件事"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)

    assert len(_gate_users(logdir)) == 2
    systems = _gate_systems(logdir)
    assert len(systems) == 1 and "停止催促" in systems[0].content
    # 熔断后没有再请求模型:assistant 恰好 4 条(脚本全部耗尽)
    manager = LogManager.load(logdir)
    assert sum(1 for m in manager.messages if m.role == "assistant") == 4
    # 熔断报告是日志最后一条消息
    assert manager.messages[-1].content == systems[0].content


def test_no_gate_when_todos_clean(logdir: Path, workspace: Path) -> None:
    """清单为空时门控零干预(正常回答不被打扰)。"""
    set_scripted_replies([_DONE])
    chat(["聊一句"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)

    assert _gate_users(logdir) == []
    assert _gate_systems(logdir) == []


def test_gate_silent_after_interrupt_marker(logdir: Path, workspace: Path) -> None:
    """回合以中断/拒绝/失败标记收尾时,门控不催(用户已表态)。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "做件事"))
    manager.append(Message("assistant", "```todo\nadd 甲\n```"))
    manager.append(Message("system", "[interrupted] 用户按了 Ctrl-C"))

    produced = list(todo.todo_gate_hook(manager))
    assert produced == []  # 清单未清也不注入


def test_replay_rebuilds_state_from_log(logdir: Path, workspace: Path) -> None:
    """恢复会话:重放日志里的 todo 调用,精确重建清单(read/坏命令跳过)。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("assistant", "```todo\nadd 任务A\n```"))
    manager.append(Message("assistant", "```todo\nupdate 1 state=in_progress\n```"))
    manager.append(Message("assistant", "```todo\nread\n```"))  # 只读,跳过
    manager.append(Message("assistant", "```todo\n飞走\n```"))  # 坏命令,静默

    chat([], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)

    (result,) = list(todo_tool.execute_todo("read", None, None))
    assert "#1 [进行中] 任务A" in result.content
    assert "未完成 1/1" in result.content


def test_replay_idempotent_across_sessions(logdir: Path, workspace: Path) -> None:
    """同一日志连续两次恢复:清单不叠加(重放前先归零)。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("assistant", "```todo\nadd 任务A\n```"))

    for _ in range(2):
        chat([], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)

    (result,) = list(todo_tool.execute_todo("read", None, None))
    assert result.content.count("#") == 1  # 仍然只有一项
