"""流式的端到端:断点后工具真的执行、中断真的停回合。

流式契约/断点检测/重试的单元测试在 tests/unit/llm/。
"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha import LogManager
from my_agent_alpha.chat import chat
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.message import INTERRUPT_CONTENT


def test_e2e_break_then_execute(logdir: Path, workspace: Path) -> None:
    set_scripted_replies(
        [
            "马上执行:\n```shell\necho streaming-e2e\n```\n\n后面的内容不该被生成",
            "完成。",
        ]
    )
    chat(["跑一下"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)
    manager = LogManager.load(logdir)
    contents = [m.content for m in manager.messages]
    # 断点:assistant 消息不含断点后的文本
    assistant = next(c for c in contents if "```shell" in c)
    assert "不该被生成" not in assistant
    # 工具真的执行了,结果进了日志
    assert any("streaming-e2e" in c for c in contents)


def test_interrupt_midstream_stops_turn(logdir: Path, workspace: Path) -> None:
    set_scripted_replies([KeyboardInterrupt()])
    chat(["触发"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)
    manager = LogManager.load(logdir)
    assert any(
        m.role == "system" and m.content.startswith(INTERRUPT_CONTENT) for m in manager.messages
    )
    # 中断后不再生成第二步
    assert sum(1 for m in manager.messages if m.role == "assistant") == 1
