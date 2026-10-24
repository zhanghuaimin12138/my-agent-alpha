"""subagent 测试:派活、真实子进程跑完回传、父代理端到端调度。"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.context import set_current_model
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.tools import execute_msg


def test_subagent_tool_spawns_and_returns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """子代理用 mock 模型离线跑:派活 → 子进程真跑 → 结论回传。"""
    monkeypatch.setenv("ALPHA_LOGS_HOME", str(tmp_path / "logs"))
    set_current_model("mock/echo")  # 子代理继承父会话的模型
    from my_agent_alpha import context

    (tmp_path / "ws").mkdir()
    with context.agent_context(tmp_path / "ws", tmp_path / "logs" / "parent"):
        msg = Message("assistant", "```subagent\n帮我说声你好\n```")
        results = execute_msg(msg)
    texts = [m.content for m in results]
    assert any("已派出子代理" in t for t in texts)
    final = texts[-1]
    assert "子代理" in final and "Echo" in final  # 子代理的回复回来了
    # 子代理的会话日志真的在磁盘上
    sub_dirs = [d for d in (tmp_path / "logs").iterdir() if d.name.startswith("subagent-")]
    assert len(sub_dirs) == 1


def test_subagent_native_kwargs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHA_LOGS_HOME", str(tmp_path / "logs"))
    set_current_model("mock/echo")
    from my_agent_alpha import context

    (tmp_path / "ws").mkdir()
    with context.agent_context(tmp_path / "ws", tmp_path / "logs" / "p"):
        msg = Message("assistant", '@subagent(s1): {"prompt": "算一下 1+1"}')
        results = execute_msg(msg)
    assert any("Echo" in m.content for m in results)


def test_e2e_parent_dispatches_subagent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """端到端:父代理(剧本)决定派子代理,真实子进程跑完回传,父收尾。"""
    monkeypatch.setenv("ALPHA_LOGS_HOME", str(tmp_path / "logs"))
    ws = tmp_path / "ws"
    ws.mkdir()
    set_scripted_replies(
        [
            "这活儿独立,我派个 subagent:\n```subagent\n报数:一到三\n```",
            "子代理已答复,任务完成。",
        ]
    )
    chat(
        ["派活"],
        logdir=tmp_path / "logs" / "parent",
        workspace=ws,
        model="mock/echo",
        non_interactive=True,
    )
    manager = LogManager.load(tmp_path / "logs" / "parent")
    contents = [m.content for m in manager.messages]
    assert any("✅ 子代理" in c and "Echo" in c for c in contents)
    assert any("任务完成" in c for c in contents)
