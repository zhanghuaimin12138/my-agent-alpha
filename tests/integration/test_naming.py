"""会话自动命名的端到端:第一回合结束后钩子真的拿模型起了名。"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager
from my_agent_alpha.chat import chat
from my_agent_alpha.hooks.naming import read_session_title
from my_agent_alpha.llm.mock import set_scripted_replies


def test_names_session_after_first_turn(
    logdir: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主剧本:对话一轮 → 钩子拿模型起名 → 写 meta、打印、不进对话日志。"""
    monkeypatch.setenv("ALPHA_AUTO_NAME", "1")
    set_scripted_replies(
        [
            # 第一回合的正常回复(措辞避开 NUDGE 的"行动宣告"标记,不加"我来帮你")
            "print 没输出是因为缓冲没刷新,加 flush=True 即可。",
            "调试打印问题",  # 命名请求的回复(钩子消耗脚本的下一条)
        ]
    )
    chat(
        ["帮我看看为什么 print 没输出"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )

    assert read_session_title(logdir) == "调试打印问题"
    out = capsys.readouterr().out
    assert "(会话已命名:调试打印问题)" in out
    # 命名是元数据:模型的命名回复不进对话日志
    manager = LogManager.load(logdir)
    assert sum(1 for m in manager.messages if m.role == "assistant") == 1
