"""原生工具调用的端到端:mock 剧本里写原生调用行,走完整 chat 循环。

参数表/调用行解析/分片拼装的单元测试在 tests/unit/ 下。
"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha import LogManager
from my_agent_alpha.chat import chat
from my_agent_alpha.llm.mock import set_scripted_replies


def test_e2e_native_tool_loop(logdir: Path, workspace: Path) -> None:
    set_scripted_replies(
        [
            '我来执行:\n@shell(call_1): {"command": "echo native-e2e"}',
            "执行完了。",
        ]
    )
    chat(
        ["跑一下"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
        tool_format="tool",  # 原生模式(系统提示词也换原生版)
    )
    manager = LogManager.load(logdir)
    roles = [m.role for m in manager.messages]
    # 阶段 5 起,每次模型回复后默认有 [用量] 钩子产出一条 system 消息
    assert roles == [
        "system",
        "system",
        "user",
        "assistant",
        "system",
        "system",
        "assistant",
        "system",
    ]
    contents = [m.content for m in manager.messages]
    assert "native-e2e" in contents[5]  # [用量] 在 4,工具输出在 5
    # 原生模式系统提示词:教的是 function calling,不是围栏约定
    assert "function calling" in contents[0]
    assert "围栏" not in contents[0]
    # 工具结果带 call_id 配对
    assert manager.messages[5].call_id == "call_1"
