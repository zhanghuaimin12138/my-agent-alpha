"""默认适配器测试:把压缩核心接到本项目的真实设施上。"""

from __future__ import annotations

from my_agent_alpha.autocompact.adapters import (
    CurrentModelInfoProvider,
    ParserToolUseDetector,
)
from my_agent_alpha.context import set_current_model
from my_agent_alpha.message import Message


def test_parser_detector_markdown_and_native() -> None:
    """两种工具调用通道(markdown 围栏 / 原生调用行)都探测得到。"""
    detector = ParserToolUseDetector()
    assert detector.contains_tool_use(Message("assistant", "```shell\nls\n```")) is True
    assert (
        detector.contains_tool_use(
            Message("assistant", '@save(c1): {"path": "a.txt", "content": "hi"}')
        )
        is True
    )
    assert detector.contains_tool_use(Message("assistant", "just plain text")) is False
    assert detector.contains_tool_use(Message("user", "```shell\nls\n```")) is False


def test_current_model_info_provider() -> None:
    """设置了当前模型 → 解析出名字和窗口;没设置 → None(核心用兜底)。"""
    provider = CurrentModelInfoProvider()
    assert provider.get_default_model() is None  # 测试环境没设过模型

    set_current_model("glm/glm-5.3")
    info = provider.get_default_model()
    assert info is not None
    assert info.name == "glm/glm-5.3"
    assert info.context == 128_000
    set_current_model("")  # 清掉,不影响别的测试
