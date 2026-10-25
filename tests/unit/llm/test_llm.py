"""llm/__init__.py 的 reply 入口测试:分发、流式契约、断点、中断(全走 mock provider)。"""

from __future__ import annotations

from my_agent_alpha import Message
from my_agent_alpha.llm import reply
from my_agent_alpha.llm.mock import STATIC_RESPONSE, set_scripted_replies

# ---- 基本分发 ----


def test_reply_mock_echo() -> None:
    messages = [Message("system", "提示词"), Message("user", "你好")]
    resp = reply(messages, "mock/echo")
    assert resp.role == "assistant"
    assert resp.content == "Echo: 你好"
    assert resp.metadata is not None
    assert resp.metadata["model"] == "mock/echo"


def test_reply_mock_static() -> None:
    resp = reply([Message("user", "x")], "mock/static")
    assert resp.content == STATIC_RESPONSE


def test_reply_mock_scripted_sequence() -> None:
    set_scripted_replies(["第一次", "第二次"])
    msgs = [Message("user", "q")]
    assert reply(msgs, "mock/any").content == "第一次"
    assert reply(msgs, "mock/any").content == "第二次"
    # 脚本耗尽后回落到 echo 行为
    assert reply(msgs, "mock/echo").content == "Echo: q"


# ---- 流式契约:分块消费 + usage 捕获 ----


def test_stream_chunks_and_metadata() -> None:
    tokens: list[str] = []
    msg = reply([Message("user", "你好")], "mock/echo", on_token=tokens.append)
    # 分块收到了(块数 > 1,证明走了流式路径),拼起来等于完整内容
    assert len(tokens) > 1
    assert msg.content == "Echo: 你好"
    assert "".join(tokens) == msg.content
    assert msg.metadata is not None
    assert msg.metadata["usage"] == {"input": 0, "output": 0}


# ---- 断点检测:完整工具调用出现即停 ----


def test_break_on_tooluse_stops_generation() -> None:
    """围栏闭合 + 空行确认后,后续文本不该被生成/保存。"""
    trailing = "这段废话在断点之后,不该出现"
    scripted = f"我先执行:\n```shell\necho b2\n```\n\n{trailing}\n\n```shell\necho never\n```"
    set_scripted_replies([scripted])
    msg = reply([Message("user", "x")], "mock/echo")
    assert "```shell\necho b2\n```" in msg.content
    assert trailing not in msg.content  # 断点之前就停了
    assert "echo never" not in msg.content
    # 排干生成器后 usage 仍被捕获(研究精华 #4)
    assert msg.metadata is not None
    assert msg.metadata["usage"] == {"input": 0, "output": 0}


def test_break_on_tooluse_disabled() -> None:
    scripted = "```shell\necho x\n```\n\n后置文本"
    set_scripted_replies([scripted])
    msg = reply([Message("user", "x")], "mock/echo", break_on_tooluse=False)
    assert "后置文本" in msg.content


def test_break_on_native_call() -> None:
    """原生调用行一凑齐就断:后面的文本不接收(mock 直接送文本行)。"""
    set_scripted_replies(['@shell(call_1): {"command": "echo break-native"}\n\n这段不该出现'])
    msg = reply([Message("user", "x")], "mock/echo")
    assert "break-native" in msg.content
    assert "这段不该出现" not in msg.content


# ---- 流式中断 ----


def test_interrupt_midstream_returns_partial() -> None:
    set_scripted_replies([KeyboardInterrupt()])
    msg = reply([Message("user", "x")], "mock/echo")
    assert "^C 已中断" in msg.content
    assert msg.metadata is not None
    assert msg.metadata.get("interrupted") is True
