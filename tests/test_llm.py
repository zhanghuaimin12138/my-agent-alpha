"""LLM 层测试(mock provider,离线零成本)。"""

from my_agent_alpha import Message
from my_agent_alpha.llm import reply, resolve_model
from my_agent_alpha.llm.mock import STATIC_RESPONSE, set_scripted_replies


def test_resolve_model_provider_split() -> None:
    meta = resolve_model("deepseek/deepseek-chat")
    assert meta.provider == "deepseek"
    assert meta.model == "deepseek-chat"
    assert meta.full == "deepseek/deepseek-chat"


def test_resolve_model_bare_name_defaults_to_openai() -> None:
    meta = resolve_model("gpt-4o-mini")
    assert meta.provider == "openai"


def test_resolve_model_provider_default_model() -> None:
    meta = resolve_model("deepseek")
    assert meta.model == "deepseek-chat"


def test_resolve_model_never_fails_on_unknown() -> None:
    meta = resolve_model("weird-gateway/some-model")
    assert meta.provider == "weird-gateway"
    assert meta.model == "some-model"


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
