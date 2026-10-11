"""LLM 层测试(mock provider,离线零成本)。"""

import json

from my_agent_alpha import Message
from my_agent_alpha.llm import mock as llm_mock
from my_agent_alpha.llm import reply, resolve_model
from my_agent_alpha.llm.mock import STATIC_RESPONSE, set_scripted_replies
from my_agent_alpha.llm.models import PROVIDERS


def test_mock_env_script_directs_subprocess(monkeypatch) -> None:
    """环境变量脚本(ALPHA_MOCK_REPLIES):让派出去的子进程也能被导演。

    set_scripted_replies 是进程内状态,子代理子进程看不到;
    要导演子代理的行为(如记忆整理代理),脚本只能走环境变量传过去。
    """
    monkeypatch.setenv("ALPHA_MOCK_REPLIES", json.dumps(["第一条脚本", "第二条脚本"]))
    msgs = [Message("user", "hi")]
    assert llm_mock.chat(msgs, "echo")[0] == "第一条脚本"
    assert llm_mock.chat(msgs, "echo")[0] == "第二条脚本"
    # 脚本用完自动回退 echo;坏 JSON 也回退 echo 而不是炸
    assert llm_mock.chat(msgs, "echo")[0].startswith("Echo")
    monkeypatch.setenv("ALPHA_MOCK_REPLIES", "{坏的json")
    llm_mock.clear_scripted()  # 清掉缓存,逼它重读环境变量
    assert llm_mock.chat(msgs, "echo")[0].startswith("Echo")


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


# ---- glm(智谱)服务商 ----


def test_resolve_model_glm_default_model() -> None:
    # 只写服务商名 glm 时,自动用表里的默认模型 glm-5.3
    meta = resolve_model("glm")
    assert meta.provider == "glm"
    assert meta.model == "glm-5.3"
    assert meta.full == "glm/glm-5.3"


def test_glm_providers_table_entries() -> None:
    # glm(按量计费)和 glm-coding(Coding Plan 订阅)端点不同,但共用 GLM_API_KEY 这把密钥
    assert PROVIDERS["glm"].base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert PROVIDERS["glm-coding"].base_url == "https://open.bigmodel.cn/api/coding/paas/v4"
    assert PROVIDERS["glm"].api_key_env == "GLM_API_KEY"
    assert PROVIDERS["glm-coding"].api_key_env == "GLM_API_KEY"


def test_resolve_model_glm_coding_explicit_model() -> None:
    # 显式写 "glm-coding/模型名" 时按写的拆;服务商名带连字符也能正确整名识别
    meta = resolve_model("glm-coding/glm-4.6")
    assert meta.provider == "glm-coding"
    assert meta.model == "glm-4.6"
