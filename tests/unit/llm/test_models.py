"""llm/models.py 测试:模型名解析(provider/model 拆分)与服务商表。"""

from __future__ import annotations

import pytest

from my_agent_alpha import Message
from my_agent_alpha.llm import reply, resolve_model
from my_agent_alpha.llm.models import PROVIDERS


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


# ---- 审查 #11/#12:模型名解析的边界 ----


def test_empty_model_resolves_to_mock() -> None:
    meta = resolve_model("")
    assert meta.provider == "mock"
    assert reply([Message("user", "x")], "").content == "Echo: x"


def test_openrouter_bare_errors_clearly() -> None:
    with pytest.raises(ValueError, match="显式指定"):
        resolve_model("openrouter")
