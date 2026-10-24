"""Provider 层测试(§9 Provider 组):注册表语义 + 内置策略。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from my_agent_alpha.autocompact.adapters import EstimateTokenCounter
from my_agent_alpha.autocompact.context_provider import (
    ContextProvider,
    DefaultContextProvider,
    get_context_provider,
    list_providers,
    register_provider,
    reset_registry,
)
from my_agent_alpha.autocompact.contracts import (
    CompactionResult,
    CompressionConfig,
    ModelInfo,
    source_digest,
)
from my_agent_alpha.message import Message

COUNTER = EstimateTokenCounter()


class _FakeToolUse:
    """测试用探测:认为含 ```shell 的消息有工具调用。"""

    def contains_tool_use(self, msg: Message) -> bool:
        return "```shell" in msg.content


class _FixedModels:
    """测试用模型信息:永远返回固定窗口。"""

    def __init__(self, context: int) -> None:
        self._context = context

    def get_default_model(self) -> ModelInfo:
        return ModelInfo("test-model", self._context)


class _NoopProvider(ContextProvider):
    """测试用自定义策略:compress 返回空结果(好认)。"""

    name = "custom-noop"

    def should_compress(self, messages: Sequence[Message], config: CompressionConfig) -> bool:
        return True

    def compress(self, messages: Sequence[Message], config: CompressionConfig) -> CompactionResult:
        return CompactionResult(messages=[], source_digest="deadbeef")


def setup_function() -> None:
    """每个测试前重置注册表(模块级全局状态,不重置会互相污染)。"""
    reset_registry()


def test_builtin_default_is_registered() -> None:
    """默认注册表里有 "default",取出来的就是内置策略。"""
    provider = get_context_provider("default")
    assert isinstance(provider, DefaultContextProvider)
    assert provider.name == "default"
    assert "default" in list_providers()


def test_custom_can_replace_default_before_first_use() -> None:
    """先注册的自定义 "default" 优先:内置的 setdefault 不会覆盖它。"""
    register_provider("default", _NoopProvider)
    provider = get_context_provider("default")
    assert isinstance(provider, _NoopProvider)  # 内置算法被整体替换


def test_later_registration_overwrites() -> None:
    """同名后写覆盖:显式再注册 default 为内置策略,自定义的让位。"""
    register_provider("default", _NoopProvider)
    register_provider("default", DefaultContextProvider)
    assert isinstance(get_context_provider("default"), DefaultContextProvider)


def test_get_unknown_provider_raises() -> None:
    """名字不存在直接报错(拼写问题早暴露)。"""
    with pytest.raises(KeyError):
        get_context_provider("no-such")


def test_default_provider_compress(logdir: Path) -> None:
    """内置策略 compress:真跑三阶段引擎,返回等长结果 + 正确元信息。"""
    provider = DefaultContextProvider(
        counter=COUNTER, models=_FixedModels(10_000), tool_use=_FakeToolUse()
    )
    msgs = [
        Message("system", "prompt"),
        Message("system", "x" * 12_000),  # 3000 token,最大
        Message("system", "x" * 11_600),  # 2900 token
        Message("system", "x" * 8_400),  # 2100 token
        Message("user", "q"),
    ]  # 总量 8003 > 目标线 8000
    result = provider.compress(msgs, CompressionConfig(store_dir=logdir))
    assert result.source_digest == source_digest(msgs)
    assert result.covered_through == len(msgs) - 1  # 全量压缩
    assert result.limitations == ["rule-based truncation of tool results"]
    assert len(result.messages) == len(msgs)  # 原位替换,等长
    assert COUNTER.len_tokens(result.messages) < COUNTER.len_tokens(msgs)
    assert result.messages[1].content.startswith("[Large tool output removed - 3000 tokens]")
    assert (logdir / "outputs" / "autocompact-001.txt").exists()  # 全量落了盘


def test_default_provider_should_compress() -> None:
    """内置策略的判定:总量超工作上限(0.8 × 窗口)才压。"""
    provider = DefaultContextProvider(
        counter=COUNTER, models=_FixedModels(10_000), tool_use=_FakeToolUse()
    )
    big = [Message("system", "x" * 33_000)]  # 8250 token > 8000
    small = [Message("system", "x" * 30_000)]  # 7500 token < 8000
    assert provider.should_compress(big, CompressionConfig()) is True
    assert provider.should_compress(small, CompressionConfig()) is False
    # 显式给了 limit 就按 limit 来
    assert provider.should_compress(small, CompressionConfig(limit=100)) is True
