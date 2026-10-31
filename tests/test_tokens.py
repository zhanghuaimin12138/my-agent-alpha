"""token 估算工具测试:三分支入口、中日韩单价、缓存命中与上限淘汰。

用例清单来自 docs/token-counting-dev.md §6。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest

from my_agent_alpha import Message
from my_agent_alpha.utils import get_context_budget, len_tokens
from my_agent_alpha.utils import tokens as tokens_mod


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    """每个用例前后清空缓存:缓存是模块级全局的,不清会互相串。"""
    tokens_mod._token_cache.clear()
    yield
    tokens_mod._token_cache.clear()


# ---- 三分支入口与单价 ----


def test_empty_content_is_zero() -> None:
    assert len_tokens("") == 0  # 空字符串
    assert len_tokens(Message("user", "")) == 0  # 空内容的消息


def test_pure_ascii() -> None:
    assert len_tokens("the") == 1  # 3 个字符,凑不满 4 个也算 1 个
    assert len_tokens("abcdefgh") == 2  # 8 个字符恰好 2 个(纯英文/代码,系数不变)


def test_pure_cjk() -> None:
    assert len_tokens("中文中文") == 4  # 每字 1 token(旧规则会算成 3,系统性低估)
    assert len_tokens("看" * 17) == 17  # 对照实测:17 个汉字真实就是 17 token


def test_mixed_cjk_and_ascii() -> None:
    # "看一下 gptme 的文件":6 个汉字 + 7 个非汉字(gptme 五个字母 + 两处空格)
    # → 6 + ceil(7/4) = 8
    assert len_tokens("看一下 gptme 的文件") == 8


def test_message_branch_counts_content() -> None:
    msg = Message("user", "中文abc")  # 2 个汉字 + 3 个字母 → 2 + 1 = 3
    assert len_tokens(msg) == 3
    assert len_tokens(msg) == len_tokens("中文abc")  # 和直接传字符串一致


def test_list_branch_sums() -> None:
    msgs = [Message("user", "中文abc"), Message("assistant", "看" * 17)]
    assert len_tokens(msgs) == 3 + 17  # 逐条相加


def test_overestimate_is_the_safe_direction() -> None:
    """对照 cl100k 实测样本:估算应略高于真实值(高估→提前压缩,安全)。"""
    assert len_tokens("a" * 52) == 13  # 英文短句真实 10,估算 13(130%)


# ---- 缓存 ----


def test_cache_hit_skips_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一内容数两次,第二次直接走缓存,不再真算。"""
    calls = 0
    real = tokens_mod._estimate

    def counting(text: str) -> int:
        """替身:数之前先记一次调用。"""
        nonlocal calls
        calls += 1
        return real(text)

    monkeypatch.setattr(tokens_mod, "_estimate", counting)
    first = len_tokens("重复出现的内容")
    second = len_tokens("重复出现的内容")
    assert first == second
    assert calls == 1  # 只真算了一次,第二次命中缓存


def test_cache_evicts_oldest() -> None:
    """灌满缓存后,最早插入的那条被删,总量不超过上限。"""
    oldest_key = hashlib.sha256("最早的一条".encode()).hexdigest()
    len_tokens("最早的一条")
    for i in range(tokens_mod.CACHE_MAX):  # 再灌 1000 条不同的内容
        len_tokens(f"内容-{i}")
    assert len(tokens_mod._token_cache) == tokens_mod.CACHE_MAX
    assert oldest_key not in tokens_mod._token_cache  # 最早的被淘汰


# ---- 预算(原 test_compaction 里的 test_budget_math 搬来) ----


def test_budget_math() -> None:
    assert get_context_budget(100_000) == 90_000  # 9 成窗口
    small = get_context_budget(5_000)  # 5000-8192-1000 < 0 → 保底链
    assert small == max(min(4500, 5000 - 8192 - 1000), 1000)  # = 1000
    assert get_context_budget(20_000) == 20_000 - 8192 - 1000  # 减法分支
