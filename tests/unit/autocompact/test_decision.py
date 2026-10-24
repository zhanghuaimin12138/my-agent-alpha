"""决策层测试(§9 决策组):什么时候不压、走规则还是走摘要。"""

from __future__ import annotations

from my_agent_alpha.autocompact.adapters import EstimateTokenCounter
from my_agent_alpha.autocompact.decision import (
    estimate_compaction_savings,
    should_auto_compact,
)
from my_agent_alpha.message import Message

COUNTER = EstimateTokenCounter()


def _big_text(tokens: int) -> str:
    """造一段恰好约 tokens 个 token 的 ascii 文本(4 字符/token)。"""
    return "x" * (tokens * 4)


def test_far_below_half_returns_none() -> None:
    """总量远低于 0.5 × 窗口:none(没什么好压的)。"""
    msgs = [Message("user", "hello") for _ in range(10)]  # 20 token
    assert should_auto_compact(msgs, context_window=1000, counter=COUNTER) == "none"


def test_close_without_massive_returns_none() -> None:
    """总量在 0.5–0.9 × 窗口之间、但没有巨型工具结果:none。"""
    msgs = [Message("system", _big_text(600)), Message("user", "hi")]  # 602 token
    assert should_auto_compact(msgs, context_window=1000, counter=COUNTER) == "none"


def test_over_limit_with_massive_returns_rule_based() -> None:
    """超 0.9 线 + 巨型工具结果:估算能省大头 → rule_based。"""
    msgs = [
        Message("system", _big_text(3000)),  # 巨型工具结果(3000 token)
        Message("user", "hi"),
    ]
    assert should_auto_compact(msgs, context_window=1000, counter=COUNTER) == "rule_based"


def test_over_limit_incompressible_returns_summarize() -> None:
    """超限但全是短 user 消息(规则压不出 10%)→ summarize。"""
    msgs = [Message("user", "hello") for _ in range(500)]  # 1000 token > 900
    assert should_auto_compact(msgs, context_window=1000, counter=COUNTER) == "summarize"


def test_pinned_massive_not_counted() -> None:
    """pinned 的巨型消息不算"巨型工具结果":同样的日志,不钉 → 候选;
    钉住 → none(没到触发线也没有 massive)。"""
    plain = [Message("system", _big_text(3000)), Message("user", "hi")]
    pinned = [Message("system", _big_text(3000), pinned=True), Message("user", "hi")]
    # 窗口 5000:总量 3002 达到 0.5 线(2500)但没到 0.9 线(4500)
    assert should_auto_compact(plain, context_window=5000, counter=COUNTER) == "summarize"
    assert should_auto_compact(pinned, context_window=5000, counter=COUNTER) == "none"


def test_estimate_reasoning_savings() -> None:
    """估算:距末尾 ≥5 且含思考区的消息,按内容 30% 计节省。"""
    # 6 条消息,第 0 条距末尾 5,恰好够老;思考区占一半
    thinky = "<think>" + _big_text(100) + "</think>" + _big_text(100)
    msgs = [Message("assistant", thinky)] + [Message("user", "hi") for _ in range(5)]
    total, savings, reason = estimate_compaction_savings(
        msgs, limit=900, context_window=1000, counter=COUNTER
    )
    tok = COUNTER.len_tokens(thinky)
    assert reason == int(tok * 0.3)
    assert savings == reason  # 这批消息没有别的可省
    assert total == COUNTER.len_tokens(msgs)


def test_estimate_tool_savings_only_when_would_remove() -> None:
    """估算:巨型工具结果只在总量达到 0.8 线后才计入节省。"""
    msgs = [Message("system", _big_text(3000)), Message("user", "hi")]
    # 窗口 5000、limit 4500:总量 3002 < 0.8 线(4000)→ 阶段 2 不计入
    _, savings_below, _ = estimate_compaction_savings(
        msgs, limit=4500, context_window=5000, counter=COUNTER
    )
    assert savings_below == 0
    # 窗口 1000、limit 900:总量 3002 > 0.8 线(800)→ 计入"原量 − 200"
    _, savings_above, _ = estimate_compaction_savings(
        msgs, limit=900, context_window=1000, counter=COUNTER
    )
    assert savings_above == 3000 - 200
