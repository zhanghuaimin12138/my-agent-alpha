"""阶段 6 上下文压缩测试:估算、预算、三阶段、兜底、溢出自愈。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from my_agent_alpha import Message
from my_agent_alpha.chat import step
from my_agent_alpha.compaction import (
    compact_messages,
    compress_long_replies,
    compress_reply,
    estimate_tokens,
    get_context_budget,
    hard_cap,
    messages_tokens,
    prepare_messages,
    strip_old_thinking,
    truncate_tool_results,
)
from my_agent_alpha.llm import ContextLengthError

# ---- token 估算与预算 ----


def test_estimate_ascii_and_cjk() -> None:
    assert estimate_tokens("abcdefgh") == 2  # 8 个英文字符 ≈ 2 token
    assert estimate_tokens("中文中文") >= 2  # 中文更"贵",同字符数 token 更多
    assert estimate_tokens("") == 0


def test_budget_math() -> None:
    assert get_context_budget(100_000) == 90_000  # 9 成窗口
    small = get_context_budget(5_000)  # 5000-8192-1000 < 0 → 保底链
    assert small == max(min(4500, 5000 - 8192 - 1000), 1000)  # = 1000
    assert get_context_budget(20_000) == 20_000 - 8192 - 1000  # 减法分支


# ---- 阶段 1:剥思考 ----


def _history_with_think() -> list[Message]:
    msgs = [Message("system", "提示词", pinned=True, hide=True)]
    msgs.append(Message("user", "问1"))
    msgs.append(Message("assistant", "<think>很长的思考过程……</think>\n结论一"))
    msgs.append(Message("system", "工具输出"))
    msgs.append(Message("assistant", "结论二"))
    msgs.append(Message("user", "问2"))
    return msgs


def test_strip_old_thinking() -> None:
    msgs = _history_with_think()
    out = strip_old_thinking(msgs)
    # 第 2 条(距结尾 ≥5?共 6 条,索引 0 距结尾 5)—— 索引 0 是 pinned system;
    # 索引 1、2 距结尾 4、3 —— 都在"近期"范围内?总长 6,range(1) 只有 i=0。
    # 造长一点才能剥到。这里直接验证:近期消息不动。
    assert out[2].content == msgs[2].content  # 近期不动


def test_strip_old_thinking_actually_strips_old() -> None:
    msgs = [Message("system", "提示词", pinned=True, hide=True)]
    for i in range(8):
        msgs.append(Message("user", f"问{i}"))
        msgs.append(Message("assistant", f"<think>第{i}轮思考,假装很长。</think>\n结论{i}"))
    out = strip_old_thinking(msgs)
    # 索引 0 是系统提示词,1=user"问0",2=最老的 assistant
    assert "<think>" not in out[2].content  # 最老的 assistant 被剥了
    assert out[-1].content == msgs[-1].content  # 最新原样
    assert "结论0" in out[2].content  # 正文保留


# ---- 阶段 2:截断工具结果 ----


def test_truncate_largest_first_with_archive(workspace: Path, logdir: Path) -> None:
    big1 = "A" * 4000  # 约 1000 token
    big2 = "B" * 12000  # 约 3000 token,最大的且超过截断门槛
    msgs = [
        Message("system", "提示词", pinned=True, hide=True),
        Message("assistant", "调工具"),
        Message("system", big1),
        Message("system", big2),
    ]
    budget = 1300  # 只够提示词 + big1 + 一点余量 → big2 必须被截
    out = truncate_tool_results(list(msgs), budget, logdir)
    truncated = [m for m in out if m.content.startswith("[内容已截断")]
    assert len(truncated) == 1
    assert "B" * 100 in truncated[0].content  # 截的是 big2(最大优先)
    # 存档真的写了完整内容
    archives = list((logdir / "tool-outputs" / "compacted").glob("*.txt"))
    assert len(archives) == 1
    assert archives[0].read_text(encoding="utf-8") == big2
    # 原列表不被修改(视图原则)
    assert msgs[3].content == big2


def test_truncate_skips_pinned_and_hidden(workspace: Path) -> None:
    secret = "X" * 9000
    msgs = [
        Message("system", secret, pinned=True, hide=True),  # 系统提示词不动
        Message("system", "小结果"),
    ]
    out = truncate_tool_results(list(msgs), 100, None)
    assert out[0].content == secret


# ---- 阶段 3:抽取式压缩 ----


def test_compress_reply_keeps_code_blocks() -> None:
    code = "```python\nprint('这段代码很长' * 200)\n```"
    text = "第一句是总结。" + "这是一句普通的话。" * 80 + code
    out = compress_reply(text, ratio=0.3)
    assert code in out  # 代码块完整保留
    assert "第一句" in out  # 开头高分保留
    assert len(out) < len(text)


def test_compress_long_replies_skips_tool_calls_and_recent() -> None:
    plain_long = "这句话用来凑长度。" * 400  # 够长且无围栏
    msgs = [
        Message("assistant", plain_long),  # 老、长、无调用 → 压
        Message("assistant", "```shell\nls\n```" + plain_long),  # 有调用 → 不压
        Message("user", "近"),
        Message("assistant", plain_long),  # 太近 → 不压
    ]
    out = compress_long_replies(list(msgs))
    assert len(out[0].content) < len(plain_long)
    assert out[1].content == msgs[1].content
    assert out[3].content == msgs[3].content


# ---- 兜底 ----


def test_hard_cap_keeps_pinned_and_tail() -> None:
    msgs = [Message("system", "提示词", pinned=True)] + [
        Message("user", f"消息{i}内容内容内容") for i in range(20)
    ]
    out = hard_cap(msgs, budget=60)
    assert out[0].pinned  # 提示词永远在
    assert len(out) < len(msgs)  # 确实丢了旧的
    assert out[-1].content.startswith("消息19")  # 最新的在
    assert messages_tokens(out) <= 60 + 30  # 约束在预算量级内(允许最后一条略超)


# ---- 总入口 ----


def test_compact_under_budget_only_strips_think() -> None:
    msgs = [
        *[Message("user", f"问{i}") for i in range(10)],
        Message("assistant", "<think>老思考</think>结论"),
        *[Message("user", f"尾{i}") for i in range(5)],
    ]
    out = compact_messages(msgs, budget=10_000)
    assert messages_tokens(out) <= 10_000
    assert all("老思考" not in m.content for m in out[:-6])  # 老的被剥(logdir None 也一样)


def test_compact_over_budget_fits(workspace: Path, logdir: Path) -> None:
    msgs = [Message("system", "提示词", pinned=True, hide=True)]
    for i in range(6):
        msgs.append(Message("user", f"问{i}"))
        msgs.append(Message("system", "C" * 5000))  # 一堆大工具结果
    out = compact_messages(msgs, budget=3000, logdir=logdir)
    assert messages_tokens(out) <= 3000 + 1500  # 收敛到预算量级
    assert out[0].pinned  # 提示词没丢


def test_prepare_messages_integration() -> None:
    msgs = [Message("system", "提示词", pinned=True, hide=True)]
    msgs += [Message("user", f"问{i}") for i in range(30)]
    view = prepare_messages(msgs, "mock/echo", budget=80)
    assert messages_tokens(view) <= 80 + 30  # 视图被压到预算量级
    # 再次确认:原列表没被动
    assert len(msgs) == 31


# ---- 溢出自愈 ----


def test_step_overflow_retry_with_smaller_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """第一次 reply 抛上下文超长 → step 以更小预算重压并重试一次。"""
    from my_agent_alpha import chat as chat_mod

    calls: list[Sequence[Message]] = []
    state = {"failed": False}
    real_reply = chat_mod.reply

    def flaky_reply(messages, model, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(messages))
        if not state["failed"]:
            state["failed"] = True
            raise ContextLengthError("This model's maximum context length is 4096 tokens")
        return real_reply(messages, model, **kwargs)

    monkeypatch.setattr(chat_mod, "reply", flaky_reply)
    log = [Message("system", "提示词", pinned=True, hide=True)] + [
        Message("user", f"消息{i}" + "内容" * 50) for i in range(20)
    ]
    out = list(step(log, "mock/echo", budget=2000))
    assert len(calls) == 2  # 失败一次、重试一次
    assert messages_tokens(list(calls[1])) <= messages_tokens(list(calls[0]))  # 重试的更小
    assert out[0].role == "assistant"


def test_step_overflow_second_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """重试仍超长 → 不再无限循环,异常抛出。"""
    from my_agent_alpha import chat as chat_mod

    def always_overflow(messages, model, **kwargs):  # type: ignore[no-untyped-def]
        raise ContextLengthError("context length exceeded")

    monkeypatch.setattr(chat_mod, "reply", always_overflow)
    with pytest.raises(ContextLengthError):
        list(step([Message("user", "hi")], "mock/echo"))
