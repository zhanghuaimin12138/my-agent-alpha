"""阶段 6 上下文压缩测试:估算、预算、三阶段、兜底、溢出自愈。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from my_agent_alpha import Message
from my_agent_alpha.chat import step
from my_agent_alpha.compaction import (
    compact_messages,
    compress_long_replies,
    compress_reply,
    hard_cap,
    prepare_messages,
    strip_old_thinking,
    truncate_tool_results,
    validate_compacted_messages,
)
from my_agent_alpha.llm import ContextLengthError
from my_agent_alpha.utils import len_tokens

# token 估算与预算的测试在 tests/test_tokens.py(函数已搬到 utils 包)


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
    assert len_tokens(out) <= 60 + 30  # 约束在预算量级内(允许最后一条略超)


# ---- 总入口 ----


def test_compact_under_budget_only_strips_think() -> None:
    msgs = [
        *[Message("user", f"问{i}") for i in range(10)],
        Message("assistant", "<think>老思考</think>结论"),
        *[Message("user", f"尾{i}") for i in range(5)],
    ]
    out = compact_messages(msgs, budget=10_000)
    assert len_tokens(out) <= 10_000
    assert all("老思考" not in m.content for m in out[:-6])  # 老的被剥(logdir None 也一样)


def test_compact_over_budget_fits(workspace: Path, logdir: Path) -> None:
    msgs = [Message("system", "提示词", pinned=True, hide=True)]
    for i in range(6):
        msgs.append(Message("user", f"问{i}"))
        msgs.append(Message("system", "C" * 5000))  # 一堆大工具结果
    out = compact_messages(msgs, budget=3000, logdir=logdir)
    assert len_tokens(out) <= 3000 + 1500  # 收敛到预算量级
    assert out[0].pinned  # 提示词没丢


def test_prepare_messages_integration() -> None:
    msgs = [Message("system", "提示词", pinned=True, hide=True)]
    msgs += [Message("user", f"问{i}") for i in range(30)]
    view = prepare_messages(msgs, "mock/echo", budget=80)
    assert len_tokens(view) <= 80 + 30  # 视图被压到预算量级
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
    assert len_tokens(list(calls[1])) <= len_tokens(list(calls[0]))  # 重试的更小
    assert out[0].role == "assistant"


def test_step_overflow_second_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """重试仍超长 → 不再无限循环,异常抛出。"""
    from my_agent_alpha import chat as chat_mod

    def always_overflow(messages, model, **kwargs):  # type: ignore[no-untyped-def]
        raise ContextLengthError("context length exceeded")

    monkeypatch.setattr(chat_mod, "reply", always_overflow)
    with pytest.raises(ContextLengthError):
        list(step([Message("user", "hi")], "mock/echo"))


# ---- V4 修复:压缩不许丢字段、不许拆工具配对 ----


def test_stages_preserve_message_fields(logdir: Path) -> None:
    """剥思考/截断/压长回复都只换 content:call_id/pinned/hide/quiet/metadata 原样保留。

    这是压缩结果能升级为持久化视图的前提 —— 手写 Message(...) 重建
    会把这些字段全丢掉,历史就失真了。
    """
    # 阶段 1:剥思考(距结尾 ≥5 的老回复)
    old = [
        Message("user", "问0"),
        Message(
            "assistant",
            "<think>推理……</think>结论",
            pinned=True,
            hide=True,
            quiet=True,
            call_id="call-x",
            metadata={"model": "m"},
        ),
        Message("user", "问1"),
        Message("user", "问2"),
        Message("user", "问3"),
        Message("user", "问4"),
        Message("user", "问5"),
    ]
    out1 = strip_old_thinking(old)
    assert "<think>" not in out1[1].content
    assert out1[1].call_id == "call-x"
    assert out1[1].pinned and out1[1].hide and out1[1].quiet
    assert out1[1].metadata == {"model": "m"}

    # 阶段 2:截断工具结果(call_id 必须跟着占位走)
    big = Message("system", "R" * 20000, call_id="call-1")
    out2 = truncate_tool_results([big], budget=100, logdir=logdir)
    assert out2[0].call_id == "call-1"
    assert out2[0].content.startswith("[内容已截断")

    # 阶段 3:压长回复(无调用的老长文)
    long_reply = Message("assistant", "开头结论。" + "普通句子。" * 900, metadata={"gen": 1})
    filler = [Message("user", "近")] * 3
    out3 = compress_long_replies([long_reply, *filler])
    assert len(out3[0].content) < len(long_reply.content)
    assert out3[0].metadata == {"gen": 1}


def test_compress_skips_native_tool_calls() -> None:
    """原生通道(@工具名(编号) 行)的回复同样是执行记录,不压。"""
    native = "我先读文件再总结。" + "说明文字。" * 300 + '\n@read(call_1): {"path": "a.txt"}\n'
    msgs = [
        Message("assistant", native),
        Message("user", "近"),
    ]
    out = compress_long_replies(list(msgs))
    assert out[0].content == native  # 带原生调用 → 原样


def _native_interaction_history() -> list[Message]:
    """构造原生模式的历史:一次回复发两个调用,[用量] 夹在中间。"""
    return [
        Message("system", "提示词", pinned=True),
        Message("user", "老问题" + "很" * 2000),
        Message(
            "assistant", '先读两个文件:\n@read(call_1): {"path": "a"}\n@read(call_2): {"path": "b"}'
        ),
        Message("system", "[用量] 输入 100 / 输出 20 tokens"),
        Message("system", "一" * 300, call_id="call_1"),
        Message("system", "二" * 300, call_id="call_2"),
        Message("user", "新问题"),
    ]


def test_hard_cap_never_splits_tool_group() -> None:
    """双调用的交互组要么整组保留、要么整组丢弃,永不拆散。"""
    msgs = _native_interaction_history()

    # 预算只够 pinned + 新问题:交互组(约 630 tokens)整组被丢
    tight = hard_cap(list(msgs), budget=100)
    contents = [m.content for m in tight]
    assert tight[0].pinned  # 提示词永远在
    assert "新问题" in contents  # 最新在
    assert not any("call_1" in c or "call_2" in c for c in contents)  # 调用整组丢
    assert all(m.call_id is None for m in tight)  # 没有半路留下的结果
    assert not any(c.startswith("[用量]") for c in contents)  # 用量随组走

    # 预算够装下交互组:调用、用量、两个结果、新问题全在,顺序不变
    # (老问题那组太大被整组丢掉,所以结果共 6 条)
    roomy = hard_cap(list(msgs), budget=800)
    assert [m.call_id for m in roomy] == [None, None, None, "call_1", "call_2", None]
    assert any(c.startswith("[用量]") for c in (m.content for m in roomy))
    assert not any(c.startswith("老问题") for c in (m.content for m in roomy))  # 老的大块丢


def test_hard_cap_result_translates_cleanly() -> None:
    """hard_cap 的结果发给接口转换层:没有孤儿 tool、没有丢主的调用。"""
    from my_agent_alpha.llm.openai_compat import msgs2dicts

    msgs = _native_interaction_history()
    roomy = hard_cap(list(msgs), budget=800)
    dicts = msgs2dicts(roomy)

    # 所有 tool 结果都有主(编号能对上某条 assistant 的 tool_calls)
    call_ids = {
        tc["id"] for d in dicts if d["role"] == "assistant" for tc in d.get("tool_calls", [])
    }
    tool_ids = [d["tool_call_id"] for d in dicts if d["role"] == "tool"]
    assert sorted(tool_ids) == sorted(call_ids)
    # 没有结果被降级成 user(孤儿降级是兼容兜底,出现在这里说明拆了组)
    assert not any(d["role"] == "user" and "一" in str(d.get("content", "")) for d in dicts)


# ---- V4:视图激活前的安检 ----


def test_validate_accepts_clean_result() -> None:
    """干净结果(pinned 在、最新 user 在、调用配对完整)顺利通过。"""
    source = _native_interaction_history()
    validate_compacted_messages(source, list(source))  # 不抛即通过
    # 剥过思考式的合法变形(只改文字,调用行与结果都原样)同样通过
    reworded = replace(source[2], content="总结:" + source[2].content)
    validate_compacted_messages(source, [source[0], source[1], reworded, *source[3:]])


def test_validate_rejects_lost_pinned() -> None:
    source = _native_interaction_history()
    victim = [m for m in source if m is not source[0]]  # 丢掉 pinned 提示词
    with pytest.raises(ValueError, match="pinned"):
        validate_compacted_messages(source, victim)


def test_validate_rejects_lost_latest_user() -> None:
    source = _native_interaction_history()
    victim = [m for m in source if m is not source[6]]  # 丢掉最后一条 user
    with pytest.raises(ValueError, match="最后一条用户"):
        validate_compacted_messages(source, victim)


def test_validate_rejects_broken_call_pairs() -> None:
    """有调用无结果 / 孤儿结果 / 同编号两条结果,一律拒绝。"""
    source = _native_interaction_history()
    # 有调用无结果:裁掉 call_2 的结果
    no_result = [m for m in source if m.call_id != "call_2"]
    with pytest.raises(ValueError, match="配对不完整"):
        validate_compacted_messages(source, no_result)
    # 孤儿结果:结果在、发起调用的 assistant 被裁掉
    orphan = [m for m in source if m is not source[2]]
    with pytest.raises(ValueError, match="孤儿"):
        validate_compacted_messages(source, orphan)
    # 同编号两条结果(重复盖号也算坏)
    dup = [*source, Message("system", "重复结果", call_id="call_1")]
    with pytest.raises(ValueError, match="配对不完整"):
        validate_compacted_messages(source, dup)
