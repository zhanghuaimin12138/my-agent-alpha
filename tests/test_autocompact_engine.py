"""三阶段引擎测试(§9 引擎组):剥思考/截工具结果/压长回复/兜底。"""

from __future__ import annotations

import re
from pathlib import Path

from my_agent_alpha.autocompact.adapters import EstimateTokenCounter
from my_agent_alpha.autocompact.engine import auto_compact_log
from my_agent_alpha.autocompact.master_context import MasterContext, read_master_range
from my_agent_alpha.autocompact.output_storage import FileOutputStore
from my_agent_alpha.logmanager import LogManager
from my_agent_alpha.message import Message

COUNTER = EstimateTokenCounter()


def _has_tool_use(msg: Message) -> bool:
    """引擎测试里用的"工具调用探测":含 ```shell 围栏就算有调用。"""
    return "```shell" in msg.content


def _big_text(tokens: int) -> str:
    """造一段恰好约 tokens 个 token 的 ascii 文本(4 字符/token)。"""
    return "x" * (tokens * 4)


def _assert_in_place(original: list[Message], result: list[Message]) -> None:
    """不变量 1:压缩是原位替换 —— 条数、顺序、角色都不变。"""
    assert len(result) == len(original)
    assert [m.role for m in result] == [m.role for m in original]


def test_phase1_strips_old_reasoning_keeps_recent() -> None:
    """阶段 1:距末尾 ≥5 条的思考区被剥,近期的保留。"""
    msgs = [
        Message("assistant", "<think>old reasoning here</think>final answer"),
        *[Message("user", f"q{i}") for i in range(1, 7)],
        Message("assistant", "<think>fresh reasoning</think>new answer"),
    ]
    result = auto_compact_log(
        msgs, context_window=128_000, counter=COUNTER, contains_tool_use=_has_tool_use
    )
    _assert_in_place(msgs, result)
    assert result[0].content == "final answer"  # 旧的(距末尾 7)剥掉
    assert "<think>" in result[7].content  # 近期(距末尾 0)原样保留
    assert msgs[0].content.startswith("<think>")  # 入参列表没被改


def test_phase1_collapses_blank_lines() -> None:
    """阶段 1:剥掉思考区后留下的大片空行压成一个空行。"""
    msgs = [Message("assistant", "<think>reason</think>\n\n\n\nbody")] + [
        Message("user", "hi") for _ in range(5)
    ]
    result = auto_compact_log(
        msgs, context_window=128_000, counter=COUNTER, contains_tool_use=_has_tool_use
    )
    assert result[0].content == "body"


def test_phase2_truncates_largest_first_and_stops() -> None:
    """阶段 2:从最大的工具结果截起,达标即停(次大的不动)。"""
    msgs = [
        Message("user", "start"),
        Message("system", _big_text(2100)),  # B:次大(在前)
        Message("system", _big_text(3000)),  # A:最大(在后)→ 先截它
        Message("system", _big_text(1900)),  # C:不到 2000,不是候选
        Message("system", _big_text(1900)),  # D:同上
        Message("user", "end"),
    ]  # 总量 8904 > 目标线(0.8 × 10000 = 8000)
    result = auto_compact_log(
        msgs, context_window=10_000, counter=COUNTER, contains_tool_use=_has_tool_use
    )
    _assert_in_place(msgs, result)
    assert result[2].content.startswith("[Large tool output removed - 3000 tokens]")
    assert result[1].content == _big_text(2100)  # 达标即停:B 没被动
    assert result[3].content == _big_text(1900)  # 小候选本来就不截
    assert COUNTER.len_tokens(result) <= 8000


def test_phase2_saves_full_output_to_files(logdir: Path) -> None:
    """阶段 2:全量内容按截断顺序落盘(001 对应第一个被截的)。"""
    msgs = [
        Message("user", "start"),
        Message("system", "Ran command: ls -la\n" + _big_text(6500)),  # B:6506 token,最大
        Message("system", _big_text(6400)),  # A:6400 token,次大
        Message("user", "end"),
    ]  # 总量 12910 > 目标线 6400:截完 B(剩 6449)仍超线,两条都得截
    result = auto_compact_log(
        msgs,
        context_window=8000,
        counter=COUNTER,
        contains_tool_use=_has_tool_use,
        store=FileOutputStore(logdir),
    )
    _assert_in_place(msgs, result)
    out1 = logdir / "outputs" / "autocompact-001.txt"
    out2 = logdir / "outputs" / "autocompact-002.txt"
    assert out1.read_text(encoding="utf-8") == msgs[1].content  # 最大(B)先截 → 001
    assert out2.read_text(encoding="utf-8") == msgs[2].content  # 其次(A)→ 002
    # 命令痕迹附在占位文本里,模型不用读文件就知道当年执行了什么
    assert "(Ran command: ls -la)" in result[1].content
    assert "Full output saved to:" in result[1].content


def test_phase2_master_reference_roundtrip(logdir: Path) -> None:
    """阶段 2:截断占位附主日志字节引用;按区间能一字不差取回原文。"""
    msgs = [
        Message("user", "start"),
        Message("system", _big_text(2100)),
        Message("system", _big_text(3000)),
        Message("system", _big_text(1900)),
        Message("system", _big_text(1900)),
        Message("user", "end"),
    ]
    # 先把同样的消息写成真实的主日志(和视图等长,下标才对得上)
    manager = LogManager.load(logdir, create=True)
    for m in msgs:
        manager.append(m)
    master = MasterContext.build(logdir)
    assert len(master.index) == 6  # 一行一条,空行不算

    result = auto_compact_log(
        msgs,
        context_window=10_000,
        counter=COUNTER,
        contains_tool_use=_has_tool_use,
        master=master,
    )
    _assert_in_place(msgs, result)
    content = result[2].content
    assert "Master context:" in content and "bytes" in content
    m = re.search(r"bytes (\d+)-(\d+)", content)
    assert m is not None
    recovered = read_master_range(master.path, int(m.group(1)), int(m.group(2)))
    assert recovered == _big_text(3000)  # 字节区间精确指回原文


def test_phase3_skips_tooluse_and_compresses_long_reply() -> None:
    """阶段 3:含工具调用的回复不压;普通长回复被抽稀且 token 下降。"""
    long_plain = " ".join(
        f"filler sentence number {i} about the general topic." for i in range(100)
    )
    long_with_tool = (
        " ".join(f"context sentence {i} before the call." for i in range(100))
        + "\n```shell\nls -la\n```"
    )
    msgs = [
        Message("assistant", long_plain),
        Message("assistant", long_with_tool),
        Message("user", "q1"),
        Message("user", "q2"),
        Message("user", "q3"),
        Message("user", "q4"),
    ]
    result = auto_compact_log(
        msgs, context_window=128_000, counter=COUNTER, contains_tool_use=_has_tool_use
    )
    _assert_in_place(msgs, result)
    # 普通长回复被压缩(token 下降)
    assert COUNTER.len_tokens(result[0].content) < COUNTER.len_tokens(long_plain)
    # 含工具调用的回复原样保留(保住调用/结果配对)
    assert result[1].content == long_with_tool


def test_phase3_code_blocks_survive() -> None:
    """阶段 3:长回复里的代码块无条件完整保留。"""
    block = "```python\nprint('keep me')\n```"
    filler = " ".join(f"plain filler sentence {i} with several words." for i in range(100))
    msgs = [
        Message("assistant", f"{block} {filler}"),
        Message("user", "q"),
        Message("user", "q"),
        Message("user", "q"),
        Message("user", "q"),
    ]
    result = auto_compact_log(
        msgs, context_window=128_000, counter=COUNTER, contains_tool_use=_has_tool_use
    )
    assert block in result[0].content
    assert COUNTER.len_tokens(result[0].content) < COUNTER.len_tokens(msgs[0].content)


def test_keep_head_pins_and_protects_head() -> None:
    """keep_head=2:前 2 条原样且输出标 pinned=True;压缩只动后面。"""
    msgs = [
        Message("system", "prompt-1"),
        Message("system", "prompt-2"),
        Message("system", _big_text(2900)),
        Message("system", _big_text(3000)),
        Message("system", _big_text(2100)),
        Message("user", "q1"),
        Message("user", "q2"),
        Message("user", "q3"),
    ]  # 总量 8010 > 目标线 8000
    result = auto_compact_log(
        msgs,
        context_window=10_000,
        counter=COUNTER,
        contains_tool_use=_has_tool_use,
        keep_head=2,
    )
    _assert_in_place(msgs, result)
    assert result[0].content == "prompt-1" and result[0].pinned is True
    assert result[1].content == "prompt-2" and result[1].pinned is True
    assert result[2].pinned is False
    assert result[3].content.startswith("[Large tool output removed")  # 最大的被截
    assert result[2].content == _big_text(2900)  # 截完达标,2900 那条不动


def test_early_exit_pins_head_only() -> None:
    """早退:没什么可压时原样返回,但头部照样钉住。"""
    msgs = [Message("system", "p1"), Message("system", "p2"), Message("user", "hi")]
    result = auto_compact_log(
        msgs,
        context_window=128_000,
        counter=COUNTER,
        contains_tool_use=_has_tool_use,
        keep_head=2,
    )
    _assert_in_place(msgs, result)
    assert result[0].pinned is True and result[1].pinned is True
    assert result[2].pinned is False
    assert result[0].content == "p1"  # 内容没人动


def test_all_pinned_unchanged_even_over_limit() -> None:
    """全部 pinned:即使超预算也原样返回(不动任何一条)。"""
    msgs = [
        Message("system", _big_text(3000), pinned=True),
        Message("system", _big_text(2900), pinned=True),
        Message("system", _big_text(2500), pinned=True),
    ]  # 总量 8400 > 8000
    result = auto_compact_log(
        msgs, context_window=10_000, counter=COUNTER, contains_tool_use=_has_tool_use
    )
    _assert_in_place(msgs, result)
    assert [m.content for m in result] == [m.content for m in msgs]


def test_reduce_fallback_truncates_code_blocks() -> None:
    """兜底:三阶段压不到位时,超长代码块被砍成 前10+[...]+后10。"""
    lines = "\n".join(f"line {i:02d} " + "y" * 90 for i in range(50))
    big_block = "result:\n```python\n" + lines + "\n```"
    msgs = [Message("user", "run"), Message("assistant", big_block), Message("user", "done")]
    result = auto_compact_log(
        msgs, context_window=300, counter=COUNTER, contains_tool_use=_has_tool_use
    )
    _assert_in_place(msgs, result)  # 兜底也是原位替换,条数不变
    block = result[1].content
    assert "[...]" in block
    assert "line 00" in block and "line 49" in block  # 头尾保留
    assert "line 15" not in block  # 中间被砍
