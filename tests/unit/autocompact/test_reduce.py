"""兜底压缩器测试(§9:M4 组)。"""

from __future__ import annotations

from my_agent_alpha.autocompact.adapters import EstimateTokenCounter
from my_agent_alpha.autocompact.reduce import reduce_log, truncate_msg
from my_agent_alpha.message import Message

COUNTER = EstimateTokenCounter()


def _no_tool(_: Message) -> bool:
    """测试用的探测:一律认为没有工具调用。"""
    return False


def _always_tool(_: Message) -> bool:
    """测试用的探测:一律认为有工具调用(谁都动不了)。"""
    return True


def _block_msg(n_lines: int) -> Message:
    """造一条只含一个 n 行代码块的消息。"""
    lines = "\n".join(f"line {i:02d} " + "y" * 90 for i in range(n_lines))
    return Message("assistant", f"```python\n{lines}\n```")


def test_under_limit_unchanged() -> None:
    """没超预算:原样返回,一个字都不动。"""
    msgs = [Message("user", "hello"), Message("assistant", "hi there")]
    assert reduce_log(msgs, 1000, counter=COUNTER, contains_tool_use=_no_tool) == msgs


def test_long_code_block_truncated_to_head_tail() -> None:
    """超过 21 行的代码块:留前 10 + [...] + 后 10(共 21 行)。"""
    msg = _block_msg(25)
    result = reduce_log([msg], 10, counter=COUNTER, contains_tool_use=_no_tool)  # limit=10 逼它动手
    block = result[0].content
    assert "[...]" in block
    assert "line 00" in block and "line 09" in block  # 前 10 行
    assert "line 15" in block and "line 24" in block  # 后 10 行
    assert "line 10" not in block  # 中间 5 行被砍
    # 压完还是超 limit 但已无可砍 → 原样(截断后的样子)返回,不抛异常
    assert COUNTER.len_tokens(result) > 10


def test_block_at_threshold_untouched() -> None:
    """恰好 21 行:不到砍的门槛 → 没有进展,原样返回。"""
    msg = _block_msg(21)
    result = reduce_log([msg], 10, counter=COUNTER, contains_tool_use=_no_tool)
    assert result[0].content == msg.content


def test_truncate_msg_returns_none_when_nothing_to_do() -> None:
    """没有可截内容(块不超长/含工具调用)→ None,而不是原样消息。"""
    assert truncate_msg(_block_msg(21), _no_tool) is None
    assert truncate_msg(_block_msg(30), _always_tool) is None


def test_all_pinned_no_candidates() -> None:
    """候选为空(全 pinned):告警并原样返回。"""
    msgs = [Message("user", "x" * 8000, pinned=True)]
    result = reduce_log(msgs, 10, counter=COUNTER, contains_tool_use=_no_tool)
    assert result == msgs


def test_tooluse_messages_not_candidates() -> None:
    """所有消息都含工具调用:谁都动不了,原样返回。"""
    msgs = [_block_msg(30)]
    result = reduce_log(msgs, 10, counter=COUNTER, contains_tool_use=_always_tool)
    assert result == msgs
