"""Message 序列化/反序列化测试。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from my_agent_alpha.message import Message, clean_bad_unicode, utcnow


def test_minimal_to_dict_only_writes_required_fields() -> None:
    msg = Message("user", "你好")
    d = msg.to_dict()
    # 只有 role/content/timestamp 恒定写出,可选字段默认值不写出
    assert set(d) == {"role", "content", "timestamp"}
    assert d["role"] == "user"
    assert d["content"] == "你好"


def test_full_round_trip() -> None:
    msg = Message(
        role="system",
        content="工具输出",
        timestamp=datetime(2026, 9, 28, 15, 30, 0, tzinfo=UTC),
        call_id="call-123",
        pinned=True,
        hide=True,
        quiet=True,
        metadata={"model": "test", "usage": {"input": 10, "output": 5}},
    )
    restored = Message.from_dict(msg.to_dict())
    assert restored == msg
    assert restored.timestamp == msg.timestamp  # 相等性不含时间戳,单独验证往返
    assert restored.call_id == "call-123"
    assert restored.pinned and restored.hide and restored.quiet
    assert restored.metadata == msg.metadata


def test_from_dict_drops_unknown_fields() -> None:
    # 向前兼容:未来版本日志里的新字段不会让旧代码崩溃
    msg = Message.from_dict(
        {"role": "user", "content": "hi", "future_field": 123, "another": {"a": 1}}
    )
    assert msg.role == "user"
    assert msg.content == "hi"


def test_from_dict_defaults_on_missing_fields() -> None:
    msg = Message.from_dict({"role": "assistant", "content": "回复"})
    assert msg.call_id is None
    assert not (msg.pinned or msg.hide or msg.quiet)
    assert msg.metadata is None
    # 时间戳缺失回退为当前时间(带时区)
    assert msg.timestamp.tzinfo is not None
    assert (utcnow() - msg.timestamp).total_seconds() < 60


def test_from_dict_invalid_role_raises() -> None:
    with pytest.raises(ValueError, match="role"):
        Message.from_dict({"role": "tool", "content": "?"})


def test_from_dict_corrupt_timestamp_falls_back() -> None:
    msg = Message.from_dict({"role": "user", "content": "x", "timestamp": "not-a-date"})
    assert (utcnow() - msg.timestamp).total_seconds() < 60


def test_constructor_validates_role() -> None:
    with pytest.raises(ValueError, match="role"):
        Message("tool", "?")  # type: ignore[arg-type]


def test_frozen() -> None:
    msg = Message("user", "x")
    with pytest.raises(FrozenInstanceError):
        msg.content = "y"  # type: ignore[misc]


def test_metadata_excluded_from_equality() -> None:
    a = Message("user", "x")
    b = Message("user", "x", metadata={"cost": 1.0})
    assert a == b
    # 且可哈希(dict 字段不会让 hash 崩溃)
    assert len({a, b}) == 1


def test_equality_ignores_call_id_and_flags() -> None:
    """相等性仿 gptme 只看 (role, content):call_id/标志位不参与同一性判定。"""
    a = Message("user", "x")
    b = Message("user", "x", call_id="call-1", pinned=True, hide=True, quiet=True)
    assert a == b
    assert hash(a) == hash(b)
    assert a != Message("assistant", "x")
    assert a != Message("user", "y")


def test_naive_timestamp_assumed_utc() -> None:
    msg = Message.from_dict({"role": "user", "content": "x", "timestamp": "2026-09-28T12:00:00"})
    assert msg.timestamp.tzinfo is not None  # 模块内时间戳恒为 aware
    assert msg.timestamp.utcoffset().total_seconds() == 0


def test_metadata_defensive_copy() -> None:
    source = {"k": "v"}
    msg = Message("user", "x", metadata=source)
    source["k"] = "被外部改动"
    source["new"] = 1
    assert msg.metadata == {"k": "v"}


def test_from_dict_non_dict_raises() -> None:
    with pytest.raises(ValueError, match="JSON 对象"):
        Message.from_dict("not a dict")  # type: ignore[arg-type]


def test_clean_bad_unicode() -> None:
    """落单代理字符(终端坏字节的解码残渣)换成 ?;正常字符(含合法表情)原样保留。"""
    assert clean_bad_unicode("\ud83c有雨吗") == "?有雨吗"
    assert clean_bad_unicode("正常中文🌧️") == "正常中文🌧️"
