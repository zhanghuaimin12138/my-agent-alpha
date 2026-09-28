"""Log / LogManager 持久化测试。

验收标准(阶段 0):日志能追加、重启能恢复;
追加游标走增量路径、被改动时正确降级为原子全量重写。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from my_agent_alpha import Log, LogManager, Message
from my_agent_alpha.logmanager.manager import generate_conversation_id


def test_append_creates_jsonl(logdir: Path) -> None:
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "列出当前目录"))
    manager.append(Message("assistant", "```shell\nls\n```"))
    manager.append(Message("system", "file1\nfile2", call_id="call-1"))

    f = logdir / "conversation.jsonl"
    assert f.exists()
    lines = f.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [p["role"] for p in parsed] == ["user", "assistant", "system"]
    assert parsed[2]["call_id"] == "call-1"
    # 中文保持人类可读(未转义成 \uXXXX)
    assert "列出当前目录" in lines[0]


def test_reload_recovers_after_restart(logdir: Path) -> None:
    manager = LogManager.load(logdir, create=True)
    for i in range(5):
        manager.append(Message("user", f"消息 {i}"))

    # 模拟"重启":用全新实例重新装载
    restored = LogManager.load(logdir)
    assert len(restored) == 5
    assert [m.content for m in restored.messages] == [f"消息 {i}" for i in range(5)]


def test_reload_then_append_is_incremental(logdir: Path) -> None:
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "第一条"))
    f = logdir / "conversation.jsonl"
    before = f.read_bytes()

    # 重启后追加:文件应当只在尾部增长(前缀字节保持不变)
    restored = LogManager.load(logdir)
    restored.append(Message("user", "第二条"))
    after = f.read_bytes()
    assert after.startswith(before)
    assert len(after) > len(before)

    # 追加后的内容可再次恢复
    again = LogManager.load(logdir)
    assert [m.content for m in again.messages] == ["第一条", "第二条"]


def test_can_append_cursor_logic(logdir: Path) -> None:
    """白盒测试:追加游标的四重校验。"""
    f = logdir / "conversation.jsonl"
    m1, m2 = Message("user", "1"), Message("user", "2")

    fresh = Log([m1, m2])
    assert not fresh._can_append(f)  # 从未落盘 → 不能增量

    fresh.write_jsonl(f)
    assert fresh._can_append(f)  # 刚写完 → 能增量

    # 文件被外部截断 → 降级
    truncated = f.read_bytes()[:-10]
    f.write_bytes(truncated)
    assert not fresh._can_append(f)
    f.write_bytes(truncated + b"\n")  # 修复行尾但字节数不同 → 仍降级
    assert not fresh._can_append(f)


def test_rewrite_when_internal_prefix_changed(logdir: Path) -> None:
    """白盒测试:内存前缀被改写时,写盘走原子全量重写而非增量。"""
    f = logdir / "conversation.jsonl"
    log = Log([Message("user", "1"), Message("user", "2")])
    log.write_jsonl(f)
    before = f.read_text()

    # 模拟"编辑"(删除首条)—— 绕过公共 API 直接操纵内部列表
    log._messages.pop(0)
    assert not log._can_append(f)
    log.write_jsonl(f)

    after = f.read_text()
    assert after != before
    # 按 content 精确校验(时间戳等字段里也可能出现 "1" 这样的字符)
    assert [m.content for m in Log.read_jsonl(f)] == ["2"]
    # 不留临时文件
    assert list(logdir.glob("*.tmp-*")) == []


def test_rewrite_is_atomic_via_temp_and_replace(logdir: Path) -> None:
    """全量重写后:目标文件存在、无残留临时文件、内容完整。"""
    f = logdir / "conversation.jsonl"
    log = Log([Message("user", f"行{i}") for i in range(10)])
    log.write_jsonl(f, sync=True)
    assert f.exists()
    assert list(logdir.glob("*.tmp-*")) == []
    restored = Log.read_jsonl(f)
    assert len(restored) == 10


def test_torn_tail_line_tolerated(logdir: Path, caplog: pytest.LogCaptureFixture) -> None:
    f = logdir / "conversation.jsonl"
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "1"))
    manager.append(Message("user", "2"))

    # 模拟崩溃:末尾写了一半的行
    with open(f, "ab") as fh:
        fh.write('{"role": "user", "content": "写'.encode())
    with caplog.at_level(logging.WARNING):
        log = Log.read_jsonl(f)
    assert len(log) == 2
    assert any("损坏" in r.message for r in caplog.records)


def test_torn_tail_splitting_multibyte_char(logdir: Path) -> None:
    """残尾切在多字节 UTF-8 字符中间:不许 UnicodeDecodeError,且能自愈。"""
    f = logdir / "conversation.jsonl"
    LogManager.load(logdir, create=True).append(Message("user", "正常消息"))

    # "正" = E6 AD A3,只写前两个字节 —— 整文件 read_text 会直接解码失败
    with open(f, "ab") as fh:
        fh.write(b'{"role": "user", "content": "\xe6\xad')

    log = Log.read_jsonl(f)  # 不抛 UnicodeDecodeError
    assert len(log) == 1

    # 装载后继续追加:游标未知 → 全量重写清掉残尾 → 重启可正常打开
    manager = LogManager.load(logdir)
    manager.append(Message("user", "追加消息"))
    restored = LogManager.load(logdir)
    assert [m.content for m in restored.messages] == ["正常消息", "追加消息"]


def test_corrupt_complete_last_line_self_heals(logdir: Path) -> None:
    """末行完整(带换行)但非法:跳过后继续追加必须走重写,否则坏行变成中间行。"""
    f = logdir / "conversation.jsonl"
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "1"))

    # 合法 JSON、完整一行,但 role 非法 → from_dict 抛 ValueError
    with open(f, "ab") as fh:
        fh.write(b'{"role": "tool", "content": "x"}\n')

    restored = LogManager.load(logdir)  # 容忍跳过
    assert len(restored) == 1
    restored.append(Message("user", "2"))  # 必须触发全量重写清掉坏行

    again = LogManager.load(logdir)  # 若走了增量追加,这里会抛 ValueError
    assert [m.content for m in again.messages] == ["1", "2"]


def test_non_dict_json_lines(logdir: Path) -> None:
    f = logdir / "conversation.jsonl"
    good = json.dumps({"role": "user", "content": "ok"})

    # 末行是非对象 JSON → 容忍跳过
    f.write_text(f"{good}\n123\n", encoding="utf-8")
    assert len(Log.read_jsonl(f)) == 1

    # 中间行是非对象 JSON → 报错
    f.write_text(f"{good}\n[1, 2]\n{good}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="损坏"):
        Log.read_jsonl(f)


def test_sync_append_path_fsyncs(logdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """增量追加路径的 sync=True 分支真的会 fsync(回归防护)。"""
    calls = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real_fsync(fd))[1], raising=True)

    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "首条"))  # 走全量重写(游标为空)
    calls.clear()

    reloaded = LogManager.load(logdir)
    reloaded.append(Message("user", "第二条"), sync=True)  # 走增量追加
    assert calls, "sync=True 的增量追加路径应当调用 os.fsync"

    monkeypatch.undo()  # 恢复真实 fsync 后再验证内容
    assert len(LogManager.load(logdir)) == 2


def test_messages_returns_defensive_copy(logdir: Path) -> None:
    """messages 属性返回拷贝:外部改动不得影响日志内部状态。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "原消息"))

    snap = manager.messages
    snap.append(Message("user", "混入"))
    snap.clear()
    assert len(manager) == 1
    assert manager.log[0].content == "原消息"

    # 内部状态未受影响 → 追加仍走增量路径
    manager.append(Message("user", "再一条"))
    assert manager.log._can_append(manager.conversation_file)


def test_corrupt_middle_line_raises(logdir: Path) -> None:
    f = logdir / "conversation.jsonl"
    lines = [
        json.dumps({"role": "user", "content": "1"}),
        "{broken json",
        json.dumps({"role": "user", "content": "3"}),
    ]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="损坏"):
        Log.read_jsonl(f)


def test_load_missing_without_create_raises(logdir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LogManager.load(logdir)


def test_conversation_id_format() -> None:
    cid = generate_conversation_id()
    # YYYY-MMDD-<6 位十六进制>
    parts = cid.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 4 and len(parts[1]) == 4 and len(parts[2]) == 6
    int(parts[2], 16)  # 合法十六进制


def test_empty_then_write_produces_valid_empty_reload(logdir: Path) -> None:
    """空会话落盘后也能重新装载(空文件边界)。"""
    manager = LogManager.load(logdir, create=True)
    manager.write()
    restored = LogManager.load(logdir)
    assert len(restored) == 0
    restored.append(Message("user", "重启后第一条"))
    assert len(LogManager.load(logdir)) == 1
