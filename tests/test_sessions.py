"""sessions.py 测试:会话列表元数据、前缀查找、无人聊过清理。"""

from __future__ import annotations

import os
from pathlib import Path

from my_agent_alpha import LogManager, Message
from my_agent_alpha.logmanager.sessions import (
    SessionMeta,
    find_by_prefix,
    format_session_row,
    is_subagent,
    list_sessions,
    prune_empty_session,
    write_session_title,
)


def _make_session(
    root: Path,
    name: str,
    *,
    user_lines: int = 0,
    assistant_lines: int = 0,
    title: str | None = None,
    mtime: float | None = None,
) -> Path:
    """造一个测试会话目录:可选写用户/助手消息、标题、改修改时间。"""
    d = root / name
    d.mkdir(parents=True)
    if user_lines or assistant_lines:
        manager = LogManager.load(d, create=True)
        for _ in range(user_lines):
            manager.append(Message("user", "问"))
        for _ in range(assistant_lines):
            manager.append(Message("assistant", "答"))
    if title:
        write_session_title(d, title)
    if mtime is not None:
        os.utime(d, (mtime, mtime))
    return d


def test_list_sessions_metadata_and_order(logdir: Path, tmp_path: Path) -> None:
    """列表按修改时间倒序;标题/用户消息数/子代理标记都齐全。"""
    root = logdir.parent  # conftest 的 logdir fixture 已建好 logs 根(内含一个空会话)
    _make_session(root, "aaaa1111", user_lines=2, title="旧会话", mtime=1_000_000_000)
    _make_session(root, "bbbb2222", user_lines=1, title="新会话", mtime=2_000_000_000)
    _make_session(root, "subagent-cccc", user_lines=3, mtime=3_000_000_000)

    metas = list_sessions(root)
    ids = [m.session_id for m in metas]
    assert ids.index("subagent-cccc") < ids.index("bbbb2222") < ids.index("aaaa1111")
    by_id = {m.session_id: m for m in metas}
    assert by_id["aaaa1111"].title == "旧会话"
    assert by_id["aaaa1111"].n_user == 2
    assert by_id["bbbb2222"].n_user == 1
    assert is_subagent(by_id["subagent-cccc"])
    assert not is_subagent(by_id["aaaa1111"])
    assert list_sessions(tmp_path / "不存在的根") == []


def test_find_by_prefix_three_outcomes(logdir: Path) -> None:
    """前缀查找的三种结局:唯一命中 / 多个命中 / 零命中。"""
    root = logdir.parent
    _make_session(root, "aaaa1111aaaa1111aaaa1111aaaa1111")
    _make_session(root, "aaaa2222aaaa2222aaaa2222aaaa2222")
    _make_session(root, "bbbb3333bbbb3333bbbb3333bbbb3333")

    assert len(find_by_prefix(root, "aaaa")) == 2  # 撞码:如实返回多个
    assert [m.session_id for m in find_by_prefix(root, "aaaa1")] == [
        "aaaa1111aaaa1111aaaa1111aaaa1111"
    ]
    assert find_by_prefix(root, "zzzz") == []


def test_prune_only_when_no_chat_content(logdir: Path) -> None:
    """清理护栏:用户和助手消息都为 0(只剩系统提示词)才删;
    只有助手说过话也算聊过、绝不删;--name 起过名但没聊照样删。
    """
    root = logdir.parent
    empty = _make_session(root, "empty1", title="起过名但没聊")  # 只有 meta.json
    alive = _make_session(root, "alive1", user_lines=1)
    bot_only = _make_session(root, "onlybot1", assistant_lines=1)  # 只有助手消息

    assert prune_empty_session(empty) is True
    assert not empty.exists()
    assert prune_empty_session(alive) is False
    assert alive.exists()
    assert prune_empty_session(bot_only) is False
    assert bot_only.exists()
    assert prune_empty_session(root / "不存在") is False  # 目录不在:安静返回


def test_format_session_row(tmp_path: Path) -> None:
    """/log list 与 /resume 选择器共用的一行格式。"""
    meta = SessionMeta(
        logdir=tmp_path,
        session_id="318a073d" + "0" * 24,
        title="测试日志",
        mtime=1_760_000_000.0,
        n_user=5,
    )
    assert "测试日志" in format_session_row(meta)
    assert "318a073d" in format_session_row(meta)
    assert "5问" in format_session_row(meta)
    no_title = SessionMeta(tmp_path, "x" * 32, None, 1_760_000_000.0, 1)
    assert "(未命名)" in format_session_row(no_title)
    assert "1) " in format_session_row(no_title, 1)  # 选择器带序号
