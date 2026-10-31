"""压缩视图(View)存储层测试 —— 对应开发方案 V1:单 View 存储骨架。

验收点(方案 §8 V1):
- 没有视图时,旧行为一分不变(views/ 目录都不创建);
- 创建视图后 manager.messages 读 View,master_messages 仍读 Master;
- Master 与 View 是两个 Log,持久化游标各记各的文件;
- 创建视图不改写 conversation.jsonl 的任何字节;
- active.json 记录视图名与双方计数,重启(load)自动恢复激活;
- 不创建 branches/ 目录(本方案明确不做分支)。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.logmanager.views import (
    is_valid_view_id,
    next_view_id,
    read_active_state,
)


def _make_manager_with_history(logdir: Path, n: int = 4) -> LogManager:
    """造一个已有 n 条消息的会话(测试小工具,内容不重要)。"""
    manager = LogManager.load(logdir, create=True)
    for i in range(n):
        manager.append(Message("user", f"消息 {i}"))
    return manager


# ---- 没有视图:旧行为不变 ----


def test_no_view_reads_master(logdir: Path) -> None:
    """没有视图时:messages/log/len 全部走 Master,也不创建 views/ 目录。"""
    manager = _make_manager_with_history(logdir)
    assert manager.current_view is None
    assert manager.log is manager.master_log
    assert len(manager.messages) == 4
    assert [m.content for m in manager.messages] == [m.content for m in manager.master_messages]
    assert not (logdir / "views").exists()


# ---- 创建视图:读取口切换 ----


def test_create_view_switches_messages(logdir: Path) -> None:
    """创建视图后:messages 读 View(压缩结果),master_messages 仍是完整历史。"""
    manager = _make_manager_with_history(logdir)
    compacted = [Message("system", "之前聊过的摘要", pinned=True), manager.master_messages[-1]]
    view_id = manager.create_compacted_view(compacted)

    assert view_id == "compacted-001"
    assert manager.current_view == "compacted-001"
    assert [m.content for m in manager.messages] == ["之前聊过的摘要", "消息 3"]
    assert len(manager.master_messages) == 4  # 完整历史不受影响
    # len/iter 同样跟随当前有效历史
    assert len(manager) == 2
    assert [m.role for m in manager] == ["system", "user"]


def test_create_view_keeps_master_bytes(logdir: Path) -> None:
    """创建视图不许碰 conversation.jsonl 的原有字节(Master 只追加,压缩不改写)。"""
    manager = _make_manager_with_history(logdir)
    before = (logdir / "conversation.jsonl").read_bytes()
    manager.create_compacted_view([Message("system", "摘要")])
    assert (logdir / "conversation.jsonl").read_bytes() == before


def test_cursors_are_independent(logdir: Path) -> None:
    """Master 与 View 两个 Log 的追加游标各记各的文件,互不认账。"""
    manager = _make_manager_with_history(logdir)
    manager.create_compacted_view([Message("system", "摘要")])
    master, view = manager.master_log, manager.log
    assert master is not view

    master_file = logdir / "conversation.jsonl"
    view_file = logdir / "views" / "compacted-001.jsonl"
    # 各自的游标指向各自的文件,且各自都能继续走增量追加
    assert master._persisted_path == master_file
    assert view._persisted_path == view_file
    assert master._can_append(master_file)
    assert view._can_append(view_file)
    # 拿 A 的游标去 B 的文件上追加必须失败(路径对不上就降级重写,防止写串)
    assert not master._can_append(view_file)
    assert not view._can_append(master_file)


def test_empty_view_rejected(logdir: Path) -> None:
    """空消息列表不许建视图:那等于把历史清零。"""
    manager = _make_manager_with_history(logdir)
    with pytest.raises(ValueError, match="空视图"):
        manager.create_compacted_view([])
    # 失败后状态原样:没有视图、没有 views/ 目录
    assert manager.current_view is None
    assert not (logdir / "views").exists()


def test_view_ids_increment(logdir: Path) -> None:
    """第二次压缩生成 compacted-002;compacted-001 原样保留成历史快照。"""
    manager = _make_manager_with_history(logdir)
    first = manager.create_compacted_view([Message("system", "第一次摘要")])
    second = manager.create_compacted_view([Message("system", "第二次摘要")])

    assert (first, second) == ("compacted-001", "compacted-002")
    assert manager.current_view == second
    # 旧视图文件还在(历史快照),但不再是读取来源
    assert (logdir / "views" / "compacted-001.jsonl").exists()
    assert [m.content for m in manager.messages] == ["第二次摘要"]


# ---- active.json:状态文件与重启恢复 ----


def test_state_records_counts(logdir: Path) -> None:
    """active.json 记录双方计数:Master 条数与 View 条数(压缩后本来就不等)。"""
    manager = _make_manager_with_history(logdir, n=6)
    manager.create_compacted_view(manager.master_messages[:2])

    state = json.loads((logdir / "views" / "active.json").read_text(encoding="utf-8"))
    assert state == {
        "version": 1,
        "active_view": "compacted-001",
        "master_committed_count": 6,
        "view_committed_count": 2,
    }


def test_reload_restores_active_view(logdir: Path) -> None:
    """重启(load)自动恢复:current_view 与 messages 都指向 compacted-001。"""
    manager = _make_manager_with_history(logdir)
    manager.create_compacted_view([Message("system", "摘要", pinned=True)])

    restored = LogManager.load(logdir)
    assert restored.current_view == "compacted-001"
    assert [m.content for m in restored.messages] == ["摘要"]
    assert len(restored.master_messages) == 4


def test_broken_state_falls_back_to_master(logdir: Path, caplog: pytest.LogCaptureFixture) -> None:
    """active.json 内容损坏:load 打警告、回退读 Master,不删任何文件。"""
    manager = _make_manager_with_history(logdir)
    manager.create_compacted_view([Message("system", "摘要")])
    (logdir / "views" / "active.json").write_text("{broken", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        restored = LogManager.load(logdir)
    assert restored.current_view is None
    assert len(restored.messages) == 4  # 回退后能读到完整历史
    assert any("视图状态" in r.message for r in caplog.records)
    # 状态文件不被顺手抹掉(坏文件也是现场证据)
    assert (logdir / "views" / "active.json").exists()


def test_missing_view_file_falls_back_to_master(
    logdir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """active.json 指向的视图文件被删了:告警回退 Master。"""
    manager = _make_manager_with_history(logdir)
    manager.create_compacted_view([Message("system", "摘要")])
    (logdir / "views" / "compacted-001.jsonl").unlink()

    with caplog.at_level(logging.WARNING):
        restored = LogManager.load(logdir)
    assert restored.current_view is None
    assert len(restored.messages) == 4


# ---- switch_to_master ----


def test_switch_to_master(logdir: Path) -> None:
    """switch_to_master:读取口回 Master,active.json 删除,视图文件保留。"""
    manager = _make_manager_with_history(logdir)
    manager.create_compacted_view([Message("system", "摘要")])
    manager.switch_to_master()

    assert manager.current_view is None
    assert len(manager.messages) == 4  # 又能读到完整历史
    assert not (logdir / "views" / "active.json").exists()  # 指针删掉
    assert (logdir / "views" / "compacted-001.jsonl").exists()  # 快照保留

    # 停用后再重启:没有激活视图,走 Master
    restored = LogManager.load(logdir)
    assert restored.current_view is None
    assert len(restored.messages) == 4


# ---- views.py 单元:编号与校验 ----


def test_view_id_validation() -> None:
    """视图 ID 只认 compacted-加数字:路径穿越、乱名、过短编号一律拒绝。"""
    assert is_valid_view_id("compacted-001")
    assert is_valid_view_id("compacted-1234")  # 超过 999 自然变 4 位,仍合法
    for bad in [
        "compacted-1",  # 不足 3 位
        "compacted-",
        "compacted-01a",
        "../conversation",
        "views/compacted-001",
        "conversation",
        "",
    ]:
        assert not is_valid_view_id(bad), bad


def test_next_view_id_empty_dir(logdir: Path) -> None:
    """views/ 不存在时从 compacted-001 起步。"""
    assert next_view_id(logdir) == "compacted-001"


def test_next_view_id_scans_existing(logdir: Path) -> None:
    """编号取现有最大 +1:接得上乱序文件,无关文件不影响。"""
    d = logdir / "views"
    d.mkdir()
    (d / "compacted-002.jsonl").write_text("", encoding="utf-8")
    (d / "compacted-007.jsonl").write_text("", encoding="utf-8")
    (d / "note.txt").write_text("", encoding="utf-8")
    (d / "active.json").write_text("{}", encoding="utf-8")
    assert next_view_id(logdir) == "compacted-008"


def test_read_state_rejects_bad_view_id(logdir: Path) -> None:
    """状态里混进非法视图名(如路径穿越)→ 整个状态作废,返回 None。"""
    d = logdir / "views"
    d.mkdir()
    (d / "active.json").write_text(
        json.dumps(
            {
                "version": 1,
                "active_view": "../conversation",
                "master_committed_count": 1,
                "view_committed_count": 1,
            }
        ),
        encoding="utf-8",
    )
    assert read_active_state(logdir) is None


def test_read_state_rejects_unknown_version(logdir: Path) -> None:
    """版本号不认识 → 状态作废(将来升级格式时,旧代码安全回退而不是猜)。"""
    d = logdir / "views"
    d.mkdir()
    (d / "active.json").write_text(
        json.dumps(
            {
                "version": 99,
                "active_view": "compacted-001",
                "master_committed_count": 1,
                "view_committed_count": 1,
            }
        ),
        encoding="utf-8",
    )
    assert read_active_state(logdir) is None


def test_no_branches_dir(logdir: Path) -> None:
    """方案明确不做分支:会话目录里不出现 branches/。"""
    manager = _make_manager_with_history(logdir)
    manager.create_compacted_view([Message("system", "摘要")])
    assert not (logdir / "branches").exists()
    assert not hasattr(manager, "_branches")
    assert not hasattr(manager, "current_branch")
