"""压缩视图(View)存储层测试 —— 对应开发方案 V1(存储骨架)与 V2(双写与崩溃恢复)。

验收点(方案 §8 V1):
- 没有视图时,旧行为一分不变(views/ 目录都不创建);
- 创建视图后 manager.messages 读 View,master_messages 仍读 Master;
- Master 与 View 是两个 Log,持久化游标各记各的文件;
- 创建视图不改写 conversation.jsonl 的任何字节;
- active.json 记录视图名与双方计数,重启(load)自动恢复激活;
- 不创建 branches/ 目录(本方案明确不做分支)。

验收点(方案 §8 V2):
- 激活后新增一条消息,恰好同时出现在 Master 与 View 各一次;
- 重启后继续追加,两边各自走正确的增量游标;
- 崩在"Master 已写、View 没写":重启补齐 View,不丢 Master 消息;
- 崩在"两边都写完、状态没更新":恢复结果无重复消息;
- 状态损坏/文件缺失/数量对不上:安全回退 Master 并告警;
- 双写中途失败不丢旧视图、不误切换;失败后本实例冻结,禁止继续追加;
- 视图文件尾部坏行的既有容忍机制仍然成立;
- pinned/hide/quiet/call_id/metadata 双写后全部字段保持一致。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from my_agent_alpha import Log, LogManager, Message
from my_agent_alpha.logmanager import manager as manager_mod
from my_agent_alpha.logmanager.views import (
    is_valid_view_id,
    next_view_id,
    read_active_state,
    view_path,
)


def _make_manager_with_history(logdir: Path, n: int = 4) -> LogManager:
    """造一个已有 n 条消息的会话(测试小工具,内容不重要)。"""
    manager = LogManager.load(logdir, create=True)
    for i in range(n):
        manager.append(Message("user", f"消息 {i}"))
    return manager


def _rewrite_state(logdir: Path, **changes: int) -> None:
    """手工改 active.json 的计数(模拟"状态没来得及更新就崩了")。"""
    path = logdir / "views" / "active.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state.update(changes)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _drop_last_line(path: Path) -> None:
    """删掉 JSONL 文件的最后一行(模拟"这一边没写成就崩了")。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")


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


# ====================================================================
# V2:双写与崩溃恢复
# ====================================================================


def _make_view_session(logdir: Path) -> LogManager:
    """造一个"4 条历史 + 已激活视图(2 条)"的会话(计数检查点 = 4/2)。"""
    manager = _make_manager_with_history(logdir)
    manager.create_compacted_view(manager.master_messages[:2])
    assert (len(manager.master_messages), len(manager.messages)) == (4, 2)
    return manager


def test_append_writes_both_sides_once(logdir: Path) -> None:
    """激活后追加一条:恰好同时进 Master 与 View 各一次,检查点同步 +1。"""
    manager = _make_view_session(logdir)
    master_file = logdir / "conversation.jsonl"
    view_file = view_path(logdir, "compacted-001")
    master_before = master_file.read_text(encoding="utf-8").count("\n")
    view_before = view_file.read_text(encoding="utf-8").count("\n")

    manager.append(Message("user", "新消息"))

    # 磁盘上两边各多恰好一行
    assert master_file.read_text(encoding="utf-8").count("\n") == master_before + 1
    assert view_file.read_text(encoding="utf-8").count("\n") == view_before + 1
    # 内存与磁盘一致
    assert len(manager.master_messages) == 5
    assert len(manager.messages) == 3
    assert manager.master_messages[-1].content == manager.messages[-1].content == "新消息"
    # 检查点推进到 (5, 3)
    state = read_active_state(logdir)
    assert state is not None
    assert (state.master_committed_count, state.view_committed_count) == (5, 3)


def test_restart_append_is_incremental_on_both(logdir: Path) -> None:
    """重启后继续追加:两边文件都只在尾部增长(游标各自正确恢复)。"""
    _make_view_session(logdir)
    master_file = logdir / "conversation.jsonl"
    view_file = view_path(logdir, "compacted-001")
    master_before = master_file.read_bytes()
    view_before = view_file.read_bytes()

    restored = LogManager.load(logdir)
    restored.append(Message("user", "重启后的一条"))

    assert master_file.read_bytes().startswith(master_before)
    assert view_file.read_bytes().startswith(view_before)
    again = LogManager.load(logdir)
    assert len(again.master_messages) == 5
    assert [m.content for m in again.messages][-1] == "重启后的一条"


def test_dual_write_preserves_all_fields(logdir: Path) -> None:
    """带全套附属字段的消息双写后,两边磁盘行的字段一字不差。"""
    manager = _make_view_session(logdir)
    rich = Message(
        "system",
        "工具输出",
        call_id="call-9",
        pinned=True,
        hide=True,
        quiet=True,
        metadata={"model": "test", "usage": {"input": 10}},
    )
    manager.append(rich)

    for f in (logdir / "conversation.jsonl", view_path(logdir, "compacted-001")):
        data = json.loads(f.read_text(encoding="utf-8").splitlines()[-1])
        assert data["call_id"] == "call-9"
        assert data["pinned"] is True
        assert data["hide"] is True
        assert data["quiet"] is True
        assert data["metadata"] == {"model": "test", "usage": {"input": 10}}


# ---- 崩溃模拟:按检查点修复 ----


def test_crash_master_written_view_missing(logdir: Path, caplog: pytest.LogCaptureFixture) -> None:
    """崩在"Master 已写、View 还没写":重启把 Master 尾部补进 View,一条不丢。"""
    manager = _make_view_session(logdir)
    manager.append(Message("user", "第 5 条"))  # 正常双写完:磁盘 (5,3)、检查点 (5,3)
    # 造崩溃现场:View 的最后一行没写进去、检查点也没推进 → 磁盘 (5,2)、检查点 (4,2)
    _drop_last_line(view_path(logdir, "compacted-001"))
    _rewrite_state(logdir, master_committed_count=4, view_committed_count=2)

    with caplog.at_level(logging.WARNING):
        restored = LogManager.load(logdir)
    assert restored.current_view == "compacted-001"
    # 修复 = View 前 2 条(检查点确认)+ Master 第 5 条起(1 条)= 3 条
    assert len(restored.master_messages) == 5  # Master 一条没丢
    assert [m.content for m in restored.messages] == ["消息 0", "消息 1", "第 5 条"]
    # 检查点已修平
    state = read_active_state(logdir)
    assert state is not None
    assert (state.master_committed_count, state.view_committed_count) == (5, 3)
    # 修完能继续双写
    restored.append(Message("user", "修复后的新消息"))
    assert len(restored.master_messages) == 6
    assert len(restored.messages) == 4


def test_crash_both_written_state_stale(logdir: Path) -> None:
    """崩在"两边都写完、检查点没更新":修复后无重复消息。"""
    manager = _make_view_session(logdir)
    manager.append(Message("user", "第 5 条"))  # 磁盘两边都有 5/3 条
    _rewrite_state(logdir, master_committed_count=4, view_committed_count=2)  # 状态停在旧值

    restored = LogManager.load(logdir)
    assert restored.current_view == "compacted-001"
    # View 里"第 5 条"只出现一次(磁盘上已有的那条被裁掉,以 Master 为准重接)
    contents = [m.content for m in restored.messages]
    assert contents == ["消息 0", "消息 1", "第 5 条"]
    assert contents.count("第 5 条") == 1


def test_crash_view_written_master_missing(logdir: Path) -> None:
    """极端顺序错乱(Master 反而落后):修复公式依然以 Master 为准修平。"""
    manager = _make_view_session(logdir)
    manager.append(Message("user", "第 5 条"))  # 磁盘 (5,3)、检查点 (5,3)
    # 造一个"View 有、Master 没有"的现场:Master 回退一条、检查点回退一条
    _drop_last_line(logdir / "conversation.jsonl")
    _rewrite_state(logdir, master_committed_count=4, view_committed_count=3)

    restored = LogManager.load(logdir)
    # Master 4 条 ≥ 检查点 4;View 3 条 ≥ 检查点 3;修复 = View 前 3 条 + Master[4:](空)
    assert len(restored.master_messages) == 4
    assert [m.content for m in restored.messages] == ["消息 0", "消息 1", "第 5 条"]


def test_master_shorter_than_checkpoint_falls_back(
    logdir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Master 比检查点还少(被截断):不许拿视图当真,告警回退 Master。"""
    manager = _make_view_session(logdir)
    manager.append(Message("user", "第 5 条"))  # 检查点 (5,3)
    _drop_last_line(logdir / "conversation.jsonl")  # Master 只剩 4 条 < 5

    with caplog.at_level(logging.WARNING):
        restored = LogManager.load(logdir)
    assert restored.current_view is None
    assert len(restored.messages) == 4  # 读完整历史
    assert any("截断" in r.message for r in caplog.records)
    # 不删文件:现场保留
    assert view_path(logdir, "compacted-001").exists()


def test_view_shorter_than_checkpoint_falls_back(
    logdir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """View 比检查点还少(已确认的消息缺了):回退 Master,不凭空补造。"""
    manager = _make_view_session(logdir)
    manager.append(Message("user", "第 5 条"))  # 检查点 (5,3)
    _drop_last_line(view_path(logdir, "compacted-001"))  # View 只剩 2 条 < 3
    # 注意 Master 仍是 5 条、检查点还是 (5,3)

    with caplog.at_level(logging.WARNING):
        restored = LogManager.load(logdir)
    assert restored.current_view is None
    assert len(restored.messages) == 5
    assert any("缺失" in r.message for r in caplog.records)


def test_view_torn_tail_still_self_heals(logdir: Path) -> None:
    """视图文件尾部坏行:跳过容忍,继续追加时全量重写清掉坏行(既有机制)。"""
    _make_view_session(logdir)
    view_file = view_path(logdir, "compacted-001")
    with open(view_file, "ab") as fh:
        fh.write('{"role": "user", "content": "写了一半'.encode())  # 崩溃残尾

    restored = LogManager.load(logdir)  # 不抛异常,坏行被跳过
    assert len(restored.messages) == 2
    restored.append(Message("user", "坏行之后的新消息"))  # 触发视图全量重写,清掉残尾

    again = LogManager.load(logdir)  # 若没清掉,这里会撞上"中间行损坏"
    assert [m.content for m in again.messages] == ["消息 0", "消息 1", "坏行之后的新消息"]


# ---- 双写中途失败:冻结与不误切换 ----


def test_partial_write_failure_freezes_manager(
    logdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """检查点没写成:Master 已带新消息,本实例冻结禁止再追加;重新 load 可恢复。"""
    manager = _make_view_session(logdir)

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(manager_mod, "write_active_state", _boom)
    with pytest.raises(OSError, match="disk full"):
        manager.append(Message("user", "写了一半的消息"))

    # 冻结:常规追加被拒(重试会把 Master 已成功的那条再写一遍)
    with pytest.raises(RuntimeError, match="重新"):
        manager.append(Message("user", "不该被写进去"))
    with pytest.raises(RuntimeError, match="重新"):
        manager.write()
    # 压缩创建视图同样禁止:它落盘的 Master 与新视图会带着分叉状态
    with pytest.raises(RuntimeError, match="重新"):
        manager.create_compacted_view([Message("system", "摘要")])

    # 重新 load:按检查点修复(Master 第 5 条被接进 View),恢复正常
    monkeypatch.undo()
    restored = LogManager.load(logdir)
    assert restored.current_view == "compacted-001"
    assert [m.content for m in restored.messages] == ["消息 0", "消息 1", "写了一半的消息"]
    restored.append(Message("user", "恢复后的消息"))
    assert len(LogManager.load(logdir).master_messages) == 6


def test_view_file_write_failure_keeps_old_view(
    logdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """创建新视图时第 2 步(写视图文件)失败:不误切换,旧视图仍是读取来源。"""
    manager = _make_view_session(logdir)
    old_view = [m.content for m in manager.messages]

    real_write = Log.write_jsonl

    def _block_view_write(self: Log, path: Path, *, sync: bool = False) -> None:
        if path.name.startswith("compacted-"):
            raise OSError("disk full")
        real_write(self, path, sync=sync)

    monkeypatch.setattr(Log, "write_jsonl", _block_view_write)
    with pytest.raises(OSError, match="disk full"):
        manager.create_compacted_view([Message("system", "新摘要")])

    # 内存读取来源没变:还是 compacted-001 的内容
    assert manager.current_view == "compacted-001"
    assert [m.content for m in manager.messages] == old_view
    # 恢复后再创建:编号继续递增(失败的半成品文件若已写出,也不复用)
    monkeypatch.undo()
    view_id = manager.create_compacted_view([Message("system", "新摘要")])
    assert view_id == "compacted-002"
    assert [m.content for m in manager.messages] == ["新摘要"]
