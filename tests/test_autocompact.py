"""自动压缩钩子测试 —— 对应开发方案 V5(turn.post 自动触发)。

验收点(方案 §8 V5):
- 不满足条件时不产生任何视图文件、零动作;
- 触发后生成 compacted-001,再次触发生成 compacted-002;
- 压缩失败不切换视图、不丢原始日志,冷却期内不再尝试;
- 尚有未返回工具调用(回合没收尾干净)时不压缩;
- 重复注册不会重复压缩;压缩后 token 减少、工具配对不破坏。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.hooks import HookType, get_hooks
from my_agent_alpha.hooks import autocompact as ac
from my_agent_alpha.hooks.autocompact import (
    autocompact_hook,
    register_autocompact_hook,
    reset_autocompact,
    should_auto_compact,
)
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.utils import len_tokens


@pytest.fixture(autouse=True)
def _mock_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """钩子默认拿 mock 模型(免得依赖会话装配的 ContextVar 空档)。"""
    monkeypatch.setattr(ac, "get_current_model", lambda: "mock/echo")


def _fat_history(logdir: Path) -> LogManager:
    """造一个"估算 tokens 已超预算"的会话(mock/echo 预算约 11.5 万,
    用 6 万汉字 ≈ 6 万 tokens 的老回复凑到阈值以上)。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("system", "提示词", pinned=True, hide=True))
    for i in range(30):
        manager.append(Message("user", f"问{i}"))
        manager.append(Message("assistant", "<think>" + "思考" * 2000 + f"</think>结论{i}"))
    manager.append(Message("user", "最新问题"))
    return manager


# ---- 触发条件 ----


def test_should_auto_compact_threshold() -> None:
    """达到 预算×比例 才触发(用小预算直测,不依赖大历史)。"""
    msgs = [Message("user", "一二三四五")]  # 5 tokens
    assert not should_auto_compact(msgs, budget=100)  # 5 < 80
    assert should_auto_compact(msgs, budget=6)  # 5 ≥ 4.8


def test_hook_noop_under_threshold(logdir: Path) -> None:
    """历史还很宽裕:零动作,不产生任何视图文件。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "就一句"))
    autocompact_hook(manager)
    assert manager.current_view is None
    assert not (logdir / "views").exists()


# ---- 触发:生成视图 ----


def test_hook_compacts_over_threshold(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """超阈值触发:生成 compacted-001,tokens 明显减少,原始日志不动。"""
    manager = _fat_history(logdir)
    master_bytes = (logdir / "conversation.jsonl").read_bytes()
    before = len_tokens(manager.messages)

    autocompact_hook(manager)

    assert manager.current_view == "compacted-001"
    assert len_tokens(manager.messages) < before * 0.5  # 老思考区被剥掉
    assert (logdir / "conversation.jsonl").read_bytes() == master_bytes  # 原样
    out = capsys.readouterr().out
    assert "自动压缩" in out and "compacted-001" in out


def test_hook_second_trigger_increments(logdir: Path) -> None:
    """再次触发生成 compacted-002;第一次的视图成为历史快照。"""
    manager = _fat_history(logdir)
    autocompact_hook(manager)
    assert manager.current_view == "compacted-001"
    # 视图还小,先补一段超阈值的新对话(直接追加,内容不重要)
    for i in range(25):
        manager.append(Message("user", f"补充问{i}" + "字" * 4000))
    assert len_tokens(manager.messages) > 115_200 * ac._AUTO_COMPACT_RATIO

    autocompact_hook(manager)
    assert manager.current_view == "compacted-002"
    assert (logdir / "views" / "compacted-001.jsonl").exists()


# ---- 失败:冷却与保护 ----


def test_hook_cooldown_after_failure(
    logdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """压缩失败:提示原因并冷却,期间不再尝试,冷却耗完再试。"""
    manager = _fat_history(logdir)
    attempts: list[str] = []

    def always_fail(m: object, model: str, **kwargs: object) -> str | None:
        attempts.append(model)
        raise ValueError("安检不过(测试模拟)")

    monkeypatch.setattr(ac, "compact_current_to_view", always_fail)
    reset_autocompact()
    autocompact_hook(manager)  # 第一次:尝试并失败
    assert len(attempts) == 1
    assert "自动压缩失败" in capsys.readouterr().out

    for _ in range(ac._COOLDOWN_TURNS):
        autocompact_hook(manager)  # 冷却期:安静跳过
    assert len(attempts) == 1  # 一次都没多试
    assert manager.current_view is None  # 没有误切换
    assert not (logdir / "views").exists()  # 没有产生任何视图文件

    autocompact_hook(manager)  # 冷却耗完:允许再试(仍败,进入新一轮冷却)
    assert len(attempts) == 2


def test_hook_skips_unfinished_tool_calls(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """回合没收尾干净(有调用无结果):压缩被安检拒绝,历史原样保留。

    无结果的调用放在尾部(压缩必保的区域):这样它一定会进压缩结果,
    安检才真的面对"配对不完整"的现场 —— 放在头部的话,裁剪会把它
    整组丢掉,结果反而合法,测不到拒绝路径。
    """
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("system", "提示词", pinned=True, hide=True))
    manager.append(Message("user", "多聊点" + "字" * 4000))
    for i in range(25):
        manager.append(Message("user", f"继续聊{i}" + "字" * 4000))  # 拉超阈值
    # 中断现场放在最后:assistant 发起原生调用,但结果还没回来
    manager.append(Message("assistant", '读文件:\n@read(call_1): {"path": "a"}'))

    autocompact_hook(manager)

    assert manager.current_view is None  # 没有切换
    assert len(manager.messages) > 20  # 历史一条没少
    assert not (logdir / "views").exists()
    assert "自动压缩失败" in capsys.readouterr().out


def test_double_registration_is_idempotent() -> None:
    """重复注册同名钩子 = 替换:TURN_POST 上 autocompact 只有一个。"""
    register_autocompact_hook()
    register_autocompact_hook()
    names = [h.name for h in get_hooks(HookType.TURN_POST)]
    assert names.count("autocompact") == 1


# ---- e2e:chat() 主链路自动压缩 ----


def test_chat_auto_compacts_between_turns(
    logdir: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """真实大历史(多轮大思考区)过触发线:回合结束自动生成 compacted-001。"""
    reset_autocompact()
    # 四轮都带 2.5 万 token 的思考区:总历史约 10 万 token,过触发线
    fat = [f"<think>{'思' * 25000}</think>第{i}轮结论。" for i in "一二三四"]
    set_scripted_replies(fat)

    chat(
        ["问一", "问二", "问三", "问四"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )

    restored = LogManager.load(logdir)
    assert restored.current_view == "compacted-001"
    assert "自动压缩" in capsys.readouterr().out
    # 会话本身正常收尾:完整历史四轮都在(思考区留在 Master 里,永不删)
    joined = "".join(m.content for m in restored.master_messages)
    assert all(q in joined for q in ["问一", "问二", "问三", "问四", "第一轮结论。"])
    # 视图里老的思考区被剥掉,但最新一轮原样
    assert "<think>" not in "".join(m.content for m in restored.messages[:-5])


def test_chat_continues_normally_after_auto_compact(
    logdir: Path,
    workspace: Path,
) -> None:
    """自动压缩后再来一回合:对话、工具调用流程都不受影响。"""
    reset_autocompact()
    # 四轮大思考区把历史顶过触发线,自动压缩在回合边界生效
    fat = [f"<think>{'思' * 25000}</think>第{i}轮结论。" for i in "一二三四"]
    fat += ["```shell\necho hi\n```", "收到结果。"]
    set_scripted_replies(fat)

    chat(
        ["问一", "问二", "问三", "问四"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )
    assert LogManager.load(logdir).current_view == "compacted-001"

    # 重新打开同一会话续聊:一轮带工具调用的回合,一切照常
    chat(["跑个命令"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)
    restored = LogManager.load(logdir)
    assert restored.current_view in {"compacted-001", "compacted-002"}
    contents = [m.content for m in restored.master_messages]
    assert "跑个命令" in contents  # 完整历史都在
    assert any("hi" in c for c in contents)  # 工具真的执行了
