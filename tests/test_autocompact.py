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
    # 视图还小,补一段超阈值的带思考区对话(软压缩有收益,仍走机械路径)
    for i in range(25):
        manager.append(Message("user", f"补充问{i}"))
        manager.append(Message("assistant", "<think>" + "思" * 4200 + f"</think>答{i}"))
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
    for i in range(25):  # 带思考区的轮次拉超阈值,让决策走机械压缩路径
        manager.append(Message("user", f"继续聊{i}"))
        manager.append(Message("assistant", "<think>" + "思" * 4200 + f"</think>答{i}"))
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


# ---- 步骤 4:三选一决策与"请模型"通道 ----

# 一份合格的最简笔记(七节齐全,给 mock 模型当回复用)
_GOOD_NOTE = "\n".join(
    [
        "## Objective\n继续任务。",
        "## Key Decisions\n无。",
        "## Current State\nDone: 前期工作。",
        "## Open Items\n1. 继续",
        "## Critical Facts\n无。",
        "## Context Files\n- 无",
        "## Next Action\n继续。",
    ]
)


def _dense_history(logdir: Path) -> LogManager:
    """造"干货型"超阈值历史:纯用户长消息,软压缩没东西可删。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("system", "提示词", pinned=True, hide=True))
    for i in range(30):
        manager.append(Message("user", f"问题{i}" + "细节" * 2000))  # 每条约 4000
    manager.append(Message("user", "最新问题"))
    return manager


def test_decide_compaction_three_ways() -> None:
    """三选一:不超→none;有思考区可剥→trim;纯干货→summarize。"""
    budget = 115_200
    target = int(budget * 0.6)

    tiny = [Message("user", "就一句")]
    assert ac.decide_compaction(tiny, budget, target) == "none"

    # 带思考区的老回复(距结尾 ≥5 才会被剥)+ 5 条垫底消息
    with_think = [
        Message("assistant", "<think>" + "思" * 100_000 + "</think>结论"),
        *[Message("user", f"垫{i}") for i in range(5)],
    ]
    assert ac.decide_compaction(with_think, budget, target) == "trim"

    dense = [Message("user", "字" * 100_000)]  # 一条超大干货:软压缩省不动
    assert ac.decide_compaction(dense, budget, target) == "summarize"


def test_hook_summarize_disabled_by_default(
    logdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """干货历史 + 自动请模型默认关:安静跳过,不花钱不建视图。"""
    manager = _dense_history(logdir)
    reset_autocompact()
    autocompact_hook(manager)
    assert manager.current_view is None
    assert not (logdir / "views").exists()
    assert capsys.readouterr().out == ""


def test_hook_summarize_when_enabled(
    logdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """开关打开 + 模型交回合格笔记:自动生成视图并提示。"""
    from my_agent_alpha.llm.mock import set_scripted_replies

    monkeypatch.setenv("ALPHA_AUTO_LLM_COMPACT", "1")
    reset_autocompact()
    manager = _dense_history(logdir)
    set_scripted_replies([_GOOD_NOTE])

    autocompact_hook(manager)

    assert manager.current_view == "compacted-001"
    out = capsys.readouterr().out
    assert "自动压缩(模型笔记)" in out


def test_hook_llm_throttle_same_digest(
    logdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """同一段历史自动压失败过:不再重复调模型(手动命令不受此限)。"""
    from my_agent_alpha import compaction_llm as clm
    from my_agent_alpha.llm.mock import set_scripted_replies

    monkeypatch.setenv("ALPHA_AUTO_LLM_COMPACT", "1")
    reset_autocompact()
    manager = _dense_history(logdir)
    set_scripted_replies(["## Objective\n只写一节"])  # 坏笔记 → ValueError

    calls: list[object] = []
    real = clm.compact_current_with_llm

    def spy(m: object, **kw: object) -> object:
        calls.append(kw)
        return real(m, **kw)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(ac, "compact_current_with_llm", spy)

    autocompact_hook(manager)  # 第一次:真调了模型,笔记不合格
    assert len(calls) == 1
    assert "不再重试" in capsys.readouterr().out

    autocompact_hook(manager)  # 第二次:同一段历史 → 不再花钱
    autocompact_hook(manager)
    assert len(calls) == 1
    assert manager.current_view is None


def test_hook_llm_throttle_retry_after_change(
    logdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """节流只在"同一段历史"内生效:来了新消息(指纹变)就允许再试。"""
    from my_agent_alpha.llm.mock import set_scripted_replies

    monkeypatch.setenv("ALPHA_AUTO_LLM_COMPACT", "1")
    reset_autocompact()
    manager = _dense_history(logdir)
    set_scripted_replies(["## Objective\n只写一节"])  # 先失败
    autocompact_hook(manager)
    assert "不再重试" in capsys.readouterr().out
    assert manager.current_view is None

    # 历史有实质变化(新消息)→ 指纹不同 → 允许再试,这次笔记合格
    manager.append(Message("user", "新进展" + "细节" * 2000))
    set_scripted_replies([_GOOD_NOTE])
    autocompact_hook(manager)
    assert manager.current_view == "compacted-001"


def test_context_command(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """/context:无视图时报完整历史;有视图时报视图与压缩方式。"""
    from my_agent_alpha.commands import builtin as builtin_commands  # noqa: F401
    from my_agent_alpha.commands import execute_cmd

    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "你好"))
    assert execute_cmd("/context", manager, "mock/echo") is True
    out = capsys.readouterr().out
    assert "完整历史:1 条" in out
    assert "没有激活视图" in out

    # 手动做一次 LLM 压缩(mock 笔记),再看 /context
    from my_agent_alpha.compaction_llm import compact_current_with_llm
    from my_agent_alpha.llm.mock import set_scripted_replies

    for i in range(12):  # 补长历史让前缀非空(尾部预算 8000 tokens)
        manager.append(Message("user", f"问题{i}" + "细节" * 1000))
    set_scripted_replies([_GOOD_NOTE])
    assert compact_current_with_llm(manager, model="mock/echo").status == "success"

    assert execute_cmd("/context", manager, "mock/echo") is True
    out = capsys.readouterr().out
    assert "compacted-001" in out
    assert "LLM 笔记" in out
    assert "预算占用" in out
