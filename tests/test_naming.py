"""会话自动命名测试:标题生成、meta 读写、/log 显示、端到端钩子触发。"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.commands import CommandContext
from my_agent_alpha.commands.builtin import log_cmd
from my_agent_alpha.context import set_current_model
from my_agent_alpha.hooks.naming import (
    _clean_title,
    naming_hook,
)
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.logmanager.sessions import (
    read_session_title,
    write_session_title,
)

# ---- 单元:标题清洗与 meta 读写 ----


def test_clean_title_strips_quotes_and_lines() -> None:
    """模型给的标题:取首行、剥引号、限长。"""
    assert _clean_title("「修登录页」") == "修登录页"
    assert _clean_title("“调试打印”\n(解释:这个标题概括了……)") == "调试打印"
    assert _clean_title("  修 bug  ") == "修 bug"
    assert len(_clean_title("字" * 50)) == 30  # 超长硬截
    assert _clean_title("") == ""


def test_title_roundtrip_and_corrupt_meta(tmp_path: Path) -> None:
    """标题写入后能读回;损坏的 meta.json 读成 None 而不是崩。"""
    assert read_session_title(tmp_path) is None  # 还没写
    write_session_title(tmp_path, "修登录页")
    assert read_session_title(tmp_path) == "修登录页"
    (tmp_path / "meta.json").write_text("{坏的 json", encoding="utf-8")
    assert read_session_title(tmp_path) is None
    (tmp_path / "meta.json").write_text('{"title": "  "}', encoding="utf-8")
    assert read_session_title(tmp_path) is None  # 空白标题当没命名


def test_hook_skips_when_titled(
    logdir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """已有标题的会话(--name 指定的)钩子直接跳过,不再请求模型。"""
    monkeypatch.setenv("ALPHA_AUTO_NAME", "1")  # conftest 默认关,这里显式开
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "hi"))
    manager.append(Message("assistant", "hello"))
    write_session_title(logdir, "用户起的名字")
    set_current_model("mock/echo")
    set_scripted_replies(["模型不该被叫到"])  # 若钩子叫了模型,脚本会被消耗/回显

    assert list(naming_hook(manager)) == []
    assert "会话已命名" not in capsys.readouterr().out
    assert read_session_title(logdir) == "用户起的名字"


def test_hook_skips_when_no_assistant_yet(logdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """对话还没成型(只有 user):不命名,等下一回合。"""
    monkeypatch.setenv("ALPHA_AUTO_NAME", "1")
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "刚开的会话"))
    set_current_model("mock/echo")

    assert list(naming_hook(manager)) == []
    assert read_session_title(logdir) is None


# ---- 端到端:第一回合结束后真的命名 ----


def test_clean_title_strips_think_block() -> None:
    """模型把思考原文混在回复开头:剥掉思考区,只留真标题。"""
    assert _clean_title("<think>推理过程…</think>测试日志") == "测试日志"
    assert _clean_title("<think>推理</think>\n修标题bug") == "修标题bug"


def test_clean_title_rejects_unclosed_think() -> None:
    """思考区没闭合 = 里面全是推理、没有标题:返回空串,钩子下回合再试。"""
    assert _clean_title("<think>The user wants a title") == ""


def test_hook_writes_clean_title_despite_think_prefix(
    logdir: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """端到端:命名回复带 <think> 前缀,写进 meta 的仍是干净标题。"""
    monkeypatch.setenv("ALPHA_AUTO_NAME", "1")
    set_scripted_replies(
        [
            "ok",  # 第一回合的正常回复
            "<think>用户在测试日志,起个名</think>测试日志",  # 命名回复带思考区
        ]
    )
    chat(
        ["测试log,返回ok即可"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )
    assert read_session_title(logdir) == "测试日志"


def test_names_session_after_first_turn(
    logdir: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主剧本:对话一轮 → 钩子拿模型起名 → 写 meta、打印、不进对话日志。"""
    monkeypatch.setenv("ALPHA_AUTO_NAME", "1")
    set_scripted_replies(
        [
            # 第一回合的正常回复
            "print 没输出是因为缓冲没刷新,加 flush=True 即可。",
            "调试打印问题",  # 命名请求的回复(钩子消耗脚本的下一条)
        ]
    )
    chat(
        ["帮我看看为什么 print 没输出"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )

    assert read_session_title(logdir) == "调试打印问题"
    out = capsys.readouterr().out
    assert "(会话已命名:调试打印问题)" in out
    # 命名是元数据:模型的命名回复不进对话日志
    manager = LogManager.load(logdir)
    assert sum(1 for m in manager.messages if m.role == "assistant") == 1


# ---- /log 显示 ----


def test_log_shows_title(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """/log 显示当前会话的标题。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "hi"))
    write_session_title(logdir, "测试会话")

    log_cmd(CommandContext(args=[], full="", manager=manager, model="mock/echo"))

    out = capsys.readouterr().out
    assert "会话标题:测试会话" in out
    assert "消息条数:1" in out


def test_log_list_shows_all_sessions(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """/log list:同级的全部会话目录,带标题的显示标题,没有的显示未命名。"""
    logs_root = logdir.parent
    write_session_title(logdir, "当前会话")
    other = logs_root / "2026-1006-abcdef"
    other.mkdir()

    manager = LogManager.load(logdir, create=True)
    # 列表只收"有人聊过"的会话:两边都补一条用户消息
    manager.append(Message("user", "hi"))
    LogManager.load(other, create=True).append(Message("user", "hi"))
    log_cmd(CommandContext(args=["list"], full="list", manager=manager, model="mock/echo"))

    out = capsys.readouterr().out
    # 列表只显示 uuid 前 8 位,认会话靠标题
    assert logdir.name[:8] in out and "当前会话" in out
    assert "2026-100" in out and "(未命名)" in out
