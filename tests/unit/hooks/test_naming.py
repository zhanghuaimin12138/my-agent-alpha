"""hooks/naming.py 单元测试:标题清洗、meta 读写、钩子跳过分支、/log 显示。

第一回合结束后真的命名的端到端剧本在 tests/integration/test_naming.py。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.commands import CommandContext
from my_agent_alpha.commands.builtin import log_cmd
from my_agent_alpha.context import set_current_model
from my_agent_alpha.hooks.naming import (
    _clean_title,
    naming_hook,
    read_session_title,
    write_session_title,
)
from my_agent_alpha.llm.mock import set_scripted_replies

# ---- 单元:标题清洗与 meta 读写 ----


def test_clean_title_strips_quotes_and_lines() -> None:
    """模型给的标题:取首行、剥引号、限长。"""
    assert _clean_title("「修登录页」") == "修登录页"
    assert _clean_title("“调试打印”\n(解释:这个标题概括了……)") == "调试打印"
    assert _clean_title("  修 bug  ") == "修 bug"
    assert len(_clean_title("字" * 50)) == 30  # 超长硬截
    assert _clean_title("") == ""


def test_clean_title_strips_think() -> None:
    """思考型模型的输出先思考再给标题:思考区不能进 meta.json。

    真机翻车样本 2026-1024-e6480f:模型在思考里复述命名指令,
    原样截 30 字存成了标题,真标题被丢掉。
    """
    # 先思考、换行、再标题:只留标题
    assert _clean_title("<think>用户要求给对话起标题</think>\n制定agent开发计划") == (
        "制定agent开发计划"
    )
    # 标题紧跟 </think> 同一行
    assert _clean_title("<think>想想…</think>修登录页") == "修登录页"
    # 只有思考、或思考没写完:剥完是空,交给上层"下回合再试"
    assert _clean_title("<think>用户要求给对话起标题") == ""


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


def test_hook_names_think_style_reply(
    logdir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """端到端:思考型回复(先 <think> 再标题)也能存进干净标题。"""
    monkeypatch.setenv("ALPHA_AUTO_NAME", "1")
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "制定开发计划"))
    manager.append(Message("assistant", "好的,计划如下……"))
    set_current_model("mock/echo")
    set_scripted_replies(["<think>用户要求起标题</think>\n制定agent开发计划"])

    assert list(naming_hook(manager)) == []
    assert read_session_title(logdir) == "制定agent开发计划"
    assert "会话已命名:制定agent开发计划" in capsys.readouterr().out


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
    log_cmd(CommandContext(args=["list"], full="list", manager=manager, model="mock/echo"))

    out = capsys.readouterr().out
    assert logdir.name in out and "当前会话" in out
    assert "2026-1006-abcdef" in out and "(未命名)" in out
