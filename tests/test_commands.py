"""阶段 5 命令框架测试:注册/别名/执行/未知命令回退 + 内置命令。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.commands import (
    command,
    execute_cmd,
    get_commands,
    unregister_command,
)
from my_agent_alpha.hooks import get_hooks  # /hooks 命令用
from my_agent_alpha.llm.mock import set_scripted_replies


def _manager(logdir: Path) -> LogManager:
    return LogManager.load(logdir, create=True)


# ---- 框架机制 ----


def test_execute_cmd_dispatch_and_args(logdir: Path) -> None:
    got: list[object] = []

    @command("probe")
    def probe(ctx) -> None:  # type: ignore[no-untyped-def]
        got.append((ctx.args, ctx.full))

    try:
        assert execute_cmd("/probe a b", _manager(logdir), "mock/echo") is True
        assert got == [(["a", "b"], "a b")]
    finally:
        unregister_command("probe")


def test_alias_works(logdir: Path) -> None:
    got: list[str] = []

    @command("salute", "hi", "你好")
    def salute(ctx) -> None:  # type: ignore[no-untyped-def]
        got.append(ctx.model)

    try:
        assert execute_cmd("/hi", _manager(logdir), "glm/glm-5.3") is True
        assert got == ["glm/glm-5.3"]
        assert execute_cmd("/你好", _manager(logdir), "glm/glm-5.3") is True
    finally:
        unregister_command("salute")


def test_unknown_or_plain_returns_false(logdir: Path) -> None:
    m = _manager(logdir)
    assert execute_cmd("普通消息", m, "mock/echo") is False  # 不带斜杠
    assert execute_cmd("/不存在", m, "mock/echo") is False  # 没注册的命令


def test_unregister_removes_aliases(logdir: Path) -> None:
    @command("temp", "t1", "t2")
    def temp(ctx) -> None:  # type: ignore[no-untyped-def]
        pass

    unregister_command("temp")
    m = _manager(logdir)
    assert execute_cmd("/temp", m, "mock/echo") is False
    assert execute_cmd("/t1", m, "mock/echo") is False


# ---- 内置命令 ----


def test_builtin_help(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert execute_cmd("/help", _manager(logdir), "mock/echo") is True
    out = capsys.readouterr().out
    assert "/tools" in out


def test_builtin_tools(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert execute_cmd("/tools", _manager(logdir), "mock/echo") is True
    out = capsys.readouterr().out
    assert "shell" in out and "save" in out and "read" in out


def test_builtin_log_and_model(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    m = _manager(logdir)
    m.append(Message("user", "hi"))
    assert execute_cmd("/log", m, "mock/echo") is True
    assert execute_cmd("/model", m, "glm/glm-5.3") is True
    out = capsys.readouterr().out
    assert "消息条数:1" in out  # 种子消息 0 + 用户 1
    assert "glm/glm-5.3" in out


def test_builtin_log_list_shows_title_and_mtime(
    logdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """/log list:按修改时间倒序(新→旧),显示标题 + uuid 前 8 位。"""
    from my_agent_alpha.logmanager.sessions import write_session_title

    root = logdir.parent
    older = root / "aaaa1111aaaa1111aaaa1111aaaa1111"
    newer = root / "bbbb2222bbbb2222bbbb2222bbbb2222"
    older.mkdir()
    newer.mkdir()
    write_session_title(older, "旧会话")
    write_session_title(newer, "新会话")
    # 列表只收"有人聊过"的会话:各补一条用户消息
    LogManager.load(older, create=True).append(Message("user", "hi"))
    LogManager.load(newer, create=True).append(Message("user", "hi"))
    os.utime(older, (1_000_000_000, 1_000_000_000))  # 2001 年
    os.utime(newer, (2_000_000_000, 2_000_000_000))  # 2033 年
    assert execute_cmd("/log list", _manager(logdir), "mock/echo") is True
    out = capsys.readouterr().out
    assert out.index("新会话") < out.index("旧会话")  # 新的排前面
    assert "bbbb2222" in out  # uuid 前 8 位


def test_log_list_filters_subagent_and_empty_sessions(
    logdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """/log list 过滤:子代理会话和没人聊过的会话不出现。"""
    root = logdir.parent
    sub = root / "subagent-aaaa1111aaaa1111aaaa1111aa"
    sub.mkdir()
    LogManager.load(sub, create=True).append(Message("user", "子代理的活"))
    empty = root / "cccc3333cccc3333cccc3333cccc3333"
    empty.mkdir()
    real = root / "bbbb2222bbbb2222bbbb2222bbbb2222"
    real.mkdir()
    LogManager.load(real, create=True).append(Message("user", "真会话"))

    assert execute_cmd("/log list", _manager(logdir), "mock/echo") is True
    out = capsys.readouterr().out
    assert "bbbb2222" in out
    assert "subagent" not in out and "aaaa1111" not in out
    assert "cccc3333" not in out


def test_resume_prefix_paths(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """/resume <码> 的判定树:找不到 / 子代理拒绝 / 撞码 / 精确命中抛切换请求。"""
    from my_agent_alpha.commands import ResumeRequested

    root = logdir.parent
    real1 = root / "aaaa1111aaaa1111aaaa1111aaaa1111"
    real2 = root / "aaaa2222aaaa2222aaaa2222aaaa2222"
    sub = root / "subagent-aaaa3333aaaa3333aaaa3333aa"
    for d in (real1, real2, sub):
        d.mkdir()
        LogManager.load(d, create=True).append(Message("user", "hi"))
    m = _manager(logdir)

    assert execute_cmd("/resume zzzz", m, "mock/echo") is True
    assert "没找到以 zzzz 开头" in capsys.readouterr().out

    assert execute_cmd("/resume subagent", m, "mock/echo") is True
    assert "子代理" in capsys.readouterr().out

    assert execute_cmd("/resume aaaa", m, "mock/echo") is True
    assert "多个会话以此开头" in capsys.readouterr().out

    with pytest.raises(ResumeRequested) as ei:
        execute_cmd("/resume aaaa1", m, "mock/echo")
    assert ei.value.meta.logdir == real1

    # 指向当前会话:提示后安静返回,不抛切换请求
    assert execute_cmd(f"/resume {logdir.name[:8]}", m, "mock/echo") is True
    assert "这就是当前会话" in capsys.readouterr().out


def test_resume_picker_selects_by_number(
    logdir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """/resume 不带码:编号列表 + 输序号选择(回车/无效序号取消)。"""
    from my_agent_alpha.commands import ResumeRequested

    root = logdir.parent
    real = root / "aaaa1111aaaa1111aaaa1111aaaa1111"
    real.mkdir()
    LogManager.load(real, create=True).append(Message("user", "hi"))
    m = _manager(logdir)

    monkeypatch.setattr("builtins.input", lambda _: "1")
    with pytest.raises(ResumeRequested) as ei:
        execute_cmd("/resume", m, "mock/echo")
    assert ei.value.meta.logdir == real
    assert "1)" in capsys.readouterr().out

    monkeypatch.setattr("builtins.input", lambda _: "99")
    assert execute_cmd("/resume", m, "mock/echo") is True  # 无效序号:不抛,取消
    assert "无效序号" in capsys.readouterr().out


def test_resume_end_to_end_switch(
    logdir: Path, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """集成:会话 A 聊一轮 → 会话 B 里 /resume 回 A → 后续消息写进 A,B 被清理。"""
    set_scripted_replies(["回复A"])
    chat(["问题A"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)
    assert "问题A" in [m.content for m in LogManager.load(logdir)]

    set_scripted_replies(["回复B"])
    logdir_b = logdir.parent / "bbbb2222bbbb2222bbbb2222bbbb2222"
    chat(
        [f"/resume {logdir.name[:8]}", "问题2"],
        logdir=logdir_b,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )

    contents = [m.content for m in LogManager.load(logdir)]
    assert "问题2" in contents and "回复B" in contents  # 新消息写进了旧会话 A
    assert not logdir_b.exists()  # B 没人聊过,切换时被清理
    out = capsys.readouterr().out
    assert "已切换" in out and "问题A" in out  # 切换横幅 + 历史回放


def test_empty_session_pruned_on_exit(logdir: Path, workspace: Path) -> None:
    """开了就退出(没有任何用户消息)的会话,退出时整个目录清掉。"""
    chat([], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)
    assert not logdir.exists()


def test_builtin_hooks(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert execute_cmd("/hooks", _manager(logdir), "mock/echo") is True
    out = capsys.readouterr().out
    # chat() 注册过内置钩子?本测试没跑 chat,直接看注册表:
    if get_hooks():
        assert "generation.post" in out
    else:
        assert "没有已注册的钩子" in out


def test_e2e_exit_command_via_chat(logdir: Path, workspace: Path) -> None:
    """chat() 装配了内置命令:/exit 结束会话(更新自旧版 test_exit_command)。"""
    with pytest.raises(SystemExit):
        chat(
            ["/exit"],
            logdir=logdir,
            workspace=workspace,
            model="mock/echo",
            non_interactive=True,
        )
    # 只输过命令 = 没人聊过:整个会话目录被清理(命令不进日志,日志里也没别的)
    assert not logdir.exists()


def test_e2e_unknown_command_treated_as_text(logdir: Path, workspace: Path) -> None:
    set_scripted_replies(["收到"])
    chat(
        ["/神秘命令"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )
    manager = LogManager.load(logdir)
    assert any(m.role == "user" and m.content == "/神秘命令" for m in manager.messages)


def test_get_commands_lists_builtins() -> None:
    names = get_commands()
    for expected in ("help", "exit", "quit", "q", "tools", "log", "model", "hooks", "?"):
        assert expected in names, expected
