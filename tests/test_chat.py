"""端到端循环测试:mock 脚本驱动完整的多步工具调用循环(离线)。"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import has_runnable_tooluse
from my_agent_alpha.llm.mock import set_scripted_replies


def test_has_runnable_tooluse(workspace: Path) -> None:
    msgs = [
        Message("system", "提示词"),
        Message("user", "跑一下"),
        Message("assistant", "执行:\n```shell\nls\n```"),
        Message("system", "$ ls\na\nReturnCode: 0"),
    ]
    assert has_runnable_tooluse(msgs) is True  # 最后的 assistant 还有工具调用
    msgs.append(Message("assistant", "完成了"))
    assert has_runnable_tooluse(msgs) is False


def test_full_turn_with_tool_call(logdir: Path, workspace: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    from my_agent_alpha.chat import chat

    set_scripted_replies(
        [
            "我来执行命令:\n```shell\necho hello-spine\n```",
            "命令输出 hello-spine,任务完成。",
        ]
    )
    chat(
        ["跑一下 echo"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )

    manager = LogManager.load(logdir)
    roles = [m.role for m in manager.messages]
    # 系统提示词 ×2、user、assistant(含调用)、system(工具输出)、assistant(收尾)
    assert roles == ["system", "system", "user", "assistant", "system", "assistant"]
    contents = [m.content for m in manager.messages]
    assert "```shell\necho hello-spine\n```" in contents[3]
    assert "hello-spine" in contents[4]
    assert "任务完成" in contents[5]
    # 系统提示词消息 pinned + hide
    assert manager.messages[0].pinned and manager.messages[0].hide
    # 终端确实渲染了工具输出(审查 #13:工具输出行带 $ 前缀,不再用会撞车的子串断言)
    out = capsys.readouterr().out
    assert "$ echo hello-spine" in out
    assert "ReturnCode: 0" in out


def test_save_and_read_roundtrip(logdir: Path, workspace: Path) -> None:
    from my_agent_alpha.chat import chat

    set_scripted_replies(
        [
            "先保存文件:\n```save hello.txt\n你好,脊柱\n```",
            "再读回来验证:\n```read\nhello.txt\n```",
            "验证完毕,内容一致。",
        ]
    )
    chat(
        ["保存并验证"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )

    # 文件真的落盘了
    assert (workspace / "hello.txt").read_text(encoding="utf-8") == "你好,脊柱"
    # 循环走了 3 步(两次工具调用)
    manager = LogManager.load(logdir)
    assistant_msgs = [m for m in manager.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 3
    tool_outputs = [m for m in manager.messages if m.role == "system" and not m.hide]
    assert any("已创建" in m.content for m in tool_outputs)
    assert any("你好,脊柱" in m.content for m in tool_outputs)


def test_max_steps_guard(logdir: Path, workspace: Path) -> None:
    """脚本无限调用工具时,步数上限必须兜底。"""
    from my_agent_alpha import chat as chat_mod

    set_scripted_replies([f"第 {i} 步:\n```shell\necho {i}\n```" for i in range(100)])
    original = chat_mod.MAX_STEPS
    chat_mod.MAX_STEPS = 3
    try:
        chat_mod.chat(
            ["无限循环"],
            logdir=logdir,
            workspace=workspace,
            model="mock/echo",
            non_interactive=True,
        )
    finally:
        chat_mod.MAX_STEPS = original

    manager = LogManager.load(logdir)
    assert any("最大步数" in m.content for m in manager.messages)
    assistant_count = sum(1 for m in manager.messages if m.role == "assistant")
    assert assistant_count == 3


def test_exit_command(logdir: Path, workspace: Path) -> None:
    import pytest

    from my_agent_alpha.chat import chat

    with pytest.raises(SystemExit):
        chat(
            ["/exit"],
            logdir=logdir,
            workspace=workspace,
            model="mock/echo",
            non_interactive=True,
        )
    # /exit 不产生用户消息
    manager = LogManager.load(logdir)
    assert not any(m.role == "user" for m in manager.messages)


def test_unknown_slash_treated_as_text(logdir: Path, workspace: Path) -> None:
    from my_agent_alpha.chat import chat

    set_scripted_replies(["收到"])
    chat(
        ["/unknown cmd"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )
    manager = LogManager.load(logdir)
    assert any(m.role == "user" and m.content == "/unknown cmd" for m in manager.messages)


def test_resume_session_reuses_log(logdir: Path, workspace: Path) -> None:
    from my_agent_alpha.chat import chat

    set_scripted_replies(["第一轮完成"])
    chat(["第一轮"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)

    set_scripted_replies(["第二轮完成"])
    chat(["第二轮"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)

    manager = LogManager.load(logdir)
    user_msgs = [m.content for m in manager.messages if m.role == "user"]
    assert user_msgs == ["第一轮", "第二轮"]
    # 系统提示词只种了一次
    system_prompt_count = sum(1 for m in manager.messages if m.hide and m.role == "system")
    assert system_prompt_count == 2  # static + dynamic 两条
