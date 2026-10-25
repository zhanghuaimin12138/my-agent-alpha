"""空回复/漏写代码块的端到端:提醒注入、重新生成、次数上限、额度重置。

剧本全部用 set_scripted_replies 导演:模型先"光说不做",
看提醒机制是否把它拉回来。检测器本身的单元测试在
tests/unit/test_chat.py(missed_tooluse_reason)。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import NUDGE_PREFIX, chat
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.tools import add_tools
from my_agent_alpha.tools.base import ToolSpec


def _make_tool(name: str, lang: str) -> ToolSpec:
    """造一个极简的可执行工具(给测试注册用):围栏语言标记为 lang。"""

    def execute(
        content: str | None, args: list[str] | None, kwargs: dict[str, str] | None
    ) -> Message:
        """被调用时返回一条固定的结果消息(内容带原样参数,方便断言)。"""
        return Message("system", f"{name} ok:{content}")

    return ToolSpec(
        name=name, desc="测试用工具", block_types=(lang,), read_only=True, execute=execute
    )


def test_think_only_reply_nudged_and_recovers(
    logdir: Path, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """实录场景复现:回复只有思考 → 注入提醒 → 模型补发调用 → 回合走完。"""
    set_scripted_replies(
        [
            "<think>用户要我跑命令,我需要调用 shell 工具。</think>",
            "补上调用:\n```shell\necho hi\n```",
            "命令输出 hi,任务完成。",
        ]
    )
    chat(
        ["跑一下 echo"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )

    contents = [m.content for m in LogManager.load(logdir).messages]
    nudge_idx = next(i for i, c in enumerate(contents) if c.startswith(NUDGE_PREFIX))
    assert "没有正文" in contents[nudge_idx]
    # 提醒生效:之后模型真的补发了调用,工具真的执行了,回合正常收尾
    assert any("$ echo hi" in c for c in contents[nudge_idx:])
    assert any(
        m.role == "assistant" and m.content == "命令输出 hi,任务完成。"
        for m in LogManager.load(logdir).messages
    )
    # 提醒在终端可见(不静默)
    assert NUDGE_PREFIX in capsys.readouterr().out


def test_empty_reply_nudged(logdir: Path, workspace: Path) -> None:
    """模型回了空字符串:同样触发一次提醒,补发后回合继续。"""
    set_scripted_replies(["", "```shell\necho ok\n```", "完成。"])
    chat(
        ["跑一下"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )
    contents = [m.content for m in LogManager.load(logdir).messages]
    assert sum(1 for c in contents if c.startswith(NUDGE_PREFIX) and "没有正文" in c) == 1
    assert any("ok" in c for c in contents)


def test_no_nudge_on_normal_final_answer(logdir: Path, workspace: Path) -> None:
    """正常的最终回答(正文里含冒号但不以冒号结尾)不该被打扰。"""
    set_scripted_replies(["这个问题直接回答:答案是 42。"])
    chat(
        ["1+41 等于几"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )
    contents = [m.content for m in LogManager.load(logdir).messages]
    assert not any(c.startswith(NUDGE_PREFIX) for c in contents)


def test_nudge_budget_exhausted_reports_and_ends(
    logdir: Path, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """两次都只回思考:默认只提醒 1 次,之后如实报告"本回合结束"。"""
    set_scripted_replies(["<think>第一次没写。</think>", "<think>还是没写。</think>"])
    chat(
        ["搜天气"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )
    manager = LogManager.load(logdir)
    contents = [m.content for m in manager.messages]
    # 提醒恰好 1 次(默认上限),然后是"本回合结束"的如实报告
    assert sum(1 for c in contents if c.startswith(NUDGE_PREFIX) and "你上一条" in c) == 1
    assert any("本回合结束" in c for c in contents if c.startswith(NUDGE_PREFIX))
    # 两次生成都真实落进了日志
    assert sum(1 for m in manager.messages if m.role == "assistant") == 2
    assert "本回合结束" in capsys.readouterr().out


def test_nudge_budget_resets_after_execution(logdir: Path, workspace: Path) -> None:
    """复刻真机复测(2026-1005-700ec7 行 12→20):提醒恢复 → 工具真实执行 →
    又漏写,提醒额度必须重置(否则模型想重试却被"本回合结束"掐死)。"""
    set_scripted_replies(
        [
            "好的,我来搜索一下:",  # 漏写 → 提醒 #1
            "```shell\necho step1\n```",  # 补发并执行成功 → 额度重置
            "超时了,我换个方式试试:",  # 又漏写 → 提醒 #2(重置生效的证明)
            "```shell\necho step2\n```",  # 再次补发并执行
            "完成。",
        ]
    )
    chat(
        ["搜天气"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )
    contents = [m.content for m in LogManager.load(logdir).messages]
    # 两次提醒、两个工具都真的执行、回合正常收尾(没有被预算掐死)
    assert sum(1 for c in contents if c.startswith(NUDGE_PREFIX) and "你上一条" in c) == 2
    assert any("$ echo step1" in c for c in contents)
    assert any("$ echo step2" in c for c in contents)
    assert not any("本回合结束" in c for c in contents)


def test_unknown_tag_nudge_end_to_end(logdir: Path, workspace: Path) -> None:
    """围栏写岔工具名:提醒点名 websearch,模型改名后工具真的执行。"""
    add_tools((_make_tool("websearch.search", "websearch.search"),))
    set_scripted_replies(
        [
            "我来搜:\n```websearch\n{}\n```",
            "改用正确名字:\n```websearch.search\n{}\n```",
            "搜完了。",
        ]
    )
    chat(
        ["搜一下"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
    )
    contents = [m.content for m in LogManager.load(logdir).messages]
    assert any(
        c.startswith(NUDGE_PREFIX) and "websearch" in c and "不是任何已知工具" in c
        for c in contents
    )
    assert any("websearch.search ok" in c for c in contents)  # 补发后真的执行了


def test_native_channel_nudge_wording(logdir: Path, workspace: Path) -> None:
    """原生通道下空回复同样提醒,措辞引导函数调用而不是围栏。"""
    set_scripted_replies(["", "任务完成,无需工具。"])
    chat(
        ["跑一下"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
        tool_format="tool",
    )
    contents = [m.content for m in LogManager.load(logdir).messages]
    nudge = next(c for c in contents if c.startswith(NUDGE_PREFIX))
    assert "函数调用" in nudge
    assert "```" not in nudge  # 原生通道不该教围栏写法


def test_calls_inside_think_nudged_and_recovers(
    logdir: Path, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """真机翻车样本 2026-1024-e6480f 复现:调用写进思考区 → 点名提醒 → 恢复。

    泛泛的"请发起函数调用"救不了这种模型 —— 它自以为已经调用了,
    提醒必须说出病因:调用在思考区里、要写在 </think> 之后的正文里。
    """
    set_scripted_replies(
        [
            # 第一次:调用行全包在 <think> 里(实录原样裁剪)
            '<think>我需要看看环境。@shell(call_1): {"command": "echo hi"}</think>',
            # 被点名提醒后:同样的调用写在正文里(思考区外)
            '@shell(call_2): {"command": "echo hi"}',
            "命令输出 hi,任务完成。",
        ]
    )
    chat(
        ["跑一下 echo"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
        tool_format="tool",
    )
    contents = [m.content for m in LogManager.load(logdir).messages]
    nudge_idx = next(i for i, c in enumerate(contents) if c.startswith(NUDGE_PREFIX))
    nudge = contents[nudge_idx]
    assert "思考区" in nudge and "</think>" in nudge  # 点中病因 + 教正确位置
    # 提醒生效:调用挪出思考区后真的执行了,回合正常收尾
    assert any("$ echo hi" in c for c in contents[nudge_idx:])
    assert any(
        m.role == "assistant" and m.content == "命令输出 hi,任务完成。"
        for m in LogManager.load(logdir).messages
    )
