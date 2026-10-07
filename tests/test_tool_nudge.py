"""空回复/漏写代码块的兜底测试:该调工具却没写成时,提醒模型补发。

背景(docs/踩坑记录.md 第 3 条的遗留、docs/tmp.jsonl 实录):真实模型
连续几轮只在 <think> 里说"我来调用工具",正文里一个围栏都没有,回合
静默结束,用户只看到空气。这组用例锁住三件事:

- 检测器认得三种"没写成"的形态(正文为空 / 工具名写岔 / 冒号结尾),
  且不打扰正常回答;
- 检测命中后真的注入提醒消息并重新生成(模型补上调用,回合继续走完);
- 提醒有次数上限,超限如实报告并结束回合,全程不静默。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import NUDGE_PREFIX, missed_tooluse_reason
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


# ---- 检测器单元测试 ----


def test_reason_empty_body_variants() -> None:
    """正文为空的几种写法(闭合思考/未闭合思考/全空/纯空白)都判"没有正文"。"""
    for content in ("<think>我来调用工具。</think>", "<think>没写完", "", "  \n  "):
        reason = missed_tooluse_reason([Message("assistant", content)])
        assert reason is not None, content
        assert "没有正文" in reason


def test_reason_normal_replies_pass() -> None:
    """正常最终回答、可执行调用、以及和工具无关的示例代码块,都不触发。"""
    assert missed_tooluse_reason([Message("assistant", "任务完成,无需工具。")]) is None
    assert missed_tooluse_reason([Message("assistant", "查:\n```shell\nls\n```")]) is None
    # 用户要示例代码,模型给 ```python 块:和任何工具名都不沾边,不打扰
    assert missed_tooluse_reason([Message("assistant", "示例:\n```python\nprint(1)\n```")]) is None


def test_reason_colon_ending() -> None:
    """以冒号结尾的半句话 = 疑似"要说接下来执行什么",要点名提醒。

    全角冒号"："也要认:中文模型几乎都用全角,真实翻车样本
    (docs/tmp.jsonl 第 6 条"现在为您查询:")就是全角 —— 开发时
    先写成了两个 ASCII 冒号导致漏检,这个用例把它锁死。
    """
    for content in ("好的,现在为您查询:", "好的,现在为您查询："):
        reason = missed_tooluse_reason([Message("assistant", content)])
        assert reason is not None, content
        assert "冒号" in reason


def test_reason_unknown_tool_tag() -> None:
    """围栏标记写岔(工具叫 websearch.search、写成 websearch)要被点名。"""
    add_tools((_make_tool("websearch.search", "websearch.search"),))
    reason = missed_tooluse_reason([Message("assistant", "```websearch\n{}\n```")])
    assert reason is not None and "websearch" in reason


def test_reason_announcement_without_colon() -> None:
    """真机实录形态(会话 2026-1005-700ec7 行 9):说要干活、但不带冒号。

    "我来帮你查一下上海今天的天气。"—— 之前检测器放行,回合静默结束,
    用户只能重问;现在必须命中"行动宣告"。
    """
    content = "<think>用户要查天气,我用 websearch 搜一下。</think>我来帮你查一下上海今天的天气。"
    reason = missed_tooluse_reason([Message("assistant", content)])
    assert reason is not None and "宣布要执行操作" in reason


def test_reason_advice_or_demo_not_announcement() -> None:
    """建议类回答、能力介绍、带示例代码块的演示,都不算行动宣告,不打扰。"""
    # "试试"刻意不在标记表里:建议句里太常见
    assert missed_tooluse_reason([Message("assistant", "建议重启服务,你可以自己试试。")]) is None
    # 真机误报原文(2026-1005-700ec7 行 3/6,曾因标记表收了"帮你"而误伤):
    # "帮你"是能力介绍/主动提议,不是"我+动作"的宣告
    assert missed_tooluse_reason([Message("assistant", "我可以帮你做这些事:\n- 读文件")]) is None
    assert missed_tooluse_reason([Message("assistant", "需要我帮你在系统里找找吗?")]) is None
    # 正文里有围栏(哪怕是纯示例)= 有"内容交付",不走宣告分支
    assert (
        missed_tooluse_reason([Message("assistant", "我来演示一下:\n```python\nprint(1)\n```")])
        is None
    )


def test_reason_conditional_offer_not_announcement() -> None:
    """条件句里的"让我"不是宣告(真机 2026-1007-2da08f 完整回答被误提醒,
    模型把整段回答原样重复了一遍)。"""
    # 真机误报句式:"想让我"是假设,不是"我现在就去干"
    assert (
        missed_tooluse_reason(
            [Message("assistant", "如果你想让我记住更多(比如称呼、技术栈偏好),告诉我即可。")]
        )
        is None
    )
    assert missed_tooluse_reason([Message("assistant", "需要的话可以让我更新这条记忆。")]) is None
    # 句首式宣告仍然必须命中(回归保护)
    assert missed_tooluse_reason([Message("assistant", "好的,我来处理这件事。")]) is not None
    assert missed_tooluse_reason([Message("assistant", "让我先看看这个文件的内容。")]) is not None


def test_reason_unparsed_native_call_line() -> None:
    """回复里有调用行、严格解析却失败(编号字符怪异):提醒而不是装没看见。

    这是原生通道的兜底:修好点号问题后,再遇到服务商给出更怪的编号格式,
    至少会被点名提醒,不再双重静默(真机实录 2026-1005-332a08)。
    """
    reason = missed_tooluse_reason(
        [Message("assistant", '正在查询:\n@shell(call/1): {"command": "ls"}')]
    )
    assert reason is not None and "没有被解析出来" in reason


def test_reason_fenced_native_example_not_flagged() -> None:
    """围栏里的 @xxx(...) 是文档示例,不该触发"未解析调用行"。"""
    reason = missed_tooluse_reason(
        [Message("assistant", '调用格式如下:\n```\n@save(call_x): {"path": "a"}\n```')]
    )
    assert reason is None


# ---- 端到端:mock 脚本驱动完整回合 ----


def test_think_only_reply_nudged_and_recovers(
    logdir: Path, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """实录场景复现:回复只有思考 → 注入提醒 → 模型补发调用 → 回合走完。"""
    from my_agent_alpha.chat import chat

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
    from my_agent_alpha.chat import chat

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
    from my_agent_alpha.chat import chat

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
    from my_agent_alpha.chat import chat

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
    from my_agent_alpha.chat import chat

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
    from my_agent_alpha.chat import chat

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
    from my_agent_alpha.chat import chat

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
