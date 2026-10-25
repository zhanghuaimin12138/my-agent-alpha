"""chat.py 单元测试:不跑完整循环,直接测 chat 模块里的各个函数。

覆盖:循环继续判定、用户输入清洗、发给模型前的压缩视图、
上下文溢出重试、流式终端渲染(_StreamPrinter)、空回复检测器。
走完整 chat() 循环的端到端测试在 tests/integration/test_chat_loop.py。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from my_agent_alpha import Message
from my_agent_alpha.chat import (
    _StreamPrinter,
    cli_confirm_hook,
    has_runnable_tooluse,
    missed_tooluse_reason,
    prepare_messages,
    step,
)
from my_agent_alpha.compaction import messages_tokens
from my_agent_alpha.llm import ContextLengthError, reply
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.tools import add_tools
from my_agent_alpha.tools.base import ToolSpec, ToolUse, parse_tooluses

# ---- 循环继续判定 ----


def test_has_runnable_tooluse() -> None:
    msgs = [
        Message("system", "提示词"),
        Message("user", "跑一下"),
        Message("assistant", "执行:\n```shell\nls\n```"),
        Message("system", "$ ls\na\nReturnCode: 0"),
    ]
    assert has_runnable_tooluse(msgs) is True  # 最后的 assistant 还有工具调用
    msgs.append(Message("assistant", "完成了"))
    assert has_runnable_tooluse(msgs) is False


# ---- 用户输入清洗 ----


def test_user_input_bad_unicode_cleaned(monkeypatch: pytest.MonkeyPatch) -> None:
    """终端传来坏字符(落单代理)时入口先清洗,不让它进日志把程序写崩。"""
    from my_agent_alpha.chat import _get_user_input

    monkeypatch.setattr("builtins.input", lambda _: "\ud800有雨吗")
    msg = _get_user_input()
    assert msg is not None
    assert msg.content == "?有雨吗"


# ---- 发给模型前的压缩视图 ----


def test_prepare_messages_view() -> None:
    msgs = [Message("system", "提示词", pinned=True, hide=True)]
    msgs += [Message("user", f"问{i}") for i in range(30)]
    view = prepare_messages(msgs, "mock/echo", budget=80)
    assert messages_tokens(view) <= 80 + 30  # 视图被压到预算量级
    # 再次确认:原列表没被动
    assert len(msgs) == 31


# ---- 溢出自愈 ----


def test_step_overflow_retry_with_smaller_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """第一次 reply 抛上下文超长 → step 以更小预算重压并重试一次。"""
    from my_agent_alpha import chat as chat_mod

    calls: list[Sequence[Message]] = []
    state = {"failed": False}
    real_reply = chat_mod.reply

    def flaky_reply(messages, model, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(messages))
        if not state["failed"]:
            state["failed"] = True
            raise ContextLengthError("This model's maximum context length is 4096 tokens")
        return real_reply(messages, model, **kwargs)

    monkeypatch.setattr(chat_mod, "reply", flaky_reply)
    log = [Message("system", "提示词", pinned=True, hide=True)] + [
        Message("user", f"消息{i}" + "内容" * 50) for i in range(20)
    ]
    out = list(step(log, "mock/echo", budget=2000))
    assert len(calls) == 2  # 失败一次、重试一次
    assert messages_tokens(list(calls[1])) <= messages_tokens(list(calls[0]))  # 重试的更小
    assert out[0].role == "assistant"


def test_step_overflow_second_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """重试仍超长 → 不再无限循环,异常抛出。"""
    from my_agent_alpha import chat as chat_mod

    def always_overflow(messages, model, **kwargs):  # type: ignore[no-untyped-def]
        raise ContextLengthError("context length exceeded")

    monkeypatch.setattr(chat_mod, "reply", always_overflow)
    with pytest.raises(ContextLengthError):
        list(step([Message("user", "hi")], "mock/echo"))


# ---- 流式终端渲染(_StreamPrinter)----


def test_think_rendered_dim_without_literal_tags(capsys: pytest.CaptureFixture[str]) -> None:
    set_scripted_replies(["<think>先想想</think>正文回答"])
    reply([Message("user", "q")], "mock/echo", on_token=_StreamPrinter().token)
    out = capsys.readouterr().out
    assert "正文回答" in out
    # think 内容在,且带暗色 ANSI(标签被消费,不打印字面量)
    # (暗色可能因标签边界分成多段,逐段断言)
    assert "先想" in out and "想" in out
    assert "\x1b[2m" in out
    assert "<think>" not in out
    assert "</think>" not in out


def test_printer_handles_tags_split_across_chunks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    printer = _StreamPrinter()
    # 手工喂会切断标签的块
    for chunk in ["<th", "ink>隐", "私推理</th", "ink>结", "论"]:
        printer.token(chunk)
    printer.finish()
    out = capsys.readouterr().out
    # 隐 / 私推理 两段都应是暗色(标签边界处分成两段输出是正常行为)
    assert "\x1b[2m隐" in out and "\x1b[2m私推理" in out
    assert "结论" in out
    assert "<think>" not in out and "</think>" not in out


# ---- 空回复/漏写检测器(missed_tooluse_reason)----
#
# 背景(docs/踩坑记录.md 第 3 条的遗留、docs/tmp.jsonl 实录):真实模型
# 连续几轮只在 <think> 里说"我来调用工具",正文里一个围栏都没有,回合
# 静默结束,用户只看到空气。这里测检测器本身;命中后注入提醒并重新
# 生成的端到端剧本在 tests/integration/test_tool_nudge.py。


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


def test_reason_empty_body_variants() -> None:
    """正文为空的几种写法(闭合思考/未闭合思考/全空/纯空白)都判"没有正文"。"""
    for content in ("<think>我来调用工具。</think>", "<think>没写完", "", "  \n  "):
        reason = missed_tooluse_reason([Message("assistant", content)])
        assert reason is not None, content
        assert "没有正文" in reason


def test_reason_calls_inside_think() -> None:
    """真机翻车样本 2026-1024-e6480f:调用写进了思考区、正文为空。

    模型两次都"以为自己在调用"(调用行其实包在 <think> 里),泛泛提醒
    没点中病因,回合死掉 —— 检测器必须点名"思考区"。
    """
    # 原生调用形态(实录原样裁剪:@read/@shell 都在思考区里,正文为空)
    native = (
        '<think>Let me actually make the tool calls.@read(call_x): {"path": "/tmp/a"}\n'
        '@shell(call_y): {"command": "ls"}\n</think>'
    )
    reason = missed_tooluse_reason([Message("assistant", native)])
    assert reason is not None and "思考区" in reason
    # markdown 通道:思考里写了语言标记对得上已知工具的围栏,同样点名
    add_tools((_make_tool("fence.demo", "fencedemo"),))
    fenced = "<think>看看目录:\n```fencedemo\nls\n```\n</think>"
    reason = missed_tooluse_reason([Message("assistant", fenced)])
    assert reason is not None and "思考区" in reason


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


# ---- 终端询问钩子(cli_confirm_hook:交互模式下问用户)----


def _fake_input(answers: list[str]):
    """把预设答案变成 input():耗尽后抛 EOFError(模拟 Ctrl-D)。"""
    it = iter(answers)

    def _input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return _input


def test_cli_hook_yes(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from my_agent_alpha.confirm import clear_confirm_hooks

    clear_confirm_hooks()
    monkeypatch.setattr("builtins.input", _fake_input(["y"]))
    assert cli_confirm_hook(_shell_use("rm file.txt")).action == "confirm"


def test_cli_hook_no(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from my_agent_alpha.confirm import clear_confirm_hooks

    clear_confirm_hooks()
    monkeypatch.setattr("builtins.input", _fake_input(["n"]))
    result = cli_confirm_hook(_shell_use("rm file.txt"))
    assert result.action == "skip"


def test_cli_hook_edit_then_a(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from my_agent_alpha.confirm import clear_confirm_hooks, get_confirmation

    clear_confirm_hooks()
    monkeypatch.setattr("builtins.input", _fake_input(["e", "echo FIXED", "x"]))
    result = cli_confirm_hook(_shell_use("rm file.txt"))
    assert result.action == "edit"
    assert result.edited_content == "echo FIXED"
    # 选 a 之后:本会话不再问(钩子退位返回 None,由链尾放行)
    monkeypatch.setattr("builtins.input", _fake_input(["a"]))
    assert cli_confirm_hook(_shell_use("rm file.txt")).action == "confirm"
    assert cli_confirm_hook(_shell_use("rm other.txt")) is None

    assert get_confirmation(_shell_use("rm other.txt")).action == "confirm"


def test_cli_hook_eof_means_skip(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from my_agent_alpha.confirm import clear_confirm_hooks

    clear_confirm_hooks()
    monkeypatch.setattr("builtins.input", _fake_input([]))  # 立即 EOF
    assert cli_confirm_hook(_shell_use("rm file.txt")).action == "skip"


def _shell_use(command: str) -> ToolUse:
    """从一条 markdown 围栏造出 shell 调用(测试辅助)。"""
    uses = parse_tooluses(f"```shell\n{command}\n```")
    assert uses and uses[0].tool.name == "shell"
    return uses[0]
