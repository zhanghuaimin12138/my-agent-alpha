"""阶段 2 流式测试:契约、断点检测、think 渲染、中断、重试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.codeblock import iter_from_markdown
from my_agent_alpha.llm import reply
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.message import INTERRUPT_CONTENT
from my_agent_alpha.tui import StreamPrinter

# ---- 流式契约:分块消费 + usage 捕获 ----


def test_stream_chunks_and_metadata() -> None:
    tokens: list[str] = []
    msg = reply([Message("user", "你好")], "mock/echo", on_token=tokens.append)
    # 分块收到了(块数 > 1,证明走了流式路径),拼起来等于完整内容
    assert len(tokens) > 1
    assert msg.content == "Echo: 你好"
    assert "".join(tokens) == msg.content
    assert msg.metadata is not None
    assert msg.metadata["usage"] == {"input": 0, "output": 0}


# ---- 断点检测:完整工具调用出现即停 ----


def test_break_on_tooluse_stops_generation() -> None:
    """围栏闭合 + 空行确认后,后续文本不该被生成/保存。"""
    trailing = "这段废话在断点之后,不该出现"
    scripted = f"我先执行:\n```shell\necho b2\n```\n\n{trailing}\n\n```shell\necho never\n```"
    set_scripted_replies([scripted])
    msg = reply([Message("user", "x")], "mock/echo")
    assert "```shell\necho b2\n```" in msg.content
    assert trailing not in msg.content  # 断点之前就停了
    assert "echo never" not in msg.content
    # 排干生成器后 usage 仍被捕获(研究精华 #4)
    assert msg.metadata is not None
    assert msg.metadata["usage"] == {"input": 0, "output": 0}


def test_break_on_tooluse_disabled() -> None:
    scripted = "```shell\necho x\n```\n\n后置文本"
    set_scripted_replies([scripted])
    msg = reply([Message("user", "x")], "mock/echo", break_on_tooluse=False)
    assert "后置文本" in msg.content


def test_streaming_fence_requires_blank_line() -> None:
    """流式规则:闭合围栏后必须有空行,块才算完整。"""
    complete = "```shell\nls\n```\n\n后续"
    assert len(list(iter_from_markdown(complete, streaming=True))) == 1

    no_blank = "```shell\nls\n```"  # 闭合栏后没空行(流式视角:可能还没写完)
    assert list(iter_from_markdown(no_blank, streaming=True)) == []
    # 非流式(最终解析)依然能拿到 —— 两套规则各司其职
    assert len(list(iter_from_markdown(no_blank))) == 1


def test_streaming_unclosed_think_no_early_stop() -> None:
    """思考区里出现的围栏不算完整调用:think 没闭合前不能触发流式早停。"""
    generating = "<think>举例 ```shell\nls\n```\n再想想"
    assert list(iter_from_markdown(generating, streaming=True)) == []


def test_e2e_break_then_execute(logdir: Path, workspace: Path) -> None:
    set_scripted_replies(
        [
            "马上执行:\n```shell\necho streaming-e2e\n```\n\n后面的内容不该被生成",
            "完成。",
        ]
    )
    chat(["跑一下"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)
    manager = LogManager.load(logdir)
    contents = [m.content for m in manager.messages]
    # 断点:assistant 消息不含断点后的文本
    assistant = next(c for c in contents if "```shell" in c)
    assert "不该被生成" not in assistant
    # 工具真的执行了,结果进了日志
    assert any("streaming-e2e" in c for c in contents)


# ---- think 渲染 ----


def test_think_rendered_dim_without_literal_tags(capsys: pytest.CaptureFixture[str]) -> None:
    set_scripted_replies(["<think>先想想</think>正文回答"])
    reply([Message("user", "q")], "mock/echo", on_token=StreamPrinter().token)
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
    printer = StreamPrinter()
    # 手工喂会切断标签的块
    for chunk in ["<th", "ink>隐", "私推理</th", "ink>结", "论"]:
        printer.token(chunk)
    printer.finish()
    out = capsys.readouterr().out
    # 隐 / 私推理 两段都应是暗色(标签边界处分成两段输出是正常行为)
    assert "\x1b[2m隐" in out and "\x1b[2m私推理" in out
    assert "结论" in out
    assert "<think>" not in out and "</think>" not in out


# ---- 流式中断 ----


def test_interrupt_midstream_returns_partial() -> None:
    set_scripted_replies([KeyboardInterrupt()])
    msg = reply([Message("user", "x")], "mock/echo")
    assert "^C 已中断" in msg.content
    assert msg.metadata is not None
    assert msg.metadata.get("interrupted") is True


def test_interrupt_midstream_stops_turn(logdir: Path, workspace: Path) -> None:
    set_scripted_replies([KeyboardInterrupt()])
    chat(["触发"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)
    manager = LogManager.load(logdir)
    assert any(
        m.role == "system" and m.content.startswith(INTERRUPT_CONTENT) for m in manager.messages
    )
    # 中断后不再生成第二步
    assert sum(1 for m in manager.messages if m.role == "assistant") == 1


# ---- reasoning_content → <think> 包装(openai_compat._stream_once) ----


def _chunk(
    reasoning: str | None = None,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
) -> Any:
    """造一个假的流式 chunk:choices[0] 里放一个增量(delta)。

    字段和真实接口一致:reasoning 是思考片段,content 是正文片段,
    tool_calls 是原生调用分片,finish_reason 是结束标记(只在最后一个
    带回答的 chunk 上出现一次)。
    """
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    reasoning_content=reasoning, content=content, tool_calls=tool_calls
                ),
                finish_reason=finish_reason,
            )
        ],
    )


def _tc_frag(
    index: int = 0, call_id: str | None = None, name: str | None = None, arguments: str = ""
) -> Any:
    """造一段原生工具调用的分片:编号和名字只在第一段出现,参数是逐段挤过来的。"""
    return SimpleNamespace(
        index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _usage_chunk(inp: int = 7, out: int = 3) -> Any:
    """造流末尾那个只带用量信息、没有回答内容的 chunk。"""
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=inp, completion_tokens=out), choices=[]
    )


def _patch_stream(monkeypatch: pytest.MonkeyPatch, events: Any) -> None:
    """把 _make_client 换成假货:create() 直接返回预制的假事件迭代器。"""
    from my_agent_alpha.llm import openai_compat

    monkeypatch.setattr(
        openai_compat,
        "_make_client",
        lambda cfg: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: events))
        ),
    )


def _fake_stream_events() -> Any:
    return iter(
        [
            _chunk(reasoning="推理A"),
            _chunk(reasoning="推理B"),
            _chunk(content="正文", finish_reason="stop"),
            _usage_chunk(),
        ]
    )


def test_reasoning_wrapped_into_think(monkeypatch: pytest.MonkeyPatch) -> None:
    from my_agent_alpha.llm import models as models_mod
    from my_agent_alpha.llm import openai_compat

    monkeypatch.setattr(
        openai_compat,
        "_make_client",
        lambda cfg: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: _fake_stream_events())
            )
        ),
    )
    cfg = models_mod.PROVIDERS["deepseek"]
    chunks = list(openai_compat.stream([], models_mod.ModelMeta("deepseek", "x"), cfg))
    text = "".join(chunks)
    assert text == "<think>推理A推理B</think>正文"
    # usage 从 return 值拿到
    gen = openai_compat._stream_once([], models_mod.ModelMeta("deepseek", "x"), cfg)
    out: list[str] = []
    usage = None
    try:
        while True:
            out.append(next(gen))
    except StopIteration as e:
        usage = e.value
    assert usage == {"input": 7, "output": 3}


# ---- 原生调用分片:think 收口 / 空参兜底 / 截断报错(2026-11-01 glm 踩坑回归) ----


def test_tool_call_line_lands_outside_think(monkeypatch: pytest.MonkeyPatch) -> None:
    """回归:思考模型"思考完直接发原生调用、全程无正文"时,调用行必须落在 </think> 之后。

    修复前:think 一直开着,调用行被包进 <think> 里,解析器按规矩把 think
    里的调用当草稿扔掉 → 工具一次都没执行(glm-5.3 真机踩坑)。
    """
    from my_agent_alpha.llm import models as models_mod
    from my_agent_alpha.llm import openai_compat

    _patch_stream(
        monkeypatch,
        iter(
            [
                _chunk(reasoning="先读需求"),
                _chunk(tool_calls=[_tc_frag(call_id="call_1", name="read", arguments='{"pa')]),
                _chunk(
                    tool_calls=[_tc_frag(arguments='th": "a.txt"}')],
                    finish_reason="tool_calls",
                ),
                _usage_chunk(),
            ]
        ),
    )
    meta = models_mod.ModelMeta("deepseek", "x")
    cfg = models_mod.PROVIDERS["deepseek"]
    text = "".join(openai_compat.stream([], meta, cfg))
    # 调用行在 think 关闭之后,参数完整可解析
    assert text == '<think>先读需求</think>@read(call_1): {"path": "a.txt"}\n'


def test_tool_call_with_empty_args_flushed(monkeypatch: pytest.MonkeyPatch) -> None:
    """无参数工具(arguments 全程是空串,永远等不到"大括号闭合"):
    流结束时兜底补出调用行,而不是静默丢掉这次调用。"""
    from my_agent_alpha.llm import models as models_mod
    from my_agent_alpha.llm import openai_compat

    _patch_stream(
        monkeypatch,
        iter(
            [
                _chunk(
                    tool_calls=[_tc_frag(call_id="call_9", name="todo")],
                    finish_reason="tool_calls",
                ),
                _usage_chunk(),
            ]
        ),
    )
    meta = models_mod.ModelMeta("deepseek", "x")
    cfg = models_mod.PROVIDERS["deepseek"]
    text = "".join(openai_compat.stream([], meta, cfg))
    assert text == "@todo(call_9): {}\n"


def test_tool_call_multiline_args_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数是格式化过的多行 JSON(内部带换行和空格,末尾还有个换行):
    去掉尾部空白后照样能凑齐调用行。"""
    from my_agent_alpha.llm import models as models_mod
    from my_agent_alpha.llm import openai_compat

    _patch_stream(
        monkeypatch,
        iter(
            [
                _chunk(tool_calls=[_tc_frag(call_id="call_1", name="read", arguments="{")]),
                _chunk(tool_calls=[_tc_frag(arguments='\n  "path":')]),
                _chunk(tool_calls=[_tc_frag(arguments=' "a.txt"\n')]),
                _chunk(tool_calls=[_tc_frag(arguments="}\n")], finish_reason="tool_calls"),
                _usage_chunk(),
            ]
        ),
    )
    meta = models_mod.ModelMeta("deepseek", "x")
    cfg = models_mod.PROVIDERS["deepseek"]
    text = "".join(openai_compat.stream([], meta, cfg))
    assert text == '@read(call_1): {\n  "path": "a.txt"\n}\n'


def test_stream_without_finish_reason_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """流结束了但接口从头到尾没给结束标记 → 视为中途被掐断:
    报错,不把半截回复冒充成完整回复存下来。"""
    from my_agent_alpha.llm import models as models_mod
    from my_agent_alpha.llm import openai_compat

    _patch_stream(monkeypatch, iter([_chunk(content="写了一半")]))
    meta = models_mod.ModelMeta("deepseek", "x")
    cfg = models_mod.PROVIDERS["deepseek"]
    with pytest.raises(RuntimeError, match="finish_reason"):
        list(openai_compat.stream([], meta, cfg))


def test_drain_error_after_break_does_not_kill_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """断点触发后读尾巴(为了拿用量)时断流:完整的调用已经到手,
    尾巴上的报错不该打死整个回合,回复照常返回。"""

    def events() -> Any:
        yield _chunk(content="```shell\necho drain-ok\n```\n\n")
        yield _chunk(content="多余的尾巴")  # 断点在这之前就该触发,这段被丢弃
        raise RuntimeError("断点后断流")

    _patch_stream(monkeypatch, events())
    from my_agent_alpha.tools import get_tools

    get_tools()  # 注册真实工具,断点检测才认得 ```shell 围栏
    msg = reply([Message("user", "x")], "deepseek/deepseek-chat")
    assert "echo drain-ok" in msg.content
    assert "多余的尾巴" not in msg.content  # 断点后断流,尾巴没进来


# ---- 重试策略 ----


def test_retry_on_transient_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from my_agent_alpha.llm import models as models_mod
    from my_agent_alpha.llm import openai_compat

    calls = {"n": 0}

    class FlakyCreate:
        def __call__(self, **kw: Any) -> Any:
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("瞬断")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="成功"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
            )

    monkeypatch.setattr(
        openai_compat,
        "_make_client",
        lambda cfg: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=FlakyCreate()))
        ),
    )
    monkeypatch.setattr(openai_compat, "_is_transient", lambda e: True)
    monkeypatch.setattr(openai_compat, "_backoff", lambda attempt: 0.0)

    meta = models_mod.ModelMeta("deepseek", "x")
    cfg = models_mod.PROVIDERS["deepseek"]
    text, usage = openai_compat.chat([Message("user", "x")], meta, cfg)
    assert text == "成功"
    assert calls["n"] == 3  # 两次失败 + 第三次成功
    assert usage == {"input": 1, "output": 2}


def test_no_retry_on_non_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    from my_agent_alpha.llm import models as models_mod
    from my_agent_alpha.llm import openai_compat

    calls = {"n": 0}

    def always_fail(**kw: Any) -> Any:
        calls["n"] += 1
        raise PermissionError("鉴权失败,重试无意义")

    monkeypatch.setattr(
        openai_compat,
        "_make_client",
        lambda cfg: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=always_fail))
        ),
    )
    with pytest.raises(PermissionError):
        meta = models_mod.ModelMeta("deepseek", "x")
        cfg = models_mod.PROVIDERS["deepseek"]
        openai_compat.chat([Message("user", "x")], meta, cfg)
    assert calls["n"] == 1  # 非瞬态:一次都不重试


def test_stream_retry_only_before_first_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    """已产出内容后报错 → 不重试直接抛(避免重复输出)。"""
    from my_agent_alpha.llm import models as models_mod
    from my_agent_alpha.llm import openai_compat

    calls = {"n": 0}

    def make_events(**kw: Any) -> Any:
        calls["n"] += 1

        # 第一次调用:吐一块后炸;若错误地重试,第二次会再吐一块
        def gen() -> Any:
            yield SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(reasoning_content=None, content="块A"))
                ],
            )
            raise RuntimeError("中途断流")

        return gen()

    monkeypatch.setattr(
        openai_compat,
        "_make_client",
        lambda cfg: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=make_events))
        ),
    )
    monkeypatch.setattr(openai_compat, "_is_transient", lambda e: True)
    monkeypatch.setattr(openai_compat, "_backoff", lambda attempt: 0.0)

    chunks: list[str] = []
    with pytest.raises(RuntimeError, match="中途断流"):
        meta = models_mod.ModelMeta("deepseek", "x")
        cfg = models_mod.PROVIDERS["deepseek"]
        for c in openai_compat.stream([], meta, cfg):
            chunks.append(c)
    assert chunks == ["块A"]
    assert calls["n"] == 1  # 已 yield → 不重试
