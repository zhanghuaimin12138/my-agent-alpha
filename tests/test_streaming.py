"""阶段 2 流式测试:契约、断点检测、think 渲染、中断、重试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import _StreamPrinter, chat
from my_agent_alpha.codeblock import iter_from_markdown
from my_agent_alpha.llm import reply
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.message import INTERRUPT_CONTENT

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


def _fake_stream_events() -> Any:
    def delta(reasoning: str | None = None, content: str | None = None) -> Any:
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(delta=SimpleNamespace(reasoning_content=reasoning, content=content))
            ],
        )

    return iter(
        [
            delta(reasoning="推理A"),
            delta(reasoning="推理B"),
            delta(content="正文"),
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3), choices=[]
            ),
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
