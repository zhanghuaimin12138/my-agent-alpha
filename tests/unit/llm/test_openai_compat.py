"""llm/openai_compat.py 测试:消息格式转换、reasoning 包装、流式分片拼装、重试策略。

全部用假 client(SimpleNamespace 伪造 SDK 对象),不发真请求。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from my_agent_alpha import Message
from my_agent_alpha.llm.openai_compat import msgs2dicts

# ---- 发请求时的格式转换(msgs2dicts)----


def test_msgs2dicts_native_round_trip() -> None:
    messages = [
        Message("system", "提示词"),
        Message("user", "保存"),
        Message("assistant", '好的\n@save(call_1): {"path": "a.txt", "content": "hi"}'),
        Message("system", "已创建 a.txt", call_id="call_1"),
        Message("assistant", "完成"),
    ]
    dicts = msgs2dicts(messages)
    assert dicts[0] == {"role": "system", "content": "提示词"}
    # assistant 的原生行 → tool_calls 字段,正文保留
    a = dicts[2]
    assert a["role"] == "assistant"
    assert a["content"] == "好的"
    assert a["tool_calls"][0]["id"] == "call_1"
    assert a["tool_calls"][0]["function"]["name"] == "save"
    # 带 call_id 的 system(工具结果)→ tool 角色
    assert dicts[3] == {"role": "tool", "tool_call_id": "call_1", "content": "已创建 a.txt"}


def test_msgs2dicts_orphan_tool_result_demoted() -> None:
    """编号找不到对应调用的工具结果:降级成 user 消息(直接发 tool 会 400)。"""
    messages = [
        Message("user", "hi"),
        Message("system", "来历不明的结果", call_id="call_ghost"),
    ]
    dicts = msgs2dicts(messages)
    assert dicts[1]["role"] == "user"


def test_msgs2dicts_markdown_mode_unchanged() -> None:
    """markdown 模式:没有原生行和 call_id,行为和从前一样。"""
    messages = [
        Message("system", "提示词"),
        Message("user", "x"),
        Message("assistant", "```shell\nls\n```"),
        Message("system", "$ ls\na\nReturnCode: 0"),
    ]
    dicts = msgs2dicts(messages)
    assert [d["role"] for d in dicts] == ["system", "user", "assistant", "system"]
    assert "tool_calls" not in dicts[2]


# ---- reasoning_content → <think> 包装(_stream_once)----


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


# ---- 原生调用分片拼装(_stream_once 的 tool_calls 增量)----


def _tool_events() -> Any:
    """模拟接口发来的原生调用分片:name 先到,arguments 逐字符到。"""

    def tc(
        index: int, id: str | None = None, name: str | None = None, args: str | None = None
    ) -> Any:
        function = SimpleNamespace(name=name, arguments=args)
        return SimpleNamespace(index=index, id=id, function=function)

    def delta(**kwargs: Any) -> Any:
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning_content=None, content=None, **kwargs)
                )
            ],
        )

    return iter(
        [
            delta(role="assistant"),
            delta(tool_calls=[tc(0, id="call_1", name="save")]),
            delta(tool_calls=[tc(0, args='{"pa')]),
            delta(tool_calls=[tc(0, args='th": "a.t')]),
            delta(tool_calls=[tc(0, args='xt", "content": "hi"}')]),
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
                choices=[],
            ),
        ]
    )


def test_native_fragments_assembled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from my_agent_alpha.llm import models as models_mod
    from my_agent_alpha.llm import openai_compat

    monkeypatch.setattr(
        openai_compat,
        "_make_client",
        lambda cfg: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: _tool_events()))
        ),
    )
    meta = models_mod.ModelMeta("deepseek", "x")
    cfg = models_mod.PROVIDERS["deepseek"]
    chunks = list(openai_compat.stream([], meta, cfg, tools=()))
    text = "".join(chunks)
    # 分片拼成了完整的原生调用行
    assert text == '@save(call_1): {"path": "a.txt", "content": "hi"}\n'
    # usage 照常从 return 值带回
    gen = openai_compat._stream_once([], meta, cfg)
    usage = None
    try:
        while True:
            next(gen)
    except StopIteration as e:
        usage = e.value
    assert usage == {"input": 5, "output": 2}


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
