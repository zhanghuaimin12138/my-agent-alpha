"""阶段 3 原生工具调用测试:参数表、@工具调用行解析、分片拼装、配对、端到端。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from my_agent_alpha import LogManager, Message
from my_agent_alpha.chat import chat
from my_agent_alpha.llm import reply
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.llm.openai_compat import msgs2dicts
from my_agent_alpha.tools import execute_msg, get_tool
from my_agent_alpha.tools.base import (
    Parameter,
    ToolUse,
    parameters_to_schema,
    spec_to_openai_tool,
)

# ---- 参数表 → JSON Schema ----


def test_parameters_to_schema() -> None:
    schema = parameters_to_schema(
        [
            Parameter("path", "string", "文件路径"),
            Parameter("content", "string", "内容"),
            Parameter("backup", "boolean", "是否备份", required=False),
        ]
    )
    assert schema["type"] == "object"
    assert schema["properties"]["path"] == {"type": "string", "description": "文件路径"}
    assert schema["required"] == ["path", "content"]  # 可选的不进必填列表
    assert schema["additionalProperties"] is False


def test_spec_to_openai_tool() -> None:
    shell = get_tool("shell")
    assert shell is not None
    payload = spec_to_openai_tool(shell)
    assert payload["type"] == "function"
    fn = payload["function"]
    assert fn["name"] == "shell"
    assert "bash" in str(fn["description"])
    assert fn["parameters"]["properties"]["command"]["type"] == "string"


def test_spec_description_truncated() -> None:
    from my_agent_alpha.tools.base import ToolSpec

    chatty = ToolSpec(name="chatty", desc="x" * 2000, block_types=("chatty",))
    payload = spec_to_openai_tool(chatty)
    assert len(str(payload["function"]["description"])) <= 1001  # 1000 + 省略号


# ---- @工具名(编号): {参数} 行解析 ----


def test_native_call_line_parsed() -> None:
    content = '保存一下:\n@save(call_1): {"path": "a.txt", "content": "你好"}'
    uses = list(ToolUse.iter_from_content(content))
    assert len(uses) == 1
    assert uses[0].tool.name == "save"
    assert uses[0].kwargs == {"path": "a.txt", "content": "你好"}
    assert uses[0].call_id == "call_1"
    assert uses[0].content is None


def test_native_unclosed_json_skipped() -> None:
    """JSON 没闭合 = 还在生成中,先不认(流式安全的自然结果)。"""
    content = '@save(call_1): {"path": "a.txt"'
    assert list(ToolUse.iter_from_content(content)) == []


def test_native_call_id_with_dot_parsed() -> None:
    """调用编号带点号也必须认(dashscope/kimi 的编号格式,真机实录踩坑)。

    dashscope 给的编号长这样 "shell.0_a475cfc6"(工具名_序号_哈希),
    编号字符集若不许点号,整行识别失败、调用被静默丢弃
    (真机会话 2026-1005-332a08 两次调用全部无结果,见踩坑记录条目 5)。
    """
    content = '我来查询:\n@shell(shell.0_a475cfc6): {"command": "echo hi"}'
    uses = list(ToolUse.iter_from_content(content))
    assert len(uses) == 1
    assert uses[0].tool.name == "shell"
    assert uses[0].call_id == "shell.0_a475cfc6"  # 编号原样保留(配对回传要用)
    assert uses[0].kwargs == {"command": "echo hi"}


def test_native_inside_fence_is_example_not_call() -> None:
    content = (
        '示范写法:\n```\n@save(call_1): {"path": "a"}\n```\n'
        '真正的调用:\n@shell(call_2): {"command": "ls"}'
    )
    uses = list(ToolUse.iter_from_content(content))
    assert [u.call_id for u in uses] == ["call_2"]  # 围栏里那个不算


def test_native_and_markdown_sorted_by_position() -> None:
    content = (
        '@shell(call_1): {"command": "echo 先"}\n'
        "然后 markdown:\n```read\na.txt\n```\n"
        '@shell(call_2): {"command": "echo 后"}'
    )
    uses = list(ToolUse.iter_from_content(content))
    assert [u.call_id or u.tool.name for u in uses] == ["call_1", "read", "call_2"]


def test_native_braces_inside_strings() -> None:
    """参数值里带大括号(引号内),配对计数不能被带偏。"""
    content = '@save(call_1): {"path": "a.txt", "content": "a{b}c"}'
    uses = list(ToolUse.iter_from_content(content))
    assert uses[0].kwargs["content"] == "a{b}c"


def test_display_command() -> None:
    uses = list(ToolUse.iter_from_content('@shell(c1): {"command": "ls -la"}'))
    assert uses[0].display_command() == "shell: command=ls -la"


# ---- 执行端:kwargs 通道 ----


def test_shell_via_native_kwargs(workspace: Path) -> None:
    msg = Message("assistant", '@shell(call_9): {"command": "echo native-shell"}')
    results = execute_msg(msg)
    assert "native-shell" in results[0].content
    assert results[0].call_id == "call_9"  # 结果盖上了调用编号


def test_save_via_native_kwargs(workspace: Path) -> None:
    msg = Message("assistant", '@save(call_1): {"path": "n.txt", "content": "原生内容"}')
    results = execute_msg(msg)
    assert "已创建" in results[0].content
    assert (workspace / "n.txt").read_text(encoding="utf-8") == "原生内容"


def test_read_via_native_kwargs(workspace: Path) -> None:
    (workspace / "r.txt").write_text("内容\n", encoding="utf-8")
    msg = Message("assistant", '@read(call_1): {"path": "r.txt"}')
    assert "内容" in execute_msg(msg)[0].content


def test_shell_missing_command_reports_error(workspace: Path) -> None:
    msg = Message("assistant", "@shell(call_1): {}")
    results = execute_msg(msg)
    assert "没有收到要执行的命令" in results[0].content


# ---- 流式分片拼装(_stream_once) ----


def _tool_events() -> Any:
    """模拟接口发来的原生调用分片:name 先到,arguments 逐字符到。"""

    def tc(
        index: int, id: str | None = None, name: str | None = None, args: str | None = None
    ) -> Any:
        function = SimpleNamespace(name=name, arguments=args)
        return SimpleNamespace(index=index, id=id, function=function)

    def delta(finish_reason: str | None = None, **kwargs: Any) -> Any:
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning_content=None, content=None, **kwargs),
                    finish_reason=finish_reason,
                )
            ],
        )

    return iter(
        [
            delta(role="assistant"),
            delta(tool_calls=[tc(0, id="call_1", name="save")]),
            delta(tool_calls=[tc(0, args='{"pa')]),
            delta(tool_calls=[tc(0, args='th": "a.t')]),
            delta(tool_calls=[tc(0, args='xt", "content": "hi"}')], finish_reason="tool_calls"),
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


def test_break_on_native_call() -> None:
    """原生调用行一凑齐就断:后面的文本不接收(mock 直接送文本行)。"""
    set_scripted_replies(['@shell(call_1): {"command": "echo break-native"}\n\n这段不该出现'])
    msg = reply([Message("user", "x")], "mock/echo")
    assert "break-native" in msg.content
    assert "这段不该出现" not in msg.content


def test_set_scripted_bare_string_is_one_reply() -> None:
    """防呆:直接传字符串(忘了包列表)也按一条回复处理,不拆成字符。"""
    set_scripted_replies("一条完整的回复")
    assert reply([Message("user", "x")], "mock/echo").content == "一条完整的回复"


# ---- 发请求时的格式转换(msgs2dicts) ----


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


# ---- 端到端(mock 剧本里写原生调用行) ----


def test_e2e_native_tool_loop(logdir: Path, workspace: Path) -> None:
    set_scripted_replies(
        [
            '我来执行:\n@shell(call_1): {"command": "echo native-e2e"}',
            "执行完了。",
        ]
    )
    chat(
        ["跑一下"],
        logdir=logdir,
        workspace=workspace,
        model="mock/echo",
        non_interactive=True,
        tool_format="tool",  # 原生模式(系统提示词也换原生版)
    )
    manager = LogManager.load(logdir)
    roles = [m.role for m in manager.messages]
    # 阶段 5 起,每次模型回复后默认有 [用量] 钩子产出一条 system 消息
    assert roles == [
        "system",
        "system",
        "user",
        "assistant",
        "system",
        "system",
        "assistant",
        "system",
    ]
    contents = [m.content for m in manager.messages]
    assert "native-e2e" in contents[5]  # [用量] 在 4,工具输出在 5
    # 原生模式系统提示词:教的是 function calling,不是围栏约定
    assert "function calling" in contents[0]
    assert "围栏" not in contents[0]
    # 工具结果带 call_id 配对
    assert manager.messages[5].call_id == "call_1"
