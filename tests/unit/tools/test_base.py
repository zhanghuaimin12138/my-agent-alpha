"""tools/base.py 测试:ToolUse 解析(两条通道 + think 剥离)、参数表转 JSON Schema。

覆盖点:
- markdown 通道:围栏在行首才认、块头参数(多词是一段)、多块按出现顺序、
  未知语言忽略、没闭合丢弃、行内代码不算围栏;
- 原生通道:@工具名(编号): {JSON} 行、JSON 没闭合忽略、围栏里的
  (含未闭合围栏)是示例、未知工具忽略、编号可省略、冒号后须紧跟 {、
  工具名可带点(MCP 前缀);
- think 剥离:思考区里的围栏/@调用是草稿不算数、粘在 </think> 后面的围栏
  剥完落到行首能认出;
- 两通道混用:按出现位置排序;
- streaming=True 透传:围栏闭合后还要空行确认(防执行半截命令);
- 参数表(Parameter 列表)→ OpenAI 的 JSON Schema,给原生通道发请求用。

跑法:uv run pytest tests/unit/tools/test_base.py -v
"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha.tools import get_tool
from my_agent_alpha.tools.base import (
    Parameter,
    ToolSpec,
    ToolUse,
    parameters_to_schema,
    spec_to_openai_tool,
)


def names(content: str) -> list[str]:
    """解析后按顺序返回工具名列表(断言用的小工具)。"""
    return [u.tool.name for u in ToolUse.iter_from_content(content)]


# ---- markdown 通道 ----


def test_md_basic_fence() -> None:
    """最基本的围栏调用:语言标记 = 工具名,块内 = content。"""
    uses = list(ToolUse.iter_from_content("```shell\nls -la\n```"))
    assert len(uses) == 1
    assert uses[0].tool.name == "shell"
    assert uses[0].content == "ls -la"
    assert uses[0].args is None  # 块头没带参数
    assert uses[0].kwargs is None  # markdown 通道没有已解析的参数字典
    assert uses[0].call_id is None  # markdown 通道没有编号


def test_md_fence_with_args() -> None:
    """块头带参数:```save a.txt``` 的 a.txt 进 args。"""
    uses = list(ToolUse.iter_from_content("```save a.txt\n文件内容\n```"))
    assert uses[0].tool.name == "save"
    assert uses[0].args == ["a.txt"]
    assert uses[0].content == "文件内容"


def test_md_multiword_args_single_element() -> None:
    """块头参数整段作为一个元素:```save a b``` 的 args 是 ["a b"]。

    围栏解析在第一个空格处把"语言标记"和"参数"切开,剩下的空格不再切,
    所以 args 列表要么没有、要么恰好一个元素 —— 执行端按"一段参数文本"理解。
    """
    uses = list(ToolUse.iter_from_content("```save a b\n内容\n```"))
    assert uses[0].args == ["a b"]


def test_md_lang_alias_matches_tool() -> None:
    """block_types 里的别名也算:sh/bash 都指向 shell 工具。"""
    assert names("```bash\necho hi\n```") == ["shell"]


def test_md_multiple_blocks_in_order() -> None:
    """多个围栏按出现顺序产出。"""
    text = "先看目录:\n```shell\nls\n```\n再存文件:\n```save a.txt\n内容\n```"
    assert names(text) == ["shell", "save"]


def test_md_unknown_lang_ignored() -> None:
    """语言标记不认识任何工具 → 不是调用,直接忽略。"""
    assert names("```python\nprint('说明')\n```") == []


def test_md_unterminated_fence_dropped() -> None:
    """没闭合的围栏丢弃 —— 绝不执行半截内容。"""
    assert names("```shell\necho 半截") == []


def test_md_inline_code_not_fence() -> None:
    """行内 `code` 不是围栏块。"""
    assert names("用 `shell` 这个标记来执行") == []


def test_md_empty_content() -> None:
    """空文本、纯文本 → 没有任何调用。"""
    assert names("") == []
    assert names("任务完成了,不需要工具。") == []


# ---- 原生通道(@工具名(编号): {JSON})----


def test_native_basic_call() -> None:
    """原生调用:kwargs 是解析好的参数字典,call_id 原样带出。"""
    uses = list(ToolUse.iter_from_content('@shell(c1): {"command": "ls -la"}'))
    assert len(uses) == 1
    assert uses[0].tool.name == "shell"
    assert uses[0].kwargs == {"command": "ls -la"}
    assert uses[0].call_id == "c1"
    assert uses[0].content is None  # 原生通道没有围栏正文


def test_native_without_call_id() -> None:
    """编号可以省略:@shell(): 也算调用(call_id 为 None)。"""
    uses = list(ToolUse.iter_from_content('@shell(): {"command": "pwd"}'))
    assert uses[0].kwargs == {"command": "pwd"}
    assert uses[0].call_id is None


def test_native_unclosed_json_ignored() -> None:
    """JSON 没闭合 → 还没写完,先忽略(流式安全)。"""
    assert names('@shell(c1): {"command": "ls"') == []


def test_native_bad_json_ignored() -> None:
    """大括号配对了但内容不是合法 JSON → 跳过。"""
    assert names("@shell(c1): {这不是json}") == []


def test_native_unknown_tool_ignored() -> None:
    """@后面不是已注册工具名 → 忽略。"""
    assert names('@nosuch(c1): {"x": 1}') == []


def test_native_inside_fence_is_example() -> None:
    """围栏代码块里写的 @xxx 只是示例,不解析。"""
    assert names('```\n@shell(c1): {"command": "ls"}\n```') == []


def test_native_id_allows_dots() -> None:
    """编号里可以带点号(dashscope 风格,踩坑记录条目 5)。"""
    uses = list(ToolUse.iter_from_content('@shell(call.0.1): {"command": "ls"}'))
    assert uses[0].call_id == "call.0.1"


# ---- think 剥离(gptme 同款,见踩坑记录条目 1)----


def test_think_fence_not_a_call() -> None:
    """思考区里的围栏是草稿,不解析。"""
    assert names("<think>要不要 ```shell\nrm -rf /\n``` 试试</think>不,我不用") == []


def test_think_native_not_a_call() -> None:
    """思考区里的 @原生调用也只是草稿,不解析。"""
    assert names('<think>比如 @save(c9): {"path": "a"}</think>我想想') == []


def test_fence_glued_to_think_close() -> None:
    """kimi 粘连格式:</think> 和开栏同一行,剥掉 think 后能认出。"""
    assert names("<think>先看看目录。</think>```shell\nls\n```") == ["shell"]


def test_unclosed_think_parses_nothing() -> None:
    """思考区没闭合 = 还在生成,整体不解析。"""
    assert names("<think>我想想 ```shell\necho 半截") == []


# ---- 两通道混用 ----


def test_mixed_channels_sorted_by_position() -> None:
    """markdown 围栏和原生调用同时出现,按在原文里的位置排序。"""
    text = '```shell\nls\n```\n说明:\n@save(c2): {"path": "a.txt", "content": "内容"}'
    assert names(text) == ["shell", "save"]


# ---- 原生通道补充(来自真机实录的编号格式)----


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


def test_native_braces_inside_strings() -> None:
    """参数值里带大括号(引号内),配对计数不能被带偏。"""
    content = '@save(call_1): {"path": "a.txt", "content": "a{b}c"}'
    uses = list(ToolUse.iter_from_content(content))
    assert uses[0].kwargs["content"] == "a{b}c"


def test_native_and_markdown_sorted_by_position() -> None:
    """原生行在前、围栏在后:同样按原文位置排序。"""
    content = (
        '@shell(call_1): {"command": "echo 先"}\n'
        "然后 markdown:\n```read\na.txt\n```\n"
        '@shell(call_2): {"command": "echo 后"}'
    )
    uses = list(ToolUse.iter_from_content(content))
    assert [u.call_id or u.tool.name for u in uses] == ["call_1", "read", "call_2"]


def test_display_command() -> None:
    """display_command:给终端展示的一行摘要。"""
    uses = list(ToolUse.iter_from_content('@shell(c1): {"command": "ls -la"}'))
    assert uses[0].display_command() == "shell: command=ls -la"


def test_native_colon_without_json_ignored() -> None:
    """冒号后面没紧跟 { → 不算调用(只是正文里提到了这个格式)。"""
    assert names("@shell(c1): 这只是普通提及") == []
    # { 必须和冒号同一行:换行后才开始的大括号不认(解析按"一行一个调用"设计)
    assert names('@shell(c1):\n{"command": "ls"}') == []


def test_native_spaces_before_json_allowed() -> None:
    """冒号和 { 之间可以隔空格(模型写调用行不一定紧贴着写)。"""
    uses = list(ToolUse.iter_from_content('@save(call_1):   {"path": "a.txt"}'))
    assert len(uses) == 1
    assert uses[0].kwargs == {"path": "a.txt"}


def test_native_inside_unclosed_fence_skipped() -> None:
    """没闭合的围栏一路占到文末,里面的 @ 调用同样当示例跳过。

    (已闭合围栏的情况上面测过;未闭合时围栏区间算到文本末尾,
    所以之后的整段都算"示例区"。)
    """
    assert names('示例:\n```\n@shell(c1): {"command": "ls"}') == []


def test_native_dotted_tool_name() -> None:
    """工具名带点号(MCP 远端工具的 "服务器名.工具名" 前缀)也要认。

    名字里的点只出现在工具名部分,和编号部分的点是两回事;
    MCP 的端到端调用在 integration/test_mcp.py,这里锁解析这一层。
    """
    from my_agent_alpha.tools import add_tools

    spec = ToolSpec(name="t.echo", desc="测试用远端工具", block_types=("t.echo",))
    add_tools((spec,))
    uses = list(ToolUse.iter_from_content('@t.echo(c1): {"text": "hi"}'))
    assert len(uses) == 1
    assert uses[0].tool.name == "t.echo"
    assert uses[0].kwargs == {"text": "hi"}
    assert uses[0].call_id == "c1"


# ---- 解析边角(来自旧 test_tools.py)----


def test_iter_from_content_maps_fences(workspace: Path) -> None:
    content = "我来执行:\n```shell\necho hi\n```\n再读文件:\n```read src/a.py\na.py\n```"
    uses = list(ToolUse.iter_from_content(content))
    assert [u.tool.name for u in uses] == ["shell", "read"]
    assert uses[0].content == "echo hi"
    assert uses[1].args == ["src/a.py"]
    assert uses[1].content == "a.py"


# ---- 审查 #8:CRLF 归一化 ----


def test_streaming_fence_needs_blank_line() -> None:
    """streaming=True 更严:闭合 ``` 后面还得跟一个空行,调用才算完整。

    规则本体在 codeblock 模块(那边有专项测试);这里锁的是
    iter_from_content 把这个开关原样透传下去、两种模式各司其职。
    """
    complete = "```shell\necho hi\n```\n\n后文"
    assert len(list(ToolUse.iter_from_content(complete, streaming=True))) == 1
    no_blank = "```shell\necho hi\n```"
    assert list(ToolUse.iter_from_content(no_blank, streaming=True)) == []
    # 默认(非流式)没有空行也认:执行时生成已结束,不需要额外确认
    assert len(list(ToolUse.iter_from_content(no_blank))) == 1


def test_crlf_fence_parsed_clean() -> None:
    content = "执行:\r\n```shell\r\nls\r\n```\r\n"
    uses = list(ToolUse.iter_from_content(content))
    assert len(uses) == 1
    assert uses[0].content == "ls"  # 不带 \r


# ---- 参数表 → JSON Schema(原生通道发请求时的工具描述)----


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
    chatty = ToolSpec(name="chatty", desc="x" * 2000, block_types=("chatty",))
    payload = spec_to_openai_tool(chatty)
    assert len(str(payload["function"]["description"])) <= 1001  # 1000 + 省略号


# ---- 工具说明书渲染(进系统提示词的文档)----


def test_tool_prompt_renders() -> None:
    shell = get_tool("shell")
    assert shell is not None
    doc = shell.get_tool_prompt()
    assert "## shell" in doc
    assert "```shell" in doc
