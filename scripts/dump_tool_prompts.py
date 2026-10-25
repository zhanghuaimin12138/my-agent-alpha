"""把所有工具的"说明书"导出到两个文件,方便人眼审阅模型到底看到了什么。

模型了解一个工具有两条渠道,对应本仓库的两条调用通道:

1. 原生模型(function calling):请求体里的 tools 参数 —— 每个工具是一段
   JSON(名字 + 说明 + 参数表),由 spec_to_openai_tool 生成,
   会发到接口那头给模型看;
2. markdown 约定:拼进系统提示词的一段文字(名字 + 说明 + 示例),
   由 ToolSpec.get_tool_prompt 生成,只发给不支持原生调用的模型。

用法(uv run 会自动把 src 里的包装进环境):

    uv run python scripts/dump_tool_prompts.py                # 写到当前目录
    uv run python scripts/dump_tool_prompts.py a.json b.md    # 指定两个输出路径
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from my_agent_alpha.tools import get_tools
from my_agent_alpha.tools.base import ToolSpec, spec_to_openai_tool

# 不传参数时的默认输出文件名(写到当前目录)
DEFAULT_NATIVE_PATH = Path("tools_native.json")
DEFAULT_MARKDOWN_PATH = Path("tools_markdown.md")


def render_native(tools: tuple[ToolSpec, ...]) -> str:
    """把工具表渲染成请求里 tools 参数的模样(美化过的 JSON 文本)。

    和 llm/openai_compat.py 真正发请求时一样,只收可执行的工具
    (没有执行函数的"纯文档"工具不会发给模型)。

    示例:输出开头形如
    [
      {
        "type": "function",
        "function": { "name": "shell", "description": "...", "parameters": {...} }
      },
      ...
    ]
    """
    payload = [spec_to_openai_tool(t) for t in tools if t.is_runnable]
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def render_markdown(tools: tuple[ToolSpec, ...]) -> str:
    """把每个工具的 markdown 说明拼成一整篇(拼法和系统提示词里一致)。

    工具之间用 --- 分隔,开头加一个带数量的标题方便核对总数。
    """
    sections = "\n\n---\n\n".join(t.get_tool_prompt() for t in tools)
    return f"# 可用工具(共 {len(tools)} 个)\n\n{sections}\n"


def main(argv: list[str]) -> None:
    """入口:取全部工具,写成两个文件(路径可用第 1、2 个参数覆盖)。"""
    tools = get_tools()
    native_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_NATIVE_PATH
    md_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_MARKDOWN_PATH
    native_path.write_text(render_native(tools), encoding="utf-8")
    md_path.write_text(render_markdown(tools), encoding="utf-8")
    print(f"共 {len(tools)} 个工具(原生格式只含可执行的)")
    print(f"已写入 {native_path} —— 原生模型的 tools 参数")
    print(f"已写入 {md_path} —— markdown 版工具说明")


if __name__ == "__main__":
    main(sys.argv)
