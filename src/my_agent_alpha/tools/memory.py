"""memory 工具:让模型"记一条"(经用户确认才写入)。

用法(markdown 通道,fence 头是记忆名,内容首行做标题):
    ```memory 部署流程
    服务器登录和发布步骤
    1. ssh admin@10.0.0.1
    2. systemctl restart app
    ```
原生通道:{"name": "部署流程", "content": "..."}。

不是只读工具 → 自动走确认链:交互模式弹预览给用户点头,
非交互/-y 自动放行(和 gptme 的"模型提议、用户确认"一致)。
"""

from __future__ import annotations

from collections.abc import Iterator

from my_agent_alpha.context import get_workspace
from my_agent_alpha.memory_store import save_memory
from my_agent_alpha.message import Message
from my_agent_alpha.tools.base import Parameter, ToolSpec


def execute_memory(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> Iterator[Message]:
    """把一条记忆写进长期记忆(默认项目层:工作区/memory/,记忆跟着项目走)。"""
    name = (kwargs.get("name") if kwargs else None) or (args[0] if args else "")
    content = (kwargs.get("content") if kwargs else None) or code or ""
    if not name or not content.strip():
        yield Message("system", "错误: memory 需要 名称(fence 头/name 参数)和 内容")
        return
    path = save_memory(name, content, get_workspace(), project_level=True)
    yield Message("system", f"已记住:{path.name}(写入 {path.parent})")


tool = ToolSpec(
    name="memory",
    desc="把一条重要信息存进长期记忆,以后每次会话都能看到(需要用户确认)。",
    instructions="名字要短且能认出内容;内容首行是标题,之后是正文。",
    examples="```memory 部署流程\n服务器发布步骤\n1. ssh admin@10.0.0.1\n```",
    block_types=("memory",),
    parameters=(
        Parameter("name", "string", "记忆名(短,能认出内容)"),
        Parameter("content", "string", "记忆内容,首行做标题"),
    ),
    execute=execute_memory,
)
