"""memory 工具:让模型"记一条"(经用户确认才写入)。

用法(markdown 通道,围栏头是记忆名,内容首行做标题):

    ```memory 部署流程
    服务器登录和发布步骤
    1. ssh admin@10.0.0.1
    ```

内容开头还可以带一段 frontmatter 把描述和类型写清楚(推荐,
markdown 通道没有参数位,描述写在这里才进索引):

    ```memory 部署流程
    ---
    description: 服务器登录与发布步骤,含 systemctl 重启命令
    type: project
    ---
    服务器发布步骤
    1. ssh admin@10.0.0.1
    ```

原生通道:{"name": "部署流程", "content": "...", "type": "project",
"description": "..."}(type/description 参数优先于内容里的 frontmatter)。

不是只读工具 → 自动走确认链:交互模式弹预览给用户点头,
非交互/-y 自动放行(和 gptme 的"模型提议、用户确认"一致)。
"""

from __future__ import annotations

from collections.abc import Iterator

from my_agent_alpha.context import get_workspace
from my_agent_alpha.memory_store import MEMORY_TYPES, save_memory
from my_agent_alpha.message import Message
from my_agent_alpha.tools.base import Parameter, ToolSpec

# 提示词协议(M1.4,仿 CC 报告 §2.4/§2.5):教模型"什么值得记"。
# 这段进系统提示词,是记忆质量的第一道闸 —— 排除清单和"从成功中学习"
# 都在这里;没有它,模型迟早把代码结构这类可推导的东西塞进记忆。
_PROTOCOL = """\
内容 = 记忆正文,首行做标题;正文开头可带 frontmatter 写 description/type(推荐)。
什么值得记(四类):
- user:用户的角色、偏好、知识背景;
- feedback:用户对工作方式的纠正或确认("不要 X"和"对,就这样"都算——只记纠正会过度保守);
- project:进行中的工作、决策及其动机(代码和 git 里查不到的);
- reference:外部资源指针(看板、文档库、服务器地址)。
不该记(读现状就能知道,记了反而污染索引):代码结构、文件路径、git 历史、
修复过程、AGENTS.md 已有内容、临时任务状态。
纪律:相对日期写成绝对日期("上周"→ 具体日期);存之前先看会话里的记忆索引,
有近似条目就用同名覆盖更新,别建重复;description 要具体——以后全靠它判断相关性。"""


def execute_memory(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> Iterator[Message]:
    """把一条记忆写进长期记忆(默认项目层:工作区/memory/,记忆跟着项目走)。"""
    # 围栏头可能是裸名字,也可能顺手带了类型("```memory 偏好 user")
    parts = args[0].split() if args and args[0] else []
    name = (kwargs.get("name") if kwargs else None) or (parts[0] if parts else "")
    content = (kwargs.get("content") if kwargs else None) or code or ""
    mem_type = (kwargs.get("type") if kwargs else None) or (parts[1] if len(parts) > 1 else None)
    description = kwargs.get("description") if kwargs else None
    if not name or not content.strip():
        yield Message("system", "错误: memory 需要 名称(围栏头/name 参数)和 内容")
        return
    if mem_type is not None and mem_type not in MEMORY_TYPES:
        yield Message(
            "system", f"错误: type 必须是 {'/'.join(MEMORY_TYPES)} 之一,收到 {mem_type!r}"
        )
        return
    path = save_memory(
        name,
        content,
        get_workspace(),
        project_level=True,
        mem_type=mem_type,
        description=description,
    )
    yield Message("system", f"已记住:{path.name}(写入 {path.parent})")


tool = ToolSpec(
    name="memory",
    desc="把一条跨会话有用的信息存进长期记忆,以后每次会话都能看到(需要用户确认)。",
    instructions=_PROTOCOL,
    examples=(
        "```memory 部署流程\n服务器发布步骤\n1. ssh admin@10.0.0.1\n```\n\n"
        "推荐带 frontmatter(描述进索引,以后好找):\n"
        "```memory 部署流程\n---\ndescription: 服务器登录与发布步骤,"
        "含 systemctl 重启命令\ntype: project\n---\n服务器发布步骤\n1. ssh admin@10.0.0.1\n```"
    ),
    block_types=("memory",),
    parameters=(
        Parameter("name", "string", "记忆名(短,能认出内容)"),
        Parameter(
            "content", "string", "记忆内容:首行标题;开头可带 description/type 的 frontmatter"
        ),
        Parameter(
            "type", "string", "user/feedback/project/reference 四选一,默认 project", required=False
        ),
        Parameter(
            "description", "string", "一行具体描述(以后靠它判断相关性),不填用标题", required=False
        ),
    ),
    execute=execute_memory,
)
