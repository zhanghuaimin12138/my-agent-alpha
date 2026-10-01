"""系统提示词组装:模型每次对话开头看到的"说明书"。

说明书分两条消息,顺序有讲究:

1. 静态部分(身份 + 工具调用规则 + 每个工具的用法示例):
   每次会话内容都一样 —— 放在前面,服务商能对它做缓存,省钱;
2. 动态部分(当前时间):每次都变 —— 放在最后,不让它弄脏前面的缓存。

两条都标了 pinned(将来压缩上下文时不删)和 hide(发给模型但不打印到终端)。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from my_agent_alpha.message import Message, utcnow
from my_agent_alpha.tools.base import ToolSpec

# 静态提示词正文(markdown 通道):告诉模型它是谁、怎么用围栏代码块调工具
_IDENTITY = """\
# 身份

你是 my-agent-alpha,一个运行在用户终端里的个人 AI 助手。
你可以通过在回复中输出 markdown 围栏代码块来调用工具。

# 工具调用协议

1. 需要执行操作时,输出一个围栏代码块,语言标记 = 工具名,例如:

```shell
ls -la
```

2. 带路径参数的工具把路径写在语言标记后面:

```save path/to/file.txt
文件完整内容
```

3. 每个代码块就是一次工具调用;调用后等待结果(以 system 消息返回),
   再根据结果决定下一步。
4. 任务完成(或无需工具)时,用自然语言直接回答 —— 不要输出代码块。
5. 多步任务一步步来:每一步等结果、看结果、再走下一步。不要臆测工具输出。
"""

# 静态提示词正文(原生通道):工具走接口的 function calling,
# 不用教围栏约定,只说明结果怎么回来
_IDENTITY_NATIVE = """\
# 身份

你是 my-agent-alpha,一个运行在用户终端里的个人 AI 助手。
你可以通过函数调用(function calling)使用下面列出的工具。

# 工具使用须知

1. 需要执行操作时,发起对应的函数调用;调用后等待结果
   (以 system 消息返回),再根据结果决定下一步。
2. 任务完成(或无需工具)时,用自然语言直接回答。
3. 多步任务一步步来:每一步等结果、看结果、再走下一步。不要臆测工具输出。

# 可用工具(参数说明见函数定义)
"""


def get_prompt(
    tools: Sequence[ToolSpec], *, now: datetime | None = None, native: bool = False
) -> list[Message]:
    """生成会话开头的系统提示词消息(两条),调用方把它们写进日志一次即可。

    native=True 用原生工具调用版说明(围栏约定不用教,参数说明在函数定义里);
    False 用 markdown 版(教模型写围栏代码块)。

    示例:返回
    [Message("system", 身份+工具文档, pinned=True, hide=True),
     Message("system", "# 当前时间\\n2026-09-29 10:00:00 UTC", pinned=True, hide=True)]
    """
    tool_docs = "\n\n---\n\n".join(t.get_tool_prompt() for t in tools)
    identity = _IDENTITY_NATIVE if native else _IDENTITY
    parts = [identity]
    if tool_docs and not native:
        parts.append("# 可用工具\n\n" + tool_docs)
    static = Message("system", "\n\n".join(parts), pinned=True, hide=True)

    dynamic = Message(
        "system",
        f"# 当前时间\n{(now or utcnow()).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        pinned=True,
        hide=True,
    )
    return [static, dynamic]
