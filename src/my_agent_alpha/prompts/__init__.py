"""系统提示词组装(最小版)。

仿 gptme 的分层原则:静态内容(身份 + 工具文档)在前,易变内容(时间)最后,
中间是天然的 prompt-cache 边界(阶段 2 接 apply_cache_control 时直接可用)。
所有提示词消息 pinned + hide:发给模型、永不裁剪、不打印。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from my_agent_alpha.message import Message, utcnow
from my_agent_alpha.tools.base import ToolSpec

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


def get_prompt(tools: Sequence[ToolSpec], *, now: datetime | None = None) -> list[Message]:
    """组装系统提示词消息(调用方在会话开头追加一次进日志)。"""
    tool_docs = "\n\n---\n\n".join(t.get_tool_prompt() for t in tools)
    parts = [_IDENTITY]
    if tool_docs:
        parts.append("# 可用工具\n\n" + tool_docs)
    static = Message("system", "\n\n".join(parts), pinned=True, hide=True)

    dynamic = Message(
        "system",
        f"# 当前时间\n{(now or utcnow()).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        pinned=True,
        hide=True,
    )
    return [static, dynamic]
