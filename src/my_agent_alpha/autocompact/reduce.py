"""兜底压缩器:三阶段引擎跑完还是超预算时的最后手段。

它非常保守,只做一件事:反复找出"最长的、没被钉住的、不含工具调用的"
消息,把它里面超过 21 行的代码块砍成"前 10 行 + [...] + 后 10 行"。
砍完重数 token,达标就停;哪一轮一点都砍不动了(没有可砍的内容)
就原样返回 —— 宁可超出预算,也不硬删消息、不弄丢工具调用的配对。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import replace

from my_agent_alpha.autocompact.contracts import TokenCounter
from my_agent_alpha.message import Message

logger = logging.getLogger(__name__)

# 超过这么多行的代码块才砍;保留的头尾行数
_BLOCK_LINES_THRESHOLD = 21
_BLOCK_KEEP_HEAD = 10
_BLOCK_KEEP_TAIL = 10

# 一个完整围栏代码块(``` 开到 ``` 收,跨行)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")


def _truncate_block(block: str) -> str | None:
    """尝试砍一个代码块:超过 21 行时留前 10 + [...] + 后 10 行。

    不到 21 行(或没有多行正文)返回 None 表示"这个块不用动"。
    """
    m = re.match(r"```[^\n]*\n([\s\S]*?)\n?```$", block)
    if m is None:
        return None  # 单行块(如 "```python```")没有可砍的正文
    lines = m.group(1).split("\n")
    if len(lines) <= _BLOCK_LINES_THRESHOLD:
        return None
    header = block[: block.index("\n")]  # 开栏那行(含语言标记)
    kept = [*lines[:_BLOCK_KEEP_HEAD], "[...]", *lines[len(lines) - _BLOCK_KEEP_TAIL :]]
    return header + "\n" + "\n".join(kept) + "\n```"


def truncate_msg(msg: Message, contains_tool_use: Callable[[Message], bool]) -> Message | None:
    """尝试截短一条消息:砍其中的超长代码块。无可砍内容返回 None。

    含工具调用的消息直接返回 None(不可动):截了它,工具调用和
    工具结果的配对就被破坏了。消息本身不可变,这里造新实例返回。
    """
    if contains_tool_use(msg):
        return None
    changed = False

    def _sub(match: re.Match[str]) -> str:
        nonlocal changed
        new_block = _truncate_block(match.group(0))
        if new_block is None:
            return match.group(0)
        changed = True
        return new_block

    new_content = _CODE_BLOCK_RE.sub(_sub, msg.content)
    if not changed:
        return None
    return replace(msg, content=new_content)


def reduce_log(
    messages: list[Message],
    limit: int,
    *,
    counter: TokenCounter,
    contains_tool_use: Callable[[Message], bool],
) -> list[Message]:
    """把消息列表压到 limit 以内(尽力而为,压不动就原样返回)。

    每一轮:找"没钉住、不含工具调用"的消息里 content 最长的那条,
    截短它的超长代码块,原位替换(条数、顺序都不变),重数 token:
    - 达标 → 返回;
    - 没达标但总量在降 → 再来一轮(递归);
    - 一点都砍不动 / 没有候选 → 原样返回,打条告警。

    示例:reduce_log(msgs, 500, counter=计数器, contains_tool_use=探测器)
    """
    before = counter.len_tokens(messages)
    if before <= limit:
        return messages
    candidates = [m for m in messages if not m.pinned and not contains_tool_use(m)]
    if not candidates:
        logger.warning("reduce_log:没有可压缩的候选消息(全被钉住或含工具调用)")
        return messages
    longest = max(candidates, key=lambda m: counter.len_tokens(m.content))
    truncated = truncate_msg(longest, contains_tool_use)
    if truncated is None:
        return messages  # 最长的都砍不动:不再有进展,止损返回
    replaced = [truncated if m is longest else m for m in messages]  # 原位替换
    after = counter.len_tokens(replaced)
    if after <= limit:
        return replaced
    if after < before:
        return reduce_log(replaced, limit, counter=counter, contains_tool_use=contains_tool_use)
    return messages
