"""markdown 围栏代码块(fenced codeblock)解析。

这是"markdown 工具调用格式"的读取端:assistant 消息里的

    ```shell
    ls -la
    ```

会被解析成 Codeblock(lang="shell", args="", content="ls -la")。

约定(仿 gptme):
- fence 头 = 语言标记 + 可选参数,如 ```save path/to/file``` →
  lang="save", args="path/to/file"(工具调用的位置参数,如文件路径);
- 围栏可变长(```、````),闭合围栏至少与开栏等长、且其后无内容;
- 未闭合的围栏一律丢弃 —— 只执行写完整的块,绝不执行半截命令
  (阶段 2 流式检测的关键规则在本阶段就以非流式形式生效);
- fence 检测容忍行首缩进(LLM 偶尔缩进代码块)。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

_FENCE_RE = re.compile(r"^(`{3,})\s*(.*?)\s*$")


@dataclass(frozen=True)
class Codeblock:
    """一个完整的围栏代码块。start 为其开栏行在原文中的字符偏移(调用排序用)。"""

    lang: str
    args: str
    content: str
    start: int


def iter_from_markdown(text: str) -> Iterator[Codeblock]:
    """按出现顺序产出所有完整围栏块;未闭合的块被丢弃。

    入口先做 CRLF 归一化(审查 #8):否则内容每行带 \\r,
    shell 命令会变成 "ls\\r" 而执行失败。
    """
    text = text.replace("\r\n", "\n")
    offset = 0  # 当前行的起始字符偏移
    in_block = False
    open_ticks = 0
    block_start = 0
    lang = ""
    args = ""
    body: list[str] = []

    for line in text.split("\n"):
        match = _FENCE_RE.match(line.lstrip())
        if not in_block:
            if match:  # 开栏
                in_block = True
                block_start = offset
                open_ticks = len(match.group(1))
                lang, _, args = match.group(2).partition(" ")
                body = []
        elif match and len(match.group(1)) >= open_ticks and not match.group(2):
            # 闭合栏:至少等长且行内无其他内容
            yield Codeblock(
                lang=lang.strip(),
                args=args.strip(),
                content="\n".join(body),
                start=block_start,
            )
            in_block = False
        else:
            body.append(line)
        offset += len(line) + 1  # +1 是被 split 吃掉的换行符

    # 循环结束仍 in_block:未闭合 → 丢弃(不产出)
