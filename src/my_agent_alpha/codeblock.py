"""markdown 围栏代码块解析:从模型的回复文本里找出"工具调用"。

模型的回复是 markdown 文本,其中形如

    ```shell
    ls -la
    ```

的三反引号块,就是我们约定好的"工具调用":语言标记是工具名,块内是要执行的内容。
本模块负责把这种块从文本里解析出来。

解析规则(仿 gptme):
- 块头 = 工具名 + 可选参数。例如 ```save path/to/file```
  解析成 lang="save"(工具名)、args="path/to/file"(参数,比如文件路径);
- 围栏可以更长(````),闭合围栏至少和开头一样长、且行内没有别的内容;
- 没闭合的块直接丢弃 —— 只执行写完整的块,绝不执行半截命令;
- 行首有缩进也能识别(模型偶尔会把代码块缩进)。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

# 匹配一行开头的三根及以上反引号,后面可跟"语言 参数"
_FENCE_RE = re.compile(r"^(`{3,})\s*(.*?)\s*$")


@dataclass(frozen=True)
class Codeblock:
    """一个完整的围栏代码块。

    示例:文本里的 "```save a.txt\\n内容\\n```" 解析为
    lang="save", args="a.txt", content="内容"。
    start 是这个块在原文中的起始位置(第几个字符),用来给多个调用排序。
    """

    lang: str
    args: str
    content: str
    start: int


def iter_from_markdown(text: str, *, streaming: bool = False) -> Iterator[Codeblock]:
    """按出现顺序逐个产出文本里所有"写完整了"的代码块。

    示例:
        list(iter_from_markdown("说明\\n```shell\\nls\\n```"))
        → [Codeblock(lang="shell", args="", content="ls", ...)]

    streaming=True 只在"边生成边检查"的场景使用(见 llm.reply 的断点检测):
    此时闭合的 ``` 后面必须再跟一个空行,块才算写完。
    为什么?生成到一半时看到 "```",它可能是真闭合,也可能只是
    更长代码块里的内层围栏 —— 等到下一行是空行,才能确定这个块真的结束了。
    执行工具时(生成已结束)用默认 streaming=False,不需要这个额外确认。
    """
    text = text.replace("\r\n", "\n")  # Windows 换行符(\r\n)统一成 \n,否则命令尾部会带看不见的 \r
    lines = text.split("\n")
    offset = 0  # 当前行的起始字符偏移
    in_block = False
    open_ticks = 0
    block_start = 0
    lang = ""
    args = ""
    body: list[str] = []

    for i, line in enumerate(lines):
        match = _FENCE_RE.match(line.lstrip())
        if not in_block:
            if match:  # 开栏:记住围栏长度和块头,开始收集正文
                in_block = True
                block_start = offset
                open_ticks = len(match.group(1))
                lang, _, args = match.group(2).partition(" ")
                body = []
        elif match and len(match.group(1)) >= open_ticks and not match.group(2):
            # 闭合栏候选:至少等长且行内无其他内容
            if streaming and not _has_blank_line_after(lines, i):
                # 流式模式:闭合栏后还没有空行 → 不能确认写完,
                # 这行暂且当正文(它可能是更长代码块的内栏)
                body.append(line)
            else:
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

    # 循环结束还在块里 → 块没闭合 → 丢弃(不产出)


def _has_blank_line_after(lines: list[str], close_index: int) -> bool:
    """看第 close_index 行(闭合栏)的下一行是不是空行;没有下一行返回 False。"""
    if close_index + 1 >= len(lines):
        return False
    return lines[close_index + 1].strip() == ""


def fenced_regions(text: str) -> list[tuple[int, int]]:
    """找出所有围栏代码块在原文中的字符区间 [(起点, 终点), ...]。

    用途:
    1.原生工具调用模式：
        在围栏代码块里的"@save(...): {...)",要跳过，这是示例，
        在围栏代码块外的，是真正的工具调用
    2.markdown格式工具调用模式:
        在围栏代码块里的是工具调用
    
    没闭合的围栏也算一个区间(同样要跳过)。

    示例:fenced_regions("说明\\n```shell\\nls\\n```\\n后文")
    → [(3, 23)]   ← "```shell" 那行开头到 "```" 那行结尾
    """
    text = text.replace("\r\n", "\n")
    regions: list[tuple[int, int]] = []
    offset = 0  # 当前行的起始字符偏移
    in_block = False
    open_ticks = 0
    start = 0
    for line in text.split("\n"):
        match = _FENCE_RE.match(line.lstrip())
        if not in_block:
            if match:
                in_block = True
                open_ticks = len(match.group(1))
                start = offset
        elif match and len(match.group(1)) >= open_ticks and not match.group(2):
            regions.append((start, offset + len(line)))
            in_block = False
        offset += len(line) + 1  # +1 是被 split 吃掉的换行符
    if in_block:  # 没闭合的围栏:一路占到文末
        regions.append((start, len(text)))
    return regions
