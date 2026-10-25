"""阶段 1 的实现:剥掉模型回复里的"思考过程"。

模型常常把推理过程包在 <think>...</think>(或 <thinking>...</thinking>)
里和正文一起输出。思考过程是"草稿" —— 模型得到结论后,旧回复里的
草稿对后续对话就没用了,白白占上下文。这个模块负责把旧消息里的
思考区删掉,只留正文。

纯函数,零依赖:不碰文件、不碰注册表,给谁都能用。
"""

from __future__ import annotations

import re
from collections.abc import Callable

# 两种思考区标签(Gemini 用 <thinking> 写法),各自完整匹配、不跨标签混搭
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)
# 三个及以上的连续空行压成两个(删掉思考区后经常留下大片空行)
_BLANK_LINES_RE = re.compile(r"\n\n\n+")

# 数 token 的函数类型:吃一段文本,返回 token 数
TokenCount = Callable[[str], int]


def strip_reasoning(content: str, count_tokens: TokenCount) -> tuple[str, int]:
    """删掉文本里的思考区,返回(新文本, 省下的 token 数)。

    处理顺序:
    1. 删 <think>...</think> 和 <thinking>...</thinking>(跨行也删);
    2. 把连续 3 行以上的空行压成 1 行空行(即最多留一个空行);
    3. 去掉首尾空白;
    4. 用传入的计数函数对比删前删后的 token 数。

    示例:
        strip_reasoning("<think>推理…</think>结论是 42", count)
        → ("结论是 42", 省下的 token 数)
    """
    stripped = _THINK_RE.sub("", content)
    stripped = _THINKING_RE.sub("", stripped)
    stripped = _BLANK_LINES_RE.sub("\n\n", stripped)
    stripped = stripped.strip()
    saved = count_tokens(content) - count_tokens(stripped)
    return stripped, saved
