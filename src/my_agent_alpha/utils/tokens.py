"""token 估算工具 —— 无分词器版(方案见 docs/token-counting-dev.md)。

估算规则:每个中日韩字符算 1 个 token,其余字符每 4 个算 1 个(向上取整)。

为什么故意往高了估:高估的代价只是"提前压缩"(浪费一点窗口);
低估的代价是"以为装得下、实际装不下",请求到接口那里直接报错。
实测(cl100k 分词表):纯中文约 1.0 字符/token,英文约 5.2,代码约 4.5 ——
所以"中文每字 1 个 + 其余除以 4"恰好落在略高估的安全区。

带一层缓存(key = 内容的 SHA-256):同一份内容反复数(压缩循环里
每轮都要数全量)不用重复算;满了删最早插入的,不加锁 ——
最坏情况是并发时重复算一次,不会算错。

和 compaction.py 的关系:那边只管"怎么压"(压缩策略),
这边只管"数多少"(数出来多少 token、历史最多能占多少)。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from my_agent_alpha.message import Message

# 缓存上限:超过就删最早插入的那条(照搬 gptme 的经验值)
CACHE_MAX = 1000
# 非中日韩字符的系数:每 4 个字符算 1 个 token
CHARS_PER_TOKEN = 4

# 缓存:内容哈希 -> token 数。不加锁:CPython 里 dict 单操作是原子的,
# 并发时最坏是两个线程各算一次,结果一样,不会出错
_token_cache: dict[str, int] = {}


def _is_cjk(ch: str) -> bool:
    """是否中日韩字符(含全角标点、日文假名)。"""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF  # 汉字基本区
        or 0x3400 <= cp <= 0x4DBF  # 汉字扩展A
        or 0x3040 <= cp <= 0x30FF  # 日文假名
        or 0x3000 <= cp <= 0x303F  # 中文标点(,。、)
        or 0xFF00 <= cp <= 0xFFEF  # 全角字符(!?:)
    )


def _estimate(text: str) -> int:
    """纯函数:按"中日韩每字 1 个,其余每 4 个 1 个"估算 token 数。

    示例:"the" → 1(3 个字符凑不满 4 个也算 1);"中文" → 2。
    """
    cjk = sum(1 for ch in text if _is_cjk(ch))
    rest = len(text) - cjk
    # 向上取整:宁可多算,不少算
    return cjk + -(-rest // CHARS_PER_TOKEN)


def len_tokens(content: str | Message | Sequence[Message]) -> int:
    """数 token(估算版),结果进缓存。

    content 可以是:
      - str        直接数这段文字;
      - Message    数它的 content 字段(别的字段不数);
      - 消息列表   逐条相加。
    空内容返回 0。

    示例:len_tokens("你好") == 2;len_tokens([msg1, msg2]) 是两条之和。
    """
    # 消息列表:逐条相加(str 也是 Sequence,要先排除掉)
    if isinstance(content, Sequence) and not isinstance(content, str):
        return sum(len_tokens(m) for m in content)
    if isinstance(content, Message):
        content = content.content
    if not content:
        return 0

    key = hashlib.sha256(content.encode()).hexdigest()  # 哈希做 key:不存原文
    cached = _token_cache.get(key)
    if cached is not None:
        return cached

    count = _estimate(content)
    if len(_token_cache) >= CACHE_MAX:
        # 满了:删最早插入的那条(dict 按插入序遍历,第一个就是最早的)
        _token_cache.pop(next(iter(_token_cache)))
    _token_cache[key] = count
    return count


def get_context_budget(context_window: int, max_output: int = 8192, headroom: int = 1000) -> int:
    """计算"历史最多能占多少 token"(预算)。

    两个约束取更严的(小的)那个:
    - 窗口的 9 成:任何窗口都不用到极限;
    - 窗口 - max_output(预留回复空间,默认 8192)- headroom
      (安全余量 1000,兜住 token 估算的误差)。

    示例:128k 窗口 → min(115200, 118808) = 115200;
         32k 窗口 → min(28800, 22808)  = 22808;
         8k 窗口  → 减完是负数,走保底 1000(至少塞得下提示词+最近几条,
                                          代价是回复可能被截断,能跑比崩强)。
    """
    return max(min(int(context_window * 0.9), context_window - max_output - headroom), 1000)
