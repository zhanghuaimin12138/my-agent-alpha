"""压缩决策层:回答"现在要不要压缩?用哪种方式压缩?"

两种压缩方式,代价完全不同:
- rule_based(规则压缩):纯本地计算,剥思考/截工具结果/压长回复,
  快、免费,但只能省下"明显冗余"的部分;
- summarize(LLM 摘要):把整个会话发给模型写一份摘要,用摘要替换历史,
  压得最狠,但要花一次模型调用,而且必然让服务商的缓存全部失效。

所以决策逻辑是:先看"总量是否接近窗口"(不接近就不动);再看
"规则压缩大概能省多少"(干跑估算,不动手) —— 能省下 10% 以上就走
规则压缩,省不出这么多(说明内容都是"干货",规则剪不动)才请模型摘要。

四条阈值线(0.5 / 0.8 / 0.9 × 窗口)故意不一样:决策层要抢在
爆窗之前动手,所以线设得激进;这是刻意设计,不要"统一参数"。
"""

from __future__ import annotations

from typing import Literal

from my_agent_alpha.autocompact.contracts import (
    HUGE_TOOL_RESULT_TOKENS,
    TokenCounter,
    is_tool_result,
)
from my_agent_alpha.message import Message

# 三种决策结果
CompactAction = Literal["none", "rule_based", "summarize"]

# 规则压缩的"最低性价比":估算能省下的量占总量的比例不到这个数,
# 就认为规则压缩不值得(缓存失效的代价大过省下的钱),改走 LLM 摘要
MIN_SAVINGS_RATIO = 0.10


def should_auto_compact(
    messages: list[Message], *, context_window: int, counter: TokenCounter
) -> CompactAction:
    """判断当前要不要自动压缩,返回三种决策之一。

    判定顺序(见模块说明):
    1. 总量超过 0.9 × 窗口 → 候选(真的快爆了);
    2. 或者:总量超过 0.5 × 窗口(接近了)且存在"巨型工具结果"
       (没被钉住、超过 2000 token 的工具输出)→ 候选(提前动手);
    3. 是候选后,干跑估算规则压缩的节省量:能省 10% 以上 → rule_based,
       否则 → summarize。

    示例:should_auto_compact(msgs, context_window=128000, counter=计数器)
    """
    limit = int(0.9 * context_window)
    total = counter.len_tokens(messages)
    close = total >= int(0.5 * context_window)
    massive = any(
        not m.pinned
        and is_tool_result(m)
        and counter.len_tokens(m.content) > HUGE_TOOL_RESULT_TOKENS
        for m in messages
    )
    if not (total > limit or (close and massive)):
        return "none"

    _, savings, _ = estimate_compaction_savings(
        messages, limit=limit, context_window=context_window, counter=counter
    )
    ratio = savings / total  # total > 0:上面 total > limit 或 close 至少一个为真
    return "rule_based" if ratio >= MIN_SAVINGS_RATIO else "summarize"


def estimate_compaction_savings(
    messages: list[Message],
    *,
    limit: int,
    context_window: int,
    counter: TokenCounter,
) -> tuple[int, int, int]:
    """干跑估算:规则压缩大约能省多少 token?不真正改动任何消息。

    按三阶段各自的启发式估:
    - 阶段 1(剥思考):距末尾 ≥5 条、含思考区的消息,按内容量的 30% 估;
    - 阶段 2(截工具结果):只在"总量已经超 0.8 × 窗口"时会计入,
      每条巨型工具结果按"原量 − 200"(截断占位的大小)估;
    - 阶段 3(压长回复):同样只在超 0.8 线时计入,距末尾 ≥3 条、
      超过 1000 token 的模型回复,按 30% 估。

    返回 (总量, 三阶段合计节省量, 其中剥思考的节省量)。
    """
    total = counter.len_tokens(messages)
    # limit 传 0/None 时用默认工作上限(0.8 × 窗口)
    eff_limit = limit or int(0.8 * context_window)
    # "会触发阶段 2/3"的判定线:总量超 limit,或已达 0.8 × 窗口
    would_remove = total > eff_limit or total >= int(0.8 * context_window)

    n = len(messages)
    savings_tool = 0
    savings_reason = 0
    savings_compress = 0
    for i, msg in enumerate(messages):
        if msg.pinned:
            continue  # 钉住的消息任何阶段都不动,自然省不出东西
        tok = counter.len_tokens(msg.content)
        dist = n - i - 1  # 距末尾还有几条
        if dist >= 5 and ("<think>" in msg.content or "<thinking>" in msg.content):
            savings_reason += int(tok * 0.3)
        if would_remove and is_tool_result(msg) and tok > HUGE_TOOL_RESULT_TOKENS:
            savings_tool += tok - 200
        if would_remove and dist >= 3 and msg.role == "assistant" and tok > 1000:
            savings_compress += int(tok * 0.3)
    return total, savings_tool + savings_reason + savings_compress, savings_reason
