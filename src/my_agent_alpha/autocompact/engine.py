"""三阶段压缩引擎:把一份消息列表压到预算以内(核心算法)。

三个阶段,由便宜到贵,顺序执行(规格见 docs/autocompact-dev.md §5.3):

阶段 1 剥思考:距末尾 ≥5 条的旧消息,删掉 <think> 思考区(草稿)。
阶段 2 截工具结果:总量超线时,把巨型工具输出(>2000 token 的 system
        消息)从最大的开始换成截断占位(全文先落盘,附 master 字节引用)。
阶段 3 压长回复:距末尾 ≥3 条、超 1000 token、不含工具调用的模型回复,
        做抽取式压缩(按句子打分保留七成,代码块完整保留)。

最重要的纪律:**所有阶段都是原位替换** —— 只改某几条消息的内容,
绝不增删消息、不换顺序。这样压缩后的第 i 条消息永远对应主日志的
第 i 行,字节引用才一直有效。三阶段压完还超预算,就交给兜底压缩器
(reduce_log);再不行原样返回,绝不抛异常。

钉住(pinned)的消息和头部 keep_head 条消息全程受保护;所有出口都把
头部标成 pinned 再返回(下游裁剪只认 pinned 标志,不认位置)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from my_agent_alpha.autocompact.contracts import (
    HUGE_TOOL_RESULT_TOKENS,
    OutputStore,
    TokenCounter,
    is_tool_result,
)
from my_agent_alpha.autocompact.master_context import MasterContext
from my_agent_alpha.autocompact.output_storage import create_tool_result_summary
from my_agent_alpha.autocompact.reasoning import strip_reasoning
from my_agent_alpha.autocompact.reduce import reduce_log
from my_agent_alpha.autocompact.scoring import compress_content
from my_agent_alpha.message import Message

# 阶段 3 的门槛:距末尾 ≥3 条、超 1000 token 的 assistant 消息才压,
# 压缩目标比例(保留七成)
_P3_AGE = 3
_P3_MIN_TOKENS = 1000
_P3_RATIO = 0.7

# 判断"一条消息里有没有工具调用"的函数类型(由外部注入)
ToolUseCheck = Callable[[Message], bool]


def _pin_head(messages: list[Message], keep_head: int) -> list[Message]:
    """把前 keep_head 条消息标成 pinned=True,其余原样(早退出口也要钉头部)。

    为什么:下游"发给模型前的最终裁剪"只认 pinned 标志不看位置,
    头部(系统提示词)不钉住,兜底裁剪时会第一个被丢掉。
    """
    return [
        replace(m, pinned=True) if i < keep_head and not m.pinned else m
        for i, m in enumerate(messages)
    ]


def auto_compact_log(
    messages: list[Message],
    *,
    context_window: int,
    counter: TokenCounter,
    contains_tool_use: ToolUseCheck,
    limit: int | None = None,
    max_tool_result_tokens: int = HUGE_TOOL_RESULT_TOKENS,
    reasoning_strip_age: int = 5,
    store: OutputStore | None = None,
    keep_head: int = 0,
    master: MasterContext | None = None,
) -> list[Message]:
    """压缩入口:返回压缩后的新消息列表(入参列表本身不被修改)。

    - context_window:当前模型的上下文窗口,所有阈值线的基数;
    - limit:工作上限,不传按 0.8 × 窗口算;
    - store:给了才做"截断内容落盘"(否则占位文本里没有文件路径);
    - master:给了才在被截消息尾部附"按字节找回原文"的引用;
    - keep_head:前几条消息永不动且输出时标 pinned(默认 0)。

    示例:
        compacted = auto_compact_log(log, context_window=128000,
                                     counter=计数器, contains_tool_use=探测器)
    """
    log = list(messages)  # 在副本上工作,不动调用方的列表
    n = len(log)
    eff_limit = limit if limit is not None else int(0.8 * context_window)
    tokens = counter.len_tokens(log)
    close = tokens >= int(0.7 * context_window)

    def _p1_target(i: int) -> bool:
        """阶段 1 会不会动第 i 条:够老(距末尾 ≥5)、没钉住、含思考区。"""
        msg = log[i]
        return (
            not msg.pinned
            and i >= keep_head
            and (n - i - 1) >= reasoning_strip_age
            and ("<think>" in msg.content or "<thinking>" in msg.content)
        )

    def _p3_target(i: int) -> bool:
        """阶段 3 会不会动第 i 条:够老、没钉住、长模型回复、不含工具调用。"""
        msg = log[i]
        return (
            not msg.pinned
            and i >= keep_head
            and (n - i - 1) >= _P3_AGE
            and msg.role == "assistant"
            and not contains_tool_use(msg)
            and counter.len_tokens(msg.content) > _P3_MIN_TOKENS
        )

    # 没有任何阶段要动手 → 早退(头部照常钉住)
    needs_p1 = any(_p1_target(i) for i in range(n))
    needs_p2 = tokens > eff_limit or close
    needs_p3 = any(_p3_target(i) for i in range(n))
    if not (needs_p1 or needs_p2 or needs_p3):
        return _pin_head(log, keep_head)

    # ── 阶段 1:按年龄剥思考区(所有角色都剥,谁都会有草稿) ──
    for i in range(n):
        msg = log[i]
        if msg.pinned or i < keep_head:
            continue
        if (n - i - 1) >= reasoning_strip_age:
            stripped, saved = strip_reasoning(msg.content, counter.len_tokens)
            if saved > 0:
                log[i] = replace(msg, content=stripped)

    # ── 阶段 2:最大的工具结果先截,降到目标线(0.8 × 窗口)即停 ──
    current = counter.len_tokens(log)
    target = int(0.8 * context_window)
    if current > target:
        candidates: list[tuple[int, int, Message]] = []
        for i, msg in enumerate(log):
            if msg.pinned or i < keep_head or not is_tool_result(msg):
                continue
            tok = counter.len_tokens(msg.content)
            if tok > max_tool_result_tokens:
                candidates.append((i, tok, msg))
        candidates.sort(key=lambda c: c[1], reverse=True)  # 最大的排前面
        for i, tok, msg in candidates:
            if current <= target:
                break  # 达标即停:能少截一条是一条
            summary = create_tool_result_summary(msg, tok, store)
            if master is not None:
                ref = master.reference_for(i, tok, msg.content)
                if ref is not None:
                    summary += "\n\n" + ref
            log[i] = replace(msg, content=summary)
            current -= tok - counter.len_tokens(summary)

    # ── 阶段 3:抽取式压缩长模型回复 ──
    # 注意:这一阶段没有 token 压力门槛 —— 只要没走早退,所有合格消息
    # 都会被压缩(哪怕当前总量远低于目标线;忠实还原 gptme 行为)。
    for i in range(n):
        if not _p3_target(i):
            continue
        msg = log[i]
        tok = counter.len_tokens(msg.content)
        compressed = compress_content(msg.content, target_ratio=_P3_RATIO)
        # 压完反而更大 → 跳过;附上字节引用后若反超原文 → 也保持原消息
        if counter.len_tokens(compressed) < tok:
            if master is not None:
                ref = master.reference_for(i, tok, msg.content)
                if ref is not None:
                    compressed += "\n\n" + ref
            if counter.len_tokens(compressed) < tok:
                log[i] = replace(msg, content=compressed)

    # ── 出口:达标就收工;不达标走兜底 ──
    final = counter.len_tokens(log)
    if final <= eff_limit:
        return _pin_head(log, keep_head)
    if 0 < keep_head < len(log):
        # 头部单独保留,剩余预算全给尾巴
        head = _pin_head(log[:keep_head], keep_head)
        return head + reduce_log(
            log[keep_head:],
            eff_limit - counter.len_tokens(head),
            counter=counter,
            contains_tool_use=contains_tool_use,
        )
    if keep_head >= len(log) and len(log) > 0:
        # 头部独占了整个预算:宁可超预算也保留任务上下文
        return _pin_head(log, keep_head)
    return reduce_log(log, eff_limit, counter=counter, contains_tool_use=contains_tool_use)
