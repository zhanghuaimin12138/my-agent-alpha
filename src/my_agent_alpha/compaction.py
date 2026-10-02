"""上下文压缩:对话太长时,把"发给模型的视图"缩小,磁盘日志一个字节不动。

为什么需要:模型每次请求都要全量重发历史(模型无记忆),对话越长请求越大,
直到超过模型的上下文窗口,直接被接口拒绝。gptme 的解法是压缩历史 ——
我们采用它的规则式三阶段(快、确定性、不需要再调一次模型),但做一个
重要简化:gptme 把压缩结果存成"视图文件"切换着用,我们**每次请求时现算**
(日志永不改写,视图永不落盘 —— 符合 AGENTS.md 的架构纪律)。

三阶段(和 gptme 的 auto_compact_log 一致,顺序执行):

阶段 1:剥思考。距结尾 ≥5 条的旧模型回复,删掉 <think>...</think> 内容
        (思考过程看过就可以扔,结论在正文里);
阶段 2:截工具结果。把超长的工具输出(system 消息)按"最大的先截"换成
        截断占位;截断前先把完整内容存档到会话目录(先存盘再截断 ——
        模型需要时可以自己去读存档文件);
阶段 3:压缩长回复。很老又很长的模型回复(且不含工具调用),做抽取式
        压缩:按句子打分保留高分的前 70%(代码块永远完整保留)。

兜底:三阶段后仍超预算 → 保留 pinned 消息 + 尽量多的尾部消息(硬截断)。
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from my_agent_alpha.message import Message

logger = logging.getLogger(__name__)

# 触发阶段 2 的工具结果长度门槛(gptme 同款默认)
_TOOL_RESULT_THRESHOLD = 2000
# 触发阶段 3 的回复长度门槛
_LONG_REPLY_THRESHOLD = 1000
# 阶段 1 起步的距离:距结尾不足 5 条的消息不动(近期上下文保持原样)
_RECENT_KEEP = 5
# 阶段 3 的压缩目标比例
_COMPRESS_RATIO = 0.7
# 思考块的正则(非贪婪匹配,含换行)
_THINK_RE = re.compile(r"<think>.*?</think>\n?", re.DOTALL)


def estimate_tokens(text: str) -> int:
    """粗略估算一段文字的 token 数(不装分词库,够用就好)。

    经验值:英文约 4 个字符一个 token,中文约 1.5 个字符一个 token。
    不同模型的分词略有差异,但对"要不要压缩"的决策来说足够准;
    真实用量以接口返回的 usage 为准(日志里每条回复都记着)。
    """
    ascii_len = sum(1 for c in text if ord(c) < 128)
    other_len = len(text) - ascii_len
    return (ascii_len + 3) // 4 + (other_len * 2 + 2) // 3


def messages_tokens(messages: list[Message]) -> int:
    """估算一批消息的总 token 数。"""
    return sum(estimate_tokens(m.content) for m in messages)


def get_context_budget(context_window: int, max_output: int = 8192, headroom: int = 1000) -> int:
    """计算"历史最多能占多少 token"(预算)。

    背景:模型的上下文窗口是输入和输出**共用**的 —— 模型是在窗口里
    接着历史往下写回复的。历史把窗口塞满,模型就没地方"说话"了
    (回复被截断,甚至一个字都写不出来直接报错)。
    所以要提前给历史设上限,给回复留好位置。

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


def strip_old_thinking(messages: list[Message]) -> list[Message]:
    """阶段 1:删掉旧回复里的思考块(近期的保留,保持上下文连贯)。

    示例:第 0 条(距结尾 ≥5)内容 "<think>推理…</think>结论"
    → 替换为 "结论";原消息对象不动,造新的。
    """
    result = list(messages)
    for i in range(len(result) - _RECENT_KEEP):
        msg = result[i]
        if msg.role == "assistant" and "<think>" in msg.content:
            stripped = _THINK_RE.sub("", msg.content).strip()
            if stripped != msg.content:
                result[i] = Message("assistant", stripped, timestamp=msg.timestamp)
    return result


def truncate_tool_results(
    messages: list[Message], budget: int, logdir: Path | None
) -> list[Message]:
    """阶段 2:最大的工具结果先截断,直到总量降到预算内。

    为什么最大的先截:一刀截掉最大的那条,省下的量最多、损失的信息相对最少。
    截断占位会注明完整内容的存档路径(先存盘再截断,存档文件名按内容哈希
    决定 —— 同一条消息反复压缩也只存一份)。
    """
    result = list(messages)
    while messages_tokens(result) > budget:
        # 找当前最长、且可截(pinned/hide 的不动,已截过的跳过)的工具结果
        target_idx = -1
        target_size = 0
        for i, msg in enumerate(result):
            if msg.role != "system" or msg.pinned or msg.hide:
                continue
            if msg.content.startswith("[内容已截断"):
                continue
            size = estimate_tokens(msg.content)
            if size > _TOOL_RESULT_THRESHOLD and size > target_size:
                target_idx, target_size = i, size
        if target_idx < 0:
            break  # 没有可截的了,交给下一阶段
        msg = result[target_idx]
        archive = _archive_content(msg.content, logdir)
        note = f"完整输出: {archive}" if archive else "完整输出见会话日志"
        result[target_idx] = Message(
            "system",
            f"[内容已截断 - 约 {target_size} tokens]\n{note}\n\n{msg.content[:400]}…",
            timestamp=msg.timestamp,
        )
    return result


def _archive_content(content: str, logdir: Path | None) -> Path | None:
    """把完整内容存进会话目录(tool-outputs/compacted/),返回路径。

    没有会话目录(纯函数调用场景)返回 None —— 那就不存,占位里说明看日志。
    文件名带内容哈希:同一条消息被多次压缩,存的还是同一个文件。
    """
    if logdir is None:
        return None
    out_dir = logdir / "tool-outputs" / "compacted"
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    path = out_dir / f"msg-{digest}.txt"
    path.write_text(content, encoding="utf-8")
    return path


# 抽取式打分用的信号(简化版 gptme scoring.py:位置 + 关键内容)
_PATH_RE = re.compile(r"[\w./-]+\.\w{1,4}")  # 像文件路径的词
_ERROR_RE = re.compile(r"错误|失败|Error|Traceback|异常", re.IGNORECASE)
# 按句末标点切句:中文标点后通常没有空格,所以零宽断言直接在标点后切;
# 英文句点要求后面跟空格(避免切开 "e.g." / "3.14" 这类)
_SENT_SPLIT = re.compile(r"(?<=[。!?;])|(?<=\.) ")


def compress_reply(content: str, ratio: float = _COMPRESS_RATIO) -> str:
    """抽取式压缩一段模型回复:按句打分,保留高分句子凑够约 ratio 的长度。

    代码块(围栏)整体抽出、无条件保留;其余句子按分排序取满为止,
    最后按原顺序拼回(保持阅读顺序)。
    """
    # 1. 摘出代码块,占位保护
    blocks: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(blocks) - 1}__"

    stashed = re.sub(r"```.*?```", _stash, content, flags=re.DOTALL)

    # 2. 切句打分
    sentences = [s for s in _SENT_SPLIT.split(stashed) if s.strip()]
    if not sentences:
        return content
    n = len(sentences)

    def score(i: int, s: str) -> float:
        val = 0.0
        if i == 0:
            val += 2.0  # 开头往往是结论/计划
        if i == n - 1:
            val += 1.5  # 结尾往往是总结
        if _PATH_RE.search(s):
            val += 1.0  # 提到文件路径,信息密度高
        if _ERROR_RE.search(s):
            val += 1.5  # 提到错误,上下文里很关键
        return val

    target_chars = max(int(len(stashed) * ratio), 1)
    ranked = sorted(range(n), key=lambda i: -score(i, sentences[i]))
    keep: set[int] = set()
    used = 0
    for i in ranked:
        keep.add(i)
        used += len(sentences[i])
        if used >= target_chars:
            break
    compressed = stashed if keep == set(range(n)) else "".join(sentences[i] for i in sorted(keep))

    # 3. 放回代码块
    for i, block in enumerate(blocks):
        compressed = compressed.replace(f"__CODE_BLOCK_{i}__", block)
    return compressed


def compress_long_replies(messages: list[Message]) -> list[Message]:
    """阶段 3:又老又长、且不含工具调用的模型回复,压缩到七成。

    含工具调用的回复不压 —— 那是执行记录,动了会破坏"日志可回放"。
    """
    result = list(messages)
    total = len(result)
    for i in range(total - 3):  # 距结尾 ≥3 的才算"老"
        msg = result[i]
        if msg.role != "assistant" or "```" in msg.content or "@" in msg.content:
            continue
        if estimate_tokens(msg.content) < _LONG_REPLY_THRESHOLD:
            continue
        result[i] = Message("assistant", compress_reply(msg.content), timestamp=msg.timestamp)
    return result


def hard_cap(messages: list[Message], budget: int) -> list[Message]:
    """兜底:还超预算的话,保留 pinned + 从尾往前的尽可能多消息。

    (走到这一步说明规则压缩救不回来,宁可丢最旧的非 pinned 历史也不让请求失败)
    """
    if messages_tokens(messages) <= budget:
        return messages
    pinned = [m for m in messages if m.pinned]
    tail: list[Message] = []
    used = messages_tokens(pinned)
    for msg in reversed(messages):
        if msg.pinned:
            continue
        cost = estimate_tokens(msg.content)
        if used + cost > budget and tail:
            break  # 至少保留一条,避免极端小预算下空上下文
        tail.append(msg)
        used += cost
    tail.reverse()
    return pinned + tail


def compact_messages(
    messages: list[Message], budget: int, logdir: Path | None = None
) -> list[Message]:
    """压缩总入口:超预算才动,不超原样返回(阶段 1 的剥思考除外,它永远划算)。

    示例:
        view = compact_messages(log.messages, budget=get_context_budget(128000))
        # view 只用于发送;log.messages 本身没变
    """
    result = strip_old_thinking(messages)
    if messages_tokens(result) <= budget:
        return result
    result = truncate_tool_results(result, budget, logdir)
    if messages_tokens(result) > budget:
        result = compress_long_replies(result)
    if messages_tokens(result) > budget:
        result = hard_cap(result, budget)
    return result
