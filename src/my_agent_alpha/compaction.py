"""上下文压缩:对话太长时,把"发给模型的历史"缩小,磁盘日志一个字节不动。

为什么需要:模型每次请求都要全量重发历史(模型无记忆),对话越长请求越大,
直到超过模型的上下文窗口,直接被接口拒绝。gptme 的解法是压缩历史 ——
我们采用它的规则式三阶段(快、确定性、不需要再调一次模型)。

压缩结果有两个去处,别混淆(开发方案 §5.1):
- 每次请求前的临时组装:prepare_messages() 现算现用,不落盘;
- 持久化视图:/compact 手动触发时,compact_current_to_view() 把一次
  成功的压缩固化成 views/compacted-NNN.jsonl,之后模型读视图、新消息
  双写。两种去处共用同一个 compact_messages(),磁盘上的完整日志
  (conversation.jsonl)永远不被压缩改写。

三阶段(和 gptme 的 auto_compact_log 一致,顺序执行):

阶段 1:剥思考。距结尾 ≥5 条的旧模型回复,删掉 <think>...</think> 内容
        (思考过程看过就可以扔,结论在正文里);
阶段 2:截工具结果。把超长的工具输出(system 消息)按"最大的先截"换成
        截断占位;截断前先把完整内容存档到会话目录(先存盘再截断 ——
        模型需要时可以自己去读存档文件);
阶段 3:压缩长回复。很老又很长的模型回复(且不含工具调用),做抽取式
        压缩:按句子打分保留高分的前 70%(代码块永远完整保留)。

兜底:三阶段后仍超预算 → 按"完整工具交互组"保留 pinned + 尽量多的
尾部(调用和结果永远同进同退,绝不拆散配对)。
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from my_agent_alpha.context import get_logdir
from my_agent_alpha.llm.models import resolve_model
from my_agent_alpha.message import Message
from my_agent_alpha.tools.base import ToolUse
from my_agent_alpha.utils import get_context_budget, len_tokens

if TYPE_CHECKING:  # 只做类型标注,不让压缩模块反向依赖日志管理(避免循环导入)
    from my_agent_alpha.logmanager import LogManager

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
                # 用 replace 只换 content,其余字段(call_id/pinned/hide/quiet/
                # metadata/时间戳)原样带到新实例 —— 手写 Message(...) 会把这些
                # 全部丢掉,压缩后的历史就"失真"了
                result[i] = replace(msg, content=stripped)
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
    while len_tokens(result) > budget:
        # 找当前最长、且可截(pinned/hide 的不动,已截过的跳过)的工具结果
        target_idx = -1
        target_size = 0
        for i, msg in enumerate(result):
            if msg.role != "system" or msg.pinned or msg.hide:
                continue
            if msg.content.startswith("[内容已截断"):
                continue
            size = len_tokens(msg.content)
            if size > _TOOL_RESULT_THRESHOLD and size > target_size:
                target_idx, target_size = i, size
        if target_idx < 0:
            break  # 没有可截的了,交给下一阶段
        msg = result[target_idx]
        archive = _archive_content(msg.content, logdir)
        note = f"完整输出: {archive}" if archive else "完整输出见会话日志"
        # replace 保住 call_id 等全部字段:原生工具调用的结果丢了编号,
        # 压缩后的历史就再也配不回它的调用(接口直接 400)
        result[target_idx] = replace(
            msg,
            content=f"[内容已截断 - 约 {target_size} tokens]\n{note}\n\n{msg.content[:400]}…",
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
    调用判断用 ToolUse 解析器(两种通道都认),不靠肉眼猜文本特征:
    只查 "@" 会漏掉 markdown 围栏调用,只查 "```" 会误伤纯展示代码块。
    """
    result = list(messages)
    total = len(result)
    for i in range(total - 3):  # 距结尾 ≥3 的才算"老"
        msg = result[i]
        if msg.role != "assistant":
            continue
        if any(True for _ in ToolUse.iter_from_content(msg.content)):
            continue  # 带工具调用(任意通道)的回复:执行记录,不动
        if len_tokens(msg.content) < _LONG_REPLY_THRESHOLD:
            continue
        # replace 保住全部附属字段(理由同阶段 1/2)
        result[i] = replace(msg, content=compress_reply(msg.content))
    return result


def tool_groups(messages: Sequence[Message]) -> list[list[Message]]:
    """把消息流切成"交互组",组是裁剪的最小单位,绝不从中间劈开。

    一组 = 一条 assistant 回复 + 它后面的全部 system 随从(工具结果/[用量]
    统计,直到下一条 user/assistant 为止),或者一条孤立的 user/system。
    为什么 assistant 的随从必须和它绑在一起:原生调用有铁律
    "一个 call_id 恰好配一条结果",把调用留在视图、结果裁掉(或反过来),
    这段历史发给接口就是 400。多条调用、多条结果同理整组进退。
    user 消息必须打断分组:它是回合边界,不是任何 assistant 的随从。

    (公开函数:LLM 语义压缩 compaction_llm.py 也用它来拆"待摘要前缀/
    原样尾部",两处必须用同一套分组规则,不能各写一份。)

    示例:[提示词, user, assistant(两个调用), 用量, 结果1, 结果2, user]
    → [[提示词], [user], [assistant,用量,结果1,结果2], [user]]
    """
    groups: list[list[Message]] = []
    followers: list[Message] | None = None  # 当前 assistant 还在收随从的组
    for msg in messages:
        if msg.role == "assistant":
            if followers is not None:
                groups.append(followers)
            followers = [msg]
        elif msg.role == "system" and followers is not None:
            followers.append(msg)  # assistant 的随从(结果/用量统计)
        else:
            if followers is not None:
                groups.append(followers)
                followers = None
            groups.append([msg])  # user 或开场的孤立 system:自己一组
    if followers is not None:
        groups.append(followers)
    return groups


def hard_cap(messages: list[Message], budget: int) -> list[Message]:
    """兜底:还超预算的话,按完整交互组保留 pinned + 从尾往前的尽可能多组。

    (走到这一步说明规则压缩救不回来,宁可整组丢掉最旧的非 pinned 历史,
    也不让请求失败 —— 但绝不把一个工具调用和它的结果拆到两边。)
    """
    if len_tokens(messages) <= budget:
        return messages
    groups = tool_groups(messages)
    # pinned 恒留:按组判(组内任一条 pinned 就整组留),保持原有先后顺序
    pinned_groups = [g for g in groups if any(m.pinned for m in g)]
    used = len_tokens([m for g in pinned_groups for m in g])
    tail: list[list[Message]] = []
    for g in reversed(groups):
        if any(m.pinned for m in g):
            continue  # 已经单独保留过了
        cost = len_tokens(g)
        if used + cost > budget and tail:
            break  # 至少保一组,避免极端小预算下空上下文
        tail.append(g)
        used += cost
    tail.reverse()
    return [m for g in pinned_groups for m in g] + [m for g in tail for m in g]


def compact_messages(
    messages: list[Message], budget: int, logdir: Path | None = None
) -> list[Message]:
    """规则式压缩总入口:超预算才动,不超原样返回(阶段 1 的剥思考除外,它永远划算)。

    示例:
        view = compact_messages(log.messages, budget=get_context_budget(128000))
        # view 只用于发送;log.messages 本身没变
    """
    result = strip_old_thinking(messages)
    if len_tokens(result) <= budget:
        return result
    result = truncate_tool_results(result, budget, logdir)
    if len_tokens(result) > budget:
        result = compress_long_replies(result)
    if len_tokens(result) > budget:
        result = hard_cap(result, budget)
    return result


def prepare_messages(
    messages: Sequence[Message], model: str, budget: int | None = None
) -> list[Message]:
    """决定"发给模型哪些消息":超预算时压缩视图,临时上下文，不落日志。

    - budget 不传时按模型的上下文窗口自动算(见上方 get_context_budget);
    - 压缩只改这份返回值(视图),log.messages 原封不动 —— 磁盘日志永不改写;
    - 详细的压缩规则见本模块上文(剥思考 → 截工具结果 → 压长回复 → 硬截断)。
    """
    meta = resolve_model(model)
    effective = budget if budget is not None else get_context_budget(meta.context)
    return compact_messages(list(messages), effective, get_logdir())


# ---------------- 压缩视图接入(V4)----------------
# prepare_messages 是"每次请求前的临时组装"(不落盘);下面这组函数把
# 一次成功的规则压缩"固化"成持久化视图。两者的区别见开发方案 §5.1。


def _is_llm_checkpoint(msg: Message) -> bool:
    """判断一条消息是不是 LLM 压缩笔记(靠 metadata 里的固定记号)。

    只在安检里用:笔记虽是 pinned system,但它本来就是压缩产物,
    下次压缩被新笔记替换是设计行为,不该按"丢了 pinned 提示词"拒绝。
    (判断逻辑内联在本模块,避免 compaction ↔ compaction_llm 循环导入。)
    """
    compaction_meta = (msg.metadata or {}).get("compaction")
    return isinstance(compaction_meta, dict) and compaction_meta.get("kind") == "llm_checkpoint"


def validate_compacted_messages(source: Sequence[Message], compacted: Sequence[Message]) -> None:
    """激活视图前的校验:传入压缩后的 msgs,判断是否是合格的压缩后产物,
    检查不通过就抛 ValueError,拒绝激活。

    检查四件事(见开发方案 §5.2.5):
    1. pinned 的系统提示词一条不少(它们是模型的身份/工具说明,丢了模型就失忆);
    2. source 的最后一条 user 消息还在(最新的用户意图不能被压没);
    3. 原生工具调用与结果按 call_id 一一配对:既没有"有调用无结果",
       也没有孤儿结果(同一编号出现两次也算坏);
    4. 以上任一不满足 → 抛错,由调用方决定继续用旧视图。

    匹配用对象同一性(is)而不是 Message 相等:pinned 提示词和最新 user
    消息不会被压缩阶段改动,原对象必须原样出现在结果里;相等性只比较
    role/content,两条不同消息内容撞车时会误判"还在"。
    一个精确的例外:LLM 压缩笔记(metadata 里 compaction.kind =
    "llm_checkpoint"的消息)虽然是 pinned system,但它本来就是压缩产物,
    下一次压缩时被新笔记替换是设计行为,不在"不可丢"之列。
    """
    for m in source:
        if (
            m.pinned
            and m.role == "system"
            and not _is_llm_checkpoint(m)
            and not any(m is c for c in compacted)
        ):
            raise ValueError(f"压缩丢了 pinned 系统提示词:{m.content[:50]}…")
    last_user = None
    for m in source:
        if m.role == "user":
            last_user = m
    if last_user is not None and not any(last_user is c for c in compacted):
        raise ValueError("压缩丢了最后一条用户消息(最新意图不能被压没)")
    calls = [
        tu.call_id
        for m in compacted
        if m.role == "assistant"
        for tu in ToolUse.iter_from_content(m.content)
        if tu.call_id
    ]
    results = [m.call_id for m in compacted if m.call_id]
    if sorted(calls) != sorted(results):
        orphan = sorted(set(results) - set(calls))
        unmatched = sorted(set(calls) - set(results))
        raise ValueError(
            f"压缩后工具调用配对不完整(未配对的调用 {unmatched} / 孤儿结果 {orphan});"
            "本回合可能还有没返回的工具调用,等它收尾再压缩"
        )


def is_meaningful_reduction(source: Sequence[Message], compacted: Sequence[Message]) -> bool:
    """压缩收益值不值得建一份新视图:至少省下一成才算,否则白多一个文件。

    示例:10 万 tokens 压到 8.5 万 → True;压到 9.5 万 → False(不值得)。
    """
    if not source:
        return False
    return len_tokens(compacted) <= len_tokens(source) * 0.9


def compact_current_to_view(
    manager: LogManager, model: str, budget: int | None = None
) -> str | None:
    """把"当前有效历史"压缩一次并固化成新视图;无收益或危险时返回 None。

    这是手动压缩(/compact)与自动压缩(turn.post 钩子)共用的服务函数,
    一个会话只在交互边界调用它:
    - 压缩源是 manager.messages(激活过视图就是上一次的视图,不是反复
      压整个 Master—— 重复压会一次次丢细节);
    - budget 缺省用模型的全预算;自动压缩会传一个更紧的目标(压到
      触发线以下,保证这次压缩真的有收益,而不是压了个寂寞);
    - 压缩结果先过 validate_compacted_messages 安检,不过就抛错不激活,
      旧视图原样继续用;
    - 收益不足一成(is_meaningful_reduction)就不建视图,返回 None。
    """
    source = list(manager.messages)
    effective = budget if budget is not None else get_context_budget(resolve_model(model).context)
    result = compact_messages(source, effective, manager.logdir)
    if not is_meaningful_reduction(source, result):
        return None
    validate_compacted_messages(source, result)
    return manager.create_compacted_view(result)
