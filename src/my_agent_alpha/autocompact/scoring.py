"""阶段 3 的实现:给句子打分,按分抽取 —— "压缩长回复"用的打分表。

思路:一条很长的模型回复,不是每句话都值得留。把回复切成句子,
按"这句话对后续对话有没有用"打分(提到了决策?提到了报错?提到了
文件路径?),保留得分最高的大约七成,再按原来的顺序拼回去。
代码块(三个反引号围起来的部分)无条件完整保留,永不参与打分。

纯函数,零依赖。
"""

from __future__ import annotations

import re
from collections.abc import Callable

# 句子切分:在句号/问号/叹号后面、且后面跟着空白的位置切开
# (英文句点必须后跟空白,避免切开 "3.14"、"e.g." 这类)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# 命中一个加 0.5 分,多个词可叠加(大小写不敏感)
_KEYWORDS = (
    "error",
    "fail",
    "success",
    "complete",
    "implement",
    "fix",
    "bug",
    "issue",
    "result",
    "output",
    "todo",
    "fixme",
    "note",
    "warning",
)

# 文件路径:以 / 或 ~ 开头的路径,或 Windows 盘符路径(C:\...)
_FILE_PATH_RE = re.compile(
    r"(?:[/~][a-zA-Z0-9_\-./]+(?:\.[a-zA-Z0-9]+)?"
    r"|[A-Za-z]:[/\\][a-zA-Z0-9_\-./\\]+(?:\.[a-zA-Z0-9]+)?)"
)
# URL 地址
_URL_RE = re.compile(r"""https?://[^\s<>"')]+""")
# 报错痕迹(error/exception/traceback/failed/failure 任一)
_ERROR_RE = re.compile(r"\b(?:error|exception|traceback)\b|\bfailed\b|\bfailure\b", re.IGNORECASE)

# 下面四组是"语义信号":每组内命中任意一个模式就整组加分一次,不叠加
# 决策类:"我们决定用……" —— 记录了技术选型,丢了后面会反复纠结
_DECISION_RE = re.compile(
    r"we'll use|we will use|decided to|going with|choosing|solution is|approach is|we chose",
    re.IGNORECASE,
)
# 结论类:"因此/总之/结果是……"
_CONCLUSION_RE = re.compile(
    r"therefore|in summary|the result is|this means|confirmed that|in conclusion|key finding",
    re.IGNORECASE,
)
# 承诺类:"我接下来会做……" —— 是待办事项,丢了任务就断了
_COMMITMENT_RE = re.compile(
    r"i'll|i will (?:implement|create|fix|add|update|write|build)"
    r"|next steps\?|action items\?|todo:"
    r"|will implement|plan to"
    r"|going to (?:implement|create|fix|add|update|write|build)",
    re.IGNORECASE,
)
# 动作结果类:"创建了文件/修好了/完成了……"
_ACTION_RESULT_RE = re.compile(
    r"created file|fixed|updated|implemented|completed|merged", re.IGNORECASE
)

# 打分函数的类型:吃(句子, 位置, 总句数),返回得分
ScoreFn = Callable[[str, int, int], float]


def score_sentence(sentence: str, position: int, total: int) -> float:
    """给一个句子打分,分越高越值得在压缩时保留。各项独立累加:

    - 位置:首句 +2.0(往往是结论/计划),末句 +1.5(往往是总结),
      前三句再 +1.0(开头信息密度高);
    - 关键词:命中 error/fix/todo…… 每个词 +0.5,可叠加;
    - 长度:太短(<10 字符)-1.0(多半是"好的。""嗯"这类),
      偏短(<50)+0.3,太长(>200)-0.2;
    - 语义:决策 +2.0 / 结论 +1.5 / 承诺 +1.5 / 动作结果 +1.0(各最多加一次);
    - 引用:文件路径 +1.0 / URL +0.5 / 报错痕迹 +1.5。

    示例:score_sentence("we'll use SQLite for storage", 4, 10) → 含决策语 +2.0
    """
    score = 0.0
    # 位置分
    if position == 0:
        score += 2.0
    if position == total - 1:
        score += 1.5
    if position < 3:
        score += 1.0
    # 关键词分(逐词独立判断,命中几个加几个 0.5)
    lowered = sentence.lower()
    for kw in _KEYWORDS:
        if kw in lowered:
            score += 0.5
    # 长度分
    if len(sentence) < 10:
        score -= 1.0
    if len(sentence) < 50:
        score += 0.3
    if len(sentence) > 200:
        score -= 0.2
    # 语义分(每组命中一次即止)
    if _DECISION_RE.search(sentence):
        score += 2.0
    if _CONCLUSION_RE.search(sentence):
        score += 1.5
    if _COMMITMENT_RE.search(sentence):
        score += 1.5
    if _ACTION_RESULT_RE.search(sentence):
        score += 1.0
    # 引用分
    if _FILE_PATH_RE.search(sentence):
        score += 1.0
    if _URL_RE.search(sentence):
        score += 0.5
    if _ERROR_RE.search(sentence):
        score += 1.5
    return score


def compress_content(content: str, target_ratio: float = 0.7) -> str:
    """抽取式压缩一段文本:保留得分最高的约 target_ratio(默认七成)句子。

    步骤:
    1. 先把代码块整体摘出来换成占位符(代码块永不参与打分和删除);
    2. 剩下的文本切句;句子总数不超过 3 → 没什么可压,原样返回;
    3. 含占位符的句子(标记句)全部保留,其余句子逐个打分;
    4. 按分数从高到低取前 max(2, 句数×比例) 句;
    5. 选中句 + 标记句按原文顺序拼回,占位符换回代码块。

    示例:十句的回复 compress_content 后只剩七句左右,
    但其中的代码块和提到决策/报错/路径的句子一定还在。
    """
    # 1. 摘出代码块,占位保护
    blocks: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(blocks) - 1}__"

    stashed = re.sub(r"```[\s\S]*?```", _stash, content)

    # 2. 切句(丢掉切出来的空白段)
    sentences = [s for s in _SENT_SPLIT_RE.split(stashed) if s.strip()]
    n = len(sentences)
    if n <= 3:
        return content  # 句子太少,压缩没有意义;content 是原文,代码块自然完好

    # 3. 分两组:标记句(含代码块占位符,全保)和可打分句
    marked_set = {i for i, s in enumerate(sentences) if "__CODE_BLOCK_" in s}
    scorable = [i for i in range(n) if i not in marked_set]

    # 4. 打分取前几名。注意:position 用句子在完整列表里的原始下标,
    #    total 也算上标记句 —— 位置分的"首句/末句"指的是原文里的位置
    ranked = sorted(scorable, key=lambda i: -score_sentence(sentences[i], i, n))
    target_count = max(2, int(len(scorable) * target_ratio))
    keep = marked_set | set(ranked[:target_count])

    # 5. 按原始位置排序拼接,占位符还原
    merged = " ".join(sentences[i] for i in sorted(keep))
    for i, block in enumerate(blocks):
        merged = merged.replace(f"__CODE_BLOCK_{i}__", block)
    return merged
