"""LLM 摘要:规则压缩救不回来时,请模型给整个会话写一份"接续文档"。

流程(见 docs/autocompact-dev.md §5.5):
1. 把当前消息发给模型,附一段固定的 RESUME 提示词(输出格式的契约);
2. 模型返回一份结构化摘要(含"该带上的上下文文件清单");
3. 摘要存成会话目录里的 RESUME.md;
4. 按清单把文件内容读进来(每个有大小上限);
5. 组装出一份全新的精简日志:系统提示词 + 上下文文件 + 引导语 + 摘要。
   原来的历史全部退出上下文 —— 摘要就是它们的"遗言"。

RESUME_PROMPT 一个字都不能改:模型端和解析端都按这个格式约定工作。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from my_agent_alpha.autocompact.contracts import LLMClient, TokenCounter
from my_agent_alpha.message import Message

logger = logging.getLogger(__name__)

# 摘要失败时返回的消息前缀:调用方(触发器)看到它就知道"这次没成功",
# 不能切换视图、要保留原日志
RESUME_FAILED_PREFIX = "[resume-failed]"

# 摘要产出文件的名字(写在会话目录下)
RESUME_FILENAME = "RESUME.md"

# 单个上下文文件的 token 上限;截断时留 50 token 的提示余量
_CONTEXT_FILE_LIMIT = 2000
_TRUNCATE_MARGIN = 50

# 输出格式的契约,不可修改(模型按它生成,解析端按它提取)
RESUME_PROMPT = """Please create a comprehensive resume of this conversation that includes:

1. **Conversation Summary**: Key topics, decisions made, and progress achieved
2. **Technical Context**: Important code changes, configurations, or technical details
3. **Current State**: What was accomplished and what remains to be done
4. **Context Files**: List the specific files that should be included in future context

For the Context Files section, use this format:
## Context Files

List each file on its own line with a bullet point and backticks:
- `path/to/file.py` - Brief rationale for including this file
- `docs/spec.md` - Contains the specification being implemented

Include files such as:
- Specs, PRDs, or design documents being referenced
- Source files being actively modified
- Configuration files relevant to the work
- Task or plan files tracking progress

Format the response as a structured document that could serve as a RESUME.md file."""

# 在摘要里找"上下文文件"小节:小节标题认四种写法,小节读到下一个标题为止
_SECTION_RE = re.compile(
    r"(?:#{1,4}\s*(?:Context Files|Files to Include|Recommended Files|Key Files)"
    r"[^\n]*\n)([\s\S]*?)(?=\n#{1,4}\s|\Z)",
    re.IGNORECASE,
)
# 小节内每一行按顺序尝试三种提取方式:反引号包裹 / 以 / ~/ . 开头的裸路径 / 常见目录开头的相对路径
_BULLET_PATTERNS = (
    re.compile(r"[-*]\s*`([^`]+)`"),
    re.compile(r"[-*]\s*([/~.][\w./\-]+(?:\.\w+)?)"),
    re.compile(
        r"[-*]\s*((?:src|docs|tests|config|scripts|tasks|journal|knowledge|lessons)/[\w./\-]+(?:\.\w+)?)"
    ),
)


def parse_context_files(summary: str) -> list[str]:
    """从摘要文本里提取"该带上的上下文文件"路径清单。

    - 先找 "Context Files"(或 Files to Include / Recommended Files / Key
      Files)小节;找不到就把全文当小节扫;
    - 小节内每个列表项按三种模式提路径(反引号包裹 → 裸路径 → 相对路径);
    - 过滤 http 链接和 # 开头的项;去掉开头的 ./;去重且保持出现顺序。

    示例:"- `docs/spec.md` - 规格文档" → 提取 "docs/spec.md"
    """
    section = _SECTION_RE.search(summary)
    text = section.group(1) if section else summary
    paths: list[str] = []
    for raw_line in text.split("\n"):
        found: str | None = None
        for pattern in _BULLET_PATTERNS:
            m = pattern.match(raw_line.lstrip())
            if m:
                found = m.group(1)
                break
        if found is None:
            continue
        if found.startswith(("http", "#")):
            continue  # 链接/锚点不是文件路径
        found = found.removeprefix("./")
        if found not in paths:
            paths.append(found)
    return paths


def load_context_files(
    paths: list[str], workspace: Path, *, counter: TokenCounter
) -> list[tuple[Path, str]]:
    """把清单里的文件读进来,返回(路径, 内容)列表;超限的截断。

    - 绝对路径直接用;~ 开头按用户主目录展开;其余都相对 workspace;
    - 不存在的、读不了的:跳过并告警(摘要里提到的文件完全可能已删);
    - 超过 2000 token 的文件:逐行累加,到(上限 − 50)停,
      末尾注明还有多少 token 没读 —— 宁可截断也不能让一个文件吃光预算。
    """
    loaded: list[tuple[Path, str]] = []
    for raw in paths:
        if raw.startswith("~"):
            path = Path(raw).expanduser()
        elif Path(raw).is_absolute():
            path = Path(raw)
        else:
            path = workspace / raw
        if not path.is_file():
            logger.warning("摘要建议的上下文文件不存在,已跳过: %s", path)
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("上下文文件读不了,已跳过(%s): %s", path, e)
            continue
        full_tokens = counter.len_tokens(content)
        if full_tokens > _CONTEXT_FILE_LIMIT:
            kept: list[str] = []
            used = 0
            for line in content.split("\n"):
                line_tokens = counter.len_tokens(line)
                if used + line_tokens > _CONTEXT_FILE_LIMIT - _TRUNCATE_MARGIN:
                    break
                kept.append(line)
                used += line_tokens
            remaining = full_tokens - used
            content = "\n".join(kept) + f"\n... (truncated, {remaining} tokens remaining)"
            logger.info("上下文文件 %s 超限,已截断(剩 %d token 未读)", path, remaining)
        loaded.append((path, content))
    return loaded


def resume_via_llm(
    messages: list[Message],
    *,
    llm: LLMClient,
    model: str,
    session_dir: Path,
    workspace: Path,
    counter: TokenCounter,
) -> list[Message]:
    """生成 LLM 摘要并组装出精简日志。失败时返回一条带标记的提示消息。

    成功的返回值结构(顺序固定):
        [开头连续的 system 消息…]        ← 保住系统提示词
        [每个上下文文件一条 system 消息]  ← 文件正文包在代码块里
        [一条引导语 system 消息]          ← 说明"历史已被摘要接续"
        [一条 assistant 消息]            ← 摘要全文(假装是模型自己说的)

    失败(消息太少 / 模型调用出错 / 摘要为空)返回
    [system: "[resume-failed] …"],调用方看到前缀就知道不要落地。
    """
    if len(messages) < 3:
        return [Message("system", f"{RESUME_FAILED_PREFIX} 消息不足 3 条,无法生成摘要")]
    # 原始消息在前、提示词在尾:服务商对前缀做缓存,历史部分还能吃到缓存
    request = [*messages, Message("user", RESUME_PROMPT)]
    try:
        summary = llm.reply(request, model)
    except Exception as e:
        logger.warning("生成摘要时模型调用失败: %s", e)
        return [Message("system", f"{RESUME_FAILED_PREFIX} 调用模型失败: {e}")]
    if not summary.strip():
        return [Message("system", f"{RESUME_FAILED_PREFIX} 模型返回了空摘要")]

    # 摘要落盘(失败只告警,不中断:正文已经在手里)
    resume_path = session_dir / RESUME_FILENAME
    try:
        resume_path.write_text(summary, encoding="utf-8")
        resume_source = str(resume_path)
    except OSError as e:
        logger.warning("摘要写盘失败,仅告警(%s)", e)
        resume_source = "LLM-generated summary"

    files = load_context_files(parse_context_files(summary), workspace, counter=counter)

    # 组装新日志
    head: list[Message] = []
    for m in messages:
        if m.role != "system":
            break
        head.append(m)
    file_messages = [
        Message("system", f"Context file `{p}`:\n```\n{content}\n```") for p, content in files
    ]
    file_note = f" (with {len(files)} context files)" if files else ""
    intro = Message("system", f"Previous conversation resumed from {resume_source}{file_note}:")
    return head + file_messages + [intro, Message("assistant", summary)]
