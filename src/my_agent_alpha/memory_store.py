"""长期记忆:跨会话记住事情(gptme memory 的骨架 + CC 式 frontmatter)。

设计(仿 gptme 的 memory 模块,格式纪律仿 Claude Code 记忆架构,
详见 docs/长期记忆改进方案.md M1/M2):

- 存储形态:一条记忆 = 一个 markdown 文件(frontmatter + 正文),
  人可以直接打开看、改、删;
- frontmatter 四字段:name / description / type / created。
  其中 description 是"向量的替代品"—— 以后判断"这条记忆和当前问题
  相不相关"全靠它(CC 报告 §4.4:CC 没有向量库,检索就是模型看描述);
- 四类封闭类型学(MEMORY_TYPES):user / feedback / project / reference;
- 两层目录(近的盖住远的,仿 gptme 的"nearest-first"):
    工作区/memory/   ← 当前项目的记忆(优先)
    数据目录/memory/ ← 用户级记忆(~/.local/share/my_agent_alpha/memory)
- MEMORY.md:自动生成的指针索引(一行一条:链接 + 描述 + 类型),给人看;
  会话开始时另渲染一份带新鲜度标注的版本注入模型(有行数/字节双上限,
  CC 报告 §2.6 的教训:单上限挡不住另一种超限);
- 写入:模型调用 memory 工具提出"想记一条",经确认链(交互时用户点头)
  才落盘 —— 记忆是要进每次上下文的,不该让模型随手乱写。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from my_agent_alpha.dirs import get_data_dir

# 四类封闭类型学(CC 报告 §2.4):记忆只允许这四种,别的都是可推导内容
MEMORY_TYPES: tuple[str, ...] = ("user", "feedback", "project", "reference")

# 注入索引的双上限(CC 报告 §2.6:200 行内塞过 197KB,所以行数字节都要卡;
# 我们两层各一份,单层配额取 CC 的一半,总量相当)
_INDEX_MAX_LINES = 100
_INDEX_MAX_BYTES = 16 * 1024

# 超过这个天数的记忆标"⚠旧"(CC 报告 §4.5:陈旧记忆要显式警示,防当事实断言)
_STALE_DAYS = 30

# 索引行里描述的截断长度(字符):索引是目录不是记忆,一行别塞下一切
_DESC_MAX_CHARS = 120


@dataclass(frozen=True)
class MemoryEntry:
    """一条记忆的元信息(从文件名和 frontmatter 解析)。

    description 是检索的钥匙:模型决定"要不要 read 这条"就看它,
    所以保存时被要求写得具体(不填则回填标题)。
    """

    name: str  # 记忆名(文件名去掉 .md)
    path: Path
    title: str  # 正文首行标题
    description: str  # frontmatter 里的一行描述
    mem_type: str  # user / feedback / project / reference


def memory_dirs(workspace: Path) -> list[Path]:
    """两层记忆目录,近的在前(工作区层盖住用户层)。"""
    return [workspace / "memory", get_data_dir() / "memory"]


def _slug(name: str) -> str:
    """把名字变成安全的文件名:只留中英文数字和连字符。"""
    slug = re.sub(r"[^\w一-鿿-]+", "-", name.strip()).strip("-")
    return slug or "memory"


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """剥出文本开头的 frontmatter,返回 (字段字典, 剩余正文)。

    没有合法 frontmatter(不是 --- 开头、或找不到闭合 ---)就返回
    ({}, 原文)—— 旧格式文件因此天然兼容,这就是"读时兼容"。

    示例:输入 "---\\ntype: user\\n---\\n# 标题\\n正文"
    → ({"type": "user"}, "# 标题\\n正文")
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    for i in range(1, len(lines)):  # 第 0 行是开头 ---,从第 1 行找闭合
        if lines[i].strip() == "---":
            fields: dict[str, str] = {}
            for ln in lines[1:i]:
                key, sep, value = ln.partition(":")
                if sep and key.strip() in ("name", "description", "type", "created"):
                    fields[key.strip()] = value.strip()
            return fields, "\n".join(lines[i + 1 :])
    return {}, text  # 没有闭合 ---:当普通正文,不当作 frontmatter


def _parse_memory_text(text: str) -> tuple[str, str, str]:
    """解析一份记忆文件,返回 (标题, 描述, 类型),新旧格式都认。

    新格式用 frontmatter 里的 description/type;
    旧格式(或字段缺失)回填默认值:描述 = 标题、类型 = project。
    """
    fields, body = _parse_frontmatter(text)
    title = ""
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped:  # 正文里第一个非空行当标题(去掉 # 前缀)
            title = stripped.lstrip("# ").strip()
            break
    description = fields.get("description") or title
    mem_type = fields.get("type") or "project"
    if mem_type not in MEMORY_TYPES:
        mem_type = "project"  # 未知类型防御性回落,不让脏数据进索引
    return title, description, mem_type


def list_memories(workspace: Path) -> list[MemoryEntry]:
    """列出两层目录的全部记忆(同名时近层覆盖远层)。"""
    by_name: dict[str, MemoryEntry] = {}
    for d in memory_dirs(workspace):  # 近的先放,远层同名不覆盖
        if not d.exists():
            continue
        for path in sorted(d.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            title, description, mem_type = _parse_memory_text(path.read_text(encoding="utf-8"))
            by_name.setdefault(
                path.stem,
                MemoryEntry(
                    path.stem,
                    path,
                    title or path.stem,
                    description or title or path.stem,
                    mem_type,
                ),
            )
    return list(by_name.values())


def save_memory(
    name: str,
    content: str,
    workspace: Path,
    project_level: bool = False,
    mem_type: str | None = None,
    description: str | None = None,
) -> Path:
    """写一条记忆并重建索引,返回文件路径。

    project_level=True 写进工作区层(项目记忆),否则用户层。
    content 首行做标题;content 开头可以自带一段 frontmatter
    (description/type),显式参数优先于它、它优先于默认值 ——
    markdown 通道的模型把描述写进内容开头即可享受同等待遇。
    落盘统一是新格式(frontmatter + # 标题 + 正文)。
    """
    fields, body = _parse_frontmatter(content)
    final_type = mem_type or fields.get("type") or "project"
    if final_type not in MEMORY_TYPES:
        final_type = "project"
    lines = body.strip().split("\n")
    title = lines[0].lstrip("# ").strip() if lines else ""
    final_desc = description or fields.get("description") or title
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    target_dir = memory_dirs(workspace)[0 if project_level else 1]
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{_slug(name)}.md"
    front = (
        f"---\nname: {path.stem}\n"
        f"description: {final_desc[:_DESC_MAX_CHARS]}\n"
        f"type: {final_type}\ncreated: {stamp}\n---\n\n"
    )
    rest = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    body_text = f"# {title}\n\n{rest}\n" if rest else f"# {title}\n"
    path.write_text(front + body_text, encoding="utf-8")
    rebuild_index(workspace)
    return path


def rebuild_index(workspace: Path) -> None:
    """重建每层的 MEMORY.md 索引(一行一条:链接 + 描述 + 类型)。

    这是给人看的镜像文件;注入模型的那份由 render_memory_index
    现算(带新鲜度),两者同行格式、数据同源。
    """
    for d in memory_dirs(workspace):
        if not d.exists():
            continue
        entries = sorted(p for p in d.glob("*.md") if p.name != "MEMORY.md")
        lines = ["# 记忆索引(自动生成,勿手改)", ""]
        for p in entries:
            title, description, mem_type = _parse_memory_text(p.read_text(encoding="utf-8"))
            desc = (description or title or p.stem)[:_DESC_MAX_CHARS]
            # 链接文本用名字(不用标题):模型"同名覆盖更新"要拿名字当钥匙
            lines.append(f"- [{p.stem}]({p.name}) — {desc}({mem_type})")
        (d / "MEMORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _age_label(mtime: float, *, now: float | None = None) -> str:
    """把文件的修改时间转成"今天 / N 天前"(超 _STALE_DAYS 天加 ⚠旧)。

    模型不擅长日期算术,裸时间戳引不起"这事可能变了"的推理,
    要转成口语年龄(CC 报告 §4.5)。

    示例:_age_label(三天前的时间戳) → "3 天前"
    """
    ref = datetime.fromtimestamp(now if now is not None else time.time())
    seen = datetime.fromtimestamp(mtime)
    days = (ref.date() - seen.date()).days
    if days <= 0:
        return "今天"
    label = f"{days} 天前"
    return f"{label}⚠旧" if days > _STALE_DAYS else label


def _cap_lines(lines: list[str]) -> tuple[list[str], str | None]:
    """给一层索引行做双上限截断,返回 (截断后的行, 触发的上限名)。

    先按行数截、再按字节截(CC 报告 §2.6 的顺序);都没超就原样返回。
    字节截断按 UTF-8 切,截断点可能劈开多字节字符或半行,忽略之。
    """
    hit: str | None = None
    if len(lines) > _INDEX_MAX_LINES:
        lines, hit = lines[:_INDEX_MAX_LINES], "行数"
    raw = "\n".join(lines).encode("utf-8")
    if len(raw) > _INDEX_MAX_BYTES:
        cut = raw[: _INDEX_MAX_BYTES - 200].decode("utf-8", errors="ignore")
        lines = cut.split("\n")[:-1]  # 丢掉可能被劈半的最后一行
        hit = "字节"
    return lines, hit


def render_memory_index(workspace: Path) -> str:
    """拼出注入系统提示词的记忆索引文本(两层合一,带新鲜度和双上限)。

    直接从记忆文件现算(不读 MEMORY.md —— 那是给人看的镜像),
    每行 = 可直接 read 的链接 + 描述 + 类型 + 年龄。

    示例:
        # 长期记忆
        ## 项目记忆
        - [部署流程](memory/部署流程.md) — 登录与发布步骤(project·今天)
        ## 用户记忆
        - [喜欢的风格](/home/xx/.local/share/.../喜欢的风格.md) — 回答要简短(user·3 天前)
    """
    entries = list_memories(workspace)
    sections: list[str] = []
    # 把两层目录和它们的名字配成对
    # 第 1 圈:d = 工作区/memory,  label = "项目记忆"
    # 第 2 圈:d = 数据目录/memory, label = "用户记忆"
    for d, label in zip(memory_dirs(workspace), ("项目记忆", "用户记忆"), strict=False):
        rows: list[str] = []
        for e in entries:  # list_memories 已按同名去重(近层盖远层)
            if e.path.parent != d:
                continue
            age = _age_label(e.path.stat().st_mtime)
            try:  # 项目层 → 相对路径(模型 read 认),用户层 → 绝对路径
                shown: str | Path = e.path.relative_to(workspace)
            except ValueError:
                shown = e.path
            desc = e.description[:_DESC_MAX_CHARS]
            rows.append(f"- [{e.name}]({shown}) — {desc}({e.mem_type}·{age})")
        if not rows:
            continue
        rows, hit = _cap_lines(rows)
        note = f"\n…({label}超过{hit}上限已截断,细节用 read 看对应文件)" if hit else ""
        sections.append(f"## {label}\n" + "\n".join(rows) + note)
    if not sections:
        return ""
    # 头部带"引用前先验证"的行为协议(CC 报告 §4.5:记忆说 X 存在 ≠ X 现在存在)
    return (
        "# 长期记忆\n(需要细节时用 read 工具查看对应文件;"
        "记忆里说的东西不一定现在还成立,引用前先验证)\n\n" + "\n\n".join(sections)
    )
