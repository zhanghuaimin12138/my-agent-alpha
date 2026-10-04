"""长期记忆:跨会话记住事情(gptme memory 的最小版)。

设计(仿 gptme 的 memory 模块,见研究报告 context-engine 节):

- 存储形态:一条记忆 = 一个 markdown 文件(YAML 头 + 正文),
  人可以直接打开看、改、删;
- 两层目录(近的盖住远的,仿 gptme 的"nearest-first"):
    工作区/memory/   ← 当前项目的记忆(优先)
    数据目录/memory/ ← 用户级记忆(~/.local/share/my_agent_alpha/memory)
- MEMORY.md:自动生成的指针索引(每条记忆一行),会话开始时注入
  系统提示词(有字节预算,gptme 是 64KB)—— 模型由此"知道"自己记过什么,
  需要细节时用 read 工具去看原文;
- 写入:模型调用 memory 工具提出"想记一条",经确认链(交互时用户点头)
  才落盘 —— 记忆是要进每次上下文的,不该让模型随手乱写。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from my_agent_alpha.dirs import get_data_dir

# 会话开始时注入的索引预算(字节);超了截断并注明(gptme 同款 64KB)
_MEMORY_BUDGET_BYTES = 64 * 1024


@dataclass(frozen=True)
class MemoryEntry:
    """一条记忆的元信息(从文件名和头部解析)。"""

    name: str  # 记忆名(文件名去掉 .md)
    path: Path
    title: str  # 首行标题


def memory_dirs(workspace: Path) -> list[Path]:
    """两层记忆目录,近的在前(工作区层盖住用户层)。"""
    return [workspace / "memory", get_data_dir() / "memory"]


def _slug(name: str) -> str:
    """把名字变成安全的文件名:只留中英文数字和连字符。"""
    slug = re.sub(r"[^\w一-鿿-]+", "-", name.strip()).strip("-")
    return slug or "memory"


def list_memories(workspace: Path) -> list[MemoryEntry]:
    """列出两层目录的全部记忆(同名时近层覆盖远层)。"""
    by_name: dict[str, MemoryEntry] = {}
    for d in memory_dirs(workspace):  # 近的先放,远层同名不覆盖
        if not d.exists():
            continue
        for path in sorted(d.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            title = path.read_text(encoding="utf-8").split("\n", 1)[0].lstrip("# ").strip()
            by_name.setdefault(path.stem, MemoryEntry(path.stem, path, title or path.stem))
    return list(by_name.values())


def save_memory(name: str, content: str, workspace: Path, project_level: bool = False) -> Path:
    """写一条记忆并重建索引,返回文件路径。

    project_level=True 写进工作区层(项目记忆),否则用户层。
    内容按"首行标题 + 空行 + 正文"约定存储。
    """
    target_dir = memory_dirs(workspace)[0 if project_level else 1]
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{_slug(name)}.md"
    lines = content.strip().split("\n")
    title = lines[0].lstrip("# ").strip()
    body = "\n".join(lines[1:]).strip()
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    path.write_text(f"# {title}\n\n{body}\n\n*记录于 {stamp}*\n", encoding="utf-8")
    rebuild_index(workspace)
    return path


def rebuild_index(workspace: Path) -> None:
    """重建每层的 MEMORY.md 索引(一行一条:链接 + 标题)。"""
    for d in memory_dirs(workspace):
        if not d.exists():
            continue
        entries = sorted(p for p in d.glob("*.md") if p.name != "MEMORY.md")
        lines = ["# 记忆索引(自动生成,勿手改)", ""]
        for p in entries:
            title = p.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
            lines.append(f"- [{p.stem}]({p.name}) — {title}")
        (d / "MEMORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _with_real_paths(body: str, layer_dir: Path, workspace: Path) -> str:
    """把索引行里的裸文件名补成模型能直接 read 的路径。

    MEMORY.md 里写的是相对名(对人友好:文件就在旁边);但注入给模型时
    它不知道目录在哪 —— 项目层补成"memory/xxx.md"(相对工作区,read 认),
    用户层补成绝对路径。
    """
    try:
        prefix = layer_dir.relative_to(workspace)  # 项目层 → "memory"
    except ValueError:
        prefix = layer_dir  # 用户层(不在工作区下)→ 绝对路径
    return re.sub(
        r"\]\(([^)/]+\.md)\)",  # ](文件.md) 且路径里还没有目录的
        lambda m: f"]({prefix}/{m.group(1)})",
        body,
    )


def render_memory_index(workspace: Path, budget: int = _MEMORY_BUDGET_BYTES) -> str:
    """拼出注入系统提示词的记忆索引文本(两层合一,带预算截断)。

    示例:
        # 长期记忆
        ## 项目记忆(工作区/memory.md)
        - [部署流程](部署流程.md) — 服务器登录和发布步骤
        ## 用户记忆(数据目录/memory.md)
        - [喜欢的风格](喜欢的风格.md) — 回答要简短
    """
    sections: list[str] = []
    # 把两层目录和它们的名字配成对
    # 第 1 圈:d = 工作区/memory,  label = "项目记忆"
    # 第 2 圈:d = 数据目录/memory, label = "用户记忆"
    for d, label in zip(memory_dirs(workspace), ("项目记忆", "用户记忆"), strict=False):
        index = d / "MEMORY.md"
        if not index.exists():
            continue
        text = index.read_text(encoding="utf-8").strip()
        # 去掉自动生成文件自己的标题行,换成层级标题
        body = "\n".join(text.split("\n")[1:]).strip()
        if body:
            # 裸文件名补成模型能直接 read 的路径(项目层相对、用户层绝对)
            body = _with_real_paths(body, d, workspace)
            sections.append(f"## {label}\n{body}")
    if not sections:
        return ""
    result = "# 长期记忆\n(需要细节时用 read 工具查看对应文件)\n\n" + "\n\n".join(sections)
    if len(result.encode("utf-8")) > budget:
        result = result[: budget - 60].rsplit("\n", 1)[0] + "\n…(记忆过多已截断,可用 /memory 查看)"
    return result
