"""会话列表与元数据(027-S1 函数化 + 032 /resume 支撑)。

会话的"目录名/标题/最后修改时间/用户消息数"这些展示用信息,从命令里
零散扫目录升级为正式函数层;标题读写也从 hooks/naming.py 搬到这里
(标题本来就是会话元数据,放 logmanager 包里依赖方向才干净)。

个人量级(会话几十个、文件都是 KB 级)不做扫描优化;"只读尾部 8KB"
的快扫已在积压表,会话多到列表变慢再做。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from my_agent_alpha.logmanager.manager import CONVERSATION_FILENAME

# meta.json 的文件名(标题存在会话目录里,目录名本身不动)
_META_FILENAME = "meta.json"

# 子代理会话目录名的前缀(它们不是给人续聊的,列表要过滤)
_SUBAGENT_PREFIX = "subagent-"


@dataclass(frozen=True)
class SessionMeta:
    """一个会话的元信息(列表展示用,不含对话内容)。

    示例:logdir = …/logs/3f2a91c0…,title = "调试打印问题",n_user = 5。
    """

    logdir: Path
    session_id: str  # 目录名(uuid4 或 subagent-uuid4)
    title: str | None  # meta.json 里的标题;可能没有(None)
    mtime: float  # 会话目录最后修改时间(排序用)
    n_user: int  # 用户消息条数(=0 的会话视为"没人聊过",列表过滤、退出清理)


def read_session_title(logdir: Path) -> str | None:
    """读会话标题;没有 meta.json、文件坏了、标题是空的,都返回 None。"""
    try:
        data = json.loads((logdir / _META_FILENAME).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    title = data.get("title")
    return title if isinstance(title, str) and title.strip() else None


def write_session_title(logdir: Path, title: str) -> None:
    """把标题原子写进 meta.json(先写临时文件再改名,中途崩溃不留半个文件)。

    先确保目录存在:main 的 --name 分支在 chat() 创建 logdir 之前就会
    调到这里(subagent 子进程正是这种场景),目录不在就得自己建。
    """
    logdir.mkdir(parents=True, exist_ok=True)
    tmp = logdir / f"{_META_FILENAME}.tmp"
    tmp.write_text(json.dumps({"title": title}, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, logdir / _META_FILENAME)


def _parse_roles(logdir: Path) -> tuple[int, int]:
    """数会话里的(用户消息数, 助手消息数);文件缺失算 (0, 0),坏行跳过(不较真)。"""
    try:
        raw = (logdir / CONVERSATION_FILENAME).read_bytes()
    except OSError:
        return 0, 0
    n_user = n_assistant = 0
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            role = json.loads(line.decode("utf-8")).get("role")
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if role == "user":
            n_user += 1
        elif role == "assistant":
            n_assistant += 1
    return n_user, n_assistant


def list_sessions(root: Path) -> list[SessionMeta]:
    """列出会话根目录下的全部会话,按最后修改时间倒序(新的在前)。

    示例:for m in list_sessions(logdir.parent): print(m.title, m.n_user)
    """
    if not root.is_dir():
        return []
    metas = [
        SessionMeta(
            logdir=d,
            session_id=d.name,
            title=read_session_title(d),
            mtime=d.stat().st_mtime,
            n_user=_parse_roles(d)[0],
        )
        for d in root.iterdir()
        if d.is_dir()
    ]
    return sorted(metas, key=lambda m: m.mtime, reverse=True)


def find_by_prefix(root: Path, prefix: str) -> list[SessionMeta]:
    """/resume 用:按会话 id 前缀找会话,0/1/多个都如实返回,由调用方处理。"""
    return [m for m in list_sessions(root) if m.session_id.startswith(prefix)]


def is_subagent(meta: SessionMeta) -> bool:
    """是不是子代理的工作记录(目录名 subagent- 开头)。"""
    return meta.session_id.startswith(_SUBAGENT_PREFIX)


def prune_empty_session(logdir: Path) -> bool:
    """删掉"没有任何对话内容"的会话目录(只剩系统提示词),返回是否删了。

    判定标准:用户消息和助手消息都为 0。护栏:只要有一句对话
    (哪怕是只有助手单方面说过的)就绝不删;没人聊过的会话
    (--name 写了标题就退出、或刚开就 /resume 跳走)是噪音,清掉。
    """
    if not logdir.is_dir():
        return False
    if _parse_roles(logdir) != (0, 0):
        return False
    shutil.rmtree(logdir)
    return True


def format_session_row(meta: SessionMeta, index: int | None = None) -> str:
    """把一条会话元信息格式化成 /log list 和 /resume 选择器共用的一行。

    示例:format_session_row(m)      → "  10-26 14:24  测试日志  (318a073d · 5问)"
          format_session_row(m, 1)  → "  1) 10-26 14:24  测试日志  (318a073d · 5问)"
    """
    ts = datetime.fromtimestamp(meta.mtime)
    title = meta.title or "(未命名)"
    lead = f"{index}) " if index is not None else "  "
    return f"{lead}{ts:%m-%d %H:%M}  {title}  ({meta.session_id[:8]} · {meta.n_user}问)"
