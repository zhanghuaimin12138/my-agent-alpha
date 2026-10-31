"""压缩视图(View)的持久化辅助:视图编号、views/active.json 的读写。

三个概念的关系(完整方案见 docs/plan/my-agent-alpha-view-development-plan.md):

- Master = conversation.jsonl:完整历史,只追加,压缩永远不改写它;
- View   = views/compacted-NNN.jsonl:压缩后的"当前有效历史",
  模型请求以它为输入,激活后新消息会继续追加进来;
- active.json = 一张小卡片,记着"当前激活哪个 View + 双方各写到了第几条"。
  没有激活 View 时这个文件直接不存在(比存一个空值少一种状态)。

为什么要记"写到了第几条"(而不是像 gptme 只记 View 名):Master 和 View
是两个文件,没法一步原子地同时写完;进程可能在"Master 写完、View 还没写"
的中间崩溃。这两个计数就是崩溃后把 View 尾部裁齐/补齐的依据
(修复逻辑在 LogManager.load,本模块只提供读写与校验)。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 会话目录下的视图子目录名与状态文件名
VIEWS_DIRNAME = "views"
ACTIVE_STATE_FILENAME = "active.json"

# active.json 的格式版本:将来字段变化时递增;读到不认识的版本就当作状态无效
STATE_VERSION = 1

# 合法 View ID 的样子:compacted- 加至少 3 位数字,如 compacted-001。
# 为什么限制得这么死:ID 会被拼进文件路径,放行 "…"、"/" 之类就能越出 views/ 目录
_VIEW_ID_RE = re.compile(r"^compacted-\d{3,}$")


@dataclass(frozen=True)
class ViewState:
    """views/active.json 的内容:当前激活的视图 + 双写进度检查点。

    两个计数字段是崩溃恢复的依据,不是压缩语义(压缩本身就会让两边条数不同):
    - master_committed_count:这张卡片签发时,Master 已确认写到第几条;
    - view_committed_count:这张卡片签发时,当前 View 已确认写到第几条。
    崩溃后磁盘上的文件可能比卡片多写了几行,凭这两个数把多写的部分裁掉或补齐。
    """

    active_view: str | None
    master_committed_count: int
    view_committed_count: int


def views_dir(logdir: Path) -> Path:
    """会话目录下的 views/ 子目录(可能还不存在,写入时会自动创建)。"""
    return logdir / VIEWS_DIRNAME


def view_path(logdir: Path, view_id: str) -> Path:
    """某个视图文件的完整路径。

    示例:view_path(logdir, "compacted-001") → …/views/compacted-001.jsonl
    """
    return views_dir(logdir) / f"{view_id}.jsonl"


def is_valid_view_id(view_id: str) -> bool:
    """判断视图 ID 是否合法(只认 compacted- 加数字,见 _VIEW_ID_RE 注释)。"""
    return _VIEW_ID_RE.match(view_id) is not None


def next_view_id(logdir: Path) -> str:
    """生成下一个视图编号:扫描 views/ 里现有的 compacted-NNN.jsonl,最大号 +1。

    示例:已有 compacted-001、compacted-002 → 返回 "compacted-003";
    目录不存在或为空 → 返回 "compacted-001"。扫描而不是记忆计数,
    这样手动删掉一个视图文件后编号也不会撞车。
    """
    biggest = 0
    d = views_dir(logdir)
    if d.is_dir():
        for f in d.iterdir():
            m = re.fullmatch(r"compacted-(\d+)\.jsonl", f.name)
            if m is not None:
                biggest = max(biggest, int(m.group(1)))
    return f"compacted-{biggest + 1:03d}"


def read_active_state(logdir: Path) -> ViewState | None:
    """读 views/active.json;文件不存在返回 None(正常:还没激活过视图)。

    内容损坏、版本不认识、字段类型不对、视图名非法 → 打警告并返回 None,
    让调用方安全回退到 Master;宁可当作"没有视图",不拿可疑状态当真。
    """
    path = views_dir(logdir) / ACTIVE_STATE_FILENAME
    if not path.exists():
        return None
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("视图状态文件读取失败,忽略(%s): %s", path, e)
        return None
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        logger.warning("视图状态文件版本不认识,忽略: %s", path)
        return None
    view_id = data.get("active_view")
    if view_id is not None and (not isinstance(view_id, str) or not is_valid_view_id(view_id)):
        logger.warning("视图状态里的视图名非法,忽略: %s", path)
        return None
    master_count = data.get("master_committed_count")
    view_count = data.get("view_committed_count")
    # isinstance(True, int) 在 Python 里是 True,所以额外排除 bool:
    # 计数字段混进 true/false 属于文件坏了,不是 1/0
    if (
        not isinstance(master_count, int)
        or isinstance(master_count, bool)
        or master_count < 0
        or not isinstance(view_count, int)
        or isinstance(view_count, bool)
        or view_count < 0
    ):
        logger.warning("视图状态里的计数字段非法,忽略: %s", path)
        return None
    return ViewState(
        active_view=view_id,
        master_committed_count=master_count,
        view_committed_count=view_count,
    )


def write_active_state(logdir: Path, state: ViewState) -> None:
    """把视图状态原子写入 views/active.json。

    写法与 Log._rewrite_jsonl 同款:先写临时文件 → fsync → os.replace 原子顶替
    → 目录 fsync。任何时刻断电,磁盘上要么是完整的新状态、要么是完整的旧状态。
    """
    d = views_dir(logdir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / ACTIVE_STATE_FILENAME
    payload = {
        "version": STATE_VERSION,
        "active_view": state.active_view,
        "master_committed_count": state.master_committed_count,
        "view_committed_count": state.view_committed_count,
    }
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_directory(d)
    finally:
        # 异常时清掉残留临时文件(正常路径下 os.replace 已把它变成目标文件)
        if tmp.exists():
            tmp.unlink()


def remove_active_state(logdir: Path) -> None:
    """删除 views/active.json(不再有激活视图时,状态文件一并消失)。

    文件不存在时静默返回;删完对目录做 fsync,让"删除"本身也不怕断电回退。
    """
    d = views_dir(logdir)
    path = d / ACTIVE_STATE_FILENAME
    if path.exists():
        path.unlink()
        _fsync_directory(d)


def _fsync_directory(path: Path) -> None:
    """把目录本身的改动强制写入磁盘,让 os.replace 的改名不会因断电而丢失。

    与 manager.py 里的同名函数同款;这里再放一份是为了让本模块不反向
    依赖 manager(否则两个文件互相 import,谁先加载都缺对方的名字)。
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # 例如 EINVAL/ENOTSUP:有的操作系统不支持对目录做 fsync
    finally:
        os.close(fd)
