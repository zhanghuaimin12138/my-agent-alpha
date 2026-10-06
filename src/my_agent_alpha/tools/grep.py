"""grep 工具:在文件内容里搜索,输出"相对路径:行号:命中行"。

为什么有了 shell 还要它(对照 Claude Code 的 Grep、round2-plan R3):
- 输出格式固定紧凑,模型拿来就能定位,配合 read 的行号输出和
  edit 的精确替换,构成"找代码 → 看代码 → 改代码"的工作流;
- 只读、免确认(shell 跑 grep 要过确认链的分诊);
- 默认忽略垃圾目录(.git/__pycache__ 等)和非文本文件。

实现是纯 Python 递归(不包装系统 rg/grep:本机没有 rg,系统 grep
又有版本差异;本项目规模的仓库毫秒级就能搜完,零外部依赖还让测试
完全稳定 —— 偏离路线图"包装系统工具"的写法,理由记录在 devlog 020)。

用法(围栏体第一行直接写要搜的词,或按 key=value 每行一个参数):

    ```grep
    def execute_
    ```

    ```grep
    pattern=TODO|FIXME
    glob=*.py
    path=src
    ```

- pattern:Python 正则(re 语法),默认区分大小写,忽略大小写写 (?i);
- glob:只搜文件名匹配它的文件,如 *.py;
- path:只在某个子目录/文件里搜,默认整个工作区。

原生通道参数:{"pattern": ..., "glob": ..., "path": ...}。
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Iterator
from pathlib import Path

from my_agent_alpha.context import get_workspace
from my_agent_alpha.message import Message
from my_agent_alpha.tools.base import Parameter, ToolSpec

# 递归时跳过的目录(版本库/缓存/构建产物,搜它们只有噪音)
_SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)

_MAX_HITS = 200  # 最多报告多少个命中行(防止一个宽泛 pattern 刷屏)
_MAX_LINE_CHARS = 200  # 单行命中最长展示多少字符(超长行截断)


def _collect_params(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> tuple[dict[str, str], str | None]:
    """从两种通道汇总 {pattern, glob, path};出错时返回 (参数, 错误说明)。

    原生通道:kwargs 三个具名参数;
    markdown 通道:围栏头第一个词可当 pattern(不含空格时),
    围栏体每行一个 "key=value",或第一行裸写 pattern。
    """
    params: dict[str, str] = {}
    if kwargs:  # 原生通道
        for key in ("pattern", "glob", "path"):
            if kwargs.get(key):
                params[key] = kwargs[key]
    else:  # markdown 通道
        if args:
            params.setdefault("pattern", args[0])  # 围栏头:```grep TODO
            if len(args) > 1:
                params.setdefault("glob", args[1])
        if code:
            for raw in code.split("\n"):
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(("pattern=", "glob=", "path=")):
                    key, _, value = line.partition("=")
                    params[key] = value
                elif "pattern" not in params:
                    params["pattern"] = line  # 第一行裸写:pattern 本身
                else:
                    return params, f"错误: 围栏里没法理解的行:{line}(参数请写 key=value 形式)"
    if not params.get("pattern", "").strip():
        usage = "文法:pattern=<正则> | glob=<如 *.py> | path=<子目录或文件>"
        return params, f"错误: grep 需要 pattern(围栏体第一行,或 pattern= 参数)。\n{usage}"
    return params, None


def _iter_files(root: Path) -> Iterator[Path]:
    """递归列出 root 下值得搜的文本文件(跳过垃圾目录和隐藏文件)。"""
    for dirpath, dirnames, filenames in os.walk(root):
        # 原地过滤 dirnames:os.walk 后面就不会再走进这些目录
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            yield Path(dirpath) / name


def execute_grep(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> Iterator[Message]:
    """grep 工具的执行入口:解析参数 → 递归搜索 → 汇报命中。"""
    params, error = _collect_params(code, args, kwargs)
    if error:
        yield Message("system", error)
        return

    pattern = params["pattern"]
    try:
        rx = re.compile(pattern)
    except re.error as e:
        yield Message("system", f"错误: pattern 不是合法的正则:{e}")
        return

    workspace = get_workspace()
    target = Path(params.get("path", ""))
    if not str(target).strip():
        target = workspace
    elif not target.is_absolute():
        target = workspace / target
    if not target.exists():
        yield Message("system", f"错误: 搜索范围不存在:{target}")
        return

    glob = params.get("glob")
    if target.is_file():
        files: Iterator[Path] = iter([target])
    else:
        files = (f for f in _iter_files(target) if not glob or fnmatch.fnmatch(f.name, glob))

    hits: list[str] = []
    truncated = False
    scanned = 0
    for file in files:
        scanned += 1
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # 二进制/权限等读不了的文件直接跳过
        rel = file.relative_to(workspace) if file.is_relative_to(workspace) else file
        for lineno, line in enumerate(text.split("\n"), start=1):
            if rx.search(line):
                shown = line.strip()
                if len(shown) > _MAX_LINE_CHARS:
                    shown = shown[:_MAX_LINE_CHARS] + "…"
                hits.append(f"{rel}:{lineno}:{shown}")
                if len(hits) >= _MAX_HITS:
                    truncated = True
                    break
        if truncated:
            break

    if not hits:
        scope = str(target)
        extra = f"(只搜 {glob})" if glob else ""
        yield Message(
            "system",
            f"没有匹配:pattern={pattern!r} 在 {scope} 下 {extra},共扫了 {scanned} 个文件。",
        )
        return
    header = (
        f"grep {pattern!r}:命中 {len(hits)} 行"
        + ("(已达上限,建议收窄 pattern 或加 glob)" if truncated else "")
        + ":"
    )
    yield Message("system", header + "\n" + "\n".join(hits))


tool = ToolSpec(
    name="grep",
    desc=(
        "在文件内容里搜索,输出 相对路径:行号:命中行,找代码/找定义/找 TODO 用它;"
        "拿到结果后可直接 read 对应文件、edit 对应行。"
    ),
    instructions=(
        "pattern 是 Python 正则,默认区分大小写(忽略大小写写 (?i));\n"
        "可选参数:glob=*.py 只搜某类文件;path=src 只搜某个子目录或文件,默认整个工作区。\n"
        "围栏体第一行直接写 pattern(或用 pattern= 每行一个参数)。\n"
        "没有匹配时会明确说没有匹配。"
    ),
    examples=("```grep\ndef execute_\n```\n```grep\npattern=TODO|FIXME\nglob=*.py\n```"),
    block_types=("grep",),
    parameters=(
        Parameter("pattern", "string", "要搜索的正则表达式"),
        Parameter("glob", "string", "只搜文件名匹配它的文件,如 *.py", required=False),
        Parameter("path", "string", "搜索范围(子目录或文件),默认工作区", required=False),
    ),
    read_only=True,
    execute=execute_grep,
)
