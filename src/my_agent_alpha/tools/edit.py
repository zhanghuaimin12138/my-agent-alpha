"""edit 工具:精确修改文件的一小段(第二轮 R1)。

为什么必须有它:save 是"整文件重写"—— 模型改一行代码也得把整个文件
重新输出一遍,输出中途断掉/漏几行,文件就毁了。edit 只要求给出
"旧文本 → 新文本",其余内容原样保留 —— 这是 agent 安全改代码的前提。

用法(对照 Claude Code 的 Edit,gptme 的 patch):

    ```edit path/to/file.py
    旧的代码(要被替换掉的原文,必须和文件内容一字不差)
    =====
    新的代码(替换上去的内容)
    ```

规则:
- 旧文本在文件中必须**恰好出现一次**:出现多次 → 报错,要求带上
  更多上下文(前后多几行)让它唯一;出现零次 → 报错,并打印文件里
  最相似的位置的行号区间(配合 read 的行号输出,模型能马上定位);
- 整个过程走确认链(非只读),预览显示改动位置和内容;
- 写回用原子写(先写临时文件再改名),中途崩溃不会毁原文件。

原生通道参数:{"path": ..., "old": ..., "new": ...}。
"""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Iterator
from pathlib import Path

from my_agent_alpha.context import get_workspace
from my_agent_alpha.message import Message
from my_agent_alpha.tools.base import Parameter, ToolSpec

# 旧/新文本的分隔行:一串 =(至少 5 个),前后容许空白
_SEPARATOR = re.compile(r"^={5,}\s*$")


def _parse_body(body: str) -> tuple[str, str] | str:
    """把围栏内容切成 (旧文本, 新文本);格式不对时返回错误说明。"""
    lines = body.split("\n")
    sep_indexes = [i for i, line in enumerate(lines) if _SEPARATOR.match(line)]
    if len(sep_indexes) == 0:
        return "错误: edit 的围栏内容需要用一行 '====='(至少5个等号)分隔旧文本和新文本"
    if len(sep_indexes) > 1:
        return "错误: edit 的围栏内容里只能有一行 '=====' 分隔线(旧/新文本内部不要出现整行等号)"
    sep = sep_indexes[0]
    old = "\n".join(lines[:sep]).rstrip("\n")
    new = "\n".join(lines[sep + 1 :])
    if not old.strip():
        return "错误: edit 的旧文本是空的 —— 必须给出要被替换的原文字"
    return old, new


def _find_match(content: str, old: str) -> tuple[int, int] | None:
    """找旧文本在文件中的 [起,止) 字符位置;恰好一次才返回,否则 None。"""
    first = content.find(old)
    if first < 0:
        return None
    second = content.find(old, first + 1)
    if second >= 0:
        return None  # 多处出现
    return first, first + len(old)


def _line_of(content: str, pos: int) -> int:
    """字符位置 → 行号(从 1 数)。"""
    return content.count("\n", 0, pos) + 1


def _nearest_hint(content: str, old: str) -> str:
    """旧文本找不到时,给出最相似位置的提示(行号 + 相似度)。"""
    old_lines = old.strip().split("\n")
    content_lines = content.split("\n")
    best_ratio, best_line = 0.0, 0
    for i in range(len(content_lines) - len(old_lines) + 1):
        window = "\n".join(content_lines[i : i + len(old_lines)])
        ratio = difflib.SequenceMatcher(None, old.strip(), window).ratio()
        if ratio > best_ratio:
            best_ratio, best_line = ratio, i + 1
    if best_ratio < 0.4:
        return "(文件里没有明显相似的内容,建议先 read 确认)"
    return (
        f"最相似的位置在第 {best_line} 行附近(相似度 {best_ratio:.0%}),"
        "可能是有细微出入 —— 建议先 read 再原样复制"
    )


def _apply_edit(path: Path, old: str, new: str) -> str:
    """执行替换并原子写回;成功返回结果说明,失败返回错误说明。"""
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"错误: 文件不存在: {path}"
    except UnicodeDecodeError:
        return f"错误: 不是 UTF-8 文本文件: {path}"

    match = _find_match(content, old)
    if match is None:
        first = content.find(old)
        if first >= 0:
            return (
                f"错误: 旧文本在文件中出现了多处,无法确定改哪一处 —— "
                f"请把前后几行一起包进旧文本,让它唯一(第一处在第 {_line_of(content, first)} 行)"
            )
        return f"错误: 旧文本在文件中找不到。\n{_nearest_hint(content, old)}"

    start, end = match
    new_content = content[:start] + new + content[end:]
    # 原子写:临时文件 → 改名(和 LogManager 同款套路)
    tmp = path.with_name(f"{path.name}.edit-tmp-{os.getpid()}")
    try:
        tmp.write_text(new_content, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    changed = "+1" if old != new else "无变化"
    return (
        f"已修改 {path}:第 {_line_of(content, start)} 行起,"
        f"替换 {end - start} 字符({changed})。\n"
        f"如需验证,read 该文件对应区域即可。"
    )


def execute_edit(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> Iterator[Message]:
    """edit 工具的执行函数:解析旧/新文本 → 精确替换 → 原子写回。

    markdown:
        code(content):内容 ; args:路径 ;kwargs: none
    native:
        kwargs: {"path": " ", "old": "content1" , "new" : "context2"}
    """
    path_str = (kwargs.get("path") if kwargs else None) or (args[0] if args else "")
    if not path_str:
        yield Message("system", "错误: edit 需要文件路径(fence 头或 path 参数)")
        return
    p = Path(path_str)
    path = p if p.is_absolute() else get_workspace() / p

    if kwargs:  # 原生通道:三个具名参数,直接用
        old = kwargs.get("old", "")
        new = kwargs.get("new", "")
        if not old.strip():
            yield Message("system", "错误: edit 的 old 参数不能为空")
            return
    else:  # markdown 通道:围栏内容按 "旧=====新" 解析
        assert code is not None
        parsed = _parse_body(code)
        if isinstance(parsed, str):
            yield Message("system", parsed)
            return
        old, new = parsed

    yield Message("system", _apply_edit(path, old, new))


tool = ToolSpec(
    name="edit",
    desc=(
        "精确修改文件的一小段:给出旧文本和新文本,只有这一段被替换,"
        "其余原样保留。改代码/改配置首选它,不要用 save 重写整个文件。"
    ),
    instructions=(
        "旧文本必须和文件内容一字不差(建议先 read,原样复制要改的那几行,\n"
        "必要时带上前后行让它唯一);新旧文本之间用一行等号(=====)分隔。"
    ),
    examples=(
        "```edit src/app.py\n"
        "def hello():\n"
        "    print('hi')\n"
        "=====\n"
        "def hello():\n"
        "    print('hello world')\n"
        "```"
    ),
    block_types=("edit",),
    parameters=(
        Parameter("path", "string", "要修改的文件路径"),
        Parameter("old", "string", "要被替换的原文(须在文件中唯一)"),
        Parameter("new", "string", "替换后的新文本"),
    ),
    execute=execute_edit,
)
