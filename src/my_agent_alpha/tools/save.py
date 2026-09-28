"""save / append 工具:写文件。

用法(仿 gptme):fence 头带路径参数,代码块即文件内容。

    ```save path/to/file
    文件内容...
    ```
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from my_agent_alpha.context import get_workspace
from my_agent_alpha.message import Message
from my_agent_alpha.tools.base import ToolSpec

# 占位符检测:LLM 常用 "# ..." / "..." 代替真实实现,直接写盘只会产出垃圾文件
_PLACEHOLDERS = ("# ...", "# TODO: implement", "...", "…")


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else get_workspace() / p


def _check_placeholder(path: Path, content: str) -> str | None:
    """整个文件内容恰好是占位符才拒绝(审查 #7:'...' 作为子串会误伤正常短文件)。"""
    stripped = content.strip()
    if stripped in _PLACEHOLDERS:
        return f"拒绝写入 {path}:内容疑似占位符({stripped!r}),请给出完整内容"
    return None


def _validate(code: str | None, args: list[str] | None) -> tuple[Path, str] | str:
    if not args:
        return "错误: save/append 的 fence 头需要路径参数,如 ```save path/to/file"
    assert code is not None
    return _resolve(args[0]), code


def execute_save(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> Iterator[Message]:
    validated = _validate(code, args)
    if isinstance(validated, str):
        yield Message("system", validated)
        return
    path, content = validated
    if reject := _check_placeholder(path, content):
        yield Message("system", reject)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    path.write_text(content, encoding="utf-8")
    verb = "覆盖" if existed else "创建"
    yield Message("system", f"已{verb} {path}({len(content.encode('utf-8'))} 字节)")


def execute_append(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> Iterator[Message]:
    validated = _validate(code, args)
    if isinstance(validated, str):
        yield Message("system", validated)
        return
    path, content = validated
    if content and not content.endswith("\n"):
        content += "\n"  # 新内容保证以换行收尾
    if path.exists() and path.stat().st_size > 0:
        # 原文件末尾没换行就先补一个,避免黏到上一行尾
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                content = "\n" + content
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
    yield Message("system", f"已追加到 {path}")


save_tool = ToolSpec(
    name="save",
    desc="创建或覆盖一个文件。fence 头带路径,代码块是完整文件内容。",
    instructions="必须写入完整内容;不确定就先 read 再改。不允许占位符。",
    examples="```save hello.py\nprint('hello')\n```",
    block_types=("save",),
    execute=execute_save,
)

append_tool = ToolSpec(
    name="append",
    desc="向文件末尾追加内容(自动补换行)。fence 头带路径。",
    examples="```append notes.md\n- 新的一行\n```",
    block_types=("append",),
    execute=execute_append,
)
