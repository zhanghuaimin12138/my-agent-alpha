"""read 工具:读取文件内容(带行号)。

用法(仿 gptme):路径既可写在 fence 头(单文件),也可写在代码块里(每行一个):

    ```read src/main.py
    ```

    ```read
    src/main.py
    pyproject.toml
    ```
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from my_agent_alpha.context import get_workspace
from my_agent_alpha.message import Message
from my_agent_alpha.tools.base import ToolSpec

_MAX_CHARS = 200_000  # 阶段 6 之前先用朴素字符上限兜底


def _read_one(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"错误: 文件不存在: {path}"
    except IsADirectoryError:
        return f"错误: 是目录不是文件: {path}"
    except UnicodeDecodeError:
        return f"错误: 不是 UTF-8 文本文件: {path}"
    except OSError as e:
        return f"错误: 读取失败: {path}: {e}"

    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + f"\n... [文件过大,已截断,共 {len(text)} 字符]"
    lines = text.split("\n")
    numbered = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(lines, start=1))
    return f"[{path}]\n{numbered}"


def execute_read(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> Iterator[Message]:
    """路径来源:fence 头参数(单文件)+ 代码块每行一个路径;相对路径相对工作区。"""
    assert code is not None
    workspace = get_workspace()
    paths = list(args or [])
    paths += [line.strip() for line in code.split("\n") if line.strip()]
    if not paths:
        yield Message("system", "错误: read 没有给出任何路径(fence 头或代码块内)")
        return
    results = [_read_one((workspace / p) if not Path(p).is_absolute() else Path(p)) for p in paths]
    yield Message("system", "\n\n".join(results))


tool = ToolSpec(
    name="read",
    desc="读取一个或多个文件的内容,输出带行号,便于之后精确引用或修改。",
    instructions="代码块里每行一个路径;相对路径相对当前工作区。",
    examples="```read\nsrc/main.py\npyproject.toml\n```",
    block_types=("read",),
    read_only=True,
    execute=execute_read,
)
