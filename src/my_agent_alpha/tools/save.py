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
from my_agent_alpha.tools.base import Parameter, ToolSpec

# 占位符检测:LLM 常用 "# ..." / "..." 代替真实实现,直接写盘只会产出垃圾文件
_PLACEHOLDERS = ("# ...", "# TODO: implement", "...", "…")


def _resolve(path_str: str) -> Path:
    """把路径解析成绝对路径:相对路径以当前工作区为基准。"""
    p = Path(path_str)
    return p if p.is_absolute() else get_workspace() / p


def _check_placeholder(path: Path, content: str) -> str | None:
    """整个文件内容恰好是占位符才拒绝(审查 #7:'...' 作为子串会误伤正常短文件)。"""
    stripped = content.strip()
    if stripped in _PLACEHOLDERS:
        return f"拒绝写入 {path}:内容疑似占位符({stripped!r}),请给出完整内容"
    return None


def _validate(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> tuple[Path, str] | str:
    """检查参数并整理出(文件路径, 文件内容);参数不对时返回错误说明字符串。

    两种通道:markdown 是 fence 头带路径、代码块当内容;
    原生是 kwargs = {"path": ..., "content": ...}。
    """
    if kwargs and kwargs.get("path") is not None:
        content = kwargs.get("content", "")
        return _resolve(str(kwargs["path"])), content
    if not args:
        return "错误: save/append 需要路径参数(markdown:fence 头;原生:path 参数)"
    assert code is not None
    return _resolve(args[0]), code


def execute_save(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> Iterator[Message]:
    """save 工具的执行函数:把内容写成文件(文件已存在就整个覆盖)。

    示例:```save a.txt\\n你好\\n``` → 在工作区创建 a.txt,内容"你好\\n"。
    """
    validated = _validate(code, args, kwargs)
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
    """append 工具的执行函数:把内容追加到文件末尾(文件不存在就新建)。

    自动处理换行:原文件末尾没换行会先补一个,新内容也保证以换行结尾,
    避免新内容黏在旧内容的行尾上。
    """
    validated = _validate(code, args, kwargs)
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
    parameters=(
        Parameter("path", "string", "要写入的文件路径"),
        Parameter("content", "string", "完整的文件内容"),
    ),
    execute=execute_save,
)

append_tool = ToolSpec(
    name="append",
    desc="向文件末尾追加内容(自动补换行)。fence 头带路径。",
    examples="```append notes.md\n- 新的一行\n```",
    block_types=("append",),
    parameters=(
        Parameter("path", "string", "要追加到的文件路径"),
        Parameter("content", "string", "要追加的内容"),
    ),
    execute=execute_append,
)
