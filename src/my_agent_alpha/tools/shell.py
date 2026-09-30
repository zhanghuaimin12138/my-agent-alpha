"""shell 工具(最小版):在工作区执行 bash 命令。

阶段 1 用 subprocess.run 逐命令执行;gptme 的持久 bash 会话、哨兵协议、
超时晋升后台、允许/拒绝列表、shellcheck 都属后续硬化(阶段 4 起逐步引入)。
当前的安全姿态 = 超时 + 输出截断 + 全量存盘,无确认(相当于 gptme 的 -y)。
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from my_agent_alpha.context import get_logdir, get_workspace
from my_agent_alpha.message import Message
from my_agent_alpha.tools.base import ToolSpec

_TIMEOUT = int(os.environ.get("ALPHA_SHELL_TIMEOUT", "60"))
_MAX_OUTPUT_CHARS = int(os.environ.get("ALPHA_SHELL_MAX_OUTPUT", "8000"))


def _truncate(text: str, limit: int) -> str:
    """输出太长时砍中间留两头(开头结尾往往最有用),并注明砍了多少。"""
    if len(text) <= limit:
        return text
    keep = limit // 2
    return f"{text[:keep]}\n... [输出过长,已截断 {len(text) - limit} 字符] ...\n{text[-keep:]}"


def _save_full_output(command: str, text: str) -> Path | None:
    """把完整输出存成一个文件(会话目录/tool-outputs/shell/ 下),返回文件路径。

    终端里显示的是截断版,但完整版不能丢 —— 模型需要时可以自己去读这个文件。
    没有会话目录(少数调用场景)就返回 None。
    """
    logdir = get_logdir()
    if logdir is None:
        return None
    out_dir = logdir / "tool-outputs" / "shell"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 微秒 + pid:同一秒内的两次调用也不能互相覆盖(审查 #6)
    path = out_dir / f"{datetime.now().strftime('%H%M%S-%f')}-{os.getpid()}.txt"
    path.write_text(f"$ {command}\n\n{text}", encoding="utf-8")
    return path


def execute_shell(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> Iterator[Message]:
    """在工作区用 bash 执行命令,产出带 ReturnCode 的结果消息。"""
    assert code is not None
    try:
        proc = subprocess.run(
            code,
            shell=True,
            cwd=get_workspace(),
            capture_output=True,
            text=True,
            errors="replace",  # 命令输出未必是 UTF-8(审查 #5),解码失败不许炸
            timeout=_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        yield Message("system", f"命令超时(>{_TIMEOUT}s)被终止:\n$ {code}")
        return

    output = ""
    if proc.stdout:
        output += proc.stdout
    if proc.stderr:
        output += ("\n" if output else "") + proc.stderr
    output = output.rstrip("\n") or "(无输出)"

    full_path = _save_full_output(code, output)
    truncated = _truncate(output, _MAX_OUTPUT_CHARS) != output
    note = f"\n(输出过长已截断,完整输出: {full_path})" if full_path and truncated else ""
    yield Message(
        "system",
        f"$ {code}\n{_truncate(output, _MAX_OUTPUT_CHARS)}\nReturnCode: {proc.returncode}{note}",
    )


tool = ToolSpec(
    name="shell",
    desc="在 bash 里执行命令并返回输出。用于检查文件、运行程序、git 操作等一切终端操作。",
    instructions=(
        "一次代码块执行一段命令(可以多行)。\n"
        "先阅读输出再决定下一步;命令失败时,根据 ReturnCode 和 stderr 修正后重试。"
    ),
    examples="```shell\nls -la\n```",
    block_types=("shell", "sh", "bash"),
    execute=execute_shell,
)
