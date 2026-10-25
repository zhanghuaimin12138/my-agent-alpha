"""tools/read.py 测试:行号渲染、缺文件报错、fence 头路径参数。"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha import Message
from my_agent_alpha.tools import execute_msg


def test_read_numbers_lines(workspace: Path) -> None:
    target = workspace / "a.txt"
    target.write_text("第一行\n第二行\n", encoding="utf-8")
    msg = Message("assistant", "```read\na.txt\n```")
    out = execute_msg(msg)[0].content
    assert "a.txt" in out
    assert "1\t第一行" in out
    assert "2\t第二行" in out


def test_read_missing_file_is_message_not_crash(workspace: Path) -> None:
    msg = Message("assistant", "```read\n不存在.txt\n```")
    out = execute_msg(msg)[0].content
    assert "不存在" in out


# ---- 审查 #18:read 支持 fence 头路径 ----


def test_read_head_arg(workspace: Path) -> None:
    (workspace / "head.txt").write_text("内容\n", encoding="utf-8")
    msg = Message("assistant", "```read head.txt\n```")
    assert "内容" in execute_msg(msg)[0].content
