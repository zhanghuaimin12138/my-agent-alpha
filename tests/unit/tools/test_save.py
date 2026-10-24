"""tools/save.py 测试:写文件、占位符拦截、覆盖分支、append 的换行行为。"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha import Message
from my_agent_alpha.tools import execute_msg

# ---- 基本执行 ----


def test_save_writes_file(workspace: Path) -> None:
    msg = Message("assistant", "```save out/hello.py\nprint('hi')\n```")
    results = execute_msg(msg)
    assert "已创建" in results[0].content
    assert (workspace / "out" / "hello.py").read_text(encoding="utf-8") == "print('hi')"


def test_save_rejects_placeholder(workspace: Path) -> None:
    msg = Message("assistant", "```save stub.py\n# ...\n```")
    results = execute_msg(msg)
    assert "占位符" in results[0].content
    assert not (workspace / "stub.py").exists()


def test_save_without_path_reports_error(workspace: Path) -> None:
    msg = Message("assistant", "```save\n没路径\n```")
    assert "路径" in execute_msg(msg)[0].content


def test_append_appends_with_newline(workspace: Path) -> None:
    target = workspace / "log.md"
    target.write_text("- 第一条", encoding="utf-8")
    msg = Message("assistant", "```append log.md\n- 第二条\n```")
    execute_msg(msg)
    assert target.read_text(encoding="utf-8") == "- 第一条\n- 第二条\n"


# ---- 审查 #7:占位符只拒"整个文件是占位符" ----


def test_placeholder_substring_allowed(workspace: Path) -> None:
    msg = Message("assistant", "```save ellipsis.txt\n正常内容包含 ... 也可以\n```")
    results = execute_msg(msg)
    assert "已创建" in results[0].content
    assert (workspace / "ellipsis.txt").exists()


def test_placeholder_whole_file_rejected(workspace: Path) -> None:
    msg = Message("assistant", "```save stub.txt\n# ...\n```")
    assert "占位符" in execute_msg(msg)[0].content


# ---- 审查 #17:save 覆盖 / append 建文件 ----


def test_save_overwrite_branch(workspace: Path) -> None:
    target = workspace / "f.txt"
    target.write_text("旧内容", encoding="utf-8")
    msg = Message("assistant", "```save f.txt\n新内容\n```")
    assert "已覆盖" in execute_msg(msg)[0].content
    # save 原样写围栏内内容,不擅自补换行
    assert target.read_text(encoding="utf-8") == "新内容"


def test_append_creates_missing_file(workspace: Path) -> None:
    msg = Message("assistant", "```append new.md\n第一行\n```")
    execute_msg(msg)
    assert (workspace / "new.md").read_text(encoding="utf-8") == "第一行\n"
