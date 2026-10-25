"""tools/shell.py 测试:真实执行命令、返回码、解码容错、超时与输出截断。"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import Message
from my_agent_alpha.tools import execute_msg

# ---- 基本执行 ----


def test_shell_executes_and_reports_rc(workspace: Path) -> None:
    msg = Message("assistant", "```shell\necho hello-spine\n```")
    results = execute_msg(msg)
    assert len(results) == 1
    assert "hello-spine" in results[0].content
    assert "ReturnCode: 0" in results[0].content


def test_shell_failure_reported(workspace: Path) -> None:
    msg = Message("assistant", "```shell\nexit 3\n```")
    results = execute_msg(msg)
    assert "ReturnCode: 3" in results[0].content


def test_shell_runs_in_workspace(workspace: Path) -> None:
    (workspace / "marker.txt").write_text("x", encoding="utf-8")
    msg = Message("assistant", "```shell\nls marker.txt\n```")
    assert "marker.txt" in execute_msg(msg)[0].content


# ---- 审查 #5/#6/#15:shell 的安全姿态 ----


def test_shell_non_utf8_output_does_not_crash(workspace: Path) -> None:
    msg = Message("assistant", '```shell\nprintf "\\xff\\xfe\\xff"\n```')
    results = execute_msg(msg)
    assert "ReturnCode: 0" in results[0].content  # 解码容错,不炸


def test_shell_timeout_branch(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from my_agent_alpha.tools import shell as shell_mod

    monkeypatch.setattr(shell_mod, "_TIMEOUT", 1)
    msg = Message("assistant", "```shell\nsleep 5\n```")
    results = execute_msg(msg)
    assert "超时" in results[0].content


def test_shell_truncation_saves_full_output(
    workspace: Path, logdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from my_agent_alpha.tools import shell as shell_mod

    monkeypatch.setattr(shell_mod, "_MAX_OUTPUT_CHARS", 200)
    msg = Message("assistant", "```shell\nseq 1 500\n```")
    results = execute_msg(msg)
    assert "已截断" in results[0].content
    # 全量输出真的存了盘,且内容完整
    files = list((logdir / "tool-outputs" / "shell").glob("*.txt"))
    assert len(files) == 1
    assert "500" in files[0].read_text(encoding="utf-8")


def test_shell_full_output_files_never_collide(workspace: Path, logdir: Path) -> None:
    for _ in range(3):
        execute_msg(Message("assistant", "```shell\necho x\n```"))
    files = list((logdir / "tool-outputs" / "shell").glob("*.txt"))
    assert len(files) == 3  # 微秒级文件名,同一秒内也不互相覆盖
