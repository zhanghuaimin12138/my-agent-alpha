"""第二轮 R1:edit 工具测试——精确替换、唯一性校验、相似位置提示、原子写。"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha import Message
from my_agent_alpha.tools import execute_msg, get_tool


def _has_edit() -> bool:
    return get_tool("edit") is not None


SAMPLE = """def hello():
    print('hi')

def bye():
    print('bye')
"""


def _write_sample(workspace: Path) -> Path:
    target = workspace / "app.py"
    target.write_text(SAMPLE, encoding="utf-8")
    return target


# ---- markdown 通道 ----


def test_basic_replace(workspace: Path) -> None:
    assert _has_edit()
    target = _write_sample(workspace)
    msg = Message(
        "assistant",
        "```edit app.py\ndef hello():\n    print('hi')\n=====\ndef hello():\n"
        "    print('hello world')\n```",
    )
    results = execute_msg(msg)
    assert "已修改" in results[0].content
    assert "print('hello world')" in target.read_text(encoding="utf-8")
    # 其余内容原样保留
    assert "def bye():" in target.read_text(encoding="utf-8")


def test_old_not_unique_rejected(workspace: Path) -> None:
    target = workspace / "dup.txt"
    target.write_text("同一行\n中间别的\n同一行\n", encoding="utf-8")
    msg = Message("assistant", "```edit dup.txt\n同一行\n=====\n改过\n```")
    results = execute_msg(msg)
    assert "多处" in results[0].content
    assert target.read_text(encoding="utf-8") == "同一行\n中间别的\n同一行\n"  # 未被改动


def test_old_not_found_with_hint(workspace: Path) -> None:
    _write_sample(workspace)
    msg = Message(
        "assistant",
        "```edit app.py\ndef hello():\n    print('hii')\n=====\n新\n```",  # 拼错一个字母
    )
    results = execute_msg(msg)
    assert "找不到" in results[0].content
    assert "第 1 行附近" in results[0].content  # 相似提示指到了正确位置


def test_missing_separator_rejected(workspace: Path) -> None:
    _write_sample(workspace)
    msg = Message("assistant", "```edit app.py\n只有旧文本没有分隔线\n```")
    results = execute_msg(msg)
    assert "=====" in results[0].content  # 报错里教了格式


def test_empty_old_rejected(workspace: Path) -> None:
    _write_sample(workspace)
    msg = Message("assistant", "```edit app.py\n=====\n新文本\n```")
    results = execute_msg(msg)
    assert "旧文本" in results[0].content


def test_missing_file(workspace: Path) -> None:
    msg = Message("assistant", "```edit nope.py\na\n=====\nb\n```")
    results = execute_msg(msg)
    assert "不存在" in results[0].content


def test_no_temp_files_left(workspace: Path) -> None:
    target = _write_sample(workspace)
    msg = Message("assistant", f"```edit {target.name}\nbye\n=====\nsee ya\n```")
    execute_msg(msg)
    assert list(workspace.glob("*.edit-tmp-*")) == []


# ---- 原生通道 ----


def test_native_kwargs(workspace: Path) -> None:
    target = _write_sample(workspace)
    msg = Message(
        "assistant",
        '@edit(c1): {"path": "app.py", "old": "print(\'bye\')", "new": "print(\'goodbye\')"}',
    )
    results = execute_msg(msg)
    assert "已修改" in results[0].content
    assert "print('goodbye')" in target.read_text(encoding="utf-8")
    assert results[0].call_id == "c1"


def test_native_empty_old(workspace: Path) -> None:
    msg = Message("assistant", '@edit(c1): {"path": "app.py", "old": "", "new": "x"}')
    results = execute_msg(msg)
    assert "old" in results[0].content and "错误" in results[0].content


# ---- 多行替换与边界 ----


def test_multiline_replace(workspace: Path) -> None:
    target = _write_sample(workspace)
    msg = Message(
        "assistant",
        "```edit app.py\ndef bye():\n    print('bye')\n=====\ndef farewell():\n"
        "    print('farewell')\n    print('done')\n```",
    )
    results = execute_msg(msg)
    assert "已修改" in results[0].content
    text = target.read_text(encoding="utf-8")
    assert "def farewell():" in text and "def bye():" not in text


def test_edit_preserves_file_outside_match(workspace: Path) -> None:
    """中文内容、空行、缩进全部原样。"""
    target = workspace / "zh.txt"
    target.write_text("第一行\n\n  缩进行\n最后一行\n", encoding="utf-8")
    msg = Message("assistant", "```edit zh.txt\n  缩进行\n=====\n  改过的行\n```")
    execute_msg(msg)
    assert target.read_text(encoding="utf-8") == "第一行\n\n  改过的行\n最后一行\n"
