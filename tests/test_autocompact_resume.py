"""LLM 摘要测试(§9 摘要组):解析上下文文件、加载、新日志组装。"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha.autocompact.adapters import EstimateTokenCounter
from my_agent_alpha.autocompact.resume import (
    RESUME_FAILED_PREFIX,
    RESUME_FILENAME,
    load_context_files,
    parse_context_files,
    resume_via_llm,
)
from my_agent_alpha.message import Message

COUNTER = EstimateTokenCounter()


class _FakeLLM:
    """测试用 LLM 客户端:回固定文本,或抛异常。"""

    def __init__(self, text: str | BaseException) -> None:
        self._text = text

    def reply(self, messages: list[Message], model: str) -> str:
        if isinstance(self._text, BaseException):
            raise self._text
        return self._text


# 一份带"上下文文件"小节的固定摘要
_RESUME_TEXT = """# Conversation Resume

## Conversation Summary
We built a parser. It works.

## Context Files

- `spec.md` - the specification
- /abs/path/legacy.py - legacy module
- src/mod.py - active module
- `./rel.txt` - relative note
- `spec.md` - duplicate entry
- `https://example.com/doc` - link, not a file
"""


def test_parse_context_files_all_patterns() -> None:
    """三种提取模式都认;http 过滤、./ 去掉、重复去重保序。"""
    paths = parse_context_files(_RESUME_TEXT)
    assert paths == ["spec.md", "/abs/path/legacy.py", "src/mod.py", "rel.txt"]


def test_parse_context_files_section_title_variants() -> None:
    """小节标题的四种写法都能找到小节。"""
    for title in ("Files to Include", "Recommended Files", "Key Files"):
        text = f"## {title}\n- `a.md` - one\n- `b.md` - two\n"
        assert parse_context_files(text) == ["a.md", "b.md"]


def test_parse_context_files_without_section_scans_all() -> None:
    """没有小节标题:把全文当小节扫。"""
    text = "Intro text.\n- `a.md` - only\n"
    assert parse_context_files(text) == ["a.md"]


def test_load_context_files_reads_and_truncates(workspace: Path) -> None:
    """存在的文件读进来;超 2000 token 的截断;不存在的跳过。"""
    (workspace / "spec.md").write_text("spec content", encoding="utf-8")
    big = "x" * 9_000  # 2250 token > 2000 → 截断
    (workspace / "src").mkdir()
    (workspace / "src" / "big.py").write_text(big, encoding="utf-8")

    loaded = load_context_files(["spec.md", "src/big.py", "missing.md"], workspace, counter=COUNTER)
    assert [p.name for p, _ in loaded] == ["spec.md", "big.py"]  # 缺的跳过
    by_name = {p.name: content for p, content in loaded}
    assert by_name["spec.md"] == "spec content"
    assert "... (truncated," in by_name["big.py"]  # 截断且带剩余提示
    assert len(by_name["big.py"]) < len(big)


def test_load_context_files_home_expansion(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """~ 开头的路径按用户主目录展开。"""
    monkeypatch.setenv("HOME", str(workspace))
    (workspace / "note.txt").write_text("home file", encoding="utf-8")
    loaded = load_context_files(["~/note.txt"], workspace, counter=COUNTER)
    assert loaded and loaded[0][1] == "home file"


def test_resume_rejects_too_few_messages(logdir: Path, workspace: Path) -> None:
    """预处理后不足 3 条:拒绝,返回带标记的提示消息。"""
    result = resume_via_llm(
        [Message("user", "hi"), Message("assistant", "hello")],
        llm=_FakeLLM("whatever"),
        model="mock/echo",
        session_dir=logdir,
        workspace=workspace,
        counter=COUNTER,
    )
    assert len(result) == 1
    assert result[0].content.startswith(RESUME_FAILED_PREFIX)


def test_resume_llm_failure_returns_marker(logdir: Path, workspace: Path) -> None:
    """模型调用失败:返回带标记的提示消息,不抛异常、不破坏原日志。"""
    result = resume_via_llm(
        [Message("system", "p"), Message("user", "q"), Message("assistant", "a")],
        llm=_FakeLLM(RuntimeError("boom")),
        model="mock/echo",
        session_dir=logdir,
        workspace=workspace,
        counter=COUNTER,
    )
    assert result[0].content.startswith(RESUME_FAILED_PREFIX)
    assert "boom" in result[0].content


def test_resume_assembles_new_log(logdir: Path, workspace: Path) -> None:
    """成功路径:新日志 = 头部 system + 文件消息 + 引导语 + assistant 摘要。"""
    (workspace / "spec.md").write_text("spec content", encoding="utf-8")
    msgs = [
        Message("system", "system prompt"),
        Message("user", "q1"),
        Message("assistant", "a1"),
        Message("user", "q2"),
    ]
    result = resume_via_llm(
        msgs,
        llm=_FakeLLM(_RESUME_TEXT),
        model="mock/echo",
        session_dir=logdir,
        workspace=workspace,
        counter=COUNTER,
    )
    roles = [m.role for m in result]
    assert roles == ["system", "system", "system", "assistant"]
    assert result[0].content == "system prompt"  # 头部 system 保住
    assert result[1].content.startswith("Context file `") and "spec.md" in result[1].content
    assert "spec content" in result[1].content  # 文件正文装进来了
    assert result[2].content.startswith("Previous conversation resumed from")
    assert "1 context file" in result[2].content
    assert result[3].content == _RESUME_TEXT  # 摘要伪装成 assistant 消息
    # 摘要落盘成 RESUME.md
    assert (logdir / RESUME_FILENAME).read_text(encoding="utf-8") == _RESUME_TEXT


def test_resume_head_stops_at_first_non_system(logdir: Path, workspace: Path) -> None:
    """头部只收"开头连续的 system":第一个非 system 之后就不再收。"""
    msgs = [
        Message("system", "prompt"),
        Message("user", "q1"),  # 从这里断开,后面的 system 不算头部
        Message("system", "tool output"),
        Message("assistant", "a"),
    ]
    result = resume_via_llm(
        msgs,
        llm=_FakeLLM("only summary, no files section"),
        model="mock/echo",
        session_dir=logdir,
        workspace=workspace,
        counter=COUNTER,
    )
    # 头部只到 prompt 为止;之后直接是引导语和摘要
    assert result[0].content == "prompt"
    assert result[1].content.startswith("Previous conversation resumed")
    assert result[2].content == "only summary, no files section"
