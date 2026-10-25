"""grep 工具执行端单测:参数文法、搜索范围、输出格式、边界情况。"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha.tools import grep


def _run(
    code: str | None, args: list[str] | None = None, kwargs: dict[str, str] | None = None
) -> str:
    """执行一次 grep,返回结果消息的文本。"""
    (msg,) = list(grep.execute_grep(code, args, kwargs))
    assert msg.role == "system"
    return msg.content


def _setup_tree(workspace: Path) -> None:
    """搭一个两层小目录:a.py / sub/b.py / sub/c.txt + 各种干扰文件。"""
    (workspace / "a.py").write_text("def run():\n    pass\n# TODO: 补测试\n", encoding="utf-8")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "b.py").write_text("print('hi')\n# TODO: 再说\n", encoding="utf-8")
    (workspace / "sub" / "c.txt").write_text("TODO in txt\n", encoding="utf-8")
    # 干扰项:应被忽略的目录/隐藏文件/非文本文件
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config.py").write_text("TODO in git\n", encoding="utf-8")
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "a.cpython-312.pyc").write_text(
        "TODO in pycache\n", encoding="utf-8"
    )
    (workspace / ".hidden.py").write_text("TODO hidden\n", encoding="utf-8")
    (workspace / "blob.bin").write_bytes(b"\xff\xfe\x00TODO binary")


def test_basic_hit_format(workspace: Path) -> None:
    """基本命中:输出是 相对路径:行号:内容。"""
    _setup_tree(workspace)
    text = _run("def run")
    assert "a.py:1:def run():" in text


def test_recursive_into_subdirs(workspace: Path) -> None:
    """递归子目录:workspace 根下能搜到 sub/ 里的内容。"""
    _setup_tree(workspace)
    text = _run("print")
    assert "sub/b.py:1:print('hi')" in text


def test_glob_filters_filenames(workspace: Path) -> None:
    """glob=*.py:txt 文件被排除。"""
    _setup_tree(workspace)
    text = _run("pattern=TODO\nglob=*.py")
    assert "a.py:3" in text
    assert "c.txt" not in text


def test_path_limits_scope_to_subdir(workspace: Path) -> None:
    """path=sub:只搜子目录,根下的 a.py 不出现。"""
    _setup_tree(workspace)
    text = _run("pattern=TODO\npath=sub")
    assert "sub/b.py:2" in text
    assert "a.py" not in text


def test_path_single_file(workspace: Path) -> None:
    """path 指向单个文件:只搜它。"""
    _setup_tree(workspace)
    text = _run("pattern=TODO\npath=sub/c.txt")
    assert "sub/c.txt:1:TODO in txt" in text
    assert "a.py" not in text


def test_no_match_is_explicit(workspace: Path) -> None:
    """无命中:明确说没有匹配(而不是空消息)。"""
    _setup_tree(workspace)
    text = _run("不存在的词xyz")
    assert "没有匹配" in text


def test_junk_dirs_and_hidden_skipped(workspace: Path) -> None:
    """.git/__pycache__/隐藏文件/二进制都被忽略。"""
    _setup_tree(workspace)
    text = _run("TODO")
    assert ".git" not in text
    assert "__pycache__" not in text
    assert ".hidden" not in text
    assert "blob.bin" not in text
    assert text.count(":") >= 3  # 真命中都在:a.py、sub/b.py、sub/c.txt


def test_bare_pattern_and_kv_grammar(workspace: Path) -> None:
    """markdown 文法:第一行裸 pattern;k=v 行逐个解析。"""
    _setup_tree(workspace)
    assert "a.py:1" in _run("def run")  # 裸 pattern
    assert "a.py:3" in _run("pattern=TODO\nglob=*.py")  # k=v


def test_fence_header_args(workspace: Path) -> None:
    """围栏头参数:```grep TODO *.py 等价于 pattern+glob。"""
    _setup_tree(workspace)
    text = _run(None, args=["TODO", "*.py"])
    assert "a.py:3" in text
    assert "c.txt" not in text


def test_native_channel_kwargs(workspace: Path) -> None:
    """原生通道:kwargs 三个具名参数。"""
    _setup_tree(workspace)
    text = _run(None, kwargs={"pattern": "TODO", "glob": "*.py"})
    assert "a.py:3" in text
    assert "c.txt" not in text


def test_missing_pattern_reports_usage() -> None:
    """没给 pattern:错误消息带文法。"""
    text = _run("glob=*.py")
    assert "pattern" in text


def test_invalid_regex_reports_error() -> None:
    """非法正则:报错而不是崩。"""
    text = _run("pattern=[unclosed")
    assert "不是合法的正则" in text


def test_missing_path_scope() -> None:
    """path 指向不存在的范围:明确报错。"""
    text = _run("pattern=x\npath=不存在的目录")
    assert "不存在" in text


def test_hit_cap_truncates(workspace: Path) -> None:
    """命中超过上限:截断到 _MAX_HITS 并提示收窄。"""
    (workspace / "big.py").write_text(
        "\n".join(f"hit_{i}" for i in range(grep._MAX_HITS + 50)), encoding="utf-8"
    )
    text = _run("pattern=hit_")
    assert f"命中 {grep._MAX_HITS} 行" in text
    assert "上限" in text
    assert "hit_250" not in text
