"""工作区键与会话根目录测试:同目录必须同桶、孪生命名必须分桶。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from my_agent_alpha.dirs import get_session_root, workspace_key


def test_same_dir_same_key_no_matter_spelling(tmp_path: Path) -> None:
    """同一个目录,不同写法(尾斜杠 / .. 绕行 / 软链)必须算出同一个键。

    这是"写法千千万,身份只有一个"的纪律:按写法算键会把一个项目拆成多个桶。
    """
    proj = tmp_path / "my_agent_alpha"
    proj.mkdir()
    link = tmp_path / "link-to-proj"
    os.symlink(proj, link)
    expected = workspace_key(proj)
    assert workspace_key(Path(str(proj) + "/")) == expected
    assert workspace_key(tmp_path / "my_agent_alpha" / ".." / "my_agent_alpha") == expected
    assert workspace_key(link) == expected  # 软链归一到真实路径


def test_twin_names_get_different_keys(tmp_path: Path) -> None:
    """孪生命名(my_agent_alpha 与 my-agent-alpha)必须分桶。

    Claude Code 只编码不防撞,孪生路径的项目记忆会串用;我们靠指纹根治。
    """
    a = tmp_path / "my_agent_alpha"
    b = tmp_path / "my-agent-alpha"
    a.mkdir()
    b.mkdir()
    assert workspace_key(a) != workspace_key(b)


def test_session_root_default_layout(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认布局:数据目录/projects/<工作区键>/logs,目录自动创建。"""
    monkeypatch.delenv("ALPHA_LOGS_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    root = get_session_root(workspace)
    assert root == (
        tmp_path / "xdg" / "my-agent-alpha" / "projects" / workspace_key(workspace) / "logs"
    )
    assert root.is_dir()


def test_session_root_logs_home_override(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ALPHA_LOGS_HOME 替代 projects 这一层:测试与真实数据隔离。"""
    monkeypatch.setenv("ALPHA_LOGS_HOME", str(tmp_path / "logs-home"))
    root = get_session_root(workspace)
    assert root == tmp_path / "logs-home" / workspace_key(workspace) / "logs"
    assert root.is_dir()
