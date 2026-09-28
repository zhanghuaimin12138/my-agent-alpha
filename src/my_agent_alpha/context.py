"""进程内上下文(ContextVar)。

仿 gptme 的 "ContextVar-everything":工具执行时不传 workspace/logdir 参数,
而是从上下文取 —— 将来子代理线程 / server 多会话天然隔离。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_workspace_var: ContextVar[Path | None] = ContextVar("alpha_workspace", default=None)
_logdir_var: ContextVar[Path | None] = ContextVar("alpha_logdir", default=None)


def get_workspace() -> Path:
    """当前工作区(agent 的 shell 在这里执行);未设置时回退 cwd。"""
    return _workspace_var.get() or Path.cwd()


def set_workspace(path: Path) -> None:
    _workspace_var.set(path)


def get_logdir() -> Path | None:
    """当前会话目录(工具全量输出等副产物存放处);未设置时为 None。"""
    return _logdir_var.get()


def set_logdir(path: Path | None) -> None:
    _logdir_var.set(path)


@contextmanager
def agent_context(workspace: Path, logdir: Path | None) -> Iterator[None]:
    """一把设置两个上下文,会话结束时还原。"""
    ws_token = _workspace_var.set(workspace)
    ld_token = _logdir_var.set(logdir)
    try:
        yield
    finally:
        _workspace_var.reset(ws_token)
        _logdir_var.reset(ld_token)
