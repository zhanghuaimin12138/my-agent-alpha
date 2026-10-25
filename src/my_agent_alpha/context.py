"""进程内的全局上下文:当前工作区目录和会话目录。

为什么不通过参数一层层传给工具?因为工具的执行函数签名是统一的
(code, args, kwargs),没有位置放 workspace。这里用 ContextVar 存"当前的环境",
工具执行时自己来取 —— 类似"当前登录用户是谁"这种贯穿全局的信息。

ContextVar 的额外好处:将来开多线程跑多个会话时,各线程各有一份,互不干扰。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_workspace_var: ContextVar[Path | None] = ContextVar("alpha_workspace", default=None)
_logdir_var: ContextVar[Path | None] = ContextVar("alpha_logdir", default=None)
_model_var: ContextVar[str | None] = ContextVar("alpha_model", default=None)


def get_workspace() -> Path:
    """返回当前工作区(agent 的 shell 命令在这里执行);没设置过就用当前目录。"""
    return _workspace_var.get() or Path.cwd()


def set_workspace(path: Path) -> None:
    """设置当前工作区。"""
    _workspace_var.set(path)


def get_logdir() -> Path | None:
    """返回当前会话目录(工具用它存放全量输出等文件);没设置过返回 None。"""
    return _logdir_var.get()


def set_logdir(path: Path | None) -> None:
    """设置当前会话目录。"""
    _logdir_var.set(path)


def get_current_model() -> str | None:
    """当前会话用的模型(subagent 工具继承它,不传就用父会话同款)。"""
    return _model_var.get()


def set_current_model(model: str) -> None:
    """设置当前模型(chat() 会话开始时设一次)。"""
    _model_var.set(model)


def clear_current_model() -> None:
    """清掉当前模型记录,恢复成"没设置过"。

    给测试隔离用:chat() 会设模型,不清掉的话,
    后面跑的测试会看到上一个测试留下的模型。
    """
    _model_var.set(None)


@contextmanager
def agent_context(workspace: Path, logdir: Path | None) -> Iterator[None]:
    """临时设置工作区和会话目录,离开 with 块后自动恢复原值。

    用法:
        with agent_context(工作区, 会话目录):
            ...  # 这段代码里工具取到的就是这两个目录
    """
    ws_token = _workspace_var.set(workspace)
    ld_token = _logdir_var.set(logdir)
    try:
        yield
    finally:
        _workspace_var.reset(ws_token)
        _logdir_var.reset(ld_token)
