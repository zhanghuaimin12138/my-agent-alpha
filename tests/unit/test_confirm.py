"""confirm.py 单元测试:确认链机制 —— 优先级、短路、自动放行状态。

钩子和工具执行装配起来之后的行为在 tests/integration/test_confirm.py。
"""

from __future__ import annotations

from my_agent_alpha.confirm import (
    ConfirmationResult,
    clear_confirm_hooks,
    get_confirmation,
    is_auto_confirm,
    register_confirm_hook,
    set_auto_confirm,
)
from my_agent_alpha.tools.base import ToolUse, parse_tooluses


def _shell_use(command: str) -> ToolUse:
    """从一条 markdown 围栏造出 shell 调用(测试辅助)。"""
    uses = parse_tooluses(f"```shell\n{command}\n```")
    assert uses and uses[0].tool.name == "shell"
    return uses[0]


# ---- 钩子链机制 ----


def test_no_hooks_auto_confirm() -> None:
    """链上没人管 → 自动放行(自主模式:测试/脚本/-y)。"""
    clear_confirm_hooks()
    assert get_confirmation(_shell_use("rm file.txt")).action == "confirm"


def test_priority_order() -> None:
    """高优先级先答;答了就不问下一家;返回 None 才往下传。"""
    clear_confirm_hooks()
    register_confirm_hook("low", lambda u: ConfirmationResult("skip"), priority=0)
    register_confirm_hook("high", lambda u: None, priority=100)  # 高优先级但不管
    assert get_confirmation(_shell_use("ls")).action == "skip"


def test_auto_confirm_state() -> None:
    clear_confirm_hooks()
    register_confirm_hook("always_ask", lambda u: ConfirmationResult("skip"))
    set_auto_confirm(True)
    assert get_confirmation(_shell_use("ls")).action == "confirm"  # 放行状态压过一切
    assert is_auto_confirm()
    set_auto_confirm(False)
    assert get_confirmation(_shell_use("ls")).action == "skip"
