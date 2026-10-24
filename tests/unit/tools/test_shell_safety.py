"""shell_safety.py 单元测试:命令安全分级 + 安全护栏/只读放行两个钩子。"""

from __future__ import annotations

from my_agent_alpha.tools.base import ToolUse, parse_tooluses
from my_agent_alpha.tools.shell_safety import (
    denied_reason,
    guardrail_hook,
    is_readonly_command,
    readonly_hook,
)


def _shell_use(command: str) -> ToolUse:
    """从一条 markdown 围栏造出 shell 调用(测试辅助)。"""
    uses = parse_tooluses(f"```shell\n{command}\n```")
    assert uses and uses[0].tool.name == "shell"
    return uses[0]


# ---- 命令安全分级 ----


def test_readonly_commands_pass() -> None:
    for cmd in [
        "ls -la",
        "ls",
        "cat a.txt",
        "ls | wc -l",
        "git status",
        "git log --oneline",
        "find . -name x",
        "echo hi",
    ]:
        assert is_readonly_command(cmd), cmd


def test_writing_commands_dont_pass() -> None:
    for cmd in [
        "rm file.txt",  # rm 不在只读名单
        "cat a > b",  # 重定向写文件
        "echo $(rm -rf x)",  # 命令替换里藏东西
        "find . -delete",  # find 带删除标志
        "git push",  # git 写子命令
        "python3 evil.py",  # 能执行任意代码的绝不放行
        "touch x",  # 会创建文件
        "ls; rm x",  # 混在管道/分号里的一段不安全
    ]:
        assert not is_readonly_command(cmd), cmd


def test_denied_patterns() -> None:
    assert denied_reason("rm -rf /") is not None
    assert denied_reason("rm -rf ~") is not None
    assert denied_reason("mkfs.ext4 /dev/sdb") is not None
    assert denied_reason("dd if=x of=/dev/sda") is not None
    assert denied_reason("shutdown now") is not None
    # 正常命令不误拦
    assert denied_reason("rm -rf ./build") is None  # 目录内清理:拦到"询问"层,不硬拒
    assert denied_reason("rm file.txt") is None
    assert denied_reason("chmod 777 file.txt") is None


# ---- 两个钩子 ----


def test_guardrail_hook() -> None:
    assert guardrail_hook(_shell_use("rm -rf /")) is not None
    assert guardrail_hook(_shell_use("rm -rf /")).action == "skip"
    assert guardrail_hook(_shell_use("ls")) is None  # 不管普通命令
    save_use = parse_tooluses("```save a.txt\n内容\n```")[0]
    assert guardrail_hook(save_use) is None  # 只管 shell


def test_readonly_hook() -> None:
    assert readonly_hook(_shell_use("ls -la")).action == "confirm"
    assert readonly_hook(_shell_use("rm file.txt")) is None  # 非只读:交给下一层
