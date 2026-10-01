"""shell 命令安全分级:哪些自动放行、哪些直接拒绝、哪些要问用户。

三层判定(从宽到严):

1. 只读命令 → 自动放行(不烦用户):ls、cat、grep 这类"只看不改"的命令;
2. 灾难命令 → 直接拒绝(不分模式,连 -y 也拦):rm -rf /、格式化硬盘这类;
3. 其余 → 交给下一条钩子(通常是"问用户")。

安全方向:判不准的一律往下走(宁可多问一句,不可错放一条)。
阶段 4 的简化(以后逐步加严):引号内容不做特殊处理(宁可误拦不可漏放)、
不接 shellcheck、不做逐命令的标志白名单。
"""

from __future__ import annotations

import re
import shlex

from my_agent_alpha.confirm import ConfirmationResult
from my_agent_alpha.tools.base import ToolUse

# 只读命令白名单:这些命令"只看不改",自动放行。
# 注意保持克制 —— python3/node 这类能干任何事的绝不能进名单
_READONLY_COMMANDS = {
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "pwd",
    "echo",
    "grep",
    "rg",
    "find",
    "file",
    "which",
    "stat",
    "du",
    "df",
    "ps",
    "printenv",
    "date",
    "whoami",
    "uname",
    "tree",
    "md5sum",
    "sha256sum",
    "basename",
    "dirname",
    "realpath",
    "seq",
    "true",
    "false",
}

# git 只读子命令(git 本身太杂,单独列)
_GIT_READONLY = {
    "status",
    "log",
    "diff",
    "show",
    "branch",
    "blame",
    "describe",
    "rev-parse",
    "remote",
    "tag",
    "ls-files",
}

# find 里能改东西的标志(有它们就不是只读)
_FIND_DANGEROUS = {"-delete", "-exec", "-execdir", "-fprintf", "-fls", "-ok", "-fprint"}

# 灾难级命令模式:直接拒绝,不给选择(宁可误拦)
_DENY_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+(-\w*\s+)*(/|~|\$HOME)(\s|/|$)", "rm 指向根目录/家目录"),
    (r"\brm\s+--no-preserve-root", "rm 绕过根目录保护"),
    (r":\(\)\s*\{.*\};\s*:", "fork 炸弹"),
    (r"\bdd\b.*\bof=/dev/(sd|hd|nvme|vd)", "dd 直写磁盘设备"),
    (r"\bmkfs(\.\w+)?\b", "格式化文件系统"),
    (r">\s*/dev/sd[a-z]", "重定向直写磁盘设备"),
    (r"\bchmod\s+(-\w+\s+)*-?R?\s*777\s+(/|~)(\s|/|$)", "对根/家目录放开全部权限"),
    (r"\b(shutdown|reboot|poweroff|halt)\b", "关机/重启"),
    (r">\s*/dev/urandom\b", "覆写内核随机源"),
    (r"\bgit\s+push\s+.*--force\b.*\b(master|main)\b", "强推主分支"),
]

# 命令分隔符:管道、顺序、条件执行 —— 每一段都要单独过检
_SEGMENT_SPLIT = re.compile(r"\|\||&&|[|;]")


def _segments(command: str) -> list[str] | None:
    """把命令按管道/分号切开,返回各段;解析不了(引号诡异等)返回 None(视为不安全)。"""
    try:
        parts = shlex.split(command, posix=True)
    except ValueError:
        return None
    segments: list[str] = []
    current: list[str] = []
    for part in parts:
        if part in ("|", "||", ";", "&&"):
            segments.append(" ".join(current))
            current = []
        else:
            current.append(part)
    segments.append(" ".join(current))
    return [s for s in segments if s]


def is_readonly_command(command: str) -> bool:
    """判断一条命令是否"只看不改"(可以不问用户直接跑)。

    示例:
        is_readonly_command("ls -la")        → True
        is_readonly_command("ls | wc -l")    → True(每段都是只读)
        is_readonly_command("cat a > b")     → False(有重定向,会写文件)
        is_readonly_command("find . -delete")→ False(find 带 -delete)
        is_readonly_command("echo $(rm x)")  → False(有命令替换)
    """
    if any(op in command for op in (">", ">>", "<", "$(", "`")):
        return False  # 重定向会写文件;命令替换里可能藏任何东西
    segments = _segments(command)
    if segments is None:
        return False
    for seg in segments:
        words = seg.split()
        base = words[0]
        if base == "git":
            if len(words) < 2 or words[1] not in _GIT_READONLY:
                return False
        elif base == "find":
            if any(w in _FIND_DANGEROUS for w in words):
                return False
        elif base not in _READONLY_COMMANDS:
            return False
    return True


def denied_reason(command: str) -> str | None:
    """命中灾难模式就返回原因说明,否则 None(交给下一层)。

    示例:denied_reason("rm -rf /") → "rm 指向根目录/家目录"
    """
    for pattern, reason in _DENY_PATTERNS:
        if re.search(pattern, command):
            return reason
    return None


def guardrail_hook(tooluse: ToolUse) -> ConfirmationResult | None:
    """优先级 200 的安全护栏:灾难命令直接拒绝,不看模式、不听用户。

    (这一层防的是"模型被误导"和"用户手滑",所以 -y 也不能绕过)
    """
    if tooluse.tool.name != "shell":
        return None
    command = tooluse.kwargs.get("command") if tooluse.kwargs else tooluse.content
    if not command:
        return None  # 没有命令内容,交给别的钩子处理
    reason = denied_reason(command)
    if reason:
        return ConfirmationResult("skip", message=f"安全护栏拒绝:{reason}")
    return None


def readonly_hook(tooluse: ToolUse) -> ConfirmationResult | None:
    """优先级 10 的只读放行:ls/grep 这类命令不用问,直接放行。

    示例:'```shell\nls -la\n```' → 放行,终端不会弹询问。
    """
    if tooluse.tool.name != "shell":
        return None  # 只管 shell;别的工具(save 等)照常走询问
    command = tooluse.kwargs.get("command") if tooluse.kwargs else tooluse.content
    if not command:
        return None
    if is_readonly_command(command):
        return ConfirmationResult("confirm")
    return None
