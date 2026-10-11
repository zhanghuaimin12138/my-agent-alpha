"""内置命令:/help /exit /tools /log /model /hooks。

全部用 @command 装饰器注册 —— 和将来插件写的命令走同一套机制。

(@ 装饰器笔记:@command(...) 的返回值必须能接受一个函数 ——
 也就是写在 @ 下面那个函数;那个函数不是传给 @command,
 而是传给 command(...) 的返回值)
"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha.commands import CommandContext, command
from my_agent_alpha.context import get_workspace
from my_agent_alpha.hooks import get_hooks
from my_agent_alpha.memory_store import list_memories
from my_agent_alpha.tools import get_tools

# 命令注册


# 等价于：help_cmd = command("help","?")(help_cmd)
@command("help", "?")
def help_cmd(ctx: CommandContext) -> None:
    """/help(别名 /?):列出全部命令。"""
    print("命令:")
    print("  /help /?     本帮助          /exit /quit /q  退出")
    print("  /tools       列出可用工具     /log            会话信息")
    print("  /model       当前模型        /hooks          已注册钩子")
    print("其余 / 开头的输入按普通对话处理。")


@command("exit", "quit", "q")
def exit_cmd(ctx: CommandContext) -> None:
    """/exit(别名 /quit /q):结束会话。"""
    print("再见!")
    raise SystemExit(0)


@command("tools")
def tools_cmd(ctx: CommandContext) -> None:
    """/tools:一行一个工具,名字 + 简介。"""
    for t in get_tools():
        mark = "(只读)" if t.read_only else ""
        print(f"  {t.name:<8} {t.desc.split('。')[0]}{mark}")


@command("log")
def log_cmd(ctx: CommandContext) -> None:
    """/log:当前会话信息;/log list 列出全部会话(目录名 + 标题)。"""
    from my_agent_alpha.hooks.naming import read_session_title

    if ctx.args and ctx.args[0] == "list":
        logs_root = ctx.manager.logdir.parent
        sessions = sorted(p for p in logs_root.iterdir() if p.is_dir())
        if not sessions:
            print("  (还没有任何会话)")
            return
        for d in sessions:
            print(f"  {d.name:<22} {read_session_title(d) or '(未命名)'}")
        return
    title = read_session_title(ctx.manager.logdir)  # 可能还没命名(None)
    print(f"  会话目录:{ctx.manager.logdir}")
    if title:
        print(f"  会话标题:{title}")
    print(f"  消息条数:{len(ctx.manager)}")


@command("model")
def model_cmd(ctx: CommandContext) -> None:
    """/model:显示当前模型(切换模型需带 -m 参数重启会话)。"""
    if ctx.args:
        print(f"  当前模型:{ctx.model}(切换请用 -m {ctx.args[0]} 重启会话)")
    else:
        print(f"  当前模型:{ctx.model}")


@command("hooks")
def hooks_cmd(ctx: CommandContext) -> None:
    """/hooks:列出已注册的钩子(学习/调试用)。"""
    hooks = get_hooks()
    if not hooks:
        print("  (没有已注册的钩子)")
        return
    for h in sorted(hooks, key=lambda x: (-x.priority, x.name)):
        print(f"  [{h.hook_type.value}] {h.name}(优先级 {h.priority})")


@command("memory", "mem")
def memory_cmd(ctx: CommandContext) -> None:
    """/memory(别名 /mem):查看记忆;/memory show <名> 看全文;/memory consolidate 整理。"""
    if ctx.args and ctx.args[0] == "consolidate":
        _consolidate_memories()
        return
    if ctx.args and ctx.args[0] == "show" and len(ctx.args) > 1:
        name = ctx.args[1]
        for entry in list_memories(get_workspace() or Path.cwd()):
            if entry.name == name:
                print(entry.path.read_text(encoding="utf-8"))
                return
        print(f"  没有叫 {name!r} 的记忆")
        return
    entries = list_memories(get_workspace() or Path.cwd())
    if not entries:
        print("  (还没有长期记忆;让模型用 memory 工具记一条)")
        return
    for entry in entries:
        print(f"  {entry.name:<20} {entry.title}  [{entry.path.parent}]")


def _consolidate_memories() -> None:
    """派一个受限子代理整理长期记忆(方案 M4,"做梦"的手动版)。

    白名单只给 read/grep/edit/memory —— 整理代理能改能删记忆文件,
    但不能开 shell(Claude Code 的 dream 代理同款约束);
    子代理带 -y 自动执行,派发这个动作本身就是用户审批。
    """
    from my_agent_alpha.memory_store import build_consolidate_prompt
    from my_agent_alpha.tools.subagent import run_subagent

    prompt = build_consolidate_prompt(get_workspace() or Path.cwd())
    for msg in run_subagent(prompt, allowed_tools=("read", "grep", "edit", "memory")):
        print(msg.content)


def register_builtin_commands() -> None:
    """初始化内置命令(幂等)。

    实际注册在上面的 @command 装饰器处、模块被导入时就完成了;
    保留这个函数是给调用方一个明确的初始化入口(和工具注册对称)。
    """
