"""内置命令:/help /exit /tools /log /model /hooks /resume。

全部用 @command 装饰器注册 —— 和将来插件写的命令走同一套机制。

(@ 装饰器笔记:@command(...) 的返回值必须能接受一个函数 ——
 也就是写在 @ 下面那个函数;那个函数不是传给 @command,
 而是传给 command(...) 的返回值)
"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha.commands import CommandContext, ResumeRequested, command
from my_agent_alpha.context import get_workspace
from my_agent_alpha.hooks import get_hooks
from my_agent_alpha.logmanager.sessions import (
    SessionMeta,
    find_by_prefix,
    format_session_row,
    is_subagent,
    list_sessions,
    read_session_title,
)
from my_agent_alpha.memory_store import list_memories
from my_agent_alpha.tools import get_tools

# 命令注册


# 等价于：help_cmd = command("help","?")(help_cmd)
@command("help", "?")
def help_cmd(ctx: CommandContext) -> None:
    """/help(别名 /?):列出全部命令。"""
    print("命令:")
    print("  /help /?     本帮助          /exit /quit /q  退出")
    print("  /tools       列出可用工具     /log list       会话列表")
    print("  /resume      恢复旧会话      /model          当前模型")
    print("  /hooks       已注册钩子      /memory         查看记忆")
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
    """/log:当前会话信息;/log list 列出全部会话(时间 + 标题,新→旧)。"""
    if ctx.args and ctx.args[0] == "list":
        # 只列"能续聊"的会话:子代理的工作记录、没人聊过的会话都过滤掉
        sessions = [
            m
            for m in list_sessions(ctx.manager.logdir.parent)
            if not is_subagent(m) and m.n_user > 0
        ]
        if not sessions:
            print("  (还没有任何会话)")
            return
        for m in sessions:
            print(format_session_row(m))
        print("  (恢复用 /resume <码>)")
        return
    title = read_session_title(ctx.manager.logdir)  # 可能还没命名(None)
    print(f"  会话目录:{ctx.manager.logdir}")
    if title:
        print(f"  会话标题:{title}")
    # 消息条数报完整历史(压缩不能让会话"看起来变短");激活了视图再补一行
    # 说明模型实际读的是哪份,避免看到两个数字不一样以为出了错
    print(f"  消息条数:{len(ctx.manager.master_messages)}(完整历史)")
    if ctx.manager.current_view is not None:
        print(
            f"  当前视图:{ctx.manager.current_view} · "
            f"{len(ctx.manager.messages)} 条(模型读这份压缩后的)"
        )


@command("resume")
def resume_cmd(ctx: CommandContext) -> None:
    """/resume <码>:恢复旧会话接着聊;不带码显示编号列表,输入序号选择。"""
    meta = _pick_resume_target(ctx)
    if meta is None:
        return  # 没选成(找不到/撞码/取消),提示语已在 _pick 里打印
    if meta.logdir == ctx.manager.logdir:
        print("  这就是当前会话")
        return
    print(f"  (当前会话已封存,回到它可用 /resume {meta.session_id[:8]})")
    raise ResumeRequested(meta)


def _pick_resume_target(ctx: CommandContext) -> SessionMeta | None:
    """解析 /resume 的参数,选出目标会话;选不成返回 None(原因已打印)。

    两条路:带码 = 按会话 id 前缀找(git 短哈希同款,撞码要多输几位);
    不带码 = 编号列表 + 输入序号(回车取消)。子代理会话一律拒绝。
    """
    root = ctx.manager.logdir.parent
    if not ctx.args:
        # 无参:列表只收"可续聊"的(排除子代理、没人聊过的、当前会话)
        candidates = [
            m
            for m in list_sessions(root)
            if not is_subagent(m) and m.n_user > 0 and m.logdir != ctx.manager.logdir
        ]
        if not candidates:
            print("  (没有可恢复的会话)")
            return None
        for i, m in enumerate(candidates, 1):
            print(format_session_row(m, i))
        choice = input("输入序号恢复(回车取消): ").strip()
        if not choice.isdigit() or not 1 <= int(choice) <= len(candidates):
            print("  (无效序号,已取消)")
            return None
        return candidates[int(choice) - 1]
    matches = find_by_prefix(root, ctx.args[0])
    if matches and all(is_subagent(m) for m in matches):
        print("  这是子代理的工作记录,不能恢复续聊")
        return None
    matches = [m for m in matches if not is_subagent(m)]
    if not matches:
        print(f"  没找到以 {ctx.args[0]} 开头的会话,用 /log list 查看")
        return None
    if len(matches) > 1:
        print("  多个会话以此开头,请多输几位:")
        for m in matches:
            print(format_session_row(m))
        return None
    return matches[0]


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
