"""核心循环:三层嵌套(仿 gptme chat.py 的结构)。

整个 agent 对话就是三层循环套在一起:

    chat()                        准备工作:打开/新建会话、写入系统提示词
    └─ _run_chat_loop()           外层:不断取用户输入(相当于普通聊天程序的主循环)
       └─ _process_message_conversation()   中层:处理一条用户输入,
                                             反复"生成→执行工具"直到模型不再要工具
          └─ step()               内层:执行一轮"问模型 + 把模型要的工具跑一遍"

中层什么时候停?就看一件事(整个程序最关键的一个判断):
    看日志里最后一条模型回复,里面还有"写完整的工具调用"吗?
    有 → 工具结果已经追加进日志,再问一轮模型;
    没有 → 这个回合结束,回到提示符等用户输入。
不需要任何额外的状态变量,翻日志就知道该不该继续。

阶段 1-2 刻意省略:hook 系统(阶段 5)、上下文压缩(阶段 6)、
更精细的中断控制、provider 分级重试。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from my_agent_alpha.commands import builtin as builtin_commands
from my_agent_alpha.commands import execute_cmd
from my_agent_alpha.compaction import compact_messages, get_context_budget
from my_agent_alpha.config import Config
from my_agent_alpha.confirm import (
    ConfirmationResult,
    clear_confirm_hooks,
    is_auto_confirm,
    register_confirm_hook,
    set_auto_confirm,
)
from my_agent_alpha.context import agent_context, get_logdir, set_current_model
from my_agent_alpha.hooks import HookType, trigger_hook
from my_agent_alpha.hooks import builtin as builtin_hooks
from my_agent_alpha.llm import ContextLengthError, reply, resolve_model
from my_agent_alpha.logmanager import LogManager
from my_agent_alpha.message import (
    DECLINED_CONTENT,
    INTERRUPT_CONTENT,
    LLM_REQUEST_FAILED_PREFIX,
    Message,
)
from my_agent_alpha.prompts import get_prompt
from my_agent_alpha.tools import add_tools, execute_msg, get_tools
from my_agent_alpha.tools.base import ToolUse
from my_agent_alpha.tools.shell_safety import guardrail_hook, readonly_hook

logger = logging.getLogger(__name__)

MAX_STEPS = int(os.environ.get("ALPHA_MAX_STEPS", "50"))

# ---- 内层:一轮"问模型 + 执行工具" ----


def step(
    log_messages: Sequence[Message],
    model: str,
    on_token: Callable[[str], None] | None = None,
    tool_format: str = "markdown",
    budget: int | None = None,
) -> Iterator[Message]:
    """执行一轮:把对话发给模型,产出模型回复,再产出工具执行结果。

    用生成器(yield)而不是返回列表,好处:调用方每拿到一条消息就能立刻保存,
    中途按 Ctrl-C 也不会丢掉已经产生的内容。

    on_token:模型每生成一小段文字就调用一次这个函数(终端实时显示用;
    传 None 就不显示)。检测到完整工具调用时提前停止生成,逻辑在 reply() 里。
    tool_format:"markdown"(默认,围栏代码块约定)或 "tool"(原生工具调用,
    工具表会随请求发给接口)。两种方式执行的还是同一批工具函数。

    示例(一轮里通常发生的事):
        yield 模型回复("我来执行:\n```shell\nls\n```")
        yield 工具输出("$ ls\na.txt\nReturnCode: 0")
    """
    msgs = prepare_messages(log_messages, model, budget=budget)
    tools = get_tools() if tool_format == "tool" else ()  # 如果支持原生调用需要拿到list[toolSpec]
    try:
        msg_response = reply(msgs, model, on_token=on_token, tools=tools, tool_format=tool_format)
    except ContextLengthError:
        # 溢出自愈:预算砍半、更狠地压缩,再试一次(只重试这一次)
        meta = resolve_model(model)
        emergency = max((budget or get_context_budget(meta.context)) // 2, 1000)
        logger.warning("上下文超长,以预算 %d 重压缩重试一次", emergency)
        msgs = prepare_messages(log_messages, model, budget=emergency)
        msg_response = reply(msgs, model, on_token=on_token, tools=tools, tool_format=tool_format)
    yield msg_response
    # 生成完毕钩子:产出的消息跟在模型回复后面进日志(如 [用量] 记录)
    yield from trigger_hook(HookType.GENERATION_POST, message=msg_response)
    yield from execute_msg(msg_response)


def prepare_messages(
    messages: Sequence[Message], model: str, budget: int | None = None
) -> list[Message]:
    """决定"发给模型哪些消息":超预算时压缩视图,日志本身不动。

    - budget 不传时按模型的上下文窗口自动算(见 compaction.get_context_budget);
    - 压缩只改这份返回值(视图),log.messages 原封不动 —— 磁盘日志永不改写;
    - 详细的压缩规则见 compaction 模块(剥思考 → 截工具结果 → 压长回复 → 硬截断)。
    """
    meta = resolve_model(model)
    effective = budget if budget is not None else get_context_budget(meta.context)
    return compact_messages(list(messages), effective, get_logdir())


def has_runnable_tooluse(messages: Sequence[Message]) -> bool:
    """回答"模型刚才的回复里,还有没执行完的工具调用吗?" —— 决定循环是否继续。

    示例:
        最后一条模型回复 = "我来:\n```shell\nls\n```"  → True(继续下一轮)
        最后一条模型回复 = "任务完成了"                → False(回合结束)
    """
    for msg in reversed(messages):
        if msg.role == "assistant":
            return any(tu.is_runnable for tu in ToolUse.iter_from_content(msg.content))
    return False


# ---- 中层:处理一条用户输入 ----


_DIM = "\x1b[2m"  # 终端"暗色"控制码(让文字看起来更淡)
_RESET = "\x1b[0m"  # 恢复正常颜色
_TAGS = ("<think>", "</think>")


class _StreamPrinter:
    """把模型流式生成的小段文字实时打印到终端。

    模型的推理内容包在 <think>...</think> 里,打印成暗色,和正文区分开。

    一个小麻烦:文字是一小段一小段到达的,"<think>" 这个标签可能被
    从中间切开(比如先到 "<thi" 再到 "nk>")。所以收到"看起来像半个标签"
    的尾巴时先扣住不发,等下一段到了能拼齐再判断;标签本身不打印,
    只用来切换"暗色/正常"两种显示状态。
    """

    def __init__(self) -> None:
        """初始化渲染状态:不在思考区、无扣留尾巴、还没打印过任何内容。"""
        self._in_think = False  # 当前是否在 <think> 区里
        self._hold = ""  # 扣住的"可能是半个标签"的尾巴
        self._started = False  # 是否已经打印过(用来在第一段前空一行)

    def token(self, chunk: str) -> None:
        """收到一段新文字,处理其中的标签并打印(作为 on_token 回调使用)。"""
        if not self._started:
            print()  # 与上一条输出空一行
            self._started = True
        buf = self._hold + chunk
        self._hold = ""
        while True:
            # 找当前状态对应的"出场标签":在思考区找 </think>,在外面找 <think>
            tag = _TAGS[1] if self._in_think else _TAGS[0]
            idx = buf.find(tag)
            if idx < 0:
                # 没找到完整标签:打印能确定安全的部分,扣住可能是半截标签的尾巴
                keep = self._partial_tag_len(buf)
                self._emit(buf[:keep])
                self._hold = buf[keep:]
                return
            self._emit(buf[:idx])
            buf = buf[idx + len(tag) :]
            self._in_think = not self._in_think

    def finish(self) -> None:
        """流结束后收尾:把扣住的尾巴打出来,再换一行。"""
        if self._hold:  # 流结束,残留尾巴不是标签,直接吐出
            self._emit(self._hold)
            self._hold = ""
        if self._started:
            print()  # 收尾换行

    def _emit(self, text: str) -> None:
        """打印一段文字:思考区用暗色,普通文字原样(不加换行,立刻刷新)。"""
        if not text:
            return
        if self._in_think:
            print(f"{_DIM}{text}{_RESET}", end="", flush=True)
        else:
            print(text, end="", flush=True)

    @staticmethod
    def _partial_tag_len(buf: str) -> int:
        """返回"可以放心打印多长"。比如 buf 以 "<thi" 结尾,
        它可能是 "<think>" 的前半截,这 4 个字符要扣住,返回 len(buf)-4。
        """
        for tag in _TAGS:
            for k in range(min(len(tag) - 1, len(buf)), 0, -1):
                if buf.endswith(tag[:k]):
                    return len(buf) - k
        return len(buf)


def _process_message_conversation(
    manager: LogManager, model: str, tool_format: str = "markdown"
) -> None:
    """处理一条用户输入:反复执行 step(),直到模型不再要工具或步数到上限。

    每拿到一条消息就立刻写进日志(而不是攒到最后一起写):
    这样中途断电或 Ctrl-C,已产生的内容(模型回复、已执行的工具结果)都在。

    模型的回复文字在流式过程中已经打印过了,这里不再重复打印;
    工具输出等其他消息由 _print_message 打印。
    """
    steps = 0
    while True:
        _dispatch_hooks(manager, HookType.STEP_PRE, model=model)
        printer = _StreamPrinter()
        interrupted = False
        declined = False
        for msg in step(manager.messages, model, on_token=printer.token, tool_format=tool_format):
            manager.append(msg)
            if msg.role == "assistant":
                printer.finish()
                if msg.metadata and msg.metadata.get("interrupted"):
                    interrupted = True
            else:
                _print_message(msg)
            if msg.role == "system" and msg.content.startswith(INTERRUPT_CONTENT):
                interrupted = True
            if msg.role == "system" and msg.content.startswith(DECLINED_CONTENT):
                declined = True
        steps += 1
        if interrupted:
            manager.append(Message("system", INTERRUPT_CONTENT, quiet=True))
            print("\n(已中断,回到提示符)")
            return  # 停止本 turn,回到提示符
        if declined:
            print("\n(已拒绝执行,本回合结束;可换个说法让模型重来)")
            return  # 用户拒绝了工具:停止本回合(仿 gptme 的 DECLINED 语义)
        if steps >= MAX_STEPS:
            manager.append(Message("system", f"已达到单 turn 最大步数({MAX_STEPS}),停止。"))
            return
        if not has_runnable_tooluse(manager.messages):
            return


# ---- 钩子分发:生命周期的统一出口 ----


def _dispatch_hooks(manager: LogManager, hook_type: HookType, **ctx: object) -> None:
    """触发一个生命周期点:钩子产出的消息立刻落盘并显示。

    这是钩子和日志系统的接合点:钩子只管产出消息,落盘/打印都由这里统一做。
    (manager 会作为上下文的一部分自动传给钩子)
    """
    for msg in trigger_hook(hook_type, manager=manager, **ctx):
        manager.append(msg)
        _print_message(msg)


# ---- 确认链:终端里问用户 ----


def _preview(tooluse: ToolUse) -> str:
    """生成给用户看的"这次要执行什么"预览。

    示例:shell 调用 → "shell: command=ls -la";
    save 调用 → "save a.txt" + 内容前几行。
    """
    if tooluse.tool.name in ("save", "append") and tooluse.content is not None:
        path = (
            (tooluse.args or ["?"])[0] if tooluse.args else (tooluse.kwargs or {}).get("path", "?")
        )
        lines = tooluse.content.split("\n")
        head = "\n".join(lines[:5])
        more = "\n…" if len(lines) > 5 else ""
        return f"{tooluse.tool.name} {path}\n{head}{more}"
    return tooluse.display_command()


def cli_confirm_hook(tooluse: ToolUse) -> ConfirmationResult | None:
    """优先级 0 的终端询问:打印预览,问用户 y/n/e/a(只在交互模式注册)。

    y 或回车 → 放行;n → 跳过;e → 用户输入改写后的内容再执行;
    a → 本会话后续都不再问。输入结束(Ctrl-D)按跳过处理(安全方向)。
    """
    if is_auto_confirm():
        return None  # 用户已选"全部放行":本钩子退位,链尾自动放行
    print(f"\n┌─ 需要确认:{tooluse.tool.name} ─────────────")
    print(f"│ {_preview(tooluse)}")
    while True:
        try:
            answer = input("└─ 执行?[y]是 / n 跳过 / e 改写 / a 全部放行: ").strip().lower()
        except EOFError:
            return ConfirmationResult("skip", message="输入结束,按跳过处理")
        if answer in ("y", ""):
            return ConfirmationResult("confirm")
        if answer == "n":
            return ConfirmationResult("skip", message="用户选择跳过")
        if answer == "a":
            set_auto_confirm(True)
            return ConfirmationResult("confirm")
        if answer == "e":
            edited = input("  改写为: ")
            if edited.strip():
                return ConfirmationResult("edit", edited_content=edited)


def _setup_confirm_hooks(interactive: bool, yes: bool) -> None:
    """会话开始时装配确认链(每次重建,防止上个会话残留)。

    - 安全护栏(200):永远注册 —— 灾难命令连 -y 也不放行;
    - 只读放行(10):永远注册 —— ls/grep 不用烦用户;
    - 终端询问(0):只在交互模式且没开 -y 时注册 ——
      非交互(脚本/测试)和 -y 走到链尾自动放行(自主模式)。
    """
    clear_confirm_hooks()
    register_confirm_hook("guardrail", guardrail_hook, priority=200)
    register_confirm_hook("readonly", readonly_hook, priority=10)
    if interactive and not yes:
        register_confirm_hook("cli", cli_confirm_hook, priority=0)


# ---- 外层:和用户来回对话的主循环 ----


def _get_user_input() -> Message | None:
    """在终端等用户输入一行字,包成 user 消息返回。

    - 空行:再问一次(用循环实现,不递归 —— 递归按几千次空行会爆栈);
    - Ctrl-C:提示一下,继续等输入;
    - Ctrl-D(输入结束):返回 None,表示用户要退出。
    """
    while True:
        try:
            content = input("\n> ")
        except EOFError:
            return None
        except KeyboardInterrupt:
            print("(Ctrl-C;Ctrl-D 退出)")
            continue
        if content.strip():
            return Message("user", content)


def _print_message(msg: Message) -> None:
    """把一条消息打印到终端(用户输入带 > 前缀,工具输出带 │ 前缀)。

    hide 或 quiet 的消息不打印(比如系统提示词、中断标记)。
    """
    if msg.hide or msg.quiet:
        return
    if msg.role == "user":
        print(f"\n> {msg.content}")
    elif msg.role == "assistant":
        print(f"\n{msg.content}")
    else:
        print(f"\n│ {msg.content}")


def _run_chat_loop(
    manager: LogManager,
    prompt_queue: list[str],
    model: str,
    *,
    non_interactive: bool,
    tool_format: str = "markdown",
) -> None:
    """外层主循环:不断取下一条输入(命令行参数或终端输入),交给中层处理。

    non_interactive=True 时:命令行带来的提示词处理完就返回(脚本模式);
    否则一直等到用户在终端按 Ctrl-D 才结束。
    """
    while True:
        if prompt_queue:
            content = prompt_queue.pop(0)
        elif non_interactive:
            return
        else:
            msg = _get_user_input()
            if msg is None:
                return
            content = msg.content

        if content.startswith("/") and execute_cmd(content, manager, model):
            continue

        manager.append(Message("user", content))
        _print_message(Message("user", content))
        _dispatch_hooks(manager, HookType.TURN_PRE, content=content)
        try:
            _process_message_conversation(manager, model, tool_format)
        except KeyboardInterrupt:
            # 中断发生在模型生成期间(工具执行期间的中断由 execute_msg 收尾)
            manager.append(Message("system", INTERRUPT_CONTENT, quiet=True))
            print("\n(已中断,回到提示符)")
        except Exception as e:
            if non_interactive:
                raise  # 脚本模式:让异常带着错误码退出
            manager.append(Message("system", f"{LLM_REQUEST_FAILED_PREFIX} {e}"))
            print(f"\n{LLM_REQUEST_FAILED_PREFIX} {e}(回到提示符)")
        _dispatch_hooks(manager, HookType.TURN_POST)
        _dispatch_hooks(manager, HookType.LOOP_CONTINUE)


def chat(
    prompts: Sequence[str],
    logdir: Path,
    workspace: Path,
    model: str,
    *,
    non_interactive: bool = False,
    tool_format: str = "markdown",
    yes: bool = False,
) -> LogManager:
    """会话入口:准备会话 → 进入对话循环 → 返回会话管理器。

    - 新会话:先把系统提示词(身份 + 工具说明 + 时间)写进日志;
    - 旧会话:重新打开,把历史消息打印一遍,接着聊;
    - prompts:开场就要发送的输入(命令行传进来的);non_interactive
      见 _run_chat_loop 的说明;
    - tool_format:"markdown" 或 "tool"(原生工具调用);
    - yes=True 相当于 -y:交互模式也不问确认(安全护栏仍然生效)。

    返回 LogManager,测试用它检查日志内容;正常调用可以忽略返回值。
    """
    # MCP:配置了外部服务器就连上,远端工具转成本地 ToolSpec(gptme 同款)
    mcp_clients: list[Any] = []
    config = Config.load(workspace)
    if config.mcp_servers:
        from my_agent_alpha.mcp_client import create_mcp_tools

        specs, mcp_clients = create_mcp_tools(config.mcp_servers)
        if specs:
            add_tools(specs)  # 内置表基础上追加(名字带"服务器."前缀,不会撞)
            print(f"(MCP:已接入 {len(specs)} 个远端工具)")

    tools = get_tools()
    # 确保内置钩子/命令就位(同名重注册=替换,幂等)。
    # 注意:不清别人的钩子!调用方在会话前注册的自定义钩子必须存活;
    # 想清场请显式调 clear_hooks()(测试的 conftest 就是这么做的)。
    builtin_commands.register_builtin_commands()
    # 注册消息hook（生命周期）
    builtin_hooks.register_builtin_hooks()
    # 激活确认链
    _setup_confirm_hooks(interactive=not non_interactive, yes=yes)
    manager = LogManager.load(logdir, create=True)
    if len(manager) == 0:
        for msg in get_prompt(
            tools, native=tool_format == "tool"
        ):  #  拿到工具使用说明（system message）
            manager.append(msg, sync=True)
        # 长期记忆索引:有记忆才注入(gptme 的做法,64KB 预算)
        from my_agent_alpha.memory_store import render_memory_index

        mem_index = render_memory_index(workspace)
        if mem_index:
            manager.append(Message("system", mem_index, pinned=True, hide=True))
    else:
        # 恢复会话:回放已打印的历史
        for msg in manager.messages:
            _print_message(msg)

    try:
        with agent_context(workspace, logdir):
            set_current_model(model)  # subagent 工具会继承这个模型
            _dispatch_hooks(manager, HookType.SESSION_START, model=model)
            try:
                _run_chat_loop(
                    manager,
                    list(prompts),
                    model,
                    non_interactive=non_interactive,
                    tool_format=tool_format,
                )
            finally:
                _dispatch_hooks(manager, HookType.SESSION_END)
    finally:
        for client in mcp_clients:
            client.close()  # 收掉 MCP 子进程
    return manager
