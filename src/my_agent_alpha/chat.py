"""核心循环:三层嵌套。

整个 agent 对话就是三层循环套在一起:

    chat()                        准备工作:打开/新建会话、写入系统提示词
    └─ _run_chat_loop()           外层:不断取用户输入(相当于普通聊天程序的主循环)
       └─ _process_message_conversation()   中层:处理一条用户输入,
                                             反复"生成→执行工具"直到模型不再要工具
          └─ step()               内层:执行一轮"问模型 + 把模型要的工具跑一遍"

本文件只留这条主链路,配套功能都在各自的家里(2026-10-30 拆分,见 devlog 037):

    终端显示/输入/回放   → tui 包(StreamPrinter、print_message、read_user_input、replay)
    发给模型前的压缩视图 → compaction.prepare_messages
    "还有没有要跑的工具" → tools.base.has_runnable_tooluse
    钩子的统一出口       → hooks.dispatch_hooks
    确认链的装配         → confirm.setup_confirm_chain
    /resume 会话切换     → logmanager.sessions.switch_session

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

from my_agent_alpha.commands import ResumeRequested, execute_cmd
from my_agent_alpha.commands import builtin as builtin_commands
from my_agent_alpha.compaction import prepare_messages
from my_agent_alpha.confirm import setup_confirm_chain
from my_agent_alpha.context import agent_context, set_current_model
from my_agent_alpha.hooks import HookType, dispatch_hooks, trigger_hook
from my_agent_alpha.hooks import builtin as builtin_hooks
from my_agent_alpha.llm import ContextLengthError, reply, resolve_model
from my_agent_alpha.logmanager import LogManager
from my_agent_alpha.logmanager.sessions import prune_empty_session, switch_session
from my_agent_alpha.message import (
    DECLINED_CONTENT,
    INTERRUPT_CONTENT,
    LLM_REQUEST_FAILED_PREFIX,
    Message,
)
from my_agent_alpha.prompts import session_opening_messages
from my_agent_alpha.tools import execute_msg, get_tools
from my_agent_alpha.tools.base import has_runnable_tooluse
from my_agent_alpha.tui import StreamPrinter, print_message, read_user_input, replay
from my_agent_alpha.utils import get_context_budget

logger = logging.getLogger(__name__)

MAX_STEPS = int(os.environ.get("ALPHA_MAX_STEPS", "50"))


def chat(
    prompts: Sequence[str],
    logdir: Path,
    workspace: Path,
    model: str,
    *,
    non_interactive: bool = False,
    tool_format: str = "tool",
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

    from my_agent_alpha.mcp_client import connect_mcp_tools

    # 连接外部mcp
    mcp_clients = connect_mcp_tools(workspace)

    # 初始化工具
    tools = get_tools()  # 拿到全部工具

    # 初始化内置命令
    builtin_commands.register_builtin_commands()

    # 初始化内置钩子
    builtin_hooks.register_builtin_hooks()

    # 激活确认链
    setup_confirm_chain(interactive=not non_interactive, yes=yes)

    manager = LogManager.load(logdir, create=True)
    if len(manager) == 0:
        # 初始化日志（系统提示词 + 长期记忆索引）
        for msg in session_opening_messages(
            tools, native=tool_format == "tool", workspace=workspace
        ):
            manager.append(msg, sync=True)
    else:
        # 恢复会话:回放已打印的历史
        replay(manager)

    try:
        with agent_context(workspace, logdir):
            set_current_model(model)  # subagent 工具会继承这个模型

            # 触发 SESSION_START 生命周期hook
            dispatch_hooks(manager, HookType.SESSION_START, model=model)
            try:
                # 接住循环返回的"最终"会话:/resume 可能中途换过,收尾对准它
                manager = _run_chat_loop(
                    manager,
                    list(prompts),
                    model,
                    non_interactive=non_interactive,
                    tool_format=tool_format,
                )
            finally:
                dispatch_hooks(manager, HookType.SESSION_END)
    finally:
        for client in mcp_clients:
            client.close()  # 收掉 MCP 子进程
        # 没人聊过的会话不留壳(起了名就退出的、刚开就 /resume 跳走的)
        if prune_empty_session(manager.logdir):
            print(f"(清理了没人聊过的会话 {manager.logdir.name[:8]})")
    return manager


# ---- 外层:和用户来回对话的主循环 ----


def _run_chat_loop(
    manager: LogManager,
    prompt_queue: list[str],
    model: str,
    *,
    non_interactive: bool,
    tool_format: str = "tool",
) -> LogManager:
    """外层主循环:不断取下一条输入(命令行参数或终端输入),交给中层处理。"""
    while True:
        if prompt_queue:
            content = prompt_queue.pop(0)
        elif non_interactive:
            return manager
        else:
            msg = read_user_input()
            if msg is None:
                return manager
            content = msg.content

        # 处理命令
        if content.startswith("/"):
            try:
                if execute_cmd(content, manager, model, non_interactive=non_interactive):
                    continue
            except ResumeRequested as r:
                # /resume:命令层只负责找到目标,切换在这里执行
                manager = switch_session(manager, r.meta, model)
                continue

        # append和钩子也纳入 try:任何一条消息的存储问题都不该打死 REPL
        # (日志层已对坏字符兜底,这里是最后一道网)
        try:
            # 用户消息落盘
            manager.append(Message("user", content))
            # 触发turn_pre hook
            dispatch_hooks(manager, HookType.TURN_PRE, content=content)

            # 处理一条用户输入
            _process_message_conversation(manager, model, tool_format)

        except KeyboardInterrupt:
            # 中断发生在模型生成期间(工具执行期间的中断由 execute_msg 收尾)
            manager.append(Message("system", INTERRUPT_CONTENT, quiet=True))
            print("\n(已中断,回到提示符)")
        except Exception as e:
            if non_interactive:
                raise  # 非交互，出错直接向上抛出异常
            manager.append(Message("system", f"{LLM_REQUEST_FAILED_PREFIX} {e}"))
            print(f"\n{LLM_REQUEST_FAILED_PREFIX} {e}(回到提示符)")

        dispatch_hooks(manager, HookType.TURN_POST)

        # loop.continue钩子，产出两类消息：user/system
        # 产出的 user 消息回灌队列(由消费端统一落日志，不在这里重复落日志）
        # 产出的 system 只落日志不回灌队列
        for msg in trigger_hook(HookType.LOOP_CONTINUE, manager=manager):
            print_message(msg)
            if msg.role == "user":
                prompt_queue.append(msg.content)
            else:
                manager.append(msg)  # system消息，直接落盘


def _process_message_conversation(
    manager: LogManager, model: str, tool_format: str = "tool"
) -> None:
    """处理一条用户输入:反复执行 step(),直到模型不再要工具或步数到上限。

    每拿到一条消息就立刻写进日志(而不是攒到最后一起写):
    这样中途断电或 Ctrl-C,已产生的内容(模型回复、已执行的工具结果)都在。

    模型的回复文字在流式过程中已经打印过了,这里不再重复打印。

    停止条件就一条:最后一条模型回复里没有任何可执行的工具调用。
    """
    steps = 0
    while True:
        dispatch_hooks(manager, HookType.STEP_PRE, model=model)

        # 初始化渲染器
        printer = StreamPrinter()

        # 中断标记
        interrupted = False
        # 工具取消标记
        declined = False

        # 取step产出的消息（llm/tool/GENERATION_POST hook）
        for msg in step(manager.messages, model, on_token=printer.token, tool_format=tool_format):
            # 产出消息先落盘
            manager.append(msg)
            if msg.role == "assistant":
                printer.finish()
                # 判断llm流式输出是否被中断
                if msg.metadata and msg.metadata.get("interrupted"):
                    interrupted = True
            else:
                print_message(msg)
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
        if has_runnable_tooluse(manager.messages):
            continue  # 带着工具结果再问一轮(继续判定规则:只看最后一条回复)
        return  # 没有任何工具调用:回合结束,回到提示符


# ---- 内层:一轮"问模型 + 执行工具" ----


def step(
    log_messages: Sequence[Message],
    model: str,
    on_token: Callable[[str], None] | None = None,
    tool_format: str = "tool",
    budget: int | None = None,
) -> Iterator[Message]:
    """执行一轮: llm调用 + 解析工具输出 + yield msgs(不落盘)。

    用生成器(yield)而不是返回列表,好处:调用方每拿到一条消息就能立刻保存,
    中途按 Ctrl-C 也不会丢掉已经产生的内容。

    on_token:模型每生成一小段文字就调用一次这个函数(终端实时显示用;
    传 None 就不显示)。检测到完整工具调用时提前停止生成,逻辑在 reply() 里。
    tool_format:"tool"(默认,原生工具调用,工具表会随请求发给接口)
    或 "markdown"(围栏代码块约定)。两种方式执行的还是同一批工具函数。

    示例(一轮里通常发生的事):
        yield 模型回复("我来执行:\n```shell\nls\n```")
        yield 工具输出("$ ls\na.txt\nReturnCode: 0")
    """
    # 组装上下文
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
    # GENERATION_POST hook 输出msg:产出的消息跟在模型回复后面进日志(如 [用量] 记录)
    yield from trigger_hook(HookType.GENERATION_POST, message=msg_response)
    # 产出工具调用消息（list[msg]得到后一个一个msg yield出去）
    yield from execute_msg(msg_response)
