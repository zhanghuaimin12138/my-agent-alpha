"""核心循环:三层嵌套(仿 gptme chat.py 的结构,阶段 1 非流式版)。

    chat()                          会话装配(装载日志、种子系统提示词、设上下文)
    └─ _run_chat_loop()             外层 REPL:取输入 / 斜杠命令 / 中断处理
       └─ _process_message_conversation()   中层:一个 turn 内反复 step,
                                             直到 assistant 无可运行工具调用
          └─ step()                 内层:一次"生成 + 工具执行"

继续判定(整个 agent 的心脏,零额外状态):
    最后一条 assistant 消息里还有可运行的工具调用吗?
    有 → 把工具结果喂回去再生成;无 → turn 结束,回到用户提示符。

阶段 1 刻意省略:流式(阶段 2)、hook(阶段 5)、压缩(阶段 6)、
中断 ContextVar 门、provider 重试、崩溃恢复标记协议。
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path

from my_agent_alpha.context import agent_context
from my_agent_alpha.llm import reply
from my_agent_alpha.logmanager import LogManager
from my_agent_alpha.message import INTERRUPT_CONTENT, LLM_REQUEST_FAILED_PREFIX, Message
from my_agent_alpha.prompts import get_prompt
from my_agent_alpha.tools import execute_msg, get_tools
from my_agent_alpha.tools.base import ToolUse

MAX_STEPS = int(os.environ.get("ALPHA_MAX_STEPS", "50"))

_COMMANDS = {"exit", "quit", "help"}


# ---- 内层:一次"生成 + 工具执行" ----


def step(log_messages: Sequence[Message], model: str) -> Iterator[Message]:
    """一次 pass:请求 LLM → yield assistant 消息 → yield 工具结果消息。

    阶段 1 的 prepare_messages 就是直通(压缩/裁剪属阶段 6),
    但保留这个显式函数名,对应 gptme logmanager 的 prepare_messages 管线入口。
    """
    msgs = prepare_messages(log_messages)
    msg_response = reply(msgs, model)
    yield msg_response
    yield from execute_msg(msg_response)


def prepare_messages(messages: Sequence[Message]) -> list[Message]:
    """日志 → LLM 输入的变换管线(阶段 1:直通)。"""
    return list(messages)


def has_runnable_tooluse(messages: Sequence[Message]) -> bool:
    """判定:最后一条 assistant 消息是否还有可运行工具调用。"""
    for msg in reversed(messages):
        if msg.role == "assistant":
            return any(tu.is_runnable for tu in ToolUse.iter_from_content(msg.content))
    return False


# ---- 中层:一个用户 turn ----


def _process_message_conversation(manager: LogManager, model: str) -> None:
    """反复 step 并追加结果,直到无待执行工具调用或达到步数上限。

    增量消费 step() 的产出:每 yield 一条就立刻落盘 ——
    中断/异常时已产生的消息(assistant 消息、已执行工具的结果)不丢失
    (审查 #1:先 list() 物化再追加会把它们整个吞掉)。
    """
    steps = 0
    while True:
        interrupted = False
        for msg in step(manager.messages, model):
            manager.append(msg)
            _print_message(msg)
            if msg.role == "system" and msg.content.startswith(INTERRUPT_CONTENT):
                interrupted = True
        steps += 1
        if interrupted:
            return  # 工具被中断:停止本 turn,回到提示符
        if steps >= MAX_STEPS:
            manager.append(Message("system", f"已达到单 turn 最大步数({MAX_STEPS}),停止。"))
            return
        if not has_runnable_tooluse(manager.messages):
            return


# ---- 外层:REPL ----


def _get_user_input() -> Message | None:
    """交互式读取用户输入;EOF(Ctrl-D)返回 None 表示退出。

    用循环而非递归处理空输入/Ctrl-C(审查 #3:递归版千次空行会栈溢出)。
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


def _handle_command(content: str) -> bool:
    """处理 / 开头的命令;返回 True 表示已消费(不进入对话)。"""
    name = content.strip().lstrip("/").split(" ", 1)[0].lower()
    if name not in _COMMANDS:
        return False
    if name in ("exit", "quit"):
        raise SystemExit(0)
    if name == "help":
        print("命令:/exit /quit 退出;/help 帮助。其余 / 开头消息按普通文本处理。")
    return True


def _print_message(msg: Message) -> None:
    """渲染一条消息(阶段 1 纯文本;hide/quiet 不打印)。"""
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
) -> None:
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

        if content.startswith("/") and _handle_command(content):
            continue

        manager.append(Message("user", content))
        _print_message(Message("user", content))
        try:
            _process_message_conversation(manager, model)
        except KeyboardInterrupt:
            # 中断发生在 LLM 生成期间(工具执行期间的中断由 execute_msg 收尾)
            manager.append(Message("system", INTERRUPT_CONTENT, quiet=True))
            print("\n(已中断,回到提示符)")
        except Exception as e:
            if non_interactive:
                raise  # 脚本/CI 模式:异常要带着退出码走出去
            manager.append(Message("system", f"{LLM_REQUEST_FAILED_PREFIX} {e}"))
            print(f"\n{LLM_REQUEST_FAILED_PREFIX} {e}(回到提示符)")


def chat(
    prompts: Sequence[str],
    logdir: Path,
    workspace: Path,
    model: str,
    *,
    non_interactive: bool = False,
) -> LogManager:
    """会话入口:装配 → 运行循环 → 返回 manager(测试断言用)。"""
    tools = get_tools()
    manager = LogManager.load(logdir, create=True)
    if len(manager) == 0:
        for msg in get_prompt(tools): #  拿到工具使用说明（system message）
            manager.append(msg, sync=True)
    else:
        # 恢复会话:回放已打印的历史
        for msg in manager.messages:
            _print_message(msg)

    with agent_context(workspace, logdir):
        _run_chat_loop(
            manager,
            list(prompts),
            model,
            non_interactive=non_interactive,
        )
    return manager
