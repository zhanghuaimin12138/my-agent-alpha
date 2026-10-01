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

import os
from collections.abc import Callable, Iterator, Sequence
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


# ---- 内层:一轮"问模型 + 执行工具" ----


def step(
    log_messages: Sequence[Message],
    model: str,
    on_token: Callable[[str], None] | None = None,
    tool_format: str = "markdown",
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
    msgs = prepare_messages(log_messages)
    tools = get_tools() if tool_format == "tool" else () # 如果支持原生调用需要拿到list[toolSpec]
    msg_response = reply(msgs, model, on_token=on_token, tools=tools, tool_format=tool_format)
    yield msg_response
    yield from execute_msg(msg_response)


def prepare_messages(messages: Sequence[Message]) -> list[Message]:
    """决定"发给模型哪些消息"。当前阶段:原样全发。

    为什么单独一个函数?以后要在这里做上下文压缩(阶段 6):
    对话太长时摘要旧消息,只把摘要发給模型。现在先占住这个位置。
    """
    return list(messages)


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
        printer = _StreamPrinter()
        interrupted = False
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
        steps += 1
        if interrupted:
            manager.append(Message("system", INTERRUPT_CONTENT, quiet=True))
            print("\n(已中断,回到提示符)")
            return  # 停止本 turn,回到提示符
        if steps >= MAX_STEPS:
            manager.append(Message("system", f"已达到单 turn 最大步数({MAX_STEPS}),停止。"))
            return
        if not has_runnable_tooluse(manager.messages):
            return


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


def _handle_command(content: str) -> bool:
    """处理 / 开头的命令(目前只有 /exit /quit /help)。

    返回 True 表示这条输入是命令、已经处理完,不要当成对话内容;
    返回 False 表示不是认识的命令,按普通对话处理。
    """
    name = content.strip().lstrip("/").split(" ", 1)[0].lower()
    if name not in _COMMANDS:
        return False
    if name in ("exit", "quit"):
        raise SystemExit(0)
    if name == "help":
        print("命令:/exit /quit 退出;/help 帮助。其余 / 开头消息按普通文本处理。")
    return True


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

        if content.startswith("/") and _handle_command(content):
            continue

        manager.append(Message("user", content))
        _print_message(Message("user", content))
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


def chat(
    prompts: Sequence[str],
    logdir: Path,
    workspace: Path,
    model: str,
    *,
    non_interactive: bool = False,
    tool_format: str = "markdown",
) -> LogManager:
    """会话入口:准备会话 → 进入对话循环 → 返回会话管理器。

    - 新会话:先把系统提示词(身份 + 工具说明 + 时间)写进日志;
    - 旧会话:重新打开,把历史消息打印一遍,接着聊;
    - prompts:开场就要发送的输入(命令行传进来的);non_interactive
      见 _run_chat_loop 的说明;
    - tool_format:"markdown" 或 "tool"(原生工具调用)。

    返回 LogManager,测试用它检查日志内容;正常调用可以忽略返回值。
    """
    tools = get_tools()
    manager = LogManager.load(logdir, create=True)
    if len(manager) == 0:
        for msg in get_prompt(
            tools, native=tool_format == "tool"
        ):  #  拿到工具使用说明（system message）
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
            tool_format=tool_format,
        )
    return manager
