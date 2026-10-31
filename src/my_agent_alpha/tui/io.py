"""终端输入输出:读用户的一行输入、把消息按固定格式打印、回放历史。

主链路(chat.py)只管循环和日志,所有"往终端读/画"的动作集中在本文件:
现场聊天时看到的消息样式,和 /resume 回放时的样式,都出自这里。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from my_agent_alpha.message import Message, clean_bad_unicode

if TYPE_CHECKING:  # 只在类型检查时导入,避免 tui → logmanager 的运行时依赖
    from my_agent_alpha.logmanager import LogManager


def read_user_input() -> Message | None:
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
            # 终端偶尔会传来坏字节(输入法/粘贴的表情序列坏掉),解码后留下
            # 编码不了的"落单代理字符"——不清理的话,这条消息写日志时会把
            # 整个程序写崩(2026-10-05 实测)。入口处先洗一遍,坏字符换 ?
            return Message("user", clean_bad_unicode(content))


def print_message(msg: Message) -> None:
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


def replay(manager: LogManager) -> None:
    """把会话历史打印到终端(恢复旧会话 / /resume 切换时用)。

    读完整历史(master_messages)而不是压缩后的视图:回放是给人看的,
    用户应当看到从头到尾的真实对话;模型上下文才用视图,两回事。
    复用 print_message 的显示规则,现场聊天时看到什么样,回放就什么样
    (系统提示词这类 hide/quiet 消息照旧不显示)。
    """
    for msg in manager.master_messages:
        print_message(msg)
