"""假模型(mock):不联网、不花钱、每次回复一模一样,专门给测试用。

为什么要它?测 agent 循环总不能真调一次模型吧 —— 又慢又花钱,
而且真模型同样的问题每次回答都不同,根本没法写断言。
mock 假扮成一家模型服务商(注册在服务商表里,名字以 mock/ 开头),
agent 代码完全不知道自己在跟"假模型"说话。

三种模式:
- mock/echo:把用户最后一句原样返回(加个 Echo: 前缀),手工试验用;
- mock/static:永远回一句固定的话;
- 脚本模式(测试主力):提前用 set_scripted_replies() 塞进去一串回复,
  每次调用按顺序弹出一条。脚本里可以写工具调用代码块,
  这样就能"导演"出完整的多步工具循环,断言每一步都真的发生了。
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator, Sequence

from my_agent_alpha.message import Message

STATIC_RESPONSE = "这是 mock/static 的固定回复。"

_scripted: list[str | BaseException] = []

# 环境变量脚本(ALPHA_MOCK_REPLIES,JSON 数组)的缓存:
# None = 还没从环境加载过;list = 已加载(每弹一条少一条)。
# 为什么需要它:set_scripted_replies 是进程内状态,派出去的子代理
# 子进程看不到;测试想导演子代理的行为,只能把脚本放环境变量里传过去。
_env_scripted: list[str] | None = None

# 流式分块大小:故意很小(3 个字符),让 "<think>" 这类标签经常被
# 从中间切开 —— 逼着渲染和断点检测代码真的处理"标签跨块"的情况
_CHUNK_SIZE = 3


def set_scripted_replies(replies: Sequence[str | BaseException] | str | BaseException) -> None:
    """设置脚本回复序列(每次调用覆盖之前的);放异常实例则到它时直接抛出。

    示例(在测试里导演一次"调工具再收尾"的对话):
        set_scripted_replies([
            "我来执行:\\n```shell\\necho hi\\n```",   # 第一轮:模型"决定"调工具
            "输出是 hi,完成。",                        # 第二轮:收尾
        ])

    防呆:直接传一个字符串时按"只有一条回复"处理 ——
    不然 extend(字符串) 会把它拆成单个字符当一堆回复,极难排查。
    """
    _scripted.clear()
    if isinstance(replies, str | BaseException):
        _scripted.append(replies)
    else:
        _scripted.extend(replies)


def clear_scripted() -> None:
    """清空脚本(测试之间的清理,防止上个测试的残留影响下个)。"""
    global _env_scripted
    _scripted.clear()
    _env_scripted = None  # 环境变量脚本一并作废(下次用时重新读环境)


def _next_reply(messages: Sequence[Message], model_name: str) -> str:
    """取出下一条回复:进程内脚本 > 环境变量脚本 > echo/static。"""
    if _scripted:
        entry = _scripted.pop(0)
        if isinstance(entry, BaseException):
            raise entry  # 脚本要求"这次调用出错"(测错误处理路径用)
        return entry
    global _env_scripted
    if _env_scripted is None:
        raw = os.environ.get("ALPHA_MOCK_REPLIES")
        try:
            data = json.loads(raw) if raw else []
            _env_scripted = (
                [s for s in data if isinstance(s, str)] if isinstance(data, list) else []
            )
        except json.JSONDecodeError:
            _env_scripted = []  # 坏 JSON:当没有脚本,走 echo(不炸测试)
    if _env_scripted:
        return _env_scripted.pop(0)
    if model_name == "static":
        return STATIC_RESPONSE
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "(无用户消息)")
    return f"Echo: {last_user}"


def chat(messages: Sequence[Message], model_name: str) -> tuple[str, dict[str, int] | None]:
    """非流式版:一次性返回 (回复文本, 假的用量信息)。"""
    return _next_reply(messages, model_name), {"input": 0, "output": 0}


def stream(
    messages: Sequence[Message], model_name: str
) -> Generator[str, None, dict[str, int] | None]:
    """流式版:把回复切成 3 字符的小段依次吐出,结束时带回用量信息。

    和真模型的流式行为一致,只是文字来自脚本/echo。
    """
    text = _next_reply(messages, model_name)
    for i in range(0, len(text), _CHUNK_SIZE):
        yield text[i : i + _CHUNK_SIZE]
    return {"input": 0, "output": 0}
