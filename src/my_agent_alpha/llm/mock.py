"""mock provider:离线、确定性、零成本(仿 gptme 的 llm_mock)。

- mock/echo:回显最后一条用户消息;
- mock/static:固定回复;
- 任何 mock/*:若设置了脚本队列(set_scripted_replies),按序弹出 ——
  这是测试 agent 循环的主力:脚本里可以写工具调用围栏,驱动完整多步循环。
"""

from __future__ import annotations

from collections.abc import Sequence

from my_agent_alpha.message import Message

STATIC_RESPONSE = "这是 mock/static 的固定回复。"

_scripted: list[str] = []


def set_scripted_replies(replies: Sequence[str]) -> None:
    """设定脚本回复序列(测试用);每次调用覆盖旧脚本。"""
    _scripted.clear()
    _scripted.extend(replies)


def clear_scripted() -> None:
    _scripted.clear()


def chat(messages: Sequence[Message], model_name: str) -> tuple[str, dict[str, int] | None]:
    """mock 的 chat 实现:返回 (文本, usage)。"""
    if _scripted:
        return _scripted.pop(0), {"input": 0, "output": 0}
    if model_name == "static":
        return STATIC_RESPONSE, {"input": 0, "output": 0}
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "(无用户消息)")
    return f"Echo: {last_user}", {"input": 0, "output": 0}
