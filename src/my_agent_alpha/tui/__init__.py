"""终端交互层:流式打印(stream)与输入/回放(io)。

主链路循环不直接碰 print/input,统一走这个包 ——
将来若要换显示方式(比如接 Web 前端),只改这里。
"""

from __future__ import annotations

from my_agent_alpha.tui.io import print_message, read_user_input, replay
from my_agent_alpha.tui.stream import StreamPrinter

__all__ = ["StreamPrinter", "print_message", "read_user_input", "replay"]
