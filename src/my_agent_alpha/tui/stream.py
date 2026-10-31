"""流式打印:模型一边生成,终端一边出字。

这里只有一件事:把模型流式吐出来的小段文字实时画到终端,
其中 <think>...</think> 里的推理过程用暗色显示,和正文区分开。
"""

from __future__ import annotations

_DIM = "\x1b[2m"  # 终端"暗色"控制码(让文字看起来更淡)
_RESET = "\x1b[0m"  # 恢复正常颜色
_TAGS = ("<think>", "</think>")


class StreamPrinter:
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
