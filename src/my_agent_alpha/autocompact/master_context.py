"""主日志的字节索引:给被压缩的内容留一条"精确找回"的路。

背景:conversation.jsonl 是一行一条消息、永不改写的完整历史。
压缩只发生在"视图"(发给模型的那份拷贝)上,而且是原位替换 ——
第 i 条消息压完还是第 i 条。所以"视图里的第 i 条"永远对应
"主日志文件的第 i 行",也就对应文件里一段固定的字节区间。

把这个区间写进压缩后的消息文本里,模型(或人)需要原文时,
按区间 seek + read 就能一字不差地取回来 —— 比"翻日志慢慢找"精确得多。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from my_agent_alpha.logmanager.manager import CONVERSATION_FILENAME

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ByteRange:
    """一条消息在主日志文件里占的字节区间(含头不含尾)。

    message_idx 是消息下标(第几条消息,从 0 数);
    byte_start/byte_end 是该行在文件里的字节位置(不含行尾换行符),
    read(byte_end - byte_start) 正好读到这一行的全部内容。
    """

    message_idx: int
    byte_start: int
    byte_end: int


def build_byte_index(master_file: Path) -> list[ByteRange]:
    """扫描主日志文件,记下每条消息占据的字节区间。

    空行不算消息(和 LogManager 读文件时的处理一致,这样下标才对得齐)。
    文件不存在返回空表 —— 调用方拿空表就不附加字节引用,压缩照常做。

    示例:conversation.jsonl 有 3 行 → 返回 3 个 ByteRange,
    第 2 个的 message_idx 是 1。
    """
    try:
        data = master_file.read_bytes()
    except OSError:
        logger.warning("读不了主日志 %s,压缩时将不附字节引用", master_file)
        return []
    index: list[ByteRange] = []
    start = 0
    while start < len(data):
        nl = data.find(b"\n", start)
        end = len(data) if nl == -1 else nl
        if end > start:  # 空行跳过
            index.append(ByteRange(message_idx=len(index), byte_start=start, byte_end=end))
        start = end + 1
    return index


def read_master_range(master_file: Path, byte_start: int, byte_end: int) -> str:
    """按字节区间从主日志里取回一条消息的原文(内容字段)。

    就是"恢复"操作本身:seek 到起点、读固定长度、按 JSON 解析出 content。
    """
    with open(master_file, "rb") as f:
        f.seek(byte_start)
        raw = f.read(byte_end - byte_start)
    return str(json.loads(raw.decode("utf-8"))["content"])


def master_reference(master_file: Path, span: ByteRange, tokens: int, content: str) -> str:
    """生成追加在被截消息尾部的"找回原文"提示文本(格式是固定契约)。

    示例输出:
        [Content truncated - 3002 tokens]
        Master context: /path/conversation.jsonl (bytes 512-12488)
        Preview: $ ls -la…
        To recover: grep or read the master context file at the byte range above.
    """
    first_line = content.split("\n", 1)[0]
    preview = first_line[:200] + ("..." if len(first_line) > 200 else "")
    return (
        f"[Content truncated - {tokens} tokens]\n"
        f"Master context: {master_file} (bytes {span.byte_start}-{span.byte_end})\n"
        f"Preview: {preview}\n"
        "To recover: grep or read the master context file at the byte range above."
    )


@dataclass(frozen=True)
class MasterContext:
    """主日志文件 + 它的字节索引,打包在一起方便传给引擎。

    用法:master = MasterContext.build(会话目录);
    master.reference_for(3, 3002, 内容) → 第 3 条消息的找回提示
    (没有对应索引时返回 None,调用方就不附加)。
    """

    path: Path
    index: list[ByteRange]

    @classmethod
    def build(cls, store_dir: Path) -> MasterContext:
        """从会话目录构建:索引主日志文件;文件不在就是空索引。"""
        master_file = store_dir / CONVERSATION_FILENAME
        return cls(path=master_file, index=build_byte_index(master_file))

    def reference_for(self, message_idx: int, tokens: int, content: str) -> str | None:
        """给出第 message_idx 条消息的找回提示;下标越界返回 None。

        越界会发生吗?会:视图比主日志短、或主日志不存在时。
        这时宁可不给引用,也不能给一个指向错误内容的引用。
        """
        if message_idx >= len(self.index):
            return None
        return master_reference(self.path, self.index[message_idx], tokens, content)
