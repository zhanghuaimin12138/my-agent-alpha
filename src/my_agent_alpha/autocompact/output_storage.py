"""被截断内容的"全量落盘":从上下文里拿掉 ≠ 删除。

阶段 2 把巨型工具输出截成一小段占位文本时,完整内容先写进一个文件,
再把文件路径写进占位文本 —— 模型需要原文时可以自己去读或 grep,
信息一点没丢。这和 master 字节引用是两条互补的找回通路:
文件适合 grep / 人直接看,字节引用适合精确取回某一条。
"""

from __future__ import annotations

import re
from pathlib import Path

from my_agent_alpha.autocompact.contracts import OutputStore
from my_agent_alpha.message import Message

# 截断占位文本的开头标记(也是"这条已被截过"的记号,重复压缩时靠它跳过)
TRUNCATED_PREFIX = "[Large tool output removed"

# 命令痕迹行的开头:内容前 10 行里如果有这样的行,把它附在占位文本里,
# 模型不用读文件就知道"当年执行的是什么命令"
_COMMAND_LINE_RE = re.compile(r"^(Ran command:|Executed:|Command:)")


class FileOutputStore:
    """OutputStore 的文件实现:写到 会话目录/outputs/autocompact-NNN.txt。

    文件名里的序号按"目录里已有文件的最大序号 + 1"递增,
    同一个会话里每截一条就多一个文件,顺序即截断顺序。
    """

    def __init__(self, base_dir: Path) -> None:
        """base_dir 是会话目录;文件会写到它下面的 outputs/ 子目录。"""
        self.base_dir = base_dir

    def save(self, content: str, meta: dict[str, object]) -> Path:
        """把全量内容写进新文件,返回文件路径(会写进截断占位文本)。

        meta 目前只做记录用途(比如带上了原消息的 token 数),
        文件内容本身就是全量原文。
        """
        out_dir = self.base_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        next_no = 1 + max(
            (int(p.stem.rsplit("-", 1)[1]) for p in out_dir.glob("autocompact-*.txt")),
            default=0,
        )
        path = out_dir / f"autocompact-{next_no:03d}.txt"
        path.write_text(content, encoding="utf-8")
        return path


def create_tool_result_summary(msg: Message, tokens: int, store: OutputStore | None) -> str:
    """生成替换被截工具输出的占位文本(格式是固定契约,别随意改)。

    两种形态:
    - 有落盘目录:注明全量输出存在哪个文件,提示可以读/grep;
    - 没有:只说明内容因超长被移除(纯函数场景,连目录都没有)。

    如果原内容前 10 行里有 "Ran command:" / "Executed:" / "Command:"
    开头的行,把它附在方括号后面 —— 模型立刻知道这段输出是干什么的。

    示例(有落盘):
        [Large tool output removed - 3002 tokens] (Ran command: ls -la).
        Full output saved to: /path/outputs/autocompact-001.txt
        You can read or grep this file if needed.
    """
    prefix = ""
    for line in msg.content.split("\n")[:10]:
        if _COMMAND_LINE_RE.match(line.strip()):
            prefix = f" ({line.strip()})"
            break
    if store is not None:
        path = store.save(msg.content, {"tokens": tokens})
        return (
            f"{TRUNCATED_PREFIX} - {tokens} tokens]{prefix}. "
            f"Full output saved to: {path}\n"
            "You can read or grep this file if needed."
        )
    return (
        f"{TRUNCATED_PREFIX} - {tokens} tokens]{prefix}. "
        "Output was automatically removed due to size to allow conversation continuation."
    )
