"""上下文压缩的"接口约定":数据类型 + 六个适配器接口。

设计动机(仿 gptme,见 docs/autocompact-dev.md §3/§4):压缩的核心算法
(决策、三阶段引擎、LLM 摘要)不直接依赖本项目任何具体设施 —— 它只认
这里定义的接口。算法怎么数 token、怎么知道模型窗口多大、怎么发现工具调用,
全部由外部通过适配器注入;换一套设施,算法一行不改。

六个接口(每个都配了一个"最小可用实现",真实实现见 adapters.py):
- TokenCounter        数 token(拿不到分词库时按字符数估)
- ModelInfoProvider   当前用的模型叫什么、上下文窗口多大
- ToolUseDetector     一条消息里有没有工具调用
- OutputStore         被截断的全量内容写到哪个文件
- LLMClient           非流式问一次模型(生成摘要用)
- ConversationStore   压缩结果的落地(视图分支:mater 日志永不压缩)
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from my_agent_alpha.message import Message

# 巨型工具结果的判定门槛(token 数):超过它的工具输出才算"巨型",
# 决策层和引擎阶段 2 共用这一个数
HUGE_TOOL_RESULT_TOKENS = 2000


@dataclass(frozen=True)
class ModelInfo:
    """一个模型的两个关键属性:名字和上下文窗口大小(token 数)。

    名字给 token 计数器挑分词器用(本项目按字符估,用不上);
    context 是压缩所有阈值比例的基数(0.9 * context = 触发线,等等)。
    """

    name: str
    context: int


# 拿不到默认模型信息时的兜底(引擎需要一个窗口大小才能算阈值)。
# context 取本项目主力模型的窗口:glm-5.3 = 128k
FALLBACK_MODEL_INFO = ModelInfo(name="glm-5.3", context=128_000)


class TokenCounter(Protocol):
    """数 token 的统一入口:字符串、单条消息、整个消息列表都要能数。

    一致性要求:同一输入必须永远返回同一个值 —— 决策层和引擎层
    各自独立调用,两边数出来的数若对不上,决策就会误判。
    """

    def len_tokens(self, obj: str | Message | list[Message]) -> int: ...


class ModelInfoProvider(Protocol):
    """回答"当前用的模型是什么"。返回 None 时核心用兜底常量。"""

    def get_default_model(self) -> ModelInfo | None: ...


class ToolUseDetector(Protocol):
    """判断一条消息里有没有工具调用。

    返回 True 的消息压缩时会跳过:截了它会破坏"调用和结果"的配对,
    模型接着看历史时会困惑(发了调用却找不到对应结果)。
    """

    def contains_tool_use(self, msg: Message) -> bool: ...


class OutputStore(Protocol):
    """把被截断消息的全量内容写进文件,返回文件路径(写进截断占位文本里)。

    目的:内容从上下文里拿掉了,但没真的丢 —— 模型需要时可以自己去读。
    """

    def save(self, content: str, meta: dict[str, object]) -> Path: ...


class LLMClient(Protocol):
    """非流式问一次模型、直接拿完整回复文本(生成会话摘要用)。"""

    def reply(self, messages: list[Message], model: str) -> str: ...


class ConversationStore(Protocol):
    """压缩结果的落地方式:视图分支 + 双写。

    语义(gptme 同款):master 日志(conversation.jsonl)是对话的
    完整历史,永远不被压缩改写;压缩结果存成"视图"(另一份消息列表),
    切到视图后,新消息同时追加进 master 和视图(双写)。
    读取时:当前在视图上就返回视图内容,否则返回 master。
    """

    @property
    def messages(self) -> Sequence[Message]: ...

    def create_view(self, name: str, messages: list[Message]) -> None:
        """注册一个视图(记进内存 + 写 views/{name}.jsonl 文件)。"""
        ...

    def switch_view(self, name: str) -> None:
        """切换到某个视图;之后的读取都以视图为准。"""
        ...

    def switch_to_master(self) -> None:
        """切回 master(完整历史),随时可以切。"""
        ...

    def append(self, msg: Message) -> None:
        """追加一条消息:在视图上 → master 和视图都追加;否则只写当前日志。"""
        ...

    def persist(self) -> None:
        """把内存里的消息写盘(双写规则同 append)。"""
        ...

    def next_view_name(self) -> str:
        """下一个视图名:扫描已有 "compacted-N" 取最大编号 +1,三位补零。"""
        ...


@dataclass
class CompressionConfig:
    """一次压缩的所有可调参数(引擎和 Provider 层共用)。

    limit 为 None 时引擎自己按模型窗口算(0.8 × 窗口);
    store_dir 给了才会做"截断内容落盘"和"master 字节索引"。
    """

    limit: int | None = None
    max_tool_result_tokens: int = HUGE_TOOL_RESULT_TOKENS
    store_dir: Path | None = None
    reasoning_strip_age_threshold: int = 5
    keep_head: int = 0


@dataclass
class CompactionResult:
    """一次压缩的产出和元信息。

    source_digest 是压缩前源消息的"指纹"(SHA-256):压缩完之后
    想知道"源消息后来又变没变",重新算一遍指纹对比即可。
    covered_through 记录压缩覆盖到源日志第几条(从 0 数起)。
    """

    messages: list[Message]
    source_digest: str
    covered_through: int = -1
    limitations: list[str] = field(default_factory=list)


def source_digest(messages: Sequence[Message]) -> str:
    """给一批消息算一个稳定的"指纹":内容不变指纹就不变,改一个字就变。

    算法:把每条消息的 role 和 content 用 \\0 分隔依次喂给 SHA-256,
    取最终的十六进制串。为什么用 \\0 分隔:它在正常文本里不会出现,
    "role=ab,content=c" 和 "role=a,content=bc" 才不会撞出同一个指纹。

    示例:source_digest(msgs) == source_digest(msgs)        # 稳定
         source_digest(改过一条内容的 msgs) != 原指纹      # 灵敏
    """
    h = hashlib.sha256()
    for m in messages:
        h.update(m.role.encode("utf-8"))
        h.update(b"\x00")
        h.update(m.content.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def is_tool_result(msg: Message) -> bool:
    """判断一条消息是不是"工具输出"(压缩阶段 2 里可以截断的那类消息)。

    本项目的约定和 gptme 一样:工具输出就是 role="system" 的消息,
    没有单独的"tool"角色。多排除一层 hide=True:系统提示词、记忆索引
    这类"发给模型的说明书"虽然也是 system,但绝不是工具输出,
    不能被当成"巨型工具结果"截断。(整个压缩包里判断这一点的函数
    只此一个,要改规则只改这里。)
    """
    return msg.role == "system" and not msg.hide
