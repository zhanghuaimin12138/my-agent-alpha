"""本项目的默认适配器:把压缩核心接到 my-agent-alpha 的真实设施上。

压缩核心(contracts.py 里那六个接口)不知道"模型客户端长什么样"、
"工具调用怎么解析" —— 这个文件负责接线:
- 数 token:按字符数估(英文约 4 字符/token,中文约 1.5 字符/token),
  不引入分词库依赖;确定性的,同一输入永远同一结果(契约要求);
- 模型信息:从全局上下文里取当前模型,查它的窗口大小;
- 工具调用探测:直接用工具系统现成的解析器(markdown 围栏 + 原生
  调用行两种通道都认);
- LLM 客户端:包装 llm.reply 做一次非流式调用。
"""

from __future__ import annotations

from my_agent_alpha.autocompact.contracts import ModelInfo
from my_agent_alpha.context import get_current_model
from my_agent_alpha.llm.models import resolve_model
from my_agent_alpha.message import Message


def estimate_tokens(text: str) -> int:
    """粗略估算一段文字的 token 数(和 compaction.estimate_tokens 同款)。

    经验值:英文约 4 个字符一个 token,中文约 1.5 个字符一个 token。
    不同模型的分词有差异,但对"要不要压缩"的决策足够准;
    真实用量以接口返回的 usage 为准(日志里每条回复都记着)。

    (这里放一份副本而不是 import:老 compaction 模块将来退役删除时,
    新压缩包不受牵连。)
    """
    ascii_len = sum(1 for c in text if ord(c) < 128)
    other_len = len(text) - ascii_len
    return (ascii_len + 3) // 4 + (other_len * 2 + 2) // 3


class EstimateTokenCounter:
    """TokenCounter 的默认实现:字符启发式估算,零依赖、完全确定。"""

    def len_tokens(self, obj: str | Message | list[Message]) -> int:
        """数 token:字符串按全文、消息按 content、列表逐条累加。

        示例:len_tokens("abcd") → 1;len_tokens(整个日志) → 各条之和
        """
        if isinstance(obj, str):
            return estimate_tokens(obj)
        if isinstance(obj, Message):
            return estimate_tokens(obj.content)
        return sum(estimate_tokens(m.content) for m in obj)


class CurrentModelInfoProvider:
    """ModelInfoProvider 的默认实现:取当前会话正在用的模型。"""

    def get_default_model(self) -> ModelInfo | None:
        """从全局上下文取当前模型名;没设置过(比如测试环境)返回 None。"""
        name = get_current_model()
        if not name:
            return None
        meta = resolve_model(name)
        return ModelInfo(name=meta.full, context=meta.context)


class ParserToolUseDetector:
    """ToolUseDetector 的默认实现:用工具系统现成的解析器。"""

    def contains_tool_use(self, msg: Message) -> bool:
        """这条消息里能不能解析出至少一个工具调用(两种通道都查)。

        只有 assistant 消息可能是工具调用;其他角色直接 False,
        省得白跑解析。
        """
        if msg.role != "assistant":
            return False
        from my_agent_alpha.tools.base import parse_tooluses  # 延迟导入避免循环依赖

        return bool(parse_tooluses(msg.content))


class ReplyLLMClient:
    """LLMClient 的默认实现:包装 llm.reply 做一次非流式、无工具的调用。"""

    def reply(self, messages: list[Message], model: str) -> str:
        """把消息发给模型,等它写完,返回回复全文(生成摘要用)。"""
        from my_agent_alpha.llm import reply as llm_reply  # 延迟导入避免循环依赖

        return llm_reply(messages, model).content
