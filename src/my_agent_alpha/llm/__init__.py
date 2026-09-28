"""LLM 门面层:外界唯一入口是 reply()(仿 gptme)。

阶段 1:非流式、markdown 工具格式(工具文档在系统提示词里,回复是纯文本)。
阶段 2 加流式契约 stream() -> Generator[str, None, metadata] 与 _StreamWithMetadata;
阶段 3 加原生工具调用(call_id 配对)。
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from my_agent_alpha.llm.mock import chat as mock_chat
from my_agent_alpha.llm.models import PROVIDERS, ModelMeta, ProviderCfg, resolve_model
from my_agent_alpha.llm.openai_compat import chat as openai_chat
from my_agent_alpha.message import Message

__all__ = ["Message", "ModelMeta", "reply", "resolve_model"]


def _custom_cfg(provider: str) -> ProviderCfg:
    """未知 provider:期望 ALPHA_<NAME>_BASE_URL 指向 OpenAI 兼容端点。"""
    base_url = os.environ.get(f"ALPHA_{provider.upper()}_BASE_URL")
    if not base_url:
        raise RuntimeError(
            f"未知 provider {provider!r}:请用已知 provider(如 deepseek/deepseek-chat),"
            f"或设置 ALPHA_{provider.upper()}_BASE_URL 指向 OpenAI 兼容端点"
        )
    return ProviderCfg(provider, base_url, f"ALPHA_{provider.upper()}_API_KEY", None)


def reply(messages: Sequence[Message], model: str) -> Message:
    """向 LLM 发起一次请求,返回完整的 assistant 消息(含 metadata)。"""
    meta = resolve_model(model)
    if meta.provider == "mock":
        text, usage = mock_chat(messages, meta.model)
    else:
        cfg = PROVIDERS.get(meta.provider) or _custom_cfg(meta.provider)
        text, usage = openai_chat(messages, meta, cfg)
    metadata: dict[str, object] = {"model": meta.full}
    if usage:
        metadata["usage"] = usage
    return Message("assistant", text, metadata=metadata)
