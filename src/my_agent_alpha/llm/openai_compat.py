"""OpenAI 兼容客户端(所有真实 provider 共用的一条路,仿 gptme 的 llm_openai 复用哲学)。

延迟导入 openai SDK:mock 路径(测试)完全不加载它。
"""

from __future__ import annotations

from collections.abc import Sequence

from my_agent_alpha.llm.models import ModelMeta, ProviderCfg
from my_agent_alpha.message import Message


def msgs2dicts(messages: Sequence[Message]) -> list[dict[str, str]]:
    """Message 序列 → chat.completions 的 messages 参数。

    三 role 直通(system 消息出现在中间是正常的:工具结果就是 system role,
    OpenAI/DeepSeek 等兼容端点均接受)。空内容消息被跳过 ——
    部分端点对空 content 直接 400(审查 #10),而空消息本身不携带信息,
    完整历史仍在日志里。原生 tool_calls 映射属阶段 3。
    """
    return [{"role": m.role, "content": m.content} for m in messages if m.content.strip()]


def chat(
    messages: Sequence[Message], meta: ModelMeta, cfg: ProviderCfg
) -> tuple[str, dict[str, int] | None]:
    """非流式调用;返回 (文本, usage)。"""
    from openai import OpenAI  # 延迟导入

    api_key: str | None = None
    if cfg.api_key_env:
        import os

        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"provider {cfg.name!r} 需要环境变量 {cfg.api_key_env},"
                f"例如:export {cfg.api_key_env}=sk-..."
            )
    else:
        api_key = "local"  # 本机服务(如 Ollama)不校验密钥,占位即可

    client = OpenAI(api_key=api_key, base_url=cfg.base_url)
    resp = client.chat.completions.create(
        model=meta.model,
        messages=msgs2dicts(messages),  # type: ignore[arg-type]
        timeout=120,
    )
    choice = resp.choices[0]
    text = choice.message.content or ""
    usage: dict[str, int] | None = None
    if resp.usage is not None:
        usage = {
            "input": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "output": getattr(resp.usage, "completion_tokens", 0) or 0,
        }
    return text, usage
