"""模型与 provider 解析(最小版)。

模型名约定仿 gptme:"provider/model",如 "deepseek/deepseek-chat"、
"mock/echo"。裸名(不含 /)按 openai 处理。
阶段 1 只需 provider 路由 + 默认模型;上下文窗口/价格/能力标志在阶段 2-3 扩充。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCfg:
    name: str
    base_url: str | None  # None = 各 SDK 默认(openai 官方)
    api_key_env: str | None  # None = 无需密钥(本机服务)
    default_model: str | None  # 只写 provider 不写模型时的默认模型


PROVIDERS: dict[str, ProviderCfg] = {
    "mock": ProviderCfg("mock", None, None, "echo"),
    "openai": ProviderCfg("openai", None, "OPENAI_API_KEY", "gpt-4o-mini"),
    "deepseek": ProviderCfg(
        "deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-chat"
    ),
    "moonshot": ProviderCfg(
        "moonshot", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", "kimi-k2-turbo-preview"
    ),
    "dashscope": ProviderCfg(
        "dashscope",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
        "qwen-plus",
    ),
    "openrouter": ProviderCfg(
        "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", None
    ),
    "local": ProviderCfg("local", "http://localhost:11434/v1", None, None),  # Ollama 等本机服务
}

DEFAULT_PROVIDER = "openai"


@dataclass(frozen=True)
class ModelMeta:
    provider: str
    model: str
    context: int = 128_000  # 阶段 1 未做 token 计数,仅占位

    @property
    def full(self) -> str:
        return f"{self.provider}/{self.model}"


def resolve_model(model: str) -> ModelMeta:
    """解析模型名。

    - "provider/model":显式指定;
    - 裸名是已知 provider(如 "deepseek"):用该 provider 的默认模型;
    - 空名:回退离线 mock(防止误打到真实计费端点);
    - 其他裸名:按 DEFAULT_PROVIDER 处理,名字原样当模型名。
    对"未知模型名"宽容(原样透传),但对"没给名字"明确报错 ——
    编造一个远端不存在的模型名只会换来更迷惑的 404。
    """
    if not model.strip():
        return ModelMeta(provider="mock", model="echo")
    if "/" in model:
        provider_name, _, model_name = model.partition("/")
    elif model in PROVIDERS:
        provider_name, model_name = model, ""
    else:
        provider_name, model_name = DEFAULT_PROVIDER, model
    if provider_name not in PROVIDERS:
        # 未知 provider:当作 openai 兼容的自定义网关,名字原样保留
        return ModelMeta(provider=provider_name, model=model_name)
    cfg = PROVIDERS[provider_name]
    if not model_name:
        if cfg.default_model is None:
            raise ValueError(
                f"provider {provider_name!r} 没有默认模型,请显式指定,如 {provider_name}/<model>"
            )
        model_name = cfg.default_model
    return ModelMeta(provider=provider_name, model=model_name)
