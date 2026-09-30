"""模型名解析:把 "deepseek/deepseek-chat" 这样的名字拆成"哪家服务商 + 什么模型"。

模型名的写法约定(仿 gptme):斜杠前是服务商(provider),斜杠后是模型名,
例如 "deepseek/deepseek-chat"、"mock/echo"。只写服务商名就用它的默认模型。

阶段 1 只需要这些;上下文窗口大小、价格等模型属性以后再补。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCfg:
    """一家服务商的连接配置。"""

    name: str
    base_url: str | None  # 接口地址;None = 用 SDK 自带的官方地址(openai)
    api_key_env: str | None  # 密钥放在哪个环境变量里;None = 不需要密钥(本机服务)
    default_model: str | None  # 只写服务商名不写模型时,默认用哪个模型


# 内置支持的服务商表(都能用 openai 这一个 SDK 访问)
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
        "qwen3.8-flash",
    ),
    # 智谱 GLM:标准按量计费端点 + Coding Plan 订阅端点,共用一把密钥
    "glm": ProviderCfg("glm", "https://open.bigmodel.cn/api/paas/v4", "GLM_API_KEY", "glm-5.3"),
    "glm-coding": ProviderCfg(
        "glm-coding", "https://open.bigmodel.cn/api/coding/paas/v4", "GLM_API_KEY", "glm-5.3"
    ),
    "openrouter": ProviderCfg(
        "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", None
    ),
    "local": ProviderCfg("local", "http://localhost:11434/v1", None, None),  # Ollama 等本机服务
}

DEFAULT_PROVIDER = "openai"


@dataclass(frozen=True)
class ModelMeta:
    """解析结果:服务商名 + 模型名。"""

    provider: str
    model: str
    context: int = 128_000  # 模型上下文窗口大小(占位,阶段 6 做长度控制时真正用上)

    @property
    def full(self) -> str:
        """完整的"服务商/模型"写法,如 "deepseek/deepseek-chat"。"""
        return f"{self.provider}/{self.model}"


def resolve_model(model: str) -> ModelMeta:
    """解析模型名,返回(服务商, 模型)。

    规则,按顺序:
    - 空字符串 → 用 mock/echo(防止不小心打到要花钱的真实接口);
    - 带 "/",如 "deepseek/chat" → 按写的拆;
    - 恰好是个服务商名,如 "deepseek" → 该服务商的默认模型;
      没有默认模型的服务商(如 openrouter)会明确报错,而不是瞎编一个名字;
    - 其他裸名,如 "gpt-4o-mini" → 当作 openai 的模型。

    示例:
        resolve_model("deepseek/deepseek-chat") → provider="deepseek"
        resolve_model("deepseek")               → provider="deepseek", model="deepseek-chat"
        resolve_model("")                       → provider="mock"
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
        # 不认识的服务商:当作自定义网关保留原名(配置方式见 llm._custom_cfg)
        return ModelMeta(provider=provider_name, model=model_name)
    cfg = PROVIDERS[provider_name]
    if not model_name:
        if cfg.default_model is None:
            raise ValueError(
                f"provider {provider_name!r} 没有默认模型,请显式指定,如 {provider_name}/<model>"
            )
        model_name = cfg.default_model
    return ModelMeta(provider=provider_name, model=model_name)
