"""Provider 层:把"压缩算法"做成可替换的组件。

为什么要这一层:内置的"规则压缩"只是众多压缩策略之一 —— 将来想换
成别的实现(比如按会话类型选策略、接第三方压缩服务),不应该去改
核心代码。这里定义统一的组件接口(ContextProvider)+ 一个注册表:
按名字注册、按名字取用;内置实现注册在 "default" 名下,后注册的
同名实现可以直接整体替换它。

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from my_agent_alpha.autocompact.contracts import (
    FALLBACK_MODEL_INFO,
    CompactionResult,
    CompressionConfig,
    ModelInfoProvider,
    TokenCounter,
    ToolUseDetector,
    source_digest,
)
from my_agent_alpha.message import Message

logger = logging.getLogger(__name__)


class ContextProvider(ABC):
    """一种压缩策略的统一抽象接口:该不该压 + 怎么压。"""

    #: 注册与查找用的名字(子类必须给出)
    name: str

    @abstractmethod
    def should_compress(self, messages: Sequence[Message], config: CompressionConfig) -> bool:
        """判断这批消息现在需不需要压缩(给手动/外部触发的场合用)。"""
        ...

    @abstractmethod
    def compress(self, messages: Sequence[Message], config: CompressionConfig) -> CompactionResult:
        """执行压缩,返回压缩后的消息流 + 元信息(源指纹、损失说明等)。"""
        ...


class DefaultContextProvider(ContextProvider):
    """内置策略:三阶段规则引擎(剥思考 → 截工具结果 → 压长回复)+ 兜底。

    三个适配器(计数器/模型信息/工具调用探测)都可以从构造函数注入,
    不注入就用本项目的默认实现(见 adapters.py)—— 方便测试,
    也方便在特殊环境里整组替换。
    """

    name = "default"

    def __init__(
        self,
        *,
        counter: TokenCounter | None = None,
        models: ModelInfoProvider | None = None,
        tool_use: ToolUseDetector | None = None,
    ) -> None:
        """记下注入的适配器;None 表示用到时再取默认实现。"""
        self._counter_injected = counter
        self._models_injected = models
        self._tool_use_injected = tool_use

    def _deps(self) -> tuple[TokenCounter, ModelInfoProvider, ToolUseDetector]:
        """取实际使用的三个适配器(没注入就用默认实现)。"""
        from my_agent_alpha.autocompact.adapters import (  # 延迟导入:不用压缩时不拖依赖链
            CurrentModelInfoProvider,
            EstimateTokenCounter,
            ParserToolUseDetector,
        )

        return (
            self._counter_injected or EstimateTokenCounter(),
            self._models_injected or CurrentModelInfoProvider(),
            self._tool_use_injected or ParserToolUseDetector(),
        )

    def should_compress(self, messages: Sequence[Message], config: CompressionConfig) -> bool:
        """内置策略的判定:总量超过工作上限(0.8 × 窗口)就要压。"""
        counter, models, _ = self._deps()
        window = self._context_window(models)
        threshold = config.limit if config.limit is not None else int(0.8 * window)
        return counter.len_tokens(list(messages)) > threshold

    def compress(self, messages: Sequence[Message], config: CompressionConfig) -> CompactionResult:
        """压缩:算源指纹 → 跑三阶段引擎 → 打包成 CompactionResult。

        covered_through = len-1:内置引擎是"全量压缩",
        输入的每一条都被覆盖到(输出与输入等长,一一对应)。
        """
        counter, models, tool_use = self._deps()
        window = self._context_window(models)
        digest = source_digest(messages)
        # 引擎在这里延迟导入:注册表初始化不因此拖起整条依赖链
        from my_agent_alpha.autocompact.engine import auto_compact_log
        from my_agent_alpha.autocompact.master_context import MasterContext
        from my_agent_alpha.autocompact.output_storage import FileOutputStore

        master = MasterContext.build(config.store_dir) if config.store_dir else None
        store = FileOutputStore(config.store_dir) if config.store_dir else None
        compacted = auto_compact_log(
            list(messages),
            context_window=window,
            counter=counter,
            contains_tool_use=tool_use.contains_tool_use,
            limit=config.limit,
            max_tool_result_tokens=config.max_tool_result_tokens,
            reasoning_strip_age=config.reasoning_strip_age_threshold,
            store=store,
            keep_head=config.keep_head,
            master=master,
        )
        return CompactionResult(
            messages=compacted,
            source_digest=digest,
            covered_through=len(messages) - 1,
            limitations=["rule-based truncation of tool results"],
        )

    @staticmethod
    def _context_window(models: ModelInfoProvider) -> int:
        """拿当前模型的窗口大小;拿不到用兜底常量。"""
        info = models.get_default_model()
        return info.context if info is not None else FALLBACK_MODEL_INFO.context


# ── 注册表 ──────────────────────────────────────────────
# 名字 → 类(注意存的是类,取的时候才实例化)
_providers: dict[str, type[ContextProvider]] = {}
# 内置的 "default" 是否已经注册过(只 setdefault 一次,之后不再碰)
_builtin_seeded = False


def register_provider(name: str, cls: type[ContextProvider]) -> None:
    """注册(或替换)一个压缩策略:同名后写覆盖先写。

    示例:register_provider("default", MyProvider) → 整体替换内置算法
    """
    _seed_builtin()
    _providers[name] = cls


def get_context_provider(name: str = "default") -> ContextProvider:
    """按名字取一个压缩策略的实例;名字不存在直接报错(拼写问题早暴露)。"""
    _seed_builtin()
    if name not in _providers:
        raise KeyError(f"未注册的压缩策略: {name!r}(已注册: {sorted(_providers)})")
    return _providers[name]()


def list_providers() -> list[str]:
    """列出所有已注册的策略名(排好序,展示用)。"""
    _seed_builtin()
    return sorted(_providers)


def _seed_builtin() -> None:
    """把内置策略以 setdefault 语义放进注册表(只放一次)。

    语义要点:如果有人在第一次取用之前已经注册了名叫 "default" 的
    自定义实现,内置实现就不放进去了 —— "先注册者优先",
    这样外部代码可以整体替换内置算法。
    """
    global _builtin_seeded
    if _builtin_seeded:
        return
    _builtin_seeded = True
    _providers.setdefault("default", DefaultContextProvider)


def reset_registry() -> None:
    """清空注册表、恢复"还没放过内置策略"的状态(测试之间重置用)。

    和 hooks.clear_hooks 同一个角色:注册表是模块级全局状态,
    测试不重置会互相污染。
    """
    global _builtin_seeded
    _providers.clear()
    _builtin_seeded = False
