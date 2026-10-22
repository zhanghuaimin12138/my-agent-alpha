"""上下文压缩包(gptme autocompact 的移植):决策、三阶段引擎、LLM 摘要。

对外的主要入口:
- should_auto_compact          要不要压、用哪种方式压(决策层)
- auto_compact_log             规则压缩(三阶段引擎,纯计算)
- resume_via_llm               LLM 摘要(生成接续文档,重建精简日志)
- get_context_provider / register_provider   压缩策略注册表

完整规格:docs/autocompact-dev.md(所有阈值、格式、边界都以它为准)。
触发器和视图分支落地(开发文档 M7)在后续阶段接入对话循环。
"""

from my_agent_alpha.autocompact.context_provider import (
    ContextProvider,
    DefaultContextProvider,
    get_context_provider,
    list_providers,
    register_provider,
    reset_registry,
)
from my_agent_alpha.autocompact.contracts import (
    FALLBACK_MODEL_INFO,
    HUGE_TOOL_RESULT_TOKENS,
    CompactionResult,
    CompressionConfig,
    ConversationStore,
    LLMClient,
    ModelInfo,
    ModelInfoProvider,
    OutputStore,
    TokenCounter,
    ToolUseDetector,
    is_tool_result,
    source_digest,
)
from my_agent_alpha.autocompact.decision import (
    CompactAction,
    estimate_compaction_savings,
    should_auto_compact,
)
from my_agent_alpha.autocompact.engine import auto_compact_log
from my_agent_alpha.autocompact.master_context import (
    MasterContext,
    build_byte_index,
    read_master_range,
)
from my_agent_alpha.autocompact.output_storage import (
    FileOutputStore,
    create_tool_result_summary,
)
from my_agent_alpha.autocompact.reasoning import strip_reasoning
from my_agent_alpha.autocompact.reduce import reduce_log, truncate_msg
from my_agent_alpha.autocompact.resume import (
    RESUME_PROMPT,
    parse_context_files,
    resume_via_llm,
)
from my_agent_alpha.autocompact.scoring import compress_content, score_sentence

__all__ = [
    "FALLBACK_MODEL_INFO",
    "HUGE_TOOL_RESULT_TOKENS",
    "RESUME_PROMPT",
    "CompactAction",
    "CompactionResult",
    "CompressionConfig",
    "ContextProvider",
    "ConversationStore",
    "DefaultContextProvider",
    "FileOutputStore",
    "LLMClient",
    "MasterContext",
    "ModelInfo",
    "ModelInfoProvider",
    "OutputStore",
    "TokenCounter",
    "ToolUseDetector",
    "auto_compact_log",
    "build_byte_index",
    "compress_content",
    "create_tool_result_summary",
    "estimate_compaction_savings",
    "get_context_provider",
    "is_tool_result",
    "list_providers",
    "parse_context_files",
    "read_master_range",
    "reduce_log",
    "register_provider",
    "reset_registry",
    "resume_via_llm",
    "score_sentence",
    "should_auto_compact",
    "source_digest",
    "strip_reasoning",
    "truncate_msg",
]
