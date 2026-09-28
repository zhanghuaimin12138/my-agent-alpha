"""消息类型 —— 整个系统唯一的数据单元。

仿 gptme 的核心约定(见 docs/gptme-study.md 支柱一):

- 只有三种 role(system/user/assistant),没有单独的"工具调用"/"工具输出"消息类型;
- 工具调用内嵌在 assistant 的 content 文本里(将来的 ToolUse 解析器负责识别);
- 工具输出是 role='system'、带 call_id 的消息,与调用一一配对;
- 序列化宽容:from_dict 丢弃未知字段(向前兼容未来版本的日志文件)。

阶段 0 相对 gptme 刻意省略的字段(详见 docs/devlog/001):
files/file_hashes(附件)、ephemeral_ttl(临时消息)、terminal_display_content。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

Role = Literal["system", "user", "assistant"]

_ROLES: tuple[str, ...] = ("system", "user", "assistant")

# 一等公民协议字符串:日志里的特殊标记消息(仿 gptme,崩溃恢复/中断识别就靠扫它们)
INTERRUPT_CONTENT = "[interrupted]"
LLM_REQUEST_FAILED_PREFIX = "[LLM 请求失败]"


def utcnow() -> datetime:
    """带时区的当前 UTC 时间(日志里的时间戳统一用它,避免 naive datetime 歧义)。"""
    return datetime.now(UTC)


@dataclass(frozen=True)
class Message:
    """一条消息。frozen:约定消息一旦产生就不再修改,修改 = 构造新实例。

    相等性仿 gptme:只基于 (role, content)。timestamp/call_id/标志位/metadata
    都是附属信息,不参与同一性判定(同时让实例可哈希)。
    注意由此推论:同 role 同 content 的两条消息"相等",即便 call_id 不同 ——
    消费方永远不得用 Message 相等性来做工具结果配对(那是 call_id 的职责)。
    """

    role: Role
    content: str
    timestamp: datetime = field(default_factory=utcnow, compare=False)
    # 工具输出与 assistant 内嵌工具调用的配对 id(原生工具调用格式的关键纪律,
    # 从第一天就预留:每个 call_id 恰好配一个结果)
    call_id: str | None = field(default=None, compare=False)
    pinned: bool = field(default=False, compare=False)  # 永不被上下文裁剪
    hide: bool = field(default=False, compare=False)  # 发给 LLM 但不打印
    quiet: bool = field(default=False, compare=False)  # 追加进日志时不打印
    # 附属元数据(model/usage/cost/timings 等);构造时浅拷贝,防外部后续改动
    metadata: dict[str, Any] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"无效的 role: {self.role!r},必须是 {_ROLES} 之一")
        if isinstance(self.metadata, dict):
            # frozen dataclass 里放可变 dict:浅拷贝隔离外部引用(嵌套值仍共享,视为只读约定)
            object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 兼容 dict。

        role/content/timestamp 恒定写出;可选字段仅在非默认值时写出,
        保持 JSONL 每行紧凑、人类可读。
        """
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.call_id is not None:
            d["call_id"] = self.call_id
        if self.pinned:
            d["pinned"] = True
        if self.hide:
            d["hide"] = True
        if self.quiet:
            d["quiet"] = True
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """宽容反序列化:未知字段直接丢弃,缺省字段回退默认值。

        - 非 dict 输入(合法 JSON 却不是对象,如数字/数组)→ ValueError;
        - 时间戳缺失或损坏回退当前时间;naive 时间戳按 UTC 解释。
        """
        if not isinstance(data, dict):
            raise ValueError(f"消息必须是 JSON 对象,得到 {type(data).__name__}")

        role = data.get("role")
        if role not in _ROLES:
            raise ValueError(f"无效的 role: {role!r},必须是 {_ROLES} 之一")

        ts_raw = data.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else utcnow()
        except ValueError:
            ts = utcnow()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)  # naive 一律按 UTC 解释,模块内时间戳恒为 aware

        return cls(
            role=cast(Role, role),
            content=data.get("content", ""),
            timestamp=ts,
            call_id=data.get("call_id"),
            pinned=bool(data.get("pinned", False)),
            hide=bool(data.get("hide", False)),
            quiet=bool(data.get("quiet", False)),
            metadata=data.get("metadata") or None,
        )
