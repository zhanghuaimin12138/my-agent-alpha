"""消息类型 —— 整个系统唯一的数据单元。

核心约定(见 docs/gptme-study.md 支柱一):

- 只有三种 role(system/user/assistant),没有单独的"工具调用"/"工具输出"消息类型;
- 工具调用内嵌在 assistant 的 content 文本里(将来的 ToolUse 解析器负责识别);
- 工具输出是 role='system'、带 call_id 的消息,与调用一一对应;
- 读取时宽容:from_dict 遇到不认识的字段直接丢弃(这样旧代码也能读新版本日志)。

阶段 0 相对 gptme 刻意省略的字段(详见 docs/devlog/001):
files/file_hashes(附件)、ephemeral_ttl(临时消息)、terminal_display_content。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

Role = Literal["system", "user", "assistant"]

_ROLES: tuple[str, ...] = ("system", "user", "assistant")

# 全局统一的特殊标记内容。日志里出现这些字符串开头 system 消息,就表示发生了对应事件
# (例如程序崩溃后重新打开会话,扫描日志看到 [interrupted] 就知道上个 turn 是被中断的)
INTERRUPT_CONTENT = "[interrupted]"
LLM_REQUEST_FAILED_PREFIX = "[LLM 请求失败]"
DECLINED_CONTENT = "[declined]"  # 用户拒绝执行某次工具调用的标记(看到它本回合停止)


def utcnow() -> datetime:
    """返回当前 UTC 时间(带时区信息)。

    日志里所有时间戳统一用它。为什么不直接 datetime.now()?
    因为那返回"不带时区的时间",和"带时区的时间"比较会直接报错,统一成 UTC 省去麻烦。
    """
    return datetime.now(UTC)


def clean_bad_unicode(text: str) -> str:
    """把文本里编码不了的"落单代理字符"换成 ?,保证 utf-8 写入永不崩溃。

    这种字符正常文本里不会有,来源通常是终端输入法/剪贴板传来的坏字节。
    不清理的话,写日志时 encode("utf-8") 会抛 UnicodeEncodeError,
    整个会话直接崩溃退到系统提示符(2026-10-05 实测:用户输入混入坏表情序列)。

    示例:clean_bad_unicode("\\ud83c有雨吗") → "?有雨吗"
    (合法表情在 Python 里本来就是完整字符,不受这个函数影响)
    """
    return text.encode("utf-8", errors="replace").decode("utf-8")


@dataclass(frozen=True)
class Message:
    """一条消息。对话里的所有内容(用户输入、模型回复、工具输出)都是它。

    frozen=True 表示消息一旦创建就不许修改(想改就新建一条),
    这样日志里的历史永远可信。

    两条消息"相等"的条件只有 role 和 content 相同(仿 gptme):
    时间戳、call_id 等都算附属信息,不参与比较。
    注意:判断"这条工具输出对应哪次调用"永远用 call_id,不要用消息相等。

    示例:
        >>> Message("user", "你好")                     # 一条用户消息
        >>> Message("system", "输出", call_id="call-1")  # 一次工具调用的结果
    """

    role: Role
    content: str
    timestamp: datetime = field(default_factory=utcnow, compare=False)
    # 当这条消息是"工具输出"时,记录它对应哪一次工具调用(原生工具调用格式的关键
    # 从第一天就预留:每个 call_id 恰好配一个结果)
    call_id: str | None = field(default=None, compare=False)
    pinned: bool = field(default=False, compare=False)  # 重要消息,将来压缩上下文时永不删
    hide: bool = field(default=False, compare=False)  # 发给模型但不显示在终端(如系统提示词)
    quiet: bool = field(default=False, compare=False)  # 写进日志但不显示在终端
    # 附加信息(用的什么模型、token 用量等);构造时会复制一份,防止外部后续修改
    metadata: dict[str, Any] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """构造后的自动检查:role 必须合法;metadata 复制一份隔离外部改动。"""
        if self.role not in _ROLES:
            raise ValueError(f"无效的 role: {self.role!r},必须是 {_ROLES} 之一")
        if isinstance(self.metadata, dict):
            # frozen 类里放 dict(可变对象):复制一份,外部再改原 dict 也不影响本消息
            object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """转成可以 json.dumps 的字典(写日志文件用)。

        示例:Message("user", "你好").to_dict()
        得到 {'role': 'user', 'content': '你好', 'timestamp': '2026-09-29T...'}

        role/content/timestamp 永远写;其他字段只在非默认值时才写,让日志每行更短。
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
        """从字典恢复一条消息(读日志文件用),和 to_dict 互为逆操作。

        读取是宽容的:
        - 不认识的字段直接丢弃 —— 以后升级日志格式,旧代码也不会崩;
        - 时间戳缺失或格式坏了,就用当前时间顶上;
        - 时间戳如果没带时区,一律当作 UTC 处理;
        - 传入的不是字典(比如一行 JSON 解析出来是个数字)→ 抛 ValueError。
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
            ts = ts.replace(tzinfo=UTC)  # 没带时区的时间一律按 UTC 处理

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
