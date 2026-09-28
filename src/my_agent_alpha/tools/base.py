"""工具系统核心抽象(最小版)。

仿 gptme 的 ToolSpec/ToolUse 三足鼎立(研究精华:一份声明同时服务
模型文档渲染、调用解析、执行分发):

- ToolSpec:工具的声明单元(fence 语言标记、说明、示例、执行函数);
- ToolUse:从 assistant 文本里解析出的一次调用;
- markdown 工具调用格式:assistant 在回复里写

      ```shell
      ls -la
      ```

  即视为一次 shell 工具调用;```save path``` 的 fence 头余部是位置参数。

阶段 1 刻意省略:Parameter/原生 JSON 工具调用(call_id 配对属阶段 3)、
hooks、确认链(阶段 4)、允许列表。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from my_agent_alpha.codeblock import iter_from_markdown
from my_agent_alpha.message import Message

# 统一执行入口:execute(代码, 位置参数, 关键字参数) -> 单条消息或消息迭代器
ExecuteFunc = Callable[
    [str | None, list[str] | None, dict[str, str] | None],
    "Message | Iterator[Message]",
]

# 工具规格说明书
@dataclass(frozen=True)
class ToolSpec:
    name: str
    desc: str = ""
    instructions: str = ""
    examples: str = ""
    block_types: tuple[str, ...] = ()  # markdown fence 语言标记 ↔ 本工具
    read_only: bool = False
    execute: ExecuteFunc | None = None
    _sort_key: tuple[int, str] = field(default=(50, ""), compare=False)

    @property
    def is_runnable(self) -> bool:
        return self.execute is not None

    # 工具提示词：name + desc + instructions + examples(分隔符\n\n)
    def get_tool_prompt(self, *, with_examples: bool = True) -> str:
        """渲染进系统提示词的 markdown 工具文档。"""
        parts = [f"## {self.name}", self.desc]
        if self.instructions:
            parts.append(self.instructions)
        if with_examples and self.examples:
            parts.append(f"### 示例\n\n{self.examples}")
        return "\n\n".join(p for p in parts if p)


@dataclass(frozen=True)
class ToolUse:
    """一次解析出的工具调用。start 为其在 assistant 文本中的字符偏移。"""

    tool: ToolSpec
    args: list[str] | None
    content: str | None
    call_id: str | None = None  # 阶段 3 原生工具调用配对用,本阶段恒为 None
    start: int = 0 # 为其在 assistant 文本中的字符偏移

    @property
    def is_runnable(self) -> bool:
        return self.tool.is_runnable

    def execute(self) -> Iterator[Message]:
        """执行本次调用;执行体抛出的异常转为 system 错误消息。

        注意 try 必须包住整个迭代:工具 execute 返回 generator 时,
        函数体在 yield from 时才真正运行,异常也从这里抛出
        (审查 #4:只包创建不包迭代是拦不住的)。
        """
        assert self.tool.execute is not None  # is_runnable 已保证
        try:
            result = self.tool.execute(self.content, self.args, None)
            if isinstance(result, Message):
                yield result
            else:
                yield from result
        except KeyboardInterrupt:
            raise  # 中断必须向上传播,不被吞成错误消息
        except Exception as e:
            yield Message("system", f"工具 {self.tool.name} 执行失败: {e}")

    @classmethod
    def iter_from_content(cls, content: str) -> Iterator[ToolUse]:
        """从 assistant 文本解析所有工具调用(按出现顺序)。

         语言标记查不到对应工具的块(如模型写的普通 ```python 说明代码)
        直接忽略 —— 它们不是工具调用。
        """
        from my_agent_alpha.tools import get_tool_for_langtag  # 延迟导入避免循环依赖

        for cb in iter_from_markdown(content): # markdown代码块解析出codeblock类
            tool = get_tool_for_langtag(cb.lang)# 通过lang->toolSpec
            if tool is not None:
                yield cls(
                    tool=tool,
                    args=[cb.args] if cb.args else None,
                    content=cb.content,
                    start=cb.start,
                )


def parse_tooluses(content: str) -> list[ToolUse]:
    """iter_from_content 的物化版本(继续判定/断言用)。"""
    return list(ToolUse.iter_from_content(content))


def tooluses_in_messages(messages: Sequence[Message]) -> list[ToolUse]:
    """便利函数:从消息序列里最后一条 assistant 消息解析工具调用。"""
    for msg in reversed(messages):
        if msg.role == "assistant":
            return parse_tooluses(msg.content)
    return []
