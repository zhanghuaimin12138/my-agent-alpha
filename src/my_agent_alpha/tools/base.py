"""工具系统的两个核心类:ToolSpec(工具是什么)和 ToolUse(一次调用)。

模型怎么"调用工具"?靠约定:它在回复里写一个 markdown 代码块,

    ```shell
    ls -la
    ```

代码块的语言标记就是工具名,块内就是执行内容;块头还可以带参数,
如 ```save path/to/file``` 中 path/to/file 是保存路径。

- ToolSpec:一个工具的"说明书"(名字、用法、示例、执行函数)。
  同一份说明书有三个用途:拼进系统提示词教模型怎么用、
  用来识别回复里的代码块是不是工具调用、提供执行函数;
- ToolUse:从模型回复里解析出来的一次具体调用(哪个工具 + 什么内容)。

阶段 1 刻意省略:参数类型表 Parameter、模型原生的 JSON 工具调用(阶段 3)、
hooks 和确认机制(阶段 4)、工具开关列表。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from my_agent_alpha.codeblock import iter_from_markdown
from my_agent_alpha.message import Message

# 所有工具的执行函数长这一个样子:
# execute(代码内容, 位置参数列表, 关键字参数字典) -> 一条消息,或者一堆消息(逐条产出)
ExecuteFunc = Callable[
    [str | None, list[str] | None, dict[str, str] | None],
    "Message | Iterator[Message]",
]


# 工具规格说明书
@dataclass(frozen=True)
class ToolSpec:
    """一个工具的声明。示例见 tools/shell.py 末尾的 tool = ToolSpec(...)。

    block_types 最关键:列出"哪些语言标记代表我",如 ("shell", "sh", "bash")。
    模型回复里的代码块语言标记命中这里,这个块就是对我的调用。
    """

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
        """有没有执行函数(没有就只能当文档展示,不能真的执行)。"""
        return self.execute is not None

    # 工具提示词：name + desc + instructions + examples(分隔符\n\n)
    def get_tool_prompt(self, *, with_examples: bool = True) -> str:
        """把说明书渲染成一段 markdown,拼进系统提示词给模型看。

        输出形如:
            ## shell
            在 bash 里执行命令……
            ### 示例
            ```shell
            ls -la
            ```
        """
        parts = [f"## {self.name}", self.desc]
        if self.instructions:
            parts.append(self.instructions)
        if with_examples and self.examples:
            parts.append(f"### 示例\n\n{self.examples}")
        return "\n\n".join(p for p in parts if p)


@dataclass(frozen=True)
class ToolUse:
    """模型回复里解析出来的一次工具调用。

    示例:回复 "我来:\n```save a.txt\n你好\n```" 解析出
    tool=save 工具、args=["a.txt"]、content="你好"。
    start 是这次调用在回复原文里的位置(多个调用按它排序)。
    """

    tool: ToolSpec
    args: list[str] | None
    content: str | None
    call_id: str | None = None  # 阶段 3 原生工具调用配对用,本阶段恒为 None
    start: int = 0  # 为其在 assistant 文本中的字符偏移

    @property
    def is_runnable(self) -> bool:
        """这次调用的工具能不能真的执行。"""
        return self.tool.is_runnable

    def execute(self) -> Iterator[Message]:
        """真正执行这次调用,逐条产出结果消息。

        工具内部出错不会炸掉程序:错误被转成一条 system 消息
        (模型看到错误信息后可以自己修正重试)。
        Ctrl-C 例外:中断必须向上传,不能被当成"工具出错"吞掉。

        实现细节:try 必须包住整个执行过程(包括逐条取结果),
        因为很多工具的执行函数是生成器 —— 调用它只是建了个生成器,
        真正的代码在取结果时才运行,错误也是那时才冒出来。
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
    def iter_from_content(cls, content: str, *, streaming: bool = False) -> Iterator[ToolUse]:
        """从模型回复文本里按顺序解析出所有工具调用。

        示例:list(ToolUse.iter_from_content("```shell\\nls\\n```"))
        → [ToolUse(tool=shell工具, args=None, content="ls")]

        - 语言标记不认识任何工具的代码块(比如模型写的普通 ```python 示例)
          直接忽略 —— 那不是工具调用;
        - streaming=True 只在流式检查时使用:代码块的 ``` 闭合后还要
          跟一个空行才算数(更严,防止执行半截命令),详见 codeblock 模块说明。
        """
        from my_agent_alpha.tools import get_tool_for_langtag  # 延迟导入避免循环依赖

        for cb in iter_from_markdown(
            content, streaming=streaming
        ):  # markdown代码块解析出codeblock类
            tool = get_tool_for_langtag(cb.lang)  # 通过lang->toolSpec
            if tool is not None:
                yield cls(
                    tool=tool,
                    args=[cb.args] if cb.args else None,
                    content=cb.content,
                    start=cb.start,
                )


def parse_tooluses(content: str) -> list[ToolUse]:
    """同 iter_from_content,但直接返回完整列表(不想写 list(...) 时用)。"""
    return list(ToolUse.iter_from_content(content))


def tooluses_in_messages(messages: Sequence[Message]) -> list[ToolUse]:
    """从消息列表里最后一条模型回复中解析工具调用(没有回复则空列表)。"""
    for msg in reversed(messages):
        if msg.role == "assistant":
            return parse_tooluses(msg.content)
    return []
