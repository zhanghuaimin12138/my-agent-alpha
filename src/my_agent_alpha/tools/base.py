"""工具系统的两个核心类:ToolSpec(工具是什么)和 ToolUse(一次调用)。

模型怎么"调用工具"?有两种通道(见 docs/learn/01 学习文档 §2.4):

通道一(markdown 约定,阶段 1 起):模型在回复里写代码块,

    ```shell
    ls -la
    ```

代码块的语言标记就是工具名,块内是执行内容;块头还可以带参数,
如 ```save path/to/file``` 中 path/to/file 是保存路径。

通道二(原生工具调用,本阶段新增):模型走接口内置的 function calling,
我们在日志里把它记成一行文本(这样日志还是纯文本、人和解析器都好处理):

    @save(call_abc123): {"path": "a.txt", "content": "你好"}

两种通道解析出来都是 ToolUse,执行的是同一个函数 —— 工具的"执行端"
(save.py 这些)完全不用关心自己是被哪种通道调用的。

- ToolSpec:一个工具的"说明书"(名字、用法、示例、参数表、执行函数);
- ToolUse:从模型回复里解析出来的一次具体调用。

阶段 3 刻意省略:hooks 和确认机制(阶段 4)、工具开关列表。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace

from my_agent_alpha.codeblock import fenced_regions, iter_from_markdown, strip_think
from my_agent_alpha.message import DECLINED_CONTENT, Message

# 所有工具的执行函数长这一个样子:
# execute(代码内容, 位置参数列表, 关键字参数字典) -> 一条消息,或者一堆消息(逐条产出)
# 两种通道在这里汇合:markdown 调用给 code/args,原生调用给 kwargs(解析好的参数字典)
ExecuteFunc = Callable[
    [str | None, list[str] | None, dict[str, str] | None],
    "Message | Iterator[Message]",
]


@dataclass(frozen=True)
class Parameter:
    """工具的一个参数声明(给原生工具调用生成 JSON Schema 用)。

    示例:Parameter("path", "string", "要保存的文件路径") 表示
    "名为 path 的字符串参数,必填"。
    """

    name: str
    type: str = "string"  # JSON Schema 的类型:string / integer / number / boolean / array
    description: str = ""
    required: bool = True


def parameters_to_schema(params: Sequence[Parameter]) -> dict[str, object]:
    """把参数表转成接口要求的 JSON Schema(原生工具调用的"参数说明书")。

    示例:两个参数 (path, content) 转成
    {"type": "object",
     "properties": {"path": {"type": "string", "description": ...},
                    "content": {"type": "string", ...}},
     "required": ["path", "content"],
     "additionalProperties": False}
    """
    properties: dict[str, object] = {}
    for p in params:
        prop: dict[str, object] = {"type": p.type}
        if p.description:
            prop["description"] = p.description
        properties[p.name] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": [p.name for p in params if p.required],
        "additionalProperties": False,
    }


def spec_to_openai_tool(spec: ToolSpec) -> dict[str, object]:
    """把 ToolSpec 转成接口的 tools 参数格式(原生工具调用的请求端)。

    示例:shell 工具转成
    {"type": "function",
     "function": {"name": "shell",
                  "description": "在 bash 里执行命令……",
                  "parameters": {"type": "object", ...}}}

    说明:description 截断到 1000 字符 —— OpenAI 对函数描述有 1024 上限,
    超长直接 400(gptme 为此专门维护"精简版说明",我们先用截断)。
    """
    description = f"{spec.desc}\n{spec.instructions}".strip()
    if len(description) > 1000:
        description = description[:1000] + "…"
    function: dict[str, object] = {
        "name": spec.name,
        "description": description,
        "parameters": parameters_to_schema(spec.parameters),
    }
    return {"type": "function", "function": function}


# 工具规格说明书
@dataclass(frozen=True)
class ToolSpec:
    """一个工具的声明。示例见 tools/shell.py 末尾的 tool = ToolSpec(...)。

    block_types 最关键:列出"哪些语言标记代表我",如 ("shell", "sh", "bash")。
    模型回复里的代码块语言标记命中这里,这个块就是对我的调用(markdown 通道)。
    parameters 是参数表,给原生工具调用生成 JSON Schema 用(两种通道共用一份声明)。
    """

    name: str
    desc: str = ""
    instructions: str = ""
    examples: str = ""
    block_types: tuple[str, ...] = ()  # markdown fence 语言标记 ↔ 本工具
    parameters: tuple[Parameter, ...] = ()  # 参数表(原生工具调用用)
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


# 匹配原生调用行的开头,如 "@save(call_abc123):"
# 编号字符集必须允许点号:有的服务商(dashscope/kimi)给的调用编号长这样
# "websearch.search_0_a475cfc6"(工具名_序号_哈希,工具名本身带点),
# 不许点号会导致整行识别失败、调用被静默丢弃(踩坑记录条目 5)
_NATIVE_CALL_RE = re.compile(r"@([A-Za-z0-9_.\-]+)\(([A-Za-z0-9_.\-]*)\)\s*:")


def find_json_end(text: str, brace_start: int) -> int:
    """从左大括号的位置开始,数配对的大括号,返回右大括号后面一格的位置。

    引号里的大括号不算(否则 "{"path": "}"} 会被数错)。
    找不到配对的右括号返回 -1(说明 JSON 还没写完,流式中间态)。

    示例:find_json_end('x {"a": 1} y', 2) → 9(指向 '}' 的下一格)
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(brace_start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


@dataclass(frozen=True)
class ToolUse:
    """模型回复里解析出来的一次工具调用。

    两种通道解析的结果:
    - markdown:回复 "我来:\n```save a.txt\n你好\n```" 解析出
      tool=save 工具、args=["a.txt"]、content="你好"、kwargs=None;
    - 原生:回复里的 "@save(call_abc): {\\"path\\": \\"a.txt\\", ...}" 行解析出
      tool=save 工具、kwargs={"path": "a.txt", ...}、call_id="call_abc"、content=None。

    start 是这次调用在回复原文里的位置(多个调用按它排序)。
    """

    tool: ToolSpec
    args: list[str] | None
    content: str | None
    kwargs: dict[str, str] | None = None  # 原生调用的参数(已解析成字典)
    call_id: str | None = None  # 原生调用的编号,执行结果要带着它配对回去
    start: int = 0  # 为其在 assistant 文本中的字符偏移

    @property
    def is_runnable(self) -> bool:
        """这次调用的工具能不能真的执行。"""
        return self.tool.is_runnable

    def display_command(self) -> str:
        """给人看的"这次调用要干什么",打印和测试断言用。

        示例:markdown 的 shell 调用 → "shell: ls -la";
        原生的 save 调用 → "save: path=a.txt"
        """
        if self.kwargs:
            kv = " ".join(f"{k}={v}" for k, v in self.kwargs.items())
            return f"{self.tool.name}: {kv}"
        if self.args:
            return f"{self.tool.name}: {' '.join(self.args)} {self.content or ''}".strip()
        return f"{self.tool.name}: {self.content or ''}"

    def execute(self) -> Iterator[Message]:
        """真正执行这次调用,逐条产出结果消息。

        执行前先过确认链:非只读工具都会被"审批"一次 ——
        放行(confirm)才执行;拒绝(skip)产出一条带 DECLINED 标记的说明,
        不执行;改写(edit)则把主参数换成改写后的内容再执行。

        工具内部出错不会炸掉程序:错误被转成一条 system 消息
        (模型看到错误信息后可以自己修正重试)。
        Ctrl-C 例外:中断必须向上传,不能被当成"工具出错"吞掉。

        实现细节:try 必须包住整个执行过程(包括逐条取结果),
        因为很多工具的执行函数是生成器 —— 调用它只是建了个生成器,
        真正的代码在取结果时才运行,错误也是那时才冒出来。
        """
        assert self.tool.execute is not None  # is_runnable 已保证
        if not self.tool.read_only:
            from my_agent_alpha.confirm import get_confirmation  # 延迟导入避免循环

            decision = get_confirmation(self)
            if decision.action == "skip":
                reason = f":{decision.message}" if decision.message else ""
                yield Message(
                    "system", f"{DECLINED_CONTENT} 已跳过工具 {self.tool.name} 调用{reason}"
                )
                return
            if decision.action == "edit" and decision.edited_content is not None:
                # 改写主参数:markdown 通道换 content,原生通道换 command/content 参数
                self = self.with_edited_content(decision.edited_content)
        execute_fn = self.tool.execute
        assert execute_fn is not None  # is_runnable 已保证
        try:
            result = execute_fn(self.content, self.args, self.kwargs)
            if isinstance(result, Message):
                yield result
            else:
                yield from result
        except KeyboardInterrupt:
            raise  # 中断必须向上传播,不被吞成错误消息
        except Exception as e:
            # 带上异常类型名:有的异常(如 TimeoutError)str() 是空字符串,
            # 只拼 {e} 会得到"执行失败: "后面一片空白,模型无从判断死因
            yield Message("system", f"工具 {self.tool.name} 执行失败: {type(e).__name__}: {e}")

    def with_edited_content(self, edited: str) -> ToolUse:
        """返回一个"主参数被改写"的新调用(消息不可变,造新的)。

        示例:markdown 的 ```shell\necho hi\n``` 被用户编辑成 "echo hello"
        → 新调用的 content = "echo hello";
        原生的 {"command": "echo hi"} → kwargs["command"] = "echo hello"。
        """
        if self.kwargs:  # 原生通道
            new_kwargs = dict(self.kwargs)
            for key in ("command", "content"):  # command是shell的参数，content是save和append的参数
                if key in new_kwargs:
                    new_kwargs[key] = edited
                    break
            return replace(self, kwargs=new_kwargs)
        return replace(self, content=edited)

    @classmethod
    def iter_from_content(cls, content: str, *, streaming: bool = False) -> Iterator[ToolUse]:
        """从模型回复文本里按出现顺序解析出所有工具调用(两种通道都认)。

        示例:list(ToolUse.iter_from_content("```shell\\nls\\n```"))
        → [ToolUse(tool=shell工具, args=None, content="ls")]

        - markdown 通道:语言标记不认识任何工具的代码块直接忽略;
        - 原生通道:@工具名(编号): {参数} 行,参数 JSON 没闭合就先忽略
          (流式生成到一半是正常现象,这本身就是"流式安全"的);
          围栏代码块里面写的 @xxx 只是示例,不算调用;
        - 两个通道都先剥掉 <think>...</think> 思考区(gptme 的做法):
          思考里写的围栏/@调用是草稿,不算真调用;
        - streaming=True 只在流式检查时使用:代码块的 ``` 闭合后还要
          跟一个空行才算数(更严,防止执行半截命令),详见 codeblock 模块说明。
        """
        from my_agent_alpha.tools import get_tool, get_tool_for_langtag  # 延迟导入避免循环依赖

        # 先剥思考区:think 里写的围栏/@调用是草稿,不算真调用;
        # 粘在 </think> 后面的围栏剥完落到行首,markdown 通道才看得见
        content = strip_think(content)
        found: list[ToolUse] = []

        for cb in iter_from_markdown(  # list[codeBlock]
            content, streaming=streaming
        ):  # markdown代码块解析出codeblock类
            tool = get_tool_for_langtag(cb.lang)  # 通过lang->toolSpec
            if tool is not None:
                found.append(  # 得到toolUse
                    cls(
                        tool=tool,
                        args=[cb.args] if cb.args else None,
                        content=cb.content,
                        start=cb.start,
                    )
                )

        regions = fenced_regions(content)  # 围栏内的 @xxx 是示例,跳过
        for match in _NATIVE_CALL_RE.finditer(content):
            if any(a <= match.start() < b for a, b in regions):
                continue
            # 冒号后面应该紧跟一个 {,开始数配对大括号
            brace = match.end()
            while brace < len(content) and content[brace] in " \t":
                brace += 1
            if brace >= len(content) or content[brace] != "{":
                continue
            end = find_json_end(content, brace)
            if end < 0:
                continue  # JSON 没闭合:还没写完,先不要
            try:
                kwargs = json.loads(content[brace:end])
            except json.JSONDecodeError:
                continue  # JSON 坏了:不是有效调用,跳过
            if not isinstance(kwargs, dict):
                continue
            tool = get_tool(match.group(1))  # 原生通道按工具名查找
            if tool is not None:
                found.append(
                    cls(
                        tool=tool,
                        args=None,
                        content=None,
                        kwargs=kwargs,
                        call_id=match.group(2) or None,
                        start=match.start(),
                    )
                )

        yield from sorted(found, key=lambda u: u.start)  # 按出现位置排序


def parse_tooluses(content: str) -> list[ToolUse]:
    """同 iter_from_content,但直接返回完整列表(不想写 list(...) 时用)。"""
    return list(ToolUse.iter_from_content(content))


def tooluses_in_messages(messages: Sequence[Message]) -> list[ToolUse]:
    """从消息列表里最后一条模型回复中解析工具调用(没有回复则空列表)。"""
    for msg in reversed(messages):
        if msg.role == "assistant":
            return parse_tooluses(msg.content)
    return []


def has_runnable_tooluse(messages: Sequence[Message]) -> bool:
    """回答"模型刚才的回复里,还有没执行完的工具调用吗?" —— 决定循环是否继续。

    找最后一条模型回复,解析其中的工具调用,看有没有可执行的
    (实现直接复用 tooluses_in_messages,它做的就是"取最后一条回复并解析")。

    示例:
        最后一条模型回复 = "我来:\n```shell\nls\n```"  → True(继续下一轮)
        最后一条模型回复 = "任务完成了"                → False(回合结束)
    """
    return any(tu.is_runnable for tu in tooluses_in_messages(messages))
