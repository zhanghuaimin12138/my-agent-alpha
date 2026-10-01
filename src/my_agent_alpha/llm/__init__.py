"""LLM 门面层:外部代码只需要调用 reply(),不用关心用的是哪家模型。

各模型服务商(openai、deepseek、mock……)的接口细节都藏在这层里面。
所有服务商统一走一种流式接口(仿 gptme):

    流式函数一边 yield 一小段文字,一边在结束时 return 用量信息。

两个关键设计:

- _StreamWithMetadata:包装流式生成器的一个小类。Python 生成器可以在
  结束时"带回一个返回值"(通过 StopIteration 异常的 value 属性),
  这个类负责把它取出来存好 —— 这样"提前停止生成"后也照样能拿到用量;

- 断点检测(break_on_tooluse):模型生成的文字是一小段一小段到达的,
  每凑满一行就检查一次:已经到达的文字里有没有"写完整的工具调用"?
  一旦发现,立刻停止接收剩余生成 —— 工具马上执行,后面的文字也不要了
  (反正下一轮会带着工具结果重新生成,留着也只是浪费钱)。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence

from my_agent_alpha.llm.mock import stream as mock_stream
from my_agent_alpha.llm.models import PROVIDERS, ModelMeta, ProviderCfg, resolve_model
from my_agent_alpha.llm.openai_compat import stream as openai_stream
from my_agent_alpha.message import Message
from my_agent_alpha.tools.base import ToolSpec

__all__ = ["Message", "ModelMeta", "reply", "resolve_model"]

# 流式渲染回调的类型:接收一小段文字的函数
OnToken = Callable[[str], None]

_INTERRUPT_MARKER = "\n^C 已中断"


def _custom_cfg(provider: str) -> ProviderCfg:
    """遇到 PROVIDERS 表里没有的服务商名时,尝试从环境变量拼出它的配置。

    约定:ALPHA_<名字大写>_BASE_URL 是接口地址,ALPHA_<名字大写>_API_KEY 是密钥。
    都没设置就报错并提示怎么配。
    """
    base_url = os.environ.get(f"ALPHA_{provider.upper()}_BASE_URL")
    if not base_url:
        raise RuntimeError(
            f"未知 provider {provider!r}:请用已知 provider(如 deepseek/deepseek-chat),"
            f"或设置 ALPHA_{provider.upper()}_BASE_URL 指向 OpenAI 兼容端点"
        )
    return ProviderCfg(provider, base_url, f"ALPHA_{provider.upper()}_API_KEY", None)


class _StreamWithMetadata:
    """包住流式生成器,让"用量信息"在流结束后随时可读(.metadata 属性)。

    背景:生成器结束时可以带一个返回值,但它藏在 StopIteration 异常里,
    平时的 for 循环拿不到。这个类把值取出来存好。
    生成器结束时的返回值(usage 账单)藏在 StopIteration 异常的.value 里
    for 循环会把它吞掉，手写 next 才有机会捕到
    _StreamWithMetadata 就是干这个的(它把 .value 存进 wrapped.metadata)

    它自己也是个迭代器,而且支持"中途停、之后接着读"(断点检测会用到:
    提前停止后,把剩余部分读完丢弃,只为在结尾拿到用量信息)。
    """

    def __init__(self, gen: Iterator[str]) -> None:
        """记下要包装的生成器,用量信息先置空(流结束时才有)。"""
        self._gen = gen
        self.metadata: dict[str, int] | None = None

    def __iter__(self) -> Iterator[str]:
        """返回自己:自己就能被 for 循环(见 __next__)。"""
        return self

    def __next__(self) -> str:
        """取下一段文字;流结束时把生成器带回的用量信息存进 .metadata。"""
        try:
            return next(self._gen)
        except StopIteration as e:
            self.metadata = e.value  # 生成器结束:把带回的用量信息存下来
            raise


def _has_complete_runnable_tooluse(accumulated: str) -> bool:
    """检查"到目前为止收到的文字"里,是否出现了写完整的工具调用。

    用流式规则检查(见 codeblock.iter_from_markdown 的 streaming 参数):
    代码块的 ``` 闭合后还要跟一个空行才算数。
    """
    from my_agent_alpha.tools.base import ToolUse  # 延迟导入避免循环依赖

    return any(tu.is_runnable for tu in ToolUse.iter_from_content(accumulated, streaming=True))


def reply(
    messages: Sequence[Message],
    model: str,
    *,
    on_token: OnToken | None = None,
    # 被工具调用打断:一次 step 只执行一个工具调用,
    # 除非模型支持并行工具调用(后续补充)
    break_on_tooluse: bool = True,
    tools: Sequence[ToolSpec] = (),
    tool_format: str = "markdown",
) -> Message:
    """把对话发给模型,等它生成完,返回一条 assistant 消息。

    - on_token:模型每生成一小段就调用一次(终端实时显示用),不传也行;
    - break_on_tooluse=True(默认):生成过程中一旦凑出完整的工具调用,
      立刻停止接收,返回已收到的部分文字。省钱,而且工具能更早开始跑;
    - tool_format="tool" 时开启原生工具调用(tools 一起发给接口);
      模型的调用会被翻译成 "@工具名(编号): {参数}" 行混在文本流里,
      断点检测同样适用。"markdown"(默认)则一切照旧;
    - 用户按 Ctrl-C:立刻停止,返回"已生成的部分 + '^C 已中断'标记",
      并在 metadata 里记上 interrupted=True(上层循环看到就结束本回合)。

    示例:msg = reply(对话消息列表, "deepseek/deepseek-chat")
    """
    meta = resolve_model(model)
    # 判断是否支持原生工具调用
    # 如果支持，传入参数tools应不为none（list[ToolSpec]）
    use_native = tool_format == "tool"
    if meta.provider == "mock":
        raw: Iterator[str] = mock_stream(messages, meta.model)
    else:
        cfg = PROVIDERS.get(meta.provider) or _custom_cfg(meta.provider)
        raw = openai_stream(messages, meta, cfg, tools if use_native else ())

    # 消费循环
    # 解析拿到的chunk(raw)
    wrapped = _StreamWithMetadata(raw)  # 包装一层，为了拿到usage
    parts: list[str] = []  # 收到的所有文字小段（攒着，最后拼完整回复）
    interrupted = False
    broke = False
    try:
        while True:
            chunk = next(wrapped)
            parts.append(chunk)
            if on_token is not None:
                on_token(chunk)  # 终端实时打印
            # 每凑满一行检查一次(检查太频繁会拖慢,行是天然的最小单位)
            if (  # 有可执行的工具,直接打断,不继续接收模型生成
                # ps:有两个可执行工具怎么办?第二个直接打断,
                # 但下一轮对话模型会再次提出调用这个工具(基于第一轮的真实结果)
                # 什么时候不打断?模型支持并行工具调用的时候,可以选择不打断
                break_on_tooluse
                and "\n" in chunk
                and _has_complete_runnable_tooluse("".join(parts))
            ):
                broke = True
                break
    except KeyboardInterrupt:
        interrupted = True
    except StopIteration:
        pass  # 正常结束:用量信息已被包装器存好

    if broke:
        # 把剩余的生成内容读完(内容直接丢弃，但是生成内容的结尾有usage，需要拿到)
        try:
            while True:
                next(wrapped)  # 在Generator的最后拿到返回值usage
        except StopIteration:
            pass

    text = "".join(parts)  # 攒起来的内容放进content（text）里
    if interrupted:
        text += _INTERRUPT_MARKER
    metadata: dict[str, object] = {"model": meta.full}
    if wrapped.metadata:
        metadata["usage"] = wrapped.metadata
    if interrupted:
        metadata["interrupted"] = True
    return Message("assistant", text, metadata=metadata)
