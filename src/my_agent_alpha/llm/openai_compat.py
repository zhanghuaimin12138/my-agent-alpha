"""调用真实模型服务商的客户端:openai、deepseek、moonshot 等全都走这一个模块。

为什么一家就够?因为这些服务商的接口都和 openai 的一样(所谓"openai 兼容"),
只是接口地址和密钥不同 —— 换服务商只是换个配置。

要点:
- openai 这个库只在真正用到时才导入(测试用 mock,完全不加载它,启动更快);
- 有些模型(DeepSeek-R 系列)会额外输出"思考过程"(reasoning_content),
  我们实时把它包进 <think>...</think> 标签,方便统一处理和暗色显示;
- 失败重试:网络抖动、限流这类"临时性错误"自动重试几次,间隔越拉越长;
  密钥错误这类"重试也没用"的错误立即报错。
  注意我们自己重试的同时,把 openai 库自带的重试关掉了 ——
  两层都在重试,次数会相乘(比如 3×3=9 次),限流时反而雪上加霜。
  流式已经吐出过内容就不再重试(内容没法"吐回去",重试会导致重复)。
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator, Sequence
from typing import Any, cast

from my_agent_alpha.llm.models import ModelMeta, ProviderCfg
from my_agent_alpha.message import Message

_MAX_RETRIES = int(os.environ.get("ALPHA_LLM_MAX_RETRIES", "5"))
_BACKOFF_CAP = 30.0



def _backoff(attempt: int) -> float:
    """第 attempt 次重试前要等多久:1秒、2秒、4秒……最多 30 秒(越错越等得久)。"""
    return min(2.0**attempt, _BACKOFF_CAP)


def _is_transient(exc: BaseException) -> bool:
    """瞬时错误才值得重试(连接/限流/服务端 5xx);鉴权/参数错误重试无意义。"""
    import openai

    return isinstance(
        exc, openai.APIConnectionError | openai.RateLimitError | openai.InternalServerError
    )


def msgs2dicts(messages: Sequence[Message]) -> list[dict[str, str]]:
    """把我们的 Message 列表转成接口要求的格式:[{"role": ..., "content": ...}, ...]。

    - 三种 role 直接照抄。system 消息出现在中间是正常的
      (工具输出就是 system 消息),服务商都接受;
    - 内容为空的消息直接跳过:有些接口遇到空内容会直接报错,
      而空消息本来也不携带信息(完整历史仍在日志文件里)。
    """
    return [{"role": m.role, "content": m.content} for m in messages if m.content.strip()]


def _make_client(cfg: ProviderCfg) -> Any:
    """按服务商配置创建一个 API 客户端;密钥缺失时给出明确的提示。

    max_retries=0:关掉 openai 库自己的重试(重试由本模块统一管理,见模块说明)。
    """
    from openai import OpenAI

    api_key: str | None = None
    if cfg.api_key_env:
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"provider {cfg.name!r} 需要环境变量 {cfg.api_key_env},"
                f"例如:export {cfg.api_key_env}=sk-..."
            )
    else:
        api_key = "local"  # 本机服务(如 Ollama)不校验密钥,占位即可
    # SDK 内部重试归零:重试预算由我们自己的 _MAX_RETRIES 独占
    return OpenAI(api_key=api_key, base_url=cfg.base_url, max_retries=0, timeout=120)


def _extract_usage(raw: object) -> dict[str, int] | None:
    """从client api返回的用量对象里取出输入/输出 token 数;没有就返回 None。"""
    if raw is None:
        return None
    return {
        "input": getattr(raw, "prompt_tokens", 0) or 0,# 输入量token
        "output": getattr(raw, "completion_tokens", 0) or 0,# 输出量token
    }


def chat(
    messages: Sequence[Message], meta: ModelMeta, cfg: ProviderCfg
) -> tuple[str, dict[str, int] | None]:
    """非流式调用:等模型把回复全部生成完,一次性返回 (回复文本, token 用量)。

    遇到临时性错误(断网、限流等)自动重试,间隔越来越长(见 _backoff)。
    """
    last_err: BaseException | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            client = _make_client(cfg)
            resp = client.chat.completions.create(
                model=meta.model,
                messages=msgs2dicts(messages),
            )
            choice = resp.choices[0]
            return choice.message.content or "", _extract_usage(resp.usage)
        except Exception as e:
            if not _is_transient(e) or attempt == _MAX_RETRIES:
                raise
            last_err = e
            time.sleep(_backoff(attempt))
    raise RuntimeError(f"不可达:{last_err}")  # pragma: no cover(循环必 raise 或 return)


def _stream_once(
    messages: Sequence[Message], meta: ModelMeta, cfg: ProviderCfg
) -> Generator[str, None, dict[str, int] | None]:
    """发起一次流式请求(不含重试逻辑)。

    模型的回复一小段一小段到达,每段立刻 yield 出去;
    思考内容(reasoning_content)在第一段到达前先发一个 "<think>" 标记,
    正文开始时发 "</think>",这样接收方就能区分思考和正文;
    请求结束时把 token 用量作为生成器的返回值带回。
    """
    client = _make_client(cfg)
    events = client.chat.completions.create( # 正式发起http请求
        model=meta.model,
        messages=msgs2dicts(messages),
        stream=True,
        stream_options={"include_usage": True},
    )# 开了流式，每次拿一部分
    """ 流式拿到的chunk(event)结构

        chunk                                   ← object = "chat.completion.chunk"
        ├── id / created / model                每个chunk都重复带(同一个请求的)
        ├── system_fingerprint
        ├── choices[] # 每次http请求可以设置参数n，代表生成几个回答，每个回答对应一个choice,默认是一个,也就是choice[0]
        │   └── [0]
        │       ├── index        0
        │       ├── delta        **增量**:只有"新到"的字段
        │       │   ├── role            只在第一个事件出现("assistant")
        │       │   ├── content         新到的正文片段(平时最主要的内容)
        │       │   ├── reasoning_content  # 新到的思考片段(思考模型才有)
        │       │   └── tool_calls      
        │       └── finish_reason  #结束原因，正常结束为stop
        └── usage """
    usage: dict[str, int] | None = None
    in_think = False
    for event in events:# 拿到一小个chunk
        if getattr(event, "usage", None) is not None: # 在最后一次event里有usage(前提是请求参数带上include_usage)
            usage = _extract_usage(event.usage)  # 解析usage
        if not event.choices:# 没有回答则continue
            continue
        delta = event.choices[0].delta  # 拿到部分回答（增量）
        reasoning = getattr(delta, "reasoning_content", None)# 尝试拿到思考过程（只有思考模型才有）
        # 整体逻辑就是先输出 <think>+思考过程 再输出 </think>+content
        if reasoning:
            if not in_think:
                yield "<think>"
                in_think = True
            yield reasoning
        if delta.content:
            if in_think:
                yield "</think>"
                in_think = False
            yield delta.content

    if in_think:
        yield "</think>"
    return usage


def stream(
    messages: Sequence[Message], meta: ModelMeta, cfg: ProviderCfg
) -> Generator[str, None, dict[str, int] | None]:
    """流式调用,外面包一层重试。

    规则:还没吐出任何内容时出错 → 等一会儿重试;
    已经吐出过内容后出错 → 直接抛出(内容没法收回,重试会导致重复)。
    """
    for attempt in range(_MAX_RETRIES + 1):
        emitted = False
        try:
            gen = _stream_once(messages, meta, cfg)
            while True:
                chunk = next(gen)
                emitted = True
                yield chunk
        except StopIteration as e:
            return cast("dict[str, int] | None", e.value)
        except Exception as e:
            if not _is_transient(e) or emitted or attempt == _MAX_RETRIES:
                raise
            time.sleep(_backoff(attempt))
    return None  # pragma: no cover(循环必 return 或 raise)
