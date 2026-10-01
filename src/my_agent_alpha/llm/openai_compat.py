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

import json
import os
import time
from collections.abc import Generator, Sequence
from typing import Any, cast

from my_agent_alpha.llm.models import ModelMeta, ProviderCfg
from my_agent_alpha.message import Message
from my_agent_alpha.tools.base import ToolSpec, find_json_end, spec_to_openai_tool

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


def msgs2dicts(messages: Sequence[Message]) -> list[dict[str, object]]:
    """把我们的 Message 列 history 转成接口要求的格式。

    除了"照抄",还要做原生工具调用的双向转换(见学习文档 §3):
    - assistant 消息里的 "@工具名(编号): {参数}" 行 → 转回接口的 tool_calls 字段
      (接口不认识我们的文本记法,要翻译回去);
    - 带 call_id 的 system 消息(工具结果)→ 转成接口的 tool 角色;
    - "孤儿"工具结果(编号找不到对应的调用,比如历史被裁过)→ 降级成
      user 消息发出去,因为接口要求每个 tool 消息必须有主,直接发会 400;
    - markdown 模式下这些情况都不存在,行为和以前完全一样;
    - 内容为空的消息直接跳过:有些接口遇到空内容会直接报错。

    示例(原生模式的一段历史转换前后):
        日志: assistant "@save(call_1): {...}"   →  {"role":"assistant","content":null,
                                                     "tool_calls":[{...,"id":"call_1"}]}
        日志: system "已创建 a.txt" + call_id    →  {"role":"tool","tool_call_id":"call_1",...}
    """
    import json as _json

    from my_agent_alpha.tools.base import ToolUse

    # 第一遍:收集所有出现过的调用编号(用来识别孤儿结果)
    call_ids = {
        tu.call_id
        for m in messages
        if m.role == "assistant"
        for tu in ToolUse.iter_from_content(m.content)
        if tu.call_id
    }

    out: list[dict[str, object]] = []
    for m in messages:
        if m.role == "system" and m.call_id:
            if m.call_id in call_ids:
                out.append({"role": "tool", "tool_call_id": m.call_id, "content": m.content})
            else:
                out.append({"role": "user", "content": m.content})
            continue
        if m.role == "assistant":
            calls = [tu for tu in ToolUse.iter_from_content(m.content) if tu.call_id]
            if calls:
                text = _strip_native_lines(m.content).strip()
                tool_calls = [
                    {
                        "id": tu.call_id,
                        "type": "function",
                        "function": {
                            "name": tu.tool.name,
                            "arguments": _json.dumps(tu.kwargs, ensure_ascii=False),
                        },
                    }
                    for tu in calls
                ]
                out.append({"role": "assistant", "content": text or None, "tool_calls": tool_calls})
                continue
        if not m.content.strip():
            continue
        out.append({"role": m.role, "content": m.content})
    return out


def _strip_native_lines(content: str) -> str:
    """把文本里的原生调用行(@工具名(编号): {参数})删掉,剩下的原样返回。

    示例:'保存一下\n@save(call_1): {"path": "a"}' → '保存一下\n'
    (围栏代码块里的 @xxx 是示例,不会被动到。)
    """
    from my_agent_alpha.codeblock import fenced_regions
    from my_agent_alpha.tools.base import _NATIVE_CALL_RE, find_json_end

    regions = fenced_regions(content)
    cuts: list[tuple[int, int]] = []
    for match in _NATIVE_CALL_RE.finditer(content):
        if any(a <= match.start() < b for a, b in regions):
            continue
        brace = match.end()
        while brace < len(content) and content[brace] in " \t":
            brace += 1
        if brace >= len(content) or content[brace] != "{":
            continue
        end = find_json_end(content, brace)
        if end > 0:
            cuts.append((match.start(), end))
    if not cuts:
        return content
    # 按区间从后往前删,避免位置变化
    parts = []
    cursor = 0
    for a, b in cuts:
        parts.append(content[cursor:a])
        cursor = b
    parts.append(content[cursor:])
    return "".join(parts)


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
        "input": getattr(raw, "prompt_tokens", 0) or 0,  # 输入量token
        "output": getattr(raw, "completion_tokens", 0) or 0,  # 输出量token
    }


def chat(
    messages: Sequence[Message],
    meta: ModelMeta,
    cfg: ProviderCfg,
    tools: Sequence[ToolSpec] = (),
) -> tuple[str, dict[str, int] | None]:
    """非流式调用:等模型把回复全部生成完,一次性返回 (回复文本, token 用量)。

    tools 非空时开启原生工具调用:把工具表发给接口;
    模型返回的 tool_calls 会翻译成 "@工具名(编号): {参数}" 行附在文本后
    (和流式路径的记法一致,后续解析/执行不用分情况)。
    遇到临时性错误(断网、限流等)自动重试,间隔越来越长(见 _backoff)。
    """
    last_err: BaseException | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            client = _make_client(cfg)
            create_kwargs: dict[str, Any] = {"model": meta.model, "messages": msgs2dicts(messages)}
            if tools:
                create_kwargs["tools"] = [spec_to_openai_tool(t) for t in tools if t.is_runnable]
            resp = client.chat.completions.create(**create_kwargs)
            choice = resp.choices[0]
            text = choice.message.content or ""
            for tc in getattr(choice.message, "tool_calls", None) or []:
                fn = tc.function
                text += f"\n@{fn.name}({tc.id}): {fn.arguments}"
            return text, _extract_usage(resp.usage)
        except Exception as e:
            if not _is_transient(e) or attempt == _MAX_RETRIES:
                raise
            last_err = e
            time.sleep(_backoff(attempt))
    raise RuntimeError(f"不可达:{last_err}")  # pragma: no cover(循环必 raise 或 return)


def _stream_once(
    messages: Sequence[Message],
    meta: ModelMeta,
    cfg: ProviderCfg,
    tools: Sequence[ToolSpec] = (),
) -> Generator[str, None, dict[str, int] | None]:
    """发起一次流式请求(不含重试逻辑)。

    模型的回复一小段一小段到达,每段立刻 yield 出去;
    思考内容(reasoning_content)在第一段到达前先发一个 "<think>" 标记,
    正文开始时发 "</think>",这样接收方就能区分思考和正文;
    原生工具调用开启时(tools 非空),模型的 tool_calls 分片在这里拼装,
    参数 JSON 一凑齐就产出 "@工具名(编号): {参数}" 行 —— 和思考/正文
    混在同一条文本流里,下游解析不用分情况;
    请求结束时把 token 用量作为生成器的返回值带回。
    """
    client = _make_client(cfg)
    create_kwargs: dict[str, Any] = {
        "model": meta.model,
        "messages": msgs2dicts(messages),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools: # 如果tools不为空，说明支持原生工具调用，要新增kwargs["tools"],内容是固定json scheme
        create_kwargs["tools"] = [spec_to_openai_tool(t) for t in tools if t.is_runnable]
    events = client.chat.completions.create(  # 正式发起http请求
        **create_kwargs
    )  # 开了流式，每次拿一部分
    """ 流式拿到的chunk(event)结构

        chunk                                   ← object = "chat.completion.chunk"
        ├── id / created / model                每个chunk都重复带(同一个请求的)
        ├── system_fingerprint
        ├── choices[] # 参数 n 代表生成几个回答,每个回答一个 choice,默认 n=1 即 choice[0]
        │   └── [0]
        │       ├── index        0
        │       ├── delta        **增量**:只有"新到"的字段
        │       │   ├── role            只在第一个事件出现("assistant")
        │       │   ├── content         新到的正文片段(平时最主要的内容)
        │       │   ├── reasoning_content  # 新到的思考片段(思考模型才有)
        │       │   └── tool_calls      # 原生工具调用的分片(见下方拼装逻辑)
        │       └── finish_reason  #结束原因，正常结束为stop
        └── usage """
    usage: dict[str, int] | None = None
    in_think = False

    # 原生调用分片的拼装处:index → 对应某一个调用，同一个调用的不同碎片，index相同
    # (id 和 name 只在第一片出现;arguments 是逐字符到达的,要拼)
    tool_acc: dict[int, dict[str, str]] = {}
    for event in events:  # 拿到一小个chunk
        if (
            getattr(event, "usage", None) is not None
        ):  # 在最后一次event里有usage(前提是请求参数带上include_usage)
            usage = _extract_usage(event.usage)  # 解析usage
        if not event.choices:  # 没有回答则continue
            continue
        delta = event.choices[0].delta  # 拿到部分回答（增量）
        reasoning = getattr(
            delta, "reasoning_content", None
        )  # 尝试拿到思考过程（只有思考模型才有）
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

        # 拿toolcall碎片tc，拼成完整的工具调用
            # 片1: index=0, id="call_1", function={name="save", arguments=""} ← 名字和编号在第一片
            # 片2: index=0, function={arguments='{"pa'}                       ← 参数开始逐字符挤牙膏
            # 片3: index=0, function={arguments='th": "a.t'}
            # 片4: index=0, function={arguments='xt", "content": "hi"}'}      ← 最后一片,大括号闭合
        for tc in getattr(delta, "tool_calls", None) or []:

            # acc：一个调用目前的状态
            acc = tool_acc.setdefault(tc.index, {"id": "", "name": "", "args": "", "done": ""})
            if tc.id:
                acc["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None and fn.name:
                acc["name"] = fn.name
            if fn is not None and fn.arguments:
                acc["args"] += fn.arguments

            # 检查一个调用是否拼装完成(大括号配对 + 能解析)就立刻产出调用行,
            args_ready = (
                acc["name"]
                and not acc["done"] # 还没yield
                and acc["args"].startswith("{") # 参数是个 JSON 对象
                and find_json_end(acc["args"], 0) == len(acc["args"]) # 大括号恰好闭合到末尾
            )
            if args_ready:
                try:
                    json.loads(acc["args"]) # json解析，出错就丢
                except json.JSONDecodeError:
                    continue
                yield f"@{acc['name']}({acc['id']}): {acc['args']}\n" # yield工具调用：str()
                acc["done"] = "1"

    if in_think:
        yield "</think>"
    return usage


def stream(
    messages: Sequence[Message],
    meta: ModelMeta,
    cfg: ProviderCfg,
    tools: Sequence[ToolSpec] = (),
) -> Generator[str, None, dict[str, int] | None]:
    """流式调用,外面包一层重试。

    规则:还没吐出任何内容时出错 → 等一会儿重试;
    已经吐出过内容后出错 → 直接抛出(内容没法收回,重试会导致重复)。
    tools 参数原样传给单次尝试(原生工具调用)。
    """
    for attempt in range(_MAX_RETRIES + 1):
        emitted = False
        try:
            gen = _stream_once(messages, meta, cfg, tools)
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
