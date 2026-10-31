# 学习文档 01 · 大模型 API 与 SDK 系统学习

> 范围:**调用大模型 API 这一层的完整知识** —— client 是什么、请求能传什么、
> 响应和流式事件里有什么字段、有哪些我们没有用到的能力、错误和限流怎么处理。
> 只讲这一层,不讲本项目 agent 循环(那部分见 devlog 和 gptme-study)。
> 以 OpenAI 的 Chat Completions 接口为纲(事实标准),标注各家兼容厂商的差异。
> 项目代码对照:src/my_agent_alpha/llm/openai_compat.py、models.py。

---

## 目录

1. [client:它到底是个什么东西](#1-client)
2. [请求参数总表:除了 model/messages 还能传什么](#2-请求参数总表)
3. [messages 的完整形态(比你想的复杂)](#3-messages-的完整形态)
4. [响应对象全解:每个字段](#4-响应对象全解)
5. [流式事件全解:每个字段](#5-流式事件全解)
6. [错误体系与限流](#6-错误体系与限流)
7. [没有用到但值得知道的能力](#7-没有用到但值得知道的能力)
8. [SDK 的其他端点](#8-sdk-的其他端点)
9. [动手实验清单](#9-动手实验清单)
10. [术语速查表](#10-术语速查表)

---

## 1. client

### 1.1 它是什么

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...", base_url="https://api.deepseek.com")
```

`client` **不是一个"连接"**,而是一个"带着配置的工具箱":
它内部保存了密钥、接口地址、超时、重试策略,并维护一组可以复用的
HTTP 连接(连接池 —— 连 TCP 握手的时间都帮你省了)。
每次调 `client.chat.completions.create(...)` 才真正发一次请求;
同一个 client 可以反复用,**线程安全,推荐全局建一个**。

(我们代码里每次都新建 client —— 功能正确但略浪费,见第 9 节实验 1。)

### 1.2 构造参数

| 参数 | 作用 | 默认 |
|---|---|---|
| `api_key` | 密钥。不传则自动读环境变量 `OPENAI_API_KEY` | 环境变量 |
| `base_url` | 接口地址。**换服务商就换它** | OpenAI 官方 |
| `timeout` | 单次请求超时(秒),也可传 `httpx.Timeout` 细分连接/读取 | 600s |
| `max_retries` | SDK 内置自动重试次数(见第 6 节;**我们设 0,自己管**) | 2 |

### 1.3 同款兄弟

- `AsyncOpenAI`:异步版,`await client.chat.completions.create(...)`,
  高并发场景(server 同时服务多会话)用;方法名相同;
- `client.with_options(timeout=30, max_retries=0)`:复制出一个改了配置的
  新 client,原 client 不受影响 —— 想给某次调用单独设超时时很方便。

---

## 2. 请求参数总表

`client.chat.completions.create(...)` 的全部常用参数,按用途分组。
**粗体 = 我们项目当前在用**。

### 2.1 必传

| 参数 | 说明 |
|---|---|
| **`model`** | 模型名,如 `"glm-5.3"`、`"deepseek-chat"` |
| **`messages`** | 对话历史列表(见第 3 节) |

### 2.2 控制生成行为(生成参数)

| 参数 | 取值 | 作用 |
|---|---|---|
| `temperature` | 0 ~ 2,默认 1 | 越低越确定/保守,越高越发散。0 ≈ 每次都挑最高概率词 |
| `top_p` | 0 ~ 1,默认 1 | 只从概率前 p 的词里采样。和 temperature 同类旋钮,**建议只调其一** |
| `max_tokens` | 正整数 | 回答最多生成多少 token(到了就停,finish_reason=length) |
| `n` | 默认 1 | 一次生成几个候选回答(choices 列表会有 n 个)—— 贵 n 倍,慎用 |
| `stop` | 字符串或列表(≤4 个) | 生成到这些文本就停(不包含它们) |
| `seed` | 整数 | 固定随机种子,配合 temperature>0 可复现结果(不保证完全一致) |
| `presence_penalty` | -2 ~ 2 | 正值鼓励谈新话题(出现了的词更难再出现) |
| `frequency_penalty` | -2 ~ 2 | 正值惩罚重复(按出现次数) |
| `logprobs` + `top_logprobs` | | 返回每个位置 top-N 候选词的概率(研究/分类置信度用) |
| `logit_bias` | {token_id: 分数} | 强行提高/压低特定词的概率(很少用) |

> 思考型模型(DeepSeek-R、GLM 思考模式)通常**忽略** temperature/top_p —— 思考过程不随机。

### 2.3 结构化输出

| 参数 | 作用 |
|---|---|
| `response_format={"type": "json_object"}` | 强制回复是合法 JSON(提示词里通常必须出现"json"字样) |
| `response_format={"type": "json_schema", "json_schema": {...}}` | 更强:回复必须符合你给的 JSON Schema(OpenAI 的 Structured Outputs) |

> 我们项目以后做"让模型按固定格式交作业"时用它(比如 subagent 的返回结构)。
> 兼容性:OpenAI/GLM/DeepSeek 支持 json_object,json_schema 各家进度不一,用前查文档。

### 2.4 原生工具调用(我们阶段 3 的主角,先混个脸熟)

| 参数 | 作用 |
|---|---|
| `tools=[{"type": "function", "function": {"name", "description", "parameters": <JSON Schema>}}]` | 告诉模型有哪些函数可调 |
| `tool_choice` | `"auto"`(默认,模型自己决定)/ `"none"`(禁用)/ `"required"`(必须调)/ 指定某个函数 |
| `parallel_tool_calls` | 是否允许一次回复里调多个工具(默认 true) |

模型不会"执行"任何东西 —— 它只是输出一段**结构化的调用请求**
(见第 4 节的 `tool_calls` 字段),执行永远是我们的活。

### 2.5 流式与其他

| 参数 | 作用 |
|---|---|
| **`stream=True`** | 开流式(见第 5 节) |
| **`stream_options={"include_usage": True}`** | 流式的最后多发一个账单事件 |
| `user` | 标识最终用户身份(服务商端风控/统计用) |
| `extra_body={...}` | 传非标参数给兼容厂商(各家私货都从这里塞,如 GLM 的思考开关) |

---

## 3. messages 的完整形态

### 3.1 role 一共有四种(不是三种!)

接口层的 role 比**我们日志里的三种**多一种:

| role | 谁的 | 备注 |
|---|---|---|
| `system` | 程序的指令/说明 | 兼容厂商通用;OpenAI 新版另有个 `developer` 等价物 |
| `user` | 用户 | |
| `assistant` | 模型 | 历史里的模型回复(**含工具调用记录**,见 3.3) |
| `tool` | **工具执行结果** | 原生工具调用的专用角色,带 `tool_call_id` |

> 我们项目把工具输出记成 system 消息(markdown 格式);
> 走原生工具调用时就必须转成 `tool` role —— 这是阶段 3 的改造点之一。

### 3.2 content 可以不只是字符串(多模态)

```json
{"role": "user", "content": [
    {"type": "text", "text": "这张图里有什么?"},
    {"type": "image_url", "image_url": {"url": "https://... 或 data:image/png;base64,....",
                                        "detail": "low"}}
]}
```

`content` 为数组时叫**多模态分段(content parts)**:文本段 + 图片段
(网址或 base64)+ 音频段。视觉能力就是从这里接的(gptme 的 vision 工具
本质就是把截图 base64 塞进 image_url)。

### 3.3 assistant 消息的特殊字段(原生工具调用的历史记录)

```json
{"role": "assistant", "content": null,
 "tool_calls": [{"id": "call_abc123", "type": "function",
                  "function": {"name": "get_weather", "arguments": "{\"city\": \"北京\"}"}}]}
```

注意两个坑:
- `content` 可以是 `null`(模型调工具时常常不说话);
- `arguments` 是**JSON 字符串**,不是对象 —— 传回去前要自己 `json.loads`。

### 3.4 tool 消息

```json
{"role": "tool", "tool_call_id": "call_abc123", "content": "晴,25 度"}
```

`tool_call_id` 必须和 assistant 里的 `id` 配对 ——
**每个调用必须恰好配一个结果**,多一个少一个接口直接 400
(gptme 的"call_id 纪律"就是防这个,我们已写进 AGENTS.md)。

---

## 4. 响应对象全解

非流式调用的返回值,`resp = client.chat.completions.create(...)`,
把它 `resp.model_dump()` 打印出来对照:

```
resp
├── id                     本次请求唯一编号(报错给客服就报它)
├── object                 "chat.completion"(流式是 "chat.completion.chunk")
├── created                时间戳(unix 秒)
├── model                  实际用的模型(有时带版本后缀)
├── system_fingerprint     模型权重版本指纹(OpenAI 用;复现实验时记录它)
├── service_tier           OpenAI 的服务档位(auto/default/flex),可忽略
├── choices                候选列表(n=1 时就一个元素)
│   └── [0]
│       ├── index          0
│       ├── message
│       │   ├── role            "assistant"
│       │   ├── content         回复正文;**调工具时可能是 null!**
│       │   ├── tool_calls      原生工具调用请求(见 3.3,同一结构)
│       │   ├── reasoning_content  思考过程(DeepSeek/GLM 等兼容厂商的字段,
│       │   │                     OpenAI 官方没有 —— 我们用 getattr 安全取)
│       │   └── annotations     引用来源(OpenAI 联网搜索时出现)
│       ├── finish_reason   为什么停(见下表)
│       └── logprobs        概率信息(请求了 logprobs 才有)
└── usage                  账单(比我们用的两个字段多)
    ├── prompt_tokens              发过去的总量
    ├── completion_tokens          生成的总量
    ├── total_tokens               合计
    ├── prompt_tokens_details
    │   ├── cached_tokens          **命中缓存的量**(见 7.1,省钱关键指标)
    │   ├── audio_tokens / image_tokens   多模态各占多少
    └── completion_tokens_details
        └── reasoning_tokens       思考型模型"想"用了多少 token(也在计费!)
```

**finish_reason 取值**(重要,决定程序下一步):

| 值 | 含义 | 程序该做什么 |
|---|---|---|
| `stop` | 正常说完 | 结束本轮 |
| `length` | 撞上 max_tokens 被截断 | 续写或调大上限 |
| `tool_calls` | 模型要求调工具 | 执行工具,把结果用 tool role 发回去 |
| `content_filter` | 内容被安全过滤 | 换个说法重试 |

---

## 5. 流式事件全解

`stream=True` 时 create 返回一个**可迭代的事件流**,`for chunk in ...` 逐个消费。
底下是 SSE(每事件一行 `data: {...}`),SDK 已帮你解析成对象。

### 5.1 单个 chunk 的完整结构

```
chunk                                   ← object = "chat.completion.chunk"
├── id / created / model                每个chunk都重复带(同一个请求的)
├── system_fingerprint
├── choices[]                           空列表 = 账单事件(见 5.3)
│   └── [0]
│       ├── index        0
│       ├── delta        **增量**:只有"新到"的字段
│       │   ├── role            只在第一个事件出现("assistant")
│       │   ├── content         新到的正文片段(平时最主要的内容)
│       │   ├── reasoning_content  新到的思考片段(兼容厂商思考模型)
│       │   └── tool_calls      原生工具调用的**分片**(见 5.2,阶段 3 的坑)
│       └── finish_reason  生成中途:null;正文的最后一个事件:stop/length/...
└── usage               只在最后的账单事件里有(需要 include_usage)
```

### 5.2 tool_calls 分片流(阶段 3 最重要的冷知识)

原生工具调用在流式里**不是一个事件给全的**,而是:

```
delta.tool_calls = [{"index": 0, "id": "call_abc", "function": {"name": "get_weather", "arguments": ""}}]
delta.tool_calls = [{"index": 0, "function": {"arguments": "{\"ci"}}]
delta.tool_calls = [{"index": 0, "function": {"arguments": "ty\": \"北京\"}"}}]
```

规则:
- `id` 和 `name` 只在第一片出现;
- `arguments` 的 JSON 文本**逐字符地分片到达**,必须自己按 `index` 拼接,
  拼完了再 `json.loads`(gptme 的括号计数 + json_repair 就是伺候这个);
- 并行多个工具调用时,靠 `index` 区分是第几个调用。

### 5.3 事件时间线(完整一轮长什么样)

```
1. delta={"role":"assistant"}                      ← 开场报身份
2. delta={"reasoning_content":"思考..."} × N        ← (思考模型才有)
3. delta={"content":"你"} ... × N                   ← 正文片段
4. delta={}, finish_reason="stop"                   ← 正文结束
5. choices=[], usage={...}                          ← 账单(需 include_usage)
```

我们的 `_stream_once` 对这条线的处理:1 跳过(role 没内容)、2 包 `<think>`、
3 直出、4 不特殊处理、5 存 usage。

### 5.4 流式特有的注意事项

- 断开/异常:流读到一半网络断了会抛 `APIConnectionError`,**已读的部分
  不会退钱也不会重传** —— 所以"已 yield 过内容就别重试"(我们代码的实现);
- 中途取消:直接 `break`/close 生成器,SDK 会关闭连接;
  服务商按已生成的部分计费(我们的断点检测正是利用这一点省钱);
- `usage` 里同样有 `cached_tokens` 等细节字段,账单事件一次性给全。

---

## 6. 错误体系与限流

### 6.1 SDK 异常类族谱(都在 `openai` 包里)

| 异常 | 触发场景 | 该重试吗 |
|---|---|---|
| `APIConnectionError` | 网络不通/DNS/超时 | ✅ |
| `APIStatusError`(带 `.status_code`) | 服务端返回了错误码 | 看码 |
| ├ `BadRequestError` 400 | 参数错了(消息格式、非法字段) | ❌ 改代码 |
| ├ `AuthenticationError` 401 | 密钥错/没传 | ❌ 改密钥 |
| ├ `PermissionDeniedError` 403 | 密钥没权限/欠费 | ❌ |
| ├ `NotFoundError` 404 | **模型名写错**/地址错 | ❌ |
| ├ `UnprocessableEntityError` 422 | 参数语义错误 | ❌ |
| ├ `RateLimitError` 429 | **限流**:请求太频/额度用尽 | ✅(等) |
| └ `InternalServerError` 5xx | 服务商内部故障 | ✅ |

排查口诀:**看类型 + 看 `.status_code`**,4xx 是自己的错,5xx/网络是对方的错。

### 6.2 限流(429)的细节

- 响应头 `Retry-After: 5` 表示"5 秒后再来",重试逻辑应尊重它;
- 限流维度通常是:每分钟请求数(RPM)+ 每分钟 token 数(TPM),按你的账户等级定;
- 正确姿势:指数退避(1s→2s→4s…)+ 尊重 Retry-After + 封顶次数。
  这正是我们 `openai_compat.py` 里 `_backoff()`/`_is_transient()` 做的事。

### 6.3 SDK 内置重试(以及为什么我们关掉它)

`OpenAI(max_retries=N)` 会对连接错误/408/429/5xx 自动重试,默认 2 次。
听起来很好,但如果**你自己的代码也重试**,两层相乘:3×3=9 次请求,
限流时火上浇油。所以要么全交给 SDK,要么 `max_retries=0` 自己管 ——
我们选了后者(gptme 同款决策,devlog 003 有记录)。

---

## 7. 没有用到但值得知道的能力

### 7.1 提示词缓存(prompt caching)—— 省钱大杀器

你的每次请求都全量重发历史(模型无记忆),但服务商发现**前缀和上次一模一样**
时,可以只按 1/10 ~ 1/4 的价格收"缓存命中"的部分。`usage.prompt_tokens_details.cached_tokens`
就是命中量。

要点:**把不变的内容放前面、易变的放后面**,命中率才高 ——
这正是我们系统提示词"静态在前、时间在后"设计的原因;
也是 gptme 里 `SYSTEM_PROMPT_CACHE_BOUNDARY` 那条"分界线消息"的意义。
各家策略:OpenAI 自动缓存(>1024 token 前缀);DeepSeek 自动硬盘缓存(极便宜);
GLM/其他兼容厂商大多自动,无需参数。

### 7.2 多候选(n)与 logprobs

`n=3` 一次拿 3 个回答(choices 有 3 个)做挑选/投票;
`logprobs+top_logprobs` 看模型每个位置的候选词概率 ——
判断"模型有多确定",做分类置信度、蒸馏数据时有用。

### 7.3 视觉(图片输入)

见 3.2 的 image_url。给 agent 加"看屏幕/看图"能力(gptme 的 vision/screenshot 工具)
全靠它。注意:图片按 tile 计费,detail="low" 便宜很多。

### 7.4 批处理(Batch API)

把几千条请求写进文件提交,服务商在 24 小时内跑完,**半价**。
适合离线评测/数据处理,不适合交互。

### 7.5 Embeddings(向量化)

`client.embeddings.create(model=..., input=[...])` 把文本变成向量,
用于语义搜索/RAG/去重。和我们 chat 这条线是**另一个端点**。
gptme 的知识库/lessons 检索就是在这一层(研究报告 §7)。

---

## 8. SDK 的其他端点

`client.` 下面还有一堆,和 `chat.completions` 平级:

| 端点 | 干什么 | 兼容厂商支持度 |
|---|---|---|
| `client.models.list()` | **拉取该服务商的模型清单** | 几乎都支持(免费、不耗 token) |
| `client.embeddings.create()` | 文本→向量 | 大多支持 |
| `client.responses.create()` | OpenAI 新一代接口(状态机式,工具更丰富) | 基本只有 OpenAI 自己 |
| `client.images.generate()` | 文生图 | OpenAI/部分厂商 |
| `client.audio.transcriptions.create()` | 语音转文字 | 部分 |
| `client.batches` / `client.files` / `client.fine_tuning` | 批处理/文件/微调 | 少 |

对项目最有用的是 `models.list()`:以后可以做"动态模型表",不用手动维护
PROVIDERS 里的默认模型名(gptme 的 get_available_models 就是这么实现的)。

---

## 9. 动手实验清单

环境统一:cd 到项目目录,先 `export GLM_API_KEY=...`,脚本用
`uv run --project /home/yby/projects/my_agent_alpha python 脚本.py` 跑
(能复用已装的 openai 库)。

**实验 1:看清 client 本体**
```python
from openai import OpenAI
import os

c = OpenAI(api_key=os.environ["GLM_API_KEY"], base_url="https://open.bigmodel.cn/api/paas/v4")
print(type(c))  # 一个普通的 Python 对象
print(c.base_url)  # 配置就存在它身上
for m in c.models.list():  # 白嫖一次接口:拉模型清单
    print(m.id)
```

**实验 2:打印完整响应,对照第 4 节的树**
```python
resp = c.chat.completions.create(
    model="glm-5.3", messages=[{"role": "user", "content": "用一句话介绍自己"}]
)
import json

print(json.dumps(resp.model_dump(), ensure_ascii=False, indent=2))
```
逐个找到:id/created/model/choices[0].message.content/finish_reason/usage。

**实验 3:temperature 的体感**
同一句"起一个中文网名"分别用 `temperature=0` 和 `temperature=1.6` 各跑 3 次,
对比稳定性。(顺便体会:为什么写代码类任务大家都用低 temperature。)

**实验 4:结构化输出**
`response_format={"type":"json_object"}`,提示词里带"json"字样
(不带会 400),拿到纯 JSON 回复;试着加 `max_tokens=20` 触发 finish_reason="length"。

**实验 5:错误体感(不用花一分钱)**
```python
c2 = OpenAI(api_key="sk-错的", base_url="https://open.bigmodel.cn/api/paas/v4")
c2.chat.completions.create(model="glm-5.3", messages=[{"role": "user", "content": "hi"}])
```
看异常类型 `AuthenticationError` 和 `.status_code == 401`;
再试不存在的模型名 → `NotFoundError`;再试 `base_url` 写个不存在的端口,
对比 `APIConnectionError` 的样子。

**实验 6:流式全字段 dump(对照第 5 节)**
把实验 2 加上 `stream=True, stream_options={"include_usage": True}`,
循环里 `print(e.model_dump())`,亲眼看 role 开场/正文分片/finish_reason/空 choices 账单事件。

**实验 7:n 与 stop**
`n=2` 看 choices 出现两个;`stop=["。"]` 看回复在句号前被掐断。

**实验 8:缓存命中观察(DeepSeek 最明显)**
同一个长提示词连续发两次,对比第二次 `usage.prompt_tokens_details.cached_tokens`
> 0(DeepSeek 自动缓存,效果最直观)。

---

## 10. 术语速查表

| 术语 | 一句话 |
|---|---|
| client | 带配置(密钥/地址/超时/重试)+ 连接池的请求工具箱,可复用、线程安全 |
| base_url | 接口地址,换服务商换它 |
| endpoint(端点) | client 下的一类功能:chat.completions、embeddings、models… |
| Chat Completions | 事实标准的对话接口,本文档的主角 |
| Responses API | OpenAI 新一代接口,兼容厂商基本不支持 |
| temperature / top_p | 随机性旋钮,二选一调 |
| max_tokens | 回答长度上限,撞线 finish_reason=length |
| n | 一次要几个候选 |
| stop | 提前停止的标志文本 |
| response_format | 强制 JSON 输出(含 JSON Schema 严格模式) |
| tools / tool_choice | 原生工具调用:把函数清单告诉模型/控制它调不调 |
| tool role / tool_call_id | 工具结果专用角色 + 一一配对的 id |
| content parts | content 数组形态,可混排文本/图片/音频 |
| finish_reason | 停止原因:stop/length/tool_calls/content_filter |
| delta | 流式事件里"新增的那点内容" |
| chunk | 流式事件的对象类型(object=chat.completion.chunk) |
| SSE | 服务器持续推送小段数据的协议,流式的底层 |
| usage / cached_tokens | 账单 / 缓存命中量(省钱指标) |
| reasoning_tokens | 思考型模型的思考部分计费量 |
| RateLimitError 429 | 限流;看 Retry-After 头决定等多久 |
| APIConnectionError | 网络层失败(没连上/断了) |
| prompt caching | 相同前缀打折计费;静态在前、易变在后 |
| embeddings | 文本→向量,语义检索用 |
| Batch API | 文件式批处理,半价,24h 内 |

---

*写于阶段 2 结束后;阶段 3(原生工具调用)会大量用到第 3.3、5.2 节的内容 —— 到时候回来翻。*
