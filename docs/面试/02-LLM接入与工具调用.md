# 面试复习:LLM 接入与工具调用

> 对应简历条目:"基于 OpenAI 兼容协议封装统一模型接入层,支持原生 Function Calling 与文本解析双模式工具调用;自主通用工具,并集成 MCP Client,实现外部工具的动态发现与调用。"
> 涉及代码:[llm/openai_compat.py](../../src/my_agent_alpha/llm/openai_compat.py)(361 行)、[llm/models.py](../../src/my_agent_alpha/llm/models.py)、[tools/base.py](../../src/my_agent_alpha/tools/base.py)(372 行)、[mcp_client.py](../../src/my_agent_alpha/mcp_client.py)(415 行)。

## 一、这条 bullet 里藏着 4 个可追问的声称

| 简历措辞 | 代码里对应什么 | 会被怎么问 |
|---|---|---|
| 统一模型接入层 | PROVIDERS 表 + 一个 OpenAI 兼容客户端 | "为什么一个客户端能接所有服务商?" |
| 原生/文本双模式工具调用 | 两条通道统一解析成 ToolUse | "为什么要两种模式?怎么统一的?"(**必问**) |
| 自主通用工具 | 8 个内置工具(shell/read/save/edit/grep/todo/memory/subagent) | "工具怎么定义的?" |
| MCP 动态发现与调用 | 远端工具转成本地 ToolSpec + 异步桥接 | "MCP 怎么集成的?"(深挖异步,见下) |

## 二、60 秒讲述模板

> 接入层的核心决定是:**所有服务商共用一个 OpenAI 兼容客户端**。deepseek、moonshot、智谱这些接口协议和 openai 一样,只是地址和密钥不同,所以一张 PROVIDERS 配置表就够了,换服务商就是换一行配置。
>
> 工具调用支持两条通道:**markdown 约定**(模型写代码块,语言标记就是工具名)和**原生 function calling**。关键设计是:原生调用到达时立刻翻译成一行文本记法 `@工具名(编号): {参数}`,和正文混在同一条流里。这样日志永远是纯文本、解析器只有一份、工具的执行端完全不用关心自己是被哪条通道调用的——N 个服务商 × M 个消费者,不会变成 N×M 份转换代码。
>
> 下一轮发回接口时再做反向翻译:文本调用行转回 tool_calls 字段,带编号的工具结果转回 tool 角色。
>
> MCP 的做法是把每个远端工具直接转成本地一等工具(命名"服务器.工具"),于是确认链、系统提示词、两条通道全部自动适用于远端工具,零特殊处理。难点是 MCP 的 SDK 是异步的而我的循环是同步的,我用"专属事件循环线程 + 一个常驻会话协程 + 队列投递"桥接。

## 三、高频追问与答法要点

### 1. "为什么一个客户端能接所有服务商?"

OpenAI 兼容协议是行业事实标准:国内主流服务商(deepseek/moonshot/dashscope/智谱)都提供和 openai 一模一样的接口,差异只有 base_url 和密钥环境变量。[PROVIDERS 表](../../src/my_agent_alpha/llm/models.py#L25-L49)就是 8 条这样的配置。

模型名解析约定([resolve_model](../../src/my_agent_alpha/llm/models.py#L68-L101)):"服务商/模型" 斜杠拆分;只写服务商用默认模型;**空字符串故意落到 mock**——防止手滑打到要花钱的真实接口;表里没有的服务商名保留原样,支持用环境变量 `ALPHA_<名字>_BASE_URL` 接自定义网关。

小细节:openai 库延迟导入——测试用 mock provider,完全不加载真实 SDK,启动更快。

### 2. "双模式怎么统一的?"(本模块最核心的设计)

两条通道,一个汇合点:

```
markdown 通道:  ```shell\nls -la\n```        ─┐
原生通道:      @save(call_1): {"path":...}   ─┴→ ToolUse.iter_from_content() → ToolUse → 同一个 execute()
```

- [ToolSpec](../../src/my_agent_alpha/tools/base.py#L109-L150) 是工具的说明书(名字/描述/示例/语言标记/参数表/执行函数);[ToolUse](../../src/my_agent_alpha/tools/base.py#L192-L359) 是从回复里解析出的一次调用。两种通道解析结果都是 ToolUse,只是参数装的位置不同(markdown 给 code/args,原生给 kwargs+call_id)。
- **为什么日志记文本行而不是结构化字段**:日志保持纯文本(人可读、解析器统一、崩溃恢复简单);原生调用只是"到达时翻译成文本记法",记法本身就是规范格式。

发回接口时的反向翻译([msgs2dicts](../../src/my_agent_alpha/llm/openai_compat.py#L62-L120)),三个细节每个都能讲:
- assistant 文本里的 `@工具名(编号): {参数}` 行 → 转回接口的 `tool_calls` 字段(接口不认识我们的文本记法);
- 带 call_id 的 system 消息(工具结果)→ 转成接口的 `tool` 角色;
- **孤儿工具结果降级**:编号在历史里找不到对应调用(比如上下文被裁剪过)→ 降级成 user 消息发出去。因为接口要求每个 tool 消息必须有主,直接发会 400。

### 3. "流式请求里,原生工具调用是怎么拼起来的?"

原生调用不是一次性到达的,是**分片挤牙膏**([_stream_once](../../src/my_agent_alpha/llm/openai_compat.py#L297-L326)):

```
片1: index=0, id="call_1", name="save", arguments=""   ← 编号和名字只在第一片
片2: index=0, arguments='{"pa'                          ← 参数逐字符到达
片3: index=0, arguments='th": "a.t'
片4: index=0, arguments='xt"}'                          ← 大括号闭合
```

按 index 聚合;什么时候算拼完?**三个条件同时满足**:大括号从开头恰好配对到末尾([find_json_end](../../src/my_agent_alpha/tools/base.py#L160-L189)——数配对括号且引号里的不算,防 `"{"path": "}"}"` 数错)、`json.loads` 校验通过、还没产出过。满足就立刻 yield 调用行,不等流结束——断点检测能更早打断。

两个真机踩坑(讲出来就是"真做过"的证据):
- **编号字符集必须允许点号**:dashscope/kimi 的编号长这样 `websearch.search_0_a475cfc6`(工具名_序号_哈希,工具名本身带点)。不许点号 → 整行正则失配 → 调用**静默丢弃**([_NATIVE_CALL_RE 注释](../../src/my_agent_alpha/tools/base.py#L153-L157),踩坑记录条目 5);
- **思考区粘连**:kimi 曾把工具调用粘在 `</think>` 后面没有换行,解析器看不见。解法:解析前先剥掉 `<think>` 区,粘在 `</think>` 后面的围栏剥完落到行首就看见了。

### 4. "重试策略是怎么设计的?"(三个细节,层层递进)

位置:[openai_compat.py](../../src/my_agent_alpha/llm/openai_compat.py#L30-L59) + [stream()](../../src/my_agent_alpha/llm/openai_compat.py#L333-L361)

1. **只重试"值得重试"的错误**:连接错误、限流、服务端 5xx(瞬时性);鉴权错、参数错立即抛——重试也没用,还浪费时间;
2. **SDK 自带重试归零**(`max_retries=0`):两层都在重试会相乘(3×3=9 次),限流时反而雪上加霜——重试预算由自己的 `_MAX_RETRIES=5` 独占,指数退避 1s/2s/4s…封顶 30 秒;
3. **流式吐过内容就不再重试**:内容已经 yield 出去收不回来,重试会导致用户看到重复文本。所以重试只发生在"一个字都还没吐"的阶段。

### 5. "MCP 怎么集成的?"(两层:设计层 + 异步桥接层)

**设计层**(一句话讲完):每个远端工具转成本地一等 ToolSpec,命名 `服务器名.工具名`,参数从服务器给的 JSON Schema 换算([_make_spec](../../src/my_agent_alpha/mcp_client.py#L236-L289))。于是确认链、只读放行、系统提示词、两条调用通道**全部自动适用**,不用为 MCP 写任何特殊分支。markdown 通道下模型看不到参数表,就按参数表**自动生成调用说明和可照抄的示例**。

**异步桥接层**(深挖时讲,这是最硬核的部分):

问题:MCP 的 Python SDK 是 asyncio 风格,我的 agent 循环是同步的;而且 stdio 传输的**取消作用域要求进出发生在同一个任务里**——不能每次调用临时起一个任务。

解法([SyncMCPClient](../../src/my_agent_alpha/mcp_client.py#L40-L153)):每个服务器一个**专属事件循环线程**,会话的连接/列举/调用/关闭全部发生在**同一个常驻协程任务**里;同步侧通过 `run_coroutine_threadsafe` 往队列投递操作单(类型/参数/回执 future/超时),协程从收件箱取单执行,结果写回 future。三个配套小件:
- `_ops` 队列 = 收件箱;
- `_ready` 事件 = 就绪门铃(后台连上服务器才创建收件箱,主线程等门铃,构造完成即可用);
- `_dead` = 失败遗言(连接失败的异常存这,主线程包成 RuntimeError 抛)。

**超时在"执行侧"取消**而不是主线程干等([_session_task](../../src/my_agent_alpha/mcp_client.py#L137-L150) 里的 `asyncio.wait_for`):到点真正取消这次调用。否则一次挂起的调用会永远占着常驻协程,堵死同一服务器的所有后续调用。主线程 `fut.result(timeout+10)` 只是兜底,10 秒余量留给取消收尾。

其他真机细节:
- **超时报错要翻译**:TimeoutError 的 `str()` 是空字符串,直接拼进"执行失败: {e}"就是一条空白消息,模型和用户都不知道发生了什么(踩坑记录条目 2);
- **env 要叠加不能替换**:MCP SDK 里 env 一旦给出就替换整个环境变量(连 PATH 都没了),必须把配置的少量变量**合并**到 os.environ 之上;
- **一个服务器连不上只告警跳过**,不拖垮整个会话;
- SDK 版本兼容:2.x 叫 `input_schema`,1.x 叫 `inputSchema`,两个都试。

### 6. "工具描述传给接口有什么坑?"

[spec_to_openai_tool](../../src/my_agent_alpha/tools/base.py#L85-L105):OpenAI 对函数描述有 **1024 字符上限,超长直接 400**。gptme 为此专门维护"精简版说明",我们先用截断到 1000 的简单方案。另外空内容消息直接跳过——有些接口遇到空 content 会报错。

## 四、数字备忘卡

- PROVIDERS 表 8 家:mock / openai / deepseek / moonshot / dashscope / glm / glm-coding / openrouter / local(Ollama)
- `_MAX_RETRIES=5`(环境变量可调),退避 2^n 封顶 30 秒,SDK 内部重试 = 0
- 工具描述截断 1000 字符(OpenAI 上限 1024)
- MCP:连接超时 60 秒(首次 npx 要下载包),单次调用超时默认 120 秒按服务器可配,主线程兜底 +10 秒
- 分片拼装判定:大括号配对到末尾 + json.loads 通过 + 未产出过
- 8 个内置工具;MCP 工具命名 `服务器名.工具名`

## 五、诚实缺口(主动说比被问出来好)

- **并行工具调用未做**:一次只执行一个调用,检测到第一个完整调用就打断(第一个模块讲过,这里是同一件事的接入侧);
- MCP 只做了 **tools 能力**,sampling、resources、prompts 没接;
- markdown 通道调 MCP 工具要求围栏内容是 JSON 参数体,靠自动生成说明缓解,但比原生通道多一层"格式写对"的负担;
- 工具描述截断是粗暴方案,没有 gptme 那种"精简版说明"。
