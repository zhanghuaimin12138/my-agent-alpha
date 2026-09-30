# gptme 架构研究报告

> 研究对象:[gptme/gptme](https://github.com/gptme/gptme) @ commit `f7bb348`(2026-09, v0.34.x)
> 研究方式:8 个并行 agent 深读源码(约 98 万 token 阅读量),本文为综合报告;每个子系统的详细报告(中文,含 file:line 引用)见 [docs/gptme-study/](gptme-study/) 目录(英文原版存于 git 历史)。
> 目的:理解 gptme 的架构,为仿造开发(`my_agent_alpha`)提供蓝图。

---

## 0. TL;DR

gptme 是一个"运行在任何有终端的地方"的个人 AI agent(2023 年春最早的 agent CLI 之一)。核心 Python 代码约 **18.7 万行** + React WebUI + Tauri 桌面端 + 26 个控制台入口。

它的全部复杂度都建立在 **三个核心设计决策** 上:

1. **一切皆消息,日志是唯一状态** —— 只有 3 种 role 的 `Message`;工具调用内嵌在 assistant 文本里,工具输出是带 `call_id` 的 system 消息;对话日志是 append-only 的 `conversation.jsonl`,崩溃可恢复、可回放、人类可读。
2. **LLM 层统一规范格式** —— 所有 provider 的流式输出都归一成纯文本:推理用 `<think>...</think>`,工具调用用 `@tool(call_id): {json}`。N 个 provider × M 个消费者不会变成 N×M 个转换器。
3. **一切横切关注点皆 Hook** —— steering(中途转向)、取消、自动回复、上下文压缩、工具确认、TTS……全部是"yield Message 的 generator hook",核心循环里没有一个 if 分支是为它们写的。

仿造的最小骨架 = **Message + JSONL 日志 + 三层循环 + ToolSpec 工具表 + 一个流式 reply()**。其余一切都是可以后加的层。

---

## 1. 项目定位与规模

| 维度 | 数据 |
|---|---|
| 定位 | 个人 AI agent,终端优先、provider 无关、本地优先 |
| 版本/规模 | v0.34.0;`gptme/` 包约 187k 行 Python |
| 核心依赖 | 极小:click、rich、tiktoken、tomlkit、platformdirs、**只有 openai + anthropic 两个 SDK**、mcp、ipython、tree-sitter |
| 可选依赖 | 16 个 extras(server/browser/tui/acp/sandbox/eval…) |
| 前端 | CLI(plain)、TUI(Textual)、HTTP Server(Flask+SSE)+ React WebUI、ACP(JSON-RPC over stdio,给 Zed 等编辑器)、A2A |
| 分发 | PyPI(26 个入口)、7 个 Docker 镜像、Tauri 桌面端(内嵌 PyInstaller 冻结的 server) |
| 内置工具 | ~40 个:shell、ipython、read/save/patch、subagent、browser、MCP、memory、todo、lessons… |

目录速览:

```
gptme/
├── chat.py            # ★ 核心循环(三层)
├── llm/               # ★ LLM 客户端层(provider 路由/流式/重试/模型元数据)
├── tools/             # ★ 工具系统(ToolSpec 注册表 + ~40 个工具)
├── message.py         # ★ 唯一的 Message 类型
├── logmanager/        # ★ 日志持久化(append-only JSONL + view/branch + WAL)
├── prompts/           # ★ 系统提示词组装
├── hooks/             # hook 注册表 + 确认链
├── context/           # 上下文选择器(可选)
├── memory/ lessons/   # 长期记忆 / 经验库
├── tools/autocompact/ # ★ 上下文压缩引擎
├── cli/ tui/ server/ acp/  # 四类前端
├── commands/ plugins/ config/ mcp/  # 扩展与配置
├── agent/             # ⚠ 命名陷阱:是 systemd/launchd 定时服务管理,不是 agent 循环!
└── checkpoint.py sandbox.py executor.py prompt_queue.py …
```

---

## 2. 三大架构支柱(灵魂)

### 支柱一:消息中心化 —— 日志即状态

`Message`(message.py:280,frozen dataclass)只有:

```python
role: 'system' | 'user' | 'assistant'   # 没有单独的 tool_call/tool_output role!
content: str
timestamp, files(附件), file_hashes, call_id
pinned(永不裁剪), hide(发给 LLM 但不打印), quiet(不打印), ephemeral_ttl
metadata(model/usage/cost/timings)
```

关键约定:
- **工具调用存在 assistant 的 content 里**(markdown codeblock / XML / `@name(call_id): {json}` 三种格式,由 `ToolUse.iter_from_content` 统一解析);
- **工具输出是 role='system' 的 Message,带 call_id** 与调用配对;
- 磁盘上每个会话是一个目录 `~/.local/share/gptme/logs/<YYYY-MMDD-slug>/`,主日志 `conversation.jsonl` **永不销毁**;压缩/分支写进 `views/`、`branches/`(双写);
- 写入用追加游标做 O(新增) 增量写,全量重写走 temp+fsync+os.replace 原子替换;
- `events.jsonl` 是 WAL(checkpoint 每 50 条),JSONL 损坏时可重放恢复。

**收益**:循环的继续判定 = "最后一条 assistant 消息是否还有可运行的 ToolUse"(chat.py:649),零额外状态;崩溃后扫日志尾部标记(INTERRUPT/DECLINED/LLM 失败)即可判断该不该重新提示用户;provider 格式变迁不影响磁盘格式(发送前才转换成各家 API 格式)。

### 支柱二:LLM 层统一规范格式

`reply()`(llm/__init__.py:362)是外界唯一入口;每个 provider 模块遵守同一个契约:

```python
chat(...)  -> tuple[str, MessageMetadata | None]                 # 非流式
stream(...) -> Generator[str, None, MessageMetadata | None]      # 流式,yield 纯文本块,return 元数据
```

- 流出的就是规范格式:`<think>...</think>` + `@tool(call_id): {json}`;
- **`_StreamWithMetadata`**(llm/__init__.py:648):包装 generator,从 `StopIteration.value` 捕获 return 的 usage/cost 元数据 —— 约 40 行,让流式的用量统计变得平凡;
- 下一轮回传时 `_handle_tools` 把文本里的工具调用重新转回各家原生 tool_calls/tool_result;
- 路由:只有 anthropic、openai-subscription、mock 有专属实现,**其余 14 个 provider 全走同一个 OpenAI 兼容客户端**;
- `ModelMeta` 静态字典(上下文窗口/价格/能力标志)+ "永不失败"的 `get_model()` 回退链(别名 → 去日期后缀 → 同族最近匹配 → 动态拉取 → 128k 兜底);
- 重试:SDK 内部重试**归零**,自家 11 次封顶指数退避(尊重 Retry-After),generator 只在未 yield 过时才重试 —— 防止两层重试相乘。

### 支柱三:Hook 体系

`HookType`(hooks/types.py)约 24 个生命周期点:`session.start/end`、`turn.pre/post`、`step.pre/post`、`generation.pre/post/chunk/interrupt`、`tool.execute.pre/post`、`tool.confirm`、`loop.continue`、`message.transform`……

- 标准 hook = **generator,直接 yield Message 注入对话**;`StopPropagation` 哨兵可中止后续 hook;
- `Hook(name, hook_type, func, priority)`,高优先级先跑,同名重注册即替换;
- **确认链是唯一例外**:TOOL_CONFIRM hook 返回 `ConfirmationResult(confirm|skip|edit)` 而非 yield,按优先级取第一个非 None(guardrails 200 > server_confirm 100 > cli_confirm 0 > 无 hook = 自主模式自动确认);
- 注册表本身、config、加载的工具集全部存 **ContextVar** —— 单进程多会话(server)与进程内 subagent 线程因此天然隔离。

---

## 3. 核心循环(逐链路)

三层嵌套,全部在 chat.py:

```
chat()                                  # :149 会话装配(init、LogManager.load、chdir、恢复输出格式)
└─ _run_chat_loop()                     # :324 外层 REPL
   ├─ drain 外部 prompt 队列(磁盘 JSONL + flock,支持其他进程注入)   # :338
   ├─ 取 prompt / prompt_toolkit 交互输入                              # :343/:415
   ├─ slash 命令 execute_cmd                                          # :349
   ├─ TURN_PRE hooks
   ├─ _process_message_conversation()   # :531 中层:一个用户 turn
   │  ├─ set_interruptible()(Ctrl-C 只在此区间生效)
   │  ├─ STEP_PRE hooks(subagent 在此排空 steer/取消)
   │  ├─ step()                         # :896 内层:一次"生成+执行"
   │  │  ├─ prepare_messages()          # 日志→提示词管线(见 §6)
   │  │  ├─ _reply_with_overflow_recovery()  # 上下文溢出→压缩→重试一次
   │  │  ├─ execute_msg()               # 解析 ToolUse 并逐个执行
   │  │  └─ yield assistant 消息 + 工具输出
   │  ├─ 追加结果、拒绝检查、turn 中压缩、GPTME_MAX_STEPS
   │  └─ 继续判定:最后一条 assistant 还有 runnable ToolUse? → 是则再 step()
   ├─ LOOP_CONTINUE hooks(auto-reply、subagent 完成通知)
   └─ 异常轨道:KeyboardInterrupt / 带来源标记的 provider 错误
```

**流式工具检测**(gptme 招牌 UX,llm/__init__.py:815/1056):
逐字符累积输出;每遇到格式相关的"断字符"(markdown 是 `\n`)就用 `streaming=True` 重新解析(codeblock.py:355 规定:**闭合 fence 后必须跟一个空行才算完整** —— 这是防半截命令被执行的关键);一旦解析出完整可运行工具调用 → 排干 generator(只为拿 usage 元数据)→ 中断返回 → 立即执行。`break_on_tooluse` 默认对不支持并行工具调用的模型开启(`supports_parallel_tool_calls` 决定)。

**中断语义**(util/interrupt.py):ContextVar 门 + SIGINT handler —— Ctrl-C 只在 step()/工具执行期间抛 KeyboardInterrupt,在用户输入时只打印提示;Ctrl-D 才退出会话。

**Provider 错误来源标记**:`mark_llm_reply_origin` 只给真正发生在 provider 调用内部的异常打标(llm/__init__.py:301),`is_provider_error` 据此区分"LLM 挂了(可恢复)"与"工具/hook 崩了(必须炸)"—— 防止 httpx 故障被吞成无限重试(issue #3668)。

**终止路径**共 8 种:非交互队列空 / EOF / 无可运行工具调用 / SessionCompleteException / MAX_STEPS / 用户拒绝 / Ctrl-C / 非交互 provider 错误。

---

## 4. 工具系统

`ToolSpec`(tools/base.py:467)是唯一声明单元,一份声明同时服务:模型文档(markdown/XML/原生三种格式渲染进 system prompt)、执行分发、确认策略、slash 命令、hook 注册、IPython 函数注入。

```python
ToolSpec(
  name, desc, instructions, examples,
  functions=[ToolFunction],            # 注册进 IPython 命名空间、不直接可执行(subagent/browser 用)
  execute(code, args, kwargs) -> Generator[Message],   # 统一入口
  block_types=['shell'],               # markdown langtag ↔ API 工具名
  parameters=[Parameter(name,type,desc)],
  available, disabled_by_default, requires_tools,      # 伴生依赖闭包(hashline_edit 需要 read)
  hints={'read-only','destructive','code-exec'},       # 供 allowlist 的 hint: 模式用
  hooks={...}, commands={...},         # 加载工具时自动注册/卸载时注销
)
```

- **发现**:扫描模块里的 `tool = ToolSpec(...)` 实例(env `TOOL_MODULES` + 插件模块 + .py 文件路径),MCP 服务器工具经 `create_mcp_tools` 转成 `server.tool` 命名的一等 ToolSpec;
- **解析**:`ToolUse.iter_from_content`(base.py:1082)统一解析三种格式(markdown fence / XML 两种方言 / `@name(call_id): {`+ 括号计数 + json_repair,且跳过 fenced code 内的误匹配);
- **执行纪律(call_id)**:`execute_msg`(tools/__init__.py:459)先物化全部 ToolUse → 每个结构化调用恰好配一个结果(不可运行的合成错误结果、中断时给"未执行"结果;多消息 generator 只在**最后一条**上盖 call_id)—— Responses API 严格要求 1:1 配对,这是防 400 的关键;
- **代表工具**:
  - `shell`(3701 行):持久 bash 子进程 + **每命令唯一哨兵标记**(`START/END_OF_COMMAND_OUTPUT_<ns>` + ReturnCode/PWD 十六进制尾车,输出无法伪造控制行)+ 状态快照恢复 + 超时梯队(杀子进程→宽限→killpg→重启)+ 前台 120s 软超时**晋升为后台任务**(不杀,注册成 BackgroundJob,结果由 step.pre hook 注回对话)+ token 感知截断(全量先存盘);
  - `python`(622 行):进程内 IPython REPL,capture 流式输出,把所有已加载工具的 functions 注册进命名空间 —— 模型可以在代码里循环/组合工具;
  - `subagent`:functions-only;`subagent(id, prompt)` 立即返回(fire-and-forget),三种后端(线程/子进程/ACP),完成通知走 hook,取消/转向走子进程 logdir 的 `control.jsonl` + STEP_PRE 检查点;预算与并发信号量封顶;
  - 安全梯(shell_validation.py):**允许标志白名单**(每个自动放行的二进制声明它允许的 flag,未知 flag 落到确认而非拦截,fail-closed 但不烦人)> 引号感知的 deny 正则(git add . / rm -rf / / pipe-to-bash…)> shellcheck > ~20 个只读命令 allowlist(全管道 + 无重定向 + 无敏感路径 + 无命令替换)。

---

## 5. LLM 客户端层(要点补充)

- `get_provider_from_model('provider/model')` → 路由;
- Anthropic 特有:thinking 签名以 `<!-- think-sig: ... -->` 注释存在 `<think>` 块里,下一轮重取出来放回签名 thinking 块(满足缺签名即 400 的规则);`apply_cache_control` 在 system、稳定边界、最后两条 user 上打 4 个 cache 断点;
- OpenAI 兼容路径是万能适配器:14 内置 + 自定义 base_url + entry-point 插件,一个 clients 字典按 Provider 分发;GPT-5 类模型走 Responses API 分支;
- 用量归一:4 种线格式(Chat Completions/Responses/OpenRouter/Codex SSE)一个函数统一,uncached = prompt − cache_read − cache_create;
- 订阅类(ChatGPT Plus/Codex、SuperGrok)与 gptme.ai 网关 = 3 套 OAuth,仿造时直接跳过;
- **mock provider**(llm_mock.py,~30 行):echo/static 脚本模型,让整个 agent 循环离线确定性可测 —— **从第一天就要抄**。

---

## 6. 日志 → 提示词管线 + 系统提示词

`prepare_messages`(logmanager/manager.py:1137)每次 LLM 调用前运行:

1. `_active_prompt_generation`:只保留最新一代生成的系统提示词(prompt_generation 标记);
2. `enrich_messages_with_context`:附件文件文本按内容寻址副本(`files/<sha256>`)内联为四反引号 codeblock —— **快照语义**:即使工作副本已改,提示词里仍是消息时刻的版本;
3. `prune_ephemeral_messages`:TTL 到期的消息丢弃 + 孤儿 tool 配对修复 + 相邻同 role 合并;
4. `limit_log`:硬截断兜底(理论上压缩应先触发,触发即告警)。

系统提示词 `get_prompt`(prompts/__init__.py:573)分三层:
- **core**(身份 + 工具文档)合并为一条;**cacheable**(workspace/agent 配置);**dynamic**(系统信息/时间/git 状态)最后;
- 中间插一条字面量消息 `SYSTEM_PROMPT_CACHE_BOUNDARY` —— 同时是 Anthropic cache 断点和合并屏障。**静态在前、易变在后 = 免费的 prompt cache 友好**;
- 所有提示词消息 pinned+hide,追加一次进日志。

---

## 7. 上下文工程(gptme 的护城河)

**预算**:`get_context_budget = min(0.9×window, window − max_output − headroom)` —— 压缩触发点 ≠ API 拒绝点,1M 窗口模型也在合理位置压缩。

**每 turn 后**(TURN_POST hook,60s 冷却):
`should_auto_compact` → 先做**节省率模拟**(估计可省 token ≥ 10% 才动 —— 压缩会打碎 prompt cache,小节省是净负):
- **规则路径** `auto_compact_log` 三阶段(engine.py,1:1 位置保持,字节范围引用才有效):
  1. 剥离距末尾 ≥5 条消息的 `<think>` 块;
  2. **最大优先**截断超 2000 token 的工具结果(全量存盘 + `[Content truncated] Master context: <path> (bytes a-b)` 恢复指针 —— 模型可以自己 grep 回来),直到达标;
  3. 对 >1000 token 的旧长 assistant 消息做抽取式压缩(句子打分:位置偏置 + 决策/结论/承诺/文件路径/错误 +2.0/1.5,代码块永保)至 70%;
- **LLM 路径** `_resume_via_llm`:生成结构化 RESUME.md(摘要/技术上下文/当前状态/建议上下文文件),重读文件(路径做 workspace 包含检查,防模型输出读 ~/.ssh);
- 输出写进**新 view 分支**,主日志不动;
- provider 真报上下文溢出时:`_reply_with_overflow_recovery` 压缩→重试一次→验证确实变小了→失败则回退 master view。

**其余机制**:
- **pruner**(可选):大工具输出捕获时让廉价模型按当前查询挑行区间,截断显示、全量存盘;
- **长期记忆**:分层根目录(explicit > project > ~/.claude 兼容 > agent > user),frontmatter markdown + MEMORY.md 指针索引(64KB 预算,兼容 Claude Code 格式),近层同名遮蔽,`supersede` 而非删除;模型经确认的 `memory` 工具写入;
- **lessons/skills**:gptme lessons、Anthropic SKILL.md、Cursor .mdc 统一解析成一个 Lesson;STEP_PRE 每步关键词匹配注入 top-5(50k token 预算;第一条强制包含,其余才计预算 —— 预算不对称,大赢家不会饿死其他人);效果分走 bandit 后验;
- **knowledge**:跨会话 JSONL 问题/解决对,会话开始一次性注入(哨兵去重);
- **todo**:内存字典 + **日志回放实现持久化**(SESSION_START 重放 write 子命令);关键在**完成门控**:turn 无工具调用时,自动回复变成"你还有未完成的 todo",连续 2 次无工具则熔断结束 —— agent 无法带着待办随便收工。

---

## 8. 界面层:一份契约,五个前端

核心契约只有:`step()` generator + `on_token/on_thinking` 回调。渲染 token 随你便,消费 yield 出的 Message 拿结构。

| 前端 | 技术要点 |
|---|---|
| CLI | click + prompt_toolkit + rich;main() 重度懒加载保 `--help` 快 |
| TUI | Textual;worker 线程跑 step(),`call_from_thread` 回 UI;确认=模态屏阻塞 worker;inline 模式借原生回滚 |
| Server | Flask 13 个 blueprint;**SSE**:`connected`(带 session_id+generating+pending_tools,断线重连可完整恢复)+ ping + 绝对索引环形缓冲(1k/10k 裁剪);**step_seq 纪元**做中断/并发控制(队列时取号,finally 里对号才清 generating);token 20 字符批量 ~20x 降事件量;确认 = pending_tools 挂起 + POST /tool/confirm(confirm/edit/skip/auto) |
| WebUI | React+Vite;一个 ApiClient 管 EventSource + 事件 switch;前端用 fence 正则重析工具卡(已知漂移债) |
| ACP | JSON-RPC over stdio 给 Zed;dup2 把 fd1 转发 stderr 保证协议纯净;确定性的 session id(cwd 派生)让编辑器重启即恢复会话 |

**ContextVar 快照过线程边界**是所有前端的公共动作(TUI `_chat_ctx.run`、server `ctx.run`、ACP `run_in_executor`)。

⚠ 已知的债:gptme 的 server 又手写了一遍 step()(session_step.py:741,~400 行)以获得逐消息事件控制 —— 仿造时应让 step() 支持可注入的 observer,避免这种分叉。

---

## 9. 扩展与配置

- **Slash 命令**:`@command` 装饰器 + `CommandContext(args, full_args, manager)`;命令注册表进程级、派发时检查 owner 工具是否在本会话加载(防多会话串);未知名回退成工具执行;
- **插件**:文件夹(tools/ hooks/ commands/ 子目录)、entry-point(`gptme.plugins` 组,**allowlist 在 ep.load() 之前检查**,禁用的重型插件零导入成本)、legacy provider 入口 —— 全部归一成 `GptmePlugin` 清单;
- **配置**:五层 TOML(operator runtime(只读) > user config.toml > config.local.toml(密钥) > project gptme.toml(+local)> 每会话 logdir/config.toml);MCP server 与 providers 按名合并;`get_env` 单函数决定 GPTME_X > X > chat > project > user;
- **TOFU 信任门**:项目配置里的 shell hook 命令集合做 sha256,存 project-trust.toml,非交互默认拒绝 —— 防克隆仓库投毒;
- **沙箱**(可选):firejail/bwrap/docker/wasmtime;**env 白名单刻意排除所有 provider API key**(LLM 调用发生在父进程,子进程泄露≠密钥泄露);
- **MCP 双向**:消费方向每远端工具 → `server.tool` ToolSpec(annotations→hints,全套确认机制免费适用);暴露方向 GptmeMCPServer 把自家工具给 Claude Desktop。同步代码用私有 asyncio loop + run_until_complete 桥接 async SDK,每 server 一个熔断器。

---

## 10. 工程实践(值得整段照搬)

1. **离线优先测试**:mock provider(echo/static)注册进正常 provider 表 + 无 key 时 `MODEL=local/test` 指向 `http://localhost:666`(保证误发真实请求会立刻失败);
2. **requires_api 契约**:收集时跳过 / 会话启动探测配额 / 运行中把配额错误改写成 skip —— 付费 API 永远不弄红 CI;
3. **conftest 当生产代码写**:restore_cwd、工具/Config 重置、线程泄漏快照对比(严格模式可 fail)、subagent 先中断退避再 join;
4. **AGENTS.md**:给 AI 开发代理的规矩 —— 永不 push master、显式 stage(禁 `git add .`)、Conventional Commits、"优先集成测试而非重度 mock"、六词术语表(Tool/ToolUse/Message/LogManager/Step/Turn)、"加一个工具 = 建 gptme/tools/x.py + 对应测试,自动被发现";
5. 文档每页强制 `:audience:` 元数据,缺失即构建失败;
6. 发布:CI 绿灯才准发;changelog 从 tag 区间生成;PyPI 最后(唯一非幂等步)。

---

## 11. 值得直接照搬的技巧清单(精华 20 条)

1. `Message` 三 role + call_id 约定(工具调用在文本里、输出是 system 消息)—— 磁盘格式 provider 无关;
2. 继续判定 = "最后一条 assistant 有无 runnable ToolUse" —— 零状态驱动循环;
3. streaming codeblock 完整性规则:闭合 fence 后必须跟空行才执行;
4. `_StreamWithMetadata` 用 `StopIteration.value` 捕获流式 usage;
5. call_id 纪律:每调用恰好一个结果(错误合成、中断排干、最后一条才盖 id);
6. provider 错误来源标记(只有 provider 调用内的异常算"可恢复 LLM 故障");
7. SDK 重试归零 + 自家封顶退避 + generator 未 yield 才重试;
8. 中断 ContextVar 门(Ctrl-C 只在 agent 工作时生效);
9. 追加游标 O(new) 写 + temp+fsync+replace 原子全量重写;
10. 主日志永不销毁,压缩写 view(可逆、可审计);
11. 截断前先把全量存盘,再给模型恢复路径(saves-then-prunes);
12. 位置保持压缩 + 字节范围恢复指针(模型能自己 grep 回原文);
13. 节省率 ≥10% 才压缩(prompt cache 经济学);
14. 静态提示词在前、时间/git 状态最后 + 显式 cache 边界标记;
15. 附件内容寻址存储 = 消息时刻快照;
16. 前台超时晋升为后台任务而非杀掉;
17. 持久 shell 哨兵协议防输出伪造 + 状态快照恢复;
18. 确认 = 优先级 hook 链(confirm/skip/**edit**),无 hook 即自主模式;
19. todo 完成门控 + 2 次熔断(agent 不能带着待办收工);
20. ContextVar 一切(配置/工具/hook/确认状态)+ 线程边界快照运行。

---

## 12. 仿造路线图(推荐)

> 原则:先让"消息→生成→工具→追加"这条脊柱跑起来,再逐层加硬。每阶段结束都有可运行的东西。

| 阶段 | 内容 | 验收 |
|---|---|---|
| **0 骨架** | `Message` dataclass + 单文件 JSONL `Log/LogManager`(append/load/原子写);项目骨架(pyproject、ruff、mypy、pytest、mock provider) | 日志能追加、重启能恢复 |
| **1 脊柱** | `step()`(非流式)+ markdown fence 工具解析 + `execute_msg`;中层循环(继续判定)+ 外层 REPL(prompt_toolkit);slash 命令骨架 | CLI 里 agent 能多步用 shell/read/save 完成任务,`--non-interactive` 能跑完退出 |
| **2 流式** | provider `stream()` 契约 + `_StreamWithMetadata` + break_on_tooluse(逐字符累积+换行重析) | 看着 agent 一个工具一个工具地干活 |
| **3 原生工具调用** | `parameters2dict` + 每家 `_spec2tool`;call_id 配对纪律;`supports_parallel_tool_calls` | 同一 ToolSpec 双格式可用 |
| **4 确认与安全** | TOOL_CONFIRM hook 链(confirm/skip/edit)+ CLI 提示 hook;deny 正则 → 只读 allowlist → shellcheck → 标志白名单(按此顺序) | `rm -rf` 必被拦,`ls` 不打扰 |
| **5 Hook/命令/配置** | Hook 注册表(session/turn/step/generation/tool 七八个点)+ 分层 TOML + `get_env` | 新功能不再改循环 |
| **6 上下文工程** | token 计数 + 预算 + 三阶段规则压缩 + saves-then-prunes + 溢出自愈 | 100+ turn 会话稳定运行 |
| **7 Server + 前端** | Flask 蓝图(append/step/events SSE)+ step_seq 纪元 + pending_tools 确认流;可选 React WebUI | 浏览器里用 |
| **8 高级** | subagent(子进程模式先行)、MCP 客户端、记忆/lessons、TUI | 按需 |

每阶段配套:该层的单测(mock provider 驱动)+ 一条真实 e2e 冒烟。

## 13. 明确不要照搬的

- `gptme/agent/` 的命名(那是 systemd/launchd 定时服务管理,循环在 chat.py);
- server 端重复实现 step() 的分叉 —— 一开始就设计可注入 observer;
- 前端重析工具调用的漂移(让 server 事件携带解析好的结构);
- 订阅 OAuth 三件套、A2A、Tauri、7 个 Dockerfile、eval 矩阵、26 入口 —— 全部推迟到核心稳定;
- view/branch/WAL 机制 MVP 阶段先跳过(先"只截断提示词、不动磁盘");
- tree-sitter 命令切分、stdin `< /dev/null` AST 重写等 shell 硬化,逐层后加。

---

## 14. 附录:子系统深读报告

| 文件 | 覆盖 |
|---|---|
| [gptme-study/agent-loop.md](gptme-study/agent-loop.md) | 三层循环逐行链路、流式检测、中断、终止路径 |
| [gptme-study/tools-system.md](gptme-study/tools-system.md) | ToolSpec/ToolUse、注册表、shell/python/subagent/MCP |
| [gptme-study/llm-layer.md](gptme-study/llm-layer.md) | provider 路由、规范格式、重试、ModelMeta |
| [gptme-study/context-logmanager.md](gptme-study/context-logmanager.md) | Message/Log/LogManager、prepare_messages、系统提示词 |
| [gptme-study/context-engine.md](gptme-study/context-engine.md) | 压缩引擎、记忆、lessons、todo、anti-slop |
| [gptme-study/interfaces.md](gptme-study/interfaces.md) | CLI/TUI/Server/WebUI/ACP 五前端 |
| [gptme-study/extensibility.md](gptme-study/extensibility.md) | 命令/hooks/插件/配置/MCP/沙箱/信任 |
| [gptme-study/project-meta.md](gptme-study/project-meta.md) | 依赖、测试策略、打包分发、AGENTS.md |

源码本地副本:`/tmp/gptme`(浅克隆,报告中的 file:line 均指向它)。
