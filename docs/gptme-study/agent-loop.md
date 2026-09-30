> 译自英文原版(见 git 历史 aaf0543),由 Claude 翻译

# agent-loop

## 目的(Purpose)
gptme 的 agent 循环(agent loop)是一个三层嵌套循环,它把终端聊天变成一个 agentic 系统:外层 REPL 循环(获取用户输入 / 排空排队的提示)、内层"回合"(turn)循环(重复"LLM 生成 + 工具执行",直到 assistant 不再产出可运行的工具使用(tool use)),以及单个 `step()` pass(准备上下文 -> 流式调用 LLM -> 在部分流中检测工具调用 -> 执行工具 -> 产出 assistant 消息 + 工具结果)。一切以消息为中心:唯一的共享状态是一份持久化的、只追加(append-only)的对话日志(`<logdir>/conversation.jsonl` 中由 `Message` 对象组成的 `LogManager`/`Log`),循环纯粹通过检查最后一条 assistant 消息是否含有可运行的工具使用来决定是否继续。横切关注点(cross-cutting behaviors:steering、取消、自动回复、压缩(compaction)、确认)通过 hook 系统(HookType 枚举)注入而非硬编码,持久的提示队列(prompt queue)文件则允许其他进程(server、编排型 subagent)向运行中的循环注入提示。

## 关键文件(Key Files)
- **gptme/__main__.py** — 极简入口:从 gptme.cli.main 导入 `main` 并调用(第 3-14 行);把启动/运行期间的 KeyboardInterrupt 映射为退出码 130。
- **gptme/chat.py** — 核心循环文件。`chat()`(chat.py:149)做会话建立(init、模型解析、LogManager.load、chdir 到 workspace、在 :273 清除 `prompt-queue-closed` 哨兵文件(sentinel)),然后委托给 `_run_chat_loop`(chat.py:279)。`_run_chat_loop`(chat.py:324)是外层 REPL:排空外部提示队列(:338)、弹出提示/获取用户输入、运行斜杠命令(slash command,`execute_cmd` :349/:438)、触发 TURN_PRE(:353/:444)与 LOOP_CONTINUE(:462)hook,并处理 KeyboardInterrupt(:482)和 provider 错误(:489)。`_process_message_conversation`(chat.py:531)是单个回合(turn)的内层循环:STEP_PRE hook(:558)、调用 `step()`(:565)、追加结果(:585)、检查 DECLINED_CONTENT(:600)、工具后压缩(compaction,:606)、GPTME_MAX_STEPS(:629),以及"最后一条 assistant 是否还有可运行工具使用"式的继续判定(continuation check,:642-654)。`step()`(chat.py:896)是一次"生成+执行"的 pass:`prepare_messages`(:924)、`_reply_with_overflow_recovery`(:933 -> :794)、`execute_msg`(:982),产出 assistant 消息 + 工具输出(:1018-1019)。另有 `_drain_external_prompt_queue`(:515)、`_get_user_input`(:750)、`_should_prompt_for_input`(:705)、`prompt_user`(:1025)、自动命名线程辅助函数(:70)、token 用量 ContextVar(:103)。
- **gptme/llm/__init__.py** — 循环使用的 LLM 门面(facade)。`reply()`(:362)触发 GENERATION_PRE hook(:380),然后调用 `_reply_after_hooks`(:420),并通过 `mark_llm_reply_origin`(:301)给 provider 调用的异常打标记。流式路径 `_reply_stream`(:815)逐字符消费流、渲染 thinking 标签,并在 `break_on_tooluse` 检查点(:1056-1072)借助 `_tooluse_break_check_char`(:49)与 `_drain_toolbreak_stream`(:61)做流式工具调用检测。`_stream`(:697)与 `_chat_complete`(:543)路由到各 provider SDK(OpenAI 兼容、Anthropic、订阅、mock),SDK 被 `_StreamWithMetadata`(:648)包装,后者从 StopIteration 捕获用量元数据。`is_provider_error`(:327)/ `is_context_length_error`(:341)为聊天循环的恢复分支分类失败类型。
- **gptme/tools/__init__.py** — `execute_msg`(tools/__init__.py:459)把 assistant 消息中的全部 ToolUse 物化(materialize)出来,通过 `tooluse.execute()`(:497)逐个执行可运行的调用,为不可运行的结构化调用配上错误结果(:522-540),以 call_id 配对的方式处理 KeyboardInterrupt(:498-515),并累积 `tool_timings`。`get_tool_for_langtag`(:543)把 markdown 围栏的 lang-tag 映射到工具。
- **gptme/tools/base.py** — `ToolUse` 数据类(base.py:839),含 `execute()`(:848:TOOL_EXECUTE_PRE/POST hook、`_current_tool_use` ContextVar、在生成器最后一条消息上盖 call_id、错误->system Message)与 `iter_from_content()`(:1083:按 tool_format 分发——markdown 代码块、XML,或带 `find_json_end` 的 JSON 工具调用正则,跳过围栏代码内部的匹配)。`ToolSpec`(:468)。
- **gptme/codeblock.py** — markdown 围栏提取:`Codeblock.iter_from_markdown`(codeblock.py:91)-> `_extract_codeblocks`(:345)。`streaming=True` 时,闭合围栏之后必须再有一个空行,代码块才算完整(:355-357)——正是这一点让流中(mid-stream)工具检测是安全的。处理 thinking 标签剥离、heredoc、嵌套围栏、变长围栏。
- **gptme/prompt_queue.py** — 持久的磁盘提示队列(`<logdir>/prompt-queue.jsonl`,通过 `.prompt-queue.lock` 做 flock)。`queue_prompt(steer=...)`(:53)追加;`drain_prompt_queue`(:83)在回合之间排空常规提示;`drain_steer_prompts`(:141)在回合中的 STEP_PRE 排空带 steer 标志的提示。steer 记录对常规排空不可见,反之亦然。
- **gptme/executor.py** — `prepare_execution_environment(workspace, tools, chat_config)`(executor.py:28):一次性引导(bootstrap)(Config.from_workspace、set_config、dotenv、init_tools、压缩 hook、init_hooks),由 server、evals 和 subagent 工具共享,使每个执行上下文获得完全相同的初始化。
- **gptme/k_best.py** — `k_best_guess(k, check, ...)` 装饰器(k_best.py:69):把一个函数跑 K 次(默认用 ThreadPoolExecutor),通过 `check(result)->float` 给候选打分,返回得分最高的赢家;Candidate/KBestResult 数据类(:45-66)。作为公开 API 导出(gptme/__init__.py);这是一个抗幻觉(anti-hallucination)工具,并未接入聊天循环本身。
- **gptme/circuit_breaker.py** — `CircuitBreaker`(circuit_breaker.py:59):经典的按资源 CLOSED->OPEN->HALF_OPEN 状态机,`call(func, *args, ignore_exceptions=...)`(:108)在锁外执行,HALF_OPEN 只放行单个在途探测;`get_breaker(name)` 注册表(:260)。被 MCP 客户端(mcp/client.py:306-388)用于对不稳定的 MCP server 快速失败(fail fast)。
- **gptme/checkpoint.py** — 基于 git 的工作区(workspace)恢复(不是对话回溯)。`classify()`(:133)把工作区分类为 clean_git/dirty_git/non_git/multi_root(BackendDecision :44);`create_checkpoint`(:240)向 `$XDG_STATE_HOME/gptme/checkpoints/<sha16-of-realpath>.jsonl` 追加一条 CheckpointRecord;`restore_checkpoint`(:355)执行 `git reset --hard`(include_dirty 时再加 `git clean -fd`);`diff_checkpoint`(:429)、`list_checkpoints`(:303)。由 cli/checkpoint.py 和 commands/checkpoint.py(`/checkpoint` 斜杠命令)消费。
- **gptme/agent/(整个目录)** — 不是 agent 循环——它是面向长期运行的定时 agent 的"自主 agent 生命周期管理"。service.py:`ServiceManager` ABC(:53)+ `SystemdManager`(:108,写 ~/.config/systemd/user/gptme-agent-<name>.service/.timer)和 `LaunchdManager`(:408,用 plistlib 写 plist),`parse_schedule`(:319)把 systemd OnCalendar 转换为 launchd 时间间隔。workspace.py:`create_workspace_from_template`(:131,克隆 gptme-agent-template + 白名单内的 fork 脚本)、`create_workspace_structure`(:261,journal/tasks/knowledge 目录)、`write_agent_config`(:304)、`generate_run_script`(:353,即服务 exec 的那个 `autonomous-run.sh`——最终运行 `gptme --non-interactive`)、`init_conversation`(:399)。cli.py:click 命令 scan/status/create/install/start/stop/run/logs/doctor。doctor.py:`run_doctor`(:498)对工作区做健康检查(CheckResult/DoctorReport)。
- **gptme/logmanager/manager.py** — `Log`(manager.py:73)消息列表包装器;`LogManager.append`(:542)持久化并打印每条消息(双写(dual-write)master + 压缩视图);`write(sync=...)`(:610);`prepare_messages`(:1137)是调用 LLM 前的变换:活跃提示生成 -> 上下文充实 -> 短命消息修剪 -> `limit_log` 兜底截断 -> 可选的证据回放(evidence replay)。
- **gptme/util/interrupt.py** — SIGINT 处理器 `handle_keyboard_interrupt`(:20)只在 ContextVar 的可中断(interruptible)标志置位时才抛出 KeyboardInterrupt(在 step()/工具执行期间置位,在等待用户输入期间清除,这样在提示符处按 Ctrl-C 只打印 'Press Ctrl-D to exit',而不是杀掉会话)。
- **gptme/llm/retry_policy.py** — 重试策略的唯一事实来源:SDK_MAX_RETRIES=0(禁用 SDK 内置重试,:29),DEFAULT_MAX_RETRIES=11 次尝试,带上限的指数退避并尊重 Retry-After(:95)。通过 llm_anthropic.py:521 的 `retry_on_overloaded` 装饰器应用(OpenAI 等价物在 llm_openai.py:928)。

## 关键抽象(Key Abstractions)
- **chat(prompt_msgs, initial_msgs, logdir, workspace, ...) -> None** — gptme/chat.py:149。会话编排器:init()、LogManager.load、chdir(workspace)、从 prompt_msgs 构建内存中的 prompt_queue、调用 _run_chat_loop、退出时写日志、处理 SessionCompleteException、在 finally 中恢复 output-format/token-accumulator 的 ContextVar。可从 CLI、server 和 subagent 线程调用。
- **_run_chat_loop(manager, prompt_queue, stream, ...)** — gptme/chat.py:324。外层 REPL:(1)排空持久队列,(2)从 prompt_queue 或交互式输入取下一条消息,(3)运行斜杠命令、TURN_PRE hook,然后调用 _process_message_conversation,(4)运行 LOOP_CONTINUE hook,其产出的消息被追加进 prompt_queue(上限 MAX_PROMPT_QUEUE_SIZE=100),(5)捕获 KeyboardInterrupt(追加 INTERRUPT_CONTENT system 消息、清空队列、继续)和带标记的 provider 错误(仅交互模式恢复,否则重新抛出)。
- **_process_message_conversation(manager, stream, tool_format, model, output_schema)** — gptme/chat.py:531。一个用户回合的内层 agentic 循环。重复:STEP_PRE hook -> step() -> 追加响应消息 -> 拒绝(declined)检查 -> 回合中自动压缩(autocompact)-> 最大步数检查 -> 继续判定(`any(tu.is_runnable for tu in ToolUse.iter_from_content(last_assistant_content))`,chat.py:649)。以 TURN_POST hook + manager.write(sync=True) 收尾。
- **step(log, stream, tool_format, workspace, model, ...) -> Generator[Message]** — gptme/chat.py:896。一次 pass:prepare_messages -> 可选的 ToolSpec 列表(仅当 tool_format=='tool')-> _reply_with_overflow_recovery -> GENERATION_POST hook -> 带 tool_timings 字典的 execute_msg(msg_response)-> 先产出 assistant 消息(除非 JSON 模式否则 quiet),再产出工具输出消息。set_interruptible/clear_interruptible 放在 try/finally 中。
- **reply(messages, model, stream, tools, ...) -> Message** — gptme/llm/__init__.py:362。触发 GENERATION_PRE hook(其输出消息会追加进请求),然后经 _reply_after_hooks 调用 provider;包装 provider 调用,使异常被打上 _gptme_from_llm_reply 来源标记(mark_llm_reply_origin,:301)以及 output_emitted 标志——正是这一点让聊天循环能区分"LLM 挂了"与工具/hook 崩溃(is_provider_error,:327)。
- **_reply_stream(...) and break_on_tooluse** — gptme/llm/__init__.py:815。把 provider 流逐字符消费进 `output`;将 <think> 块渲染为暗色、抑制 think-sig 注释、按行/chunk 缓冲以实现 O(chunks) 的终端写入;在每个格式特定的断点字符(:1056)处用 ToolUse.iter_from_content(output, streaming=True) 重新解析已累积的输出,若存在可运行的 tooluse,则排空生成器剩余部分以获取用量元数据(_drain_toolbreak_stream,:61)并 break——这条部分完成的消息被返回并立即执行。break_on_tooluse 默认为 not model_meta.supports_parallel_tool_calls(:443-447),可用 GPTME_BREAK_ON_TOOLUSE 覆盖。
- **execute_msg(msg, log, workspace, tool_timings) -> Generator[Message]** — gptme/tools/__init__.py:459。预先物化全部 ToolUse(保证每个结构化 call_id 恰好得到一个结果),经 ToolUse.execute 顺序执行可运行者,为不可运行的结构化调用发出配对的错误结果,把 KeyboardInterrupt 转换为 INTERRUPT_CONTENT + 'was not executed' 的配对消息,并按工具名把墙钟毫秒数累积进 tool_timings。
- **ToolUse** — gptme/tools/base.py:839 近似 frozen 的数据类:tool、args、content、kwargs、call_id、start、_format。`execute()`(:848)运行 TOOL_EXECUTE_PRE hook、设置 _current_tool_use ContextVar(供确认流程使用)、调用 tool.execute(content, args, kwargs)——接受单个 Message 或一个 Generator(当 call_id 已设置时缓冲生成器,使只有最后一条(LAST)消息被盖上 call_id + 工具元数据,因为 Responses API 要求每个 call_id 恰好一个 function_call_output)——记录遥测、触发 TOOL_EXECUTE_POST hook、把异常转换为 system 错误 Message(仅在 pytest 下重新抛出)。`iter_from_content(content, streaming=False)`(:1083)解析 markdown 围栏、XML(<tool-use> 与 Haiku 的 <function_calls>)以及原生 JSON 工具调用(正则 + find_json_end + json_repair,跳过围栏代码内部的匹配),按位置排序。
- **Message** — gptme/message.py:281 frozen 数据类:role 取值 {system,user,assistant}、content、timestamp、files、file_hashes、call_id(把工具结果与结构化 tool_use 配对)、pinned(永不被上下文裁剪)、hide(发给 LLM 但不打印)、quiet(追加时不打印)、ephemeral_ttl(N 个 assistant 回合后修剪)、metadata(usage/cost/timings,含 tool_ms_by_name)。
- **LogManager / Log** — gptme/logmanager/manager.py:219/73。持久化 + 视图(view)。append(msg)(:542)按哈希存储附件、双写到 master 分支与当前压缩视图、写 JSONL、发出 event-log、除非 quiet 否则打印。prepare_messages(:1137)是每次 LLM 调用前运行的上下文流水线。视图(create_view/switch_view :875/:893)实现"看起来无损"的压缩:master 保留一切,视图才是 LLM 看到的东西。
- **提示队列(持久 steering,Prompt queue / durable steering)** — gptme/prompt_queue.py:logdir 下带 flock 的 JSONL 记录 {content, queued_at, steer?, skill_invocation_id?}。常规记录 -> 由 drain_prompt_queue 在回合之间排空(由 _drain_external_prompt_queue 在每次外层循环迭代开头调用);steer 记录 -> 由 subagent 控制 hook(tools/subagent/hooks.py:40,注册于 tools/subagent/execution.py:269)在每个 STEP_PRE 排空,使编排者(orchestrator)的指引能在回合中直达下一次 LLM 调用。`prompt-queue-closed` 哨兵文件(chat.py:273/312/379/402)在循环退出时封掉排空竞态。
- **Hook(HookType / trigger_hook / register_hook)** — gptme/hooks/types.py:66 + registry.py:654。基于生成器的 hook,产出被注入对话的 Message。与循环相关的类型:SESSION_START/END、TURN_PRE/POST、STEP_PRE/POST、GENERATION_PRE/POST/CHUNK/INTERRUPT、TOOL_EXECUTE_PRE/POST、TOOL_CONFIRM、LOOP_CONTINUE。steering、取消(控制 hook 抛出的 SessionCompleteException)、自动回复、自动压缩、TTS 都是以这种方式叠加到循环上,而无需改动循环本身。
- **SessionCompleteException** — gptme/tools/complete.py。由 `complete` 工具、自动回复退出(complete.py:534 'No tools used after 2 auto-reply confirmations')以及 subagent 取消(tools/subagent/hooks.py:102)抛出。在 _run_chat_loop 中它首先于 chat.py:369 被处理:写哨兵文件、再排空一次队列——若仍有链式提示则循环继续,否则传播到 chat(),后者运行 SESSION_END hook 并干净退出。
- **可中断标志(Interruptible flag,set_interruptible/clear_interruptible)** — gptme/util/interrupt.py:44/49 + SIGINT 处理器 :20。ContextVar 门闸:仅当循环处于 step() 或工具执行中时,Ctrl-C 才抛出真正的 KeyboardInterrupt;等待用户输入期间只打印提示。注册于 cli/main.py:1466。

## 数据流(Data Flow)
一个用户回合的完整追踪(链路上每个函数都给出 file:line):

1. 入口(ENTRY):`python -m gptme` -> gptme/__main__.py:11 的 `main()`(gptme/cli/main.py:762)解析 CLI、构建 logdir/workspace/model、构建 `prompt_msgs = [Message('user', p) for p in prompts]`(cli/main.py:1136)、安装 SIGINT 处理器(cli/main.py:1466),然后调用 `chat(prompt_msgs, initial_msgs, logdir, workspace, model, stream, no_confirm, interactive, ...)`(cli/main.py:1478)。

2. 会话建立(SESSION SETUP):`chat()` gptme/chat.py:149 —— `_reset_token_accumulator()`(:200)、`set_output_format`(:201)、`init(model, interactive, tool_allowlist, tool_format, no_confirm)`(:205,gptme/init.py:41:加载 dotenv,每进程一次性初始化模型+工具+hook)、`set_session_allow_hosts`(:210)、SESSION_START hook(:213)、模型解析 + 流式能力检查(:223-239)、`LogManager.load(logdir, initial_msgs, create=True)`(:243)、skill 会话上下文(:247)、`os.chdir(workspace)`(:255)、打印过往日志(:260)、unlink 过期的 `prompt-queue-closed` 哨兵(:273)、`prompt_queue = list(prompt_msgs)`(:276)、`_run_chat_loop(...)`(:279)。

3. 外层循环(OUTER LOOP):`_run_chat_loop` gptme/chat.py:337 `while True`:
   a. `_drain_external_prompt_queue(manager, prompt_queue)`(:338 -> :515)调用 `drain_prompt_queue(logdir, max_items=MAX_PROMPT_QUEUE_SIZE-len)`(prompt_queue.py:83):flock 加锁、读取 prompt-queue.jsonl、跳过 steer 记录、返回 user Message、重写/删除该文件。
   b. 若 prompt_queue 非空(:342):`msg = prompt_queue.pop(0)`(:343);`include_paths(msg, workspace)`(:345,展开 @file 引用);`manager.append(msg)`(:346 -> logmanager/manager.py:542:按哈希存储文件、双写分支+视图、写 JSONL、除非 quiet 否则打印);若为 user 角色且 `execute_cmd(msg, manager)`(:349 -> commands/base.py:195:若内容是 /命令则运行处理器并返回 True)-> `continue`(:350);TURN_PRE hook(:353);`_process_message_conversation(...)`(:362)。
   c. 否则若非交互模式(:393):touch 哨兵(:402)、再排空一次,若仍为空则 break(:413)——`gptme --non-interactive 'prompt'` 就是这样终止的。
   d. 否则为交互模式:`_get_user_input(manager.log, manager.workspace)`(:415 -> :750):`clear_interruptible()`(:752)、`_should_prompt_for_input(log)`(:755 -> :705:正常情况下提示输入,除非处于崩溃恢复(crash-recovery)状态,即最后一条消息是未应答的 user 消息,且最后一条 assistant 之后没有 interrupt/decline/LLM-failed 标记);若返回 None -> 退出(break :420),或直接进入 `_process_message_conversation` 做崩溃恢复(:423);否则 `prompt_user()`(:768 -> chat.py:1025:响铃、flush stdin、prompt_toolkit 带历史的 `get_input`、Ctrl-C 打印提示、Ctrl-D 抛出 EOFError -> None -> 退出)。输入长度钳制到 MAX_MESSAGE_LENGTH=100k(:771-779)。然后 append + execute_cmd + TURN_PRE + `_process_message_conversation`(:432-458)。
   e. LOOP_CONTINUE hook(:462-480,例如 tools/complete.py:479 的自动回复和 subagent 完成消息):产出的消息被追加进 prompt_queue(超过 MAX_PROMPT_QUEUE_SIZE=100 的丢弃),外层循环 `continue` 去处理它们。
   f. 异常护栏:KeyboardInterrupt(:482)-> 记日志、`manager.append(Message('system', INTERRUPT_CONTENT))`、`prompt_queue.clear()`、continue。Provider 错误(:489):`if not interactive or not is_provider_error(e): raise`,否则记日志 + system 消息 `LLM_REQUEST_FAILED_PREFIX` + 清空队列 + continue(issue #3668:只有在 reply() 内部打过标记的异常才算数)。

4. 内层循环(INNER LOOP,一个回合):`_process_message_conversation` gptme/chat.py:553 `while True`:
   a. `set_interruptible()`(:555)。
   b. STEP_PRE hook(:558)——含 subagent 控制 hook(tools/subagent/hooks.py:40):排空 `control.jsonl` 中的取消操作(抛出 SessionCompleteException,hooks.py:102),并排空 steer 提示(hooks.py:109 -> prompt_queue.py:141),后者成为 LLM 接下来看到的 user 消息。
   c. `response_msgs = list(step(manager.log, stream, tool_format=..., workspace=..., model=..., output_schema=..., logdir=...))`(:565-575)。
   d. KeyboardInterrupt(:576)-> INTERRUPT_CONTENT 消息,break(回到用户提示符)。
   e. 对每条响应消息:`manager.append(response_msg)`(:586);user 角色的命令经 `execute_cmd`(:588);assistant 消息做 `print_inline_cost`(:593)。
   f. DECLINED_CONTENT 检查(:600)——用户在工具确认处回答 'n' -> break(行为类似 Ctrl-C)。
   g. `pending_continuation = _has_pending_tooluse(manager.log)`(:605 -> :668);`_run_post_tool_compaction(manager)`(:606 -> :679:在工具结果之后、甚至回合中也运行自动压缩 hook;若活跃视图发生了切换则返回 True)。
   h. 每个会话一次的后台自动命名线程(:611-626)。
   i. `step_count += 1`;若达到 GPTME_MAX_STEPS -> system 消息 + break(:629-636)。
   j. 继续判定(CONTINUATION DECISION,:642-654):若回合中已压缩则使用压缩前的答案,否则重新计算 `has_runnable = any(tu.is_runnable for tu in ToolUse.iter_from_content(last_assistant_content))`——由于工具结果是 'system' 角色,最后一条 assistant 消息仍持有刚刚执行过的那些工具调用,因此循环会继续生成;当最新的 assistant 回复不含可运行的工具使用时,`break`。
   k. 退出时:TURN_POST hook(:658)+ `manager.write(sync=True)`(:665)。

5. 单次 pass(ONE PASS):`step()` gptme/chat.py:896:
   a. 模型解析(:909-915)。
   b. `msgs = prepare_messages(log.messages, workspace, logdir)`(:924 -> logmanager/manager.py:1137:`_active_prompt_generation` -> `enrich_messages_with_context` -> `prune_ephemeral_messages` -> `limit_log` 硬截断警告 -> 可选的 evidence replay)。
   c. 仅当 tool_format == 'tool' 时 `tools = [t for t in get_tools() if t.is_runnable]`(:926-928)。
   d. `msg_response = _reply_with_overflow_recovery(...)`(:933 -> :794):调用 `reply()`;若 `is_context_length_error`(且尚未发出可见输出)则运行 `compact_for_overflow(manager)`、创建+切换压缩视图、重新准备消息、重试一次、记录压缩事件,失败则回退到 master 视图。
   e. GENERATION_POST hook(:962)。
   f. `tool_outputs = list(execute_msg(msg_response, log=log, workspace=workspace, tool_timings=tool_timings))`(:982-989)。
   g. 把 `timings.tool_ms`/`tool_ms_by_name` 附到 assistant 消息的 metadata(:990-1014)。
   h. `yield msg_response.replace(quiet=not is_output_json())`,然后 `yield from tool_outputs`(:1018-1019)。

6. LLM 调用(LLM CALL):`reply()` gptme/llm/__init__.py:362 -> GENERATION_PRE hook(:380)-> `_reply_after_hooks`(:420):`init_llm(provider)`(:433);流式 -> `_reply_stream(msgs, model, tools, break_on_tooluse, ...)`(:448);否则 `_chat_complete`(:464)按 provider 路由(Anthropic :612、OpenAI 兼容/custom/plugin :596、openai-subscription :624、mock :631)并经 ToolCallDisplay 美化打印(:476-479)。Provider SDK 用 `retry_on_overloaded` 包装(llm_anthropic.py:521;llm_openai.py 有等价物),采用 llm/retry_policy.py(11 次尝试、带上限的指数退避、尊重 Retry-After、禁用 SDK 重试)。返回的异常经过 `mark_llm_reply_origin`(:409-417)。

7. 流式 + 工具调用检测(STREAMING + TOOL-CALL DETECTION):`_reply_stream` gptme/llm/__init__.py:815 —— `_stream()`(:889 -> :697)返回包装 provider 生成器的 `_StreamWithMetadata`;:907 的 for 循环逐字符迭代(`_chars_with_chunk_end` :899);每个字符追加进 `output`(:1006);thinking 标签状态机(:919-1003)加上 on_token/on GENERATION_CHUNK 发射与按行缓冲(:1013-1041);在 `break_on_tooluse and _tooluse_break_check_char(char)`(:1056——markdown 为 '\\n',tool 格式为 '}'/'\\n',xml 为 '>'/'\\n')处运行 `ToolUse.iter_from_content(output, streaming=True)`,若发现任何 `is_runnable` 的 tooluse,则调用 `_drain_toolbreak_stream(stream)`(:1071,排空生成器,使 StopIteration.value 把 message_delta 的用量带进 stream.metadata)并 break。流中 KeyboardInterrupt(:1094)返回一条以 '... ^C Interrupted' 结尾、带部分 timings 的 Message。返回 `Message('assistant', output, metadata=stream.metadata)`(:1160)。

8. 工具执行(TOOL EXECUTION):`execute_msg` gptme/tools/__init__.py:459 -> `ToolUse.execute` gptme/tools/base.py:848 -> 具体的 `tool.execute(content, args, kwargs)`(各工具模块位于 gptme/tools/ 之下)。结果以 Message 返回(工具输出为 'system' 角色;结构化调用时 `call_id` 恰好盖在生成器的最后一条消息上,base.py:907-945)。这些消息沿 yield 链向上流入 chat.py:586 的 `manager.append`、被持久化,并成为下一次 `prepare_messages` 输入的一部分——就此闭环。

9. 终止路径(TERMINATION PATHS):(a)非交互 + 空队列 -> break(chat.py:413);(b)提示符处 EOF/Ctrl-D -> `_get_user_input` 返回 None 且 `_should_prompt_for_input` 为 True -> break(chat.py:416-420);(c)最后一条 assistant 消息无可运行的工具使用 -> 内层 break(chat.py:653);(d)SessionCompleteException -> 链式提示检查后传播到 chat(),后者运行 SESSION_END hook 并退出(chat.py:293-302、369-390);(e)GPTME_MAX_STEPS(chat.py:630);(f)DECLINED_CONTENT(chat.py:600);(g)KeyboardInterrupt(chat.py:482/576);(h)非交互模式下的 provider 错误重新抛出(chat.py:497)。

## 设计亮点(Design Highlights)
- 循环唯一的继续判定(continuation test)就是"最后一条 assistant 消息是否仍含有可运行的工具使用?"(chat.py:649)。由于工具结果是追加在 assistant 消息之后的 'system' 角色消息,这个纯粹基于内容的单一检查以零额外状态驱动整个 agentic 周期——崩溃安全,且可从持久化日志轻松重放。
- 工具调用在流式过程中即被检测:在廉价的、格式特定的断点字符处重新解析已累积的文本(llm/__init__.py:1056,_tooluse_break_check_char :49),并且 streaming=True 的代码块提取要求闭合围栏后跟一个空行(codeblock.py:355),以避免执行写了一半的命令。工具使用中断后,生成器仍会被排空(_drain_toolbreak_stream :61),纯粹是为了从 provider 的最终事件捕获 usage/cost 元数据。
- Provider 失败的来源标记(provenance tagging)(mark_llm_reply_origin llm/__init__.py:301 + is_provider_error :327):只有 provider 调用内部抛出的异常才会被聊天循环视为"可恢复的 LLM 故障",因此工具或 hook 内部的 httpx 失败仍会大声崩溃,而不是被静默吞进无限重试(issue #3668)。
- 一切横切的东西都是产出 Message 的 hook(hooks/types.py:66):steering、取消、自动回复、自动压缩、成本显示、TTS。核心循环对它们没有任何 if 分支——同一批 LOOP_CONTINUE/STEP_PRE/TURN_POST 触发点服务于所有这些功能。
- 持久的跨进程 steering:prompt-queue.jsonl 带 flock 与 steer 标志,实现两种排空节奏(回合之间 vs 回合中的 STEP_PRE),外加在最终排空之前写入的 'prompt-queue-closed' 哨兵文件,以封掉"编排者的 steer 被虚假确认"的竞态(chat.py:369-413 注释)。
- 可中断性是显式的 ContextVar 门闸(util/interrupt.py),而非裸信号行为:Ctrl-C 只在 step()/工具执行内部抛出,在用户提示符处刻意保持无效,从而给出"Ctrl-C 中断 agent、Ctrl-D 退出会话"的语义,并辅以细致的部分输出刷新(llm/__init__.py:1094-1123)。
- 上下文长度溢出在 step() 内部自愈:_reply_with_overflow_recovery(chat.py:794)压缩进一个具名的 LogManager 视图、重试一次、验证重试确实缩小了 provider 输入、记录带前后 token 数的压缩事件,若重试失败则回退到 master 视图。
- 结构化(原生 JSON)工具调用在任何失败模式下都保持 tool_use/tool_result 配对有效:不可运行的调用获得合成的错误结果(tools/__init__.py:522-540),中断以 'was not executed' 结果排空剩余调用(:498-515),多条消息的工具生成器只有最后一条(LAST)消息被盖上 call_id(base.py:907-945),因为 Responses API 要求每个 call_id 恰好一个 function_call_output。
- 重试策略集中管理且 SDK 重试归零(llm/retry_policy.py:29),因此 gptme 自己的 11 次带上限退避的预算才是真正的预算——防止限流期间 SDK 重试 x 自身重试的乘法放大。
- 持久化模型:append-only JSONL 加 master/视图双写(logmanager/manager.py:542)意味着压缩永不销毁历史,而 _should_prompt_for_input(chat.py:705)能通过检查最后一个 assistant 回合之后的标记消息(INTERRUPT/DECLINED/LLM-failed)来恢复崩溃的会话。

## MVP 指南(MVP Guidance)
- 先复刻主干(spine):Message 数据类 -> append-only JSONL Log -> 外层 REPL 循环 -> 内层"当最后一条 assistant 还有可运行工具使用时继续"的循环 -> step() = prepare+reply+execute_msg -> 把结果 yield 进 manager.append。其余一切都是润色。
- 即使在 MVP 里也要尽早实现流式工具检测:累积流式文本,每逢换行就用"围栏后需空行"的规则重跑一遍 markdown 代码块解析器;一旦出现完整且可运行的块就立即执行。这是 gptme 的标志性 UX(你能看着 agent 一个工具一个工具地干活)。
- 只从 markdown 围栏工具调用开始(就像 gptme 默认的 'markdown' ToolFormat);循环跑通后,再把原生 JSON 工具调用(call_id 配对)作为第二种格式加上。配对不变量(每个 call_id 恰好一个结果、为不可运行/被中断的调用合成错误结果)才是难点——照抄 tools/__init__.py:459 的"先物化"结构。
- 抄最小 hook 集合:GENERATION_PRE/POST、STEP_PRE、TURN_POST、LOOP_CONTINUE,做成"产出 Message 的生成器函数"。v1 可以把自动回复('No tool call detected... complete?')和 max-steps 硬编码在循环里,以后再抽成 hook。
- 保留可中断标志模式(ContextVar + SIGINT 处理器)——它很小,却免费给出正确的 Ctrl-C 语义;同样把三个标记内容 INTERRUPT_CONTENT / DECLINED_CONTENT / LLM_REQUEST_FAILED_PREFIX 当作一等协议字符串。
- v1 跳过:视图/压缩机制(直接截断最旧的未 pin 消息即可)、k_best.py、circuit_breaker.py(只有接 MCP 才有意义)、checkpoint.py、整个 gptme/agent/ 服务管理目录(systemd/launchd 自动化)、evidence replay、自动命名线程、ACP/TUI 特有的缓冲,以及 prompt-queue 的持久化/哨兵逻辑(加 subagent 之前,进程内队列就够了)。
- provider 重试保持简单但集中:一个带上限指数退避 + Retry-After 的重试装饰器,并禁用 SDK 内部重试——多层重试相乘的问题是真实存在的。
- 把 LLM 门面设计成 reply(messages, model, stream, tools),返回一条完整构建的 Message,元数据从流包装器捕获——_StreamWithMetadata 的"捕获 StopIteration.value"技巧(:648)值得抄,用于 usage/cost 记账。
- 工作顺序:(1)Message+Log 持久化,(2)step() 配非流式 reply + markdown 工具执行,(3)内层循环继续判定,(4)外层 REPL + 斜杠命令,(5)流式 + break_on_tooluse,(6)中断/拒绝,(7)provider 重试,(8)hook,(9)压缩,(10)steering/subagent。

## 遗留问题(Open Questions)
- gptme/agent/ 尽管名字如此,并不包含 agent 循环——它管理用于自主定时运行的 systemd/launchd 服务与模板工作区。真正的循环在 chat.py 里。重新实现时不应照搬这个命名。
- k_best.py 在公开 API(gptme/__init__.py)中导出,但我在循环内部没有找到调用点——它看起来是给用户/evals 用的库工具;不清楚是否有内置路径用它装饰生成过程。
- STEP_POST 和 MESSAGE_TRANSFORM 两个 hook 类型存在(hooks/types.py:92/99),但主 CLI 循环从不直接触发 STEP_POST——它可能仅限 server 使用,或是残留物;请在 gptme/server/session_step.py 中验证(超出本子系统的范围)。
- server(gptme/server/session_step.py)运行本循环的一个平行变体:立即持久化并用 _attach_tool_timings 打补丁,而非 CLI 的"先缓冲再 yield"方式(chat.py:975-981 注释)——确切差异未在此处追踪。
- break_on_tooluse 会丢弃第一个被检测到的工具调用之后流出的 assistant 文本(排空只为拿元数据);因此不支持 supports_parallel_tool_calls 的模型会在后续 step 中重新生成其余工具调用——这是用 token 成本换正确性,并依赖准确的 ModelMeta 能力标志。
- _should_prompt_for_input 把带 pinned 或没有 user 消息的日志视为"应提示输入"(chat.py:741-747);pinned 情形与视图切换的交互方式未在此处充分探究。
