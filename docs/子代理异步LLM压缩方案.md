# 子代理异步 LLM 压缩方案(仅设计,未开发)

> 目标:把 LLM 语义压缩(compact_current_with_llm)从"回合边界同步执行"改为"派子进程后台执行、稍后收割",消除主线程阻塞。
> 依据:本项目 main 分支源码 + gptme 源码研究报告(git 历史里的 docs/gptme-study/,源码副本 /tmp/gptme 已不在磁盘)。

---

## 1. 背景与动机

### 1.1 现状的阻塞点

当前自动压缩的 summarize 通道是同步的,调用链:

```
chat.py:194  dispatch_hooks(TURN_POST)          ← 用户已经可以打字了吗?不行,钩子没返回
  └─ autocompact_hook (autocompact.py:111)
       └─ _auto_summarize (autocompact.py:155)
            └─ compact_current_with_llm (compaction_llm.py:449)
                 └─ reply(...) 调模型写笔记      ← 一次 30-60 秒;分块时最多 6+1 次
```

turn.post 在 `_run_chat_loop` 的循环体内(chat.py:194):钩子不返回,循环就不进入下一轮,用户拿不到输入提示符。**规则压缩(trim)是纯计算,毫秒级,不是问题;真正阻塞的是请模型写笔记的这一步。** 这也是 `ALPHA_AUTO_LLM_COMPACT` 默认关闭的原因之一——同步等待的体验太差。

### 1.2 用户想法

用子代理(独立执行体)承载 LLM 压缩,主线程不等待,聊天继续,压缩在后台完成后应用结果。

---

## 2. gptme 的对应设计(研究报告要点)

gptme 没有把 LLM 压缩本身异步化(它的 `_resume_via_llm` 也在 TURN_POST 同步跑),但它的三套机制给了完整样板:

| 机制 | gptme 做法 | 对本方案的启示 |
|---|---|---|
| subagent 派生 | `subagent(id, prompt)` **立即返回**(fire-and-forget),三种后端:线程 / 子进程 / ACP | 子进程模式我们已有雏形(subagent.py),缺的只是"不等待" |
| 完成通知 | 完成结果进队列,`_subagent_completion_hook` 在 **loop.continue / step.pre** 钩子排空队列,以 "✅ Subagent X completed" 注入父会话 | **收割放在生命周期边界钩子上**,核心循环一行不改 |
| 后台 shell | 前台命令软超时后不杀,注册成 BackgroundJob;`background_job_completion_hook`(step.pre)把已完成任务结果注入下一步 | 同款"后台跑 + 边界收割"思想,证明该模式可靠 |
| 过期摘要检测 | `_resume_via_llm` 用 (view, len, last-content) + 文件大小/mtime 快照检测并发追加,**过期就丢弃** | 异步下"过期"几乎必然发生,不能照搬"丢弃"——见 §5.3 的改造 |
| 取消/转向 | 子代理 logdir 的 control.jsonl + STEP_PRE 检查点,跨进程发控制指令 | MVP 不需要主动取消,用"单飞行 + 冲突作废"替代(§5.5) |
| 重启恢复 | 子代理元数据持久化在 logdir,重启后重建注册表 | **job 状态全放磁盘**,父进程死了也能下次收割 |

核心结论:gptme 的可复用模式是"**派生即返回、状态落磁盘、边界钩子收割**",而不是它同步的压缩本身。

---

## 3. 设计总览

```
回合 N 结束,turn.post 触发 autocompact_hook
  │
  ├─ ① 先收割:会话目录里有已完成的压缩 job 吗?
  │     有 → 读笔记 → 三道验货 → 组装(源前缀 + 新笔记 + 新增消息接尾)
  │        → 过安检 → 激活新视图(§5.4)          [纯计算 + 磁盘,毫秒级]
  │
  ├─ ② 再评估:decide_compaction 三选一(逻辑不变)
  │     none / trim → 照旧(trim 仍同步,它是免费的)
  │     summarize 且没有在飞 job → ③ 派发
  │
  └─ ③ 派发:
        快照当前历史 → compaction-jobs/<job-id>/source.jsonl
        写 job.json(状态/指纹/切分点/模型/pid)
        Popen 派子进程(python -m my_agent_alpha --compact-job <job-dir>)
        打一行提示 "(上下文快满了,已在后台开始压缩…)"   ← 立即返回
        ══════════ 主线程回到输入提示符,聊天继续 ══════════

子进程(后台):读 source.jsonl → 调模型写笔记(含分块)→ 写 notes.md + result.json
              → 把 job.json 状态改成 done/failed → 退出

回合 N+1/N+2/… 结束,turn.post 又触发 → 回到 ① 收割
```

---

## 4. 关键设计决策

### D1. 异步的范围:只外包"调模型写笔记"

慢的部分(调模型,分钟级)交给子进程;危险的部分(改读取口)留在主进程。

子进程产出的是**笔记文本**;验货(validate_summary)、组装(assemble_semantic_view)、安检(validate_compacted_messages)、收益检查、视图激活(create_compacted_view 四步落盘)**全部仍在主进程收割时做**。理由:

- 视图激活要写 active.json、要和主进程的内存 LogManager 状态一致,只能在持有它的进程里做;
- 三道验货一道不少,异步不降低任何安全标准;
- 子进程交的货不行,主进程直接丢弃 job,现状保持——失败语义和现在完全一样。

### D2. 执行体:专用压缩子进程,不是通用工具子代理

两个候选:

| | A. 复用 run_subagent(通用子代理) | B. 专用压缩子进程(推荐) |
|---|---|---|
| 形态 | `python -m my_agent_alpha -n -y`,带完整 agent 循环 | `python -m my_agent_alpha --compact-job <dir>`,只跑"读历史→调模型→写笔记" |
| 工具 | 有工具表(要专门裁剪) | 无工具(记录员本就该 tools=(),现有 _request_summary 同款) |
| 行为确定性 | agent 可能自作主张多轮、调工具、超时 | 一次(或分块多次)模型调用,路径确定 |
| 输入 | 历史塞 prompt(stdin 可解决超长) | 读 source.jsonl 文件,无长度限制 |
| 代码复用 | 少 | 多:直接复用 compaction_llm 的指令文本、分块、_request_summary |

**推荐 B**。所谓"子代理"在这里的正确粒度是 gptme 子进程后端的特例——一个无工具、单任务的子进程。派生基础设施(PythonPath 处理、超时、进程环境)从 subagent.py 复用。将来若通用 subagent 做成异步版(gptme 的 fire-and-forget),两者可以共享同一套"后台 job + 边界收割"骨架。

### D3. 收割点:turn.post 开头,不新增生命周期点

派发和收割都发生在 autocompact_hook 内部(先收割、后派发),符合架构纪律"横切关注点一律做成 hook、不改核心循环"。回合边界也天然保证收割时历史是完整的(没有半截工具交互)。收割成功激活了新视图后,同一次 hook 里的 decide_compaction 基于新视图计算,大概率返回 none——天然避免连环派发。

### D4. 防"过期":前缀快照不可变 + 新增消息接尾(本方案最重要的改造)

同步版的规则是"提交前指纹必须一致,变了就作废"(compaction_llm.py:577)。异步下用户在等待期间继续聊,**指纹几乎必然变**——照搬"作废"会让异步压缩几乎永远失败。

改为**分层处理**:

- **前缀(被摘要的部分)是派发时的快照,不可变**——子进程总结的就是它,永远不会过期;
- **派发后新增的消息不参与摘要,原样接在组装结果的尾部**。这本来就是 assemble_semantic_view 的结构(pinned 头 + 笔记 + 原样尾部),只是"尾部"从"派发时的尾部"扩展为"派发时的尾部 + 派发后新增的全部消息";
- 收割时只做**一个**前提检查:当前有效历史的前 N 条(N = 快照条数)的指纹 == 快照指纹。相等 → 当前历史确实是快照的延伸,可以安全接尾;不等 → 说明期间发生过别的压缩/手动 /compact/视图切换,旧 job 的前缀已经不在当前历史里,**作废**(不是重试)。

新增消息都是完整回合产生的(turn 边界收割),调用/结果配对完整性由现有安检兜住,不会出现半截交互。

### D5. 并发规则:同会话单飞行

- 同一会话**同时最多一个在飞的压缩 job**;已有在飞时,本回合自动压缩整体跳过(trim 也不做),避免互相作废;
- 请求级防线二(超窗砍半重试,chat.py:292)是**临时压缩、不碰视图**,与在飞 job 无任何冲突,照常兜底——异步化不削弱任何现有防线;
- 手动 `/compact`(trim)与在飞 job 冲突:手动命令优先(用户意图最高),执行后将在飞 job 作废(它的前缀前提已被破坏);
- 手动 `/compact summarize` **保持同步**:用户主动触发,愿意等,即时反馈比异步简单得多。

### D6. 状态全放磁盘,收割幂等,可跨会话

job 的全部状态在会话目录里,不依赖父进程活着:

```
<会话目录>/compaction-jobs/<job-id>/
├── job.json     # 状态机 + 元数据(见 §5.1)
├── source.jsonl # 派发时的历史快照(子进程唯一输入,收割时比对指纹)
├── notes.md     # 子进程写出的笔记(收割时的验货对象)
└── result.json  # 子进程执行报告(成功/失败/分块数/耗时)
```

父进程在 job 派发后崩溃也没关系:`/resume` 重新打开会话,下一个回合的 turn.post 扫描 compaction-jobs/,发现已完成的 job 照常收割——前缀比对过了就激活,和没崩过一样。子进程与父进程之间**零通信通道**(没有队列、没有 socket),一切通过文件系统交接,这是 gptme"元数据持久化在 logdir 可重启恢复"的同款思路。

### D7. 三道验货原样保留,全在主进程收割时执行

1. `validate_summary`:七章节齐全 / 不超 2048 tokens / 无伪造工具调用行(compaction_llm.py:209);
2. `validate_compacted_messages`:pinned 提示词一条不少 / 最后一条 user 还在 / call_id 一一配对(compaction.py:322)——注意源改用**收割时的当前历史**,"最新 user 在"的检查自然覆盖派发后新增的输入;
3. 收益检查:省不足 15% 不激活(LLMCompactionConfig.min_savings_ratio)。

收割时组装结果超预算 → 放弃整个 job,保持现状(不打折激活);下回合按新指纹重新评估派发。

### D8. 节流与超时

- **节流照搬**:job 失败(模型故障 / 验货不过)记源指纹,指纹不变不重派(_failed_llm_digest 机制不变);
- **超时兜底**:子进程自己把 job.json 改成终态(原子写);收割时发现 running 超过上限(建议 10 分钟)且进程已死 → 标记 failed。不做跨进程主动杀(gptme 的 control.jsonl 是为长任务取消/转向设计的,压缩任务一次几分钟,作废等价于取消,不值得引入控制通道)。

---

## 5. 详细设计

### 5.1 job.json 状态机

```jsonc
{
  "status": "running",          // pending → running → done | failed | harvested(收割完成)
  "pid": 12345,                 // 子进程 pid(超时探测用)
  "model": "gpt-4o",            // 摘要模型(派发时锁定)
  "created_at": "2026-11-02T10:00:00",
  "source_digest": "sha256:…",  // 派发时全量历史的指纹(收割时前缀比对)
  "source_count": 182,          // 快照条数(收割时取"新增消息"的切点)
  "split_at": 150,              // 派发时算好的切分点:前 150 条摘要,后 32 条原样保留
  "source_view": "compacted-001", // 派发时的激活视图(审计用)
  "config": { "keep_recent_tokens": 8000, "summary_max_output_tokens": 2048, "max_chunk_requests": 6 }
}
```

状态流转:主进程写 pending+running;子进程结束(无论成败)原子改写为 done/failed(带 reason);主进程收割成功后改成 harvested(目录保留作审计;笔记拷贝进 views/summaries/ 与现有习惯一致)。

**split_at 在派发时算好并锁定**:子进程按它切前缀/尾部,收割时按它组装——两边用同一个数,收割逻辑不重新 split(避免主/子进程 config 不一致导致组装错位)。

### 5.2 子进程入口(--compact-job)

新增一个 CLI 入口(如 `python -m my_agent_alpha --compact-job <job-dir>`),只做四件事:

```
读 job.json(model/config/split_at) + source.jsonl
  → split_for_semantic_compaction 按锁定切分点取前缀      (复用 compaction_llm)
  → bound_summary_input 裁超长消息 → 装得下就一次调用,
    装不下就 chunk_prefix 分块 → 逐块小结 → 合并           (全部复用现有纯函数)
  → validate_summary 本地预验一遍(不合格直接标 failed,省得收割一趟)
  → notes.md + result.json 落盘,job.json 改 done
```

子进程**没有**工具注册、确认链、钩子、会话日志——它不是 agent 循环,是一次结构化模型调用(理由见 D2)。

### 5.3 派发(turn.post 的 summarize 分支改造)

```
_auto_summarize 改为 _dispatch_compaction_job:
  1. 有在飞 job? → 跳过(D5)
  2. source = manager.messages
     digest = digest_messages(source)
     digest == _failed_llm_digest? → 节流跳过(D8)
  3. split_at = split_for_semantic_compaction(...) 的切点   ← 主进程算,锁定
  4. 写 job 目录(source.jsonl + job.json)
  5. Popen 派子进程(不等待;复用 subagent.py 的 env/PYTHONPATH 处理)
  6. print("(上下文接近预算,已在后台压缩,不影响聊天;下个回合生效)")
```

### 5.4 收割(turn.post 开头)

```
_harvest_compaction_job(manager, model):
  1. 扫 compaction-jobs/:没有 running/done 的 job → 返回
  2. running 且未超时 → 返回(还在飞)
  3. running 且超时 → 标 failed,走 6
  4. done:
     a. 前缀比对:digest(manager.messages[:source_count]) == source_digest?
        不等 → 作废,标 harvested(附作废原因)
     b. 读 notes.md → validate_summary
     c. 新尾部 = manager.messages[split_at:](含派发后新增消息,D4)
        新历史 = assemble_semantic_view(当前历史, 笔记, 新尾部, …)
     d. validate_compacted_messages(当前历史, 新历史)
     e. 收益检查(tokens_after ≤ before × 0.85 且 ≤ 预算)
     f. manager.create_compacted_view(新历史)          ← 现有四步安全落盘,零改动
        笔记拷贝 views/summaries/,job 标 harvested
  5. 成功 → print 一行(现有格式);失败 → 记节流指纹 + 一行原因
  6. 失败/作废都保持现状,旧视图原样继续用
```

整条收割路径没有一次模型调用,毫秒级,turn.post 保持非阻塞。

### 5.5 冲突矩阵

| 事件 | 在飞 job 的命运 |
|---|---|
| 等待期间用户继续聊(最常见) | 前提不破坏 → 收割时新增消息接尾,**正常激活** |
| 等待期间预算爆了(请求被拒) | 防线二临时压缩救场(不碰视图);job 继续飞 |
| 等待期间手动 /compact(trim) | 新视图激活 → 前缀比对失败 → job 作废 |
| 等待期间手动 /compact summarize | 同步通道先完成 → job 作废(同上) |
| 父进程崩溃/退出 | 子进程跑完落盘;resume 后首个 turn.post 收割 |
| 子进程崩溃/超时 | job 标 failed,记节流指纹,现状保持 |
| 下个回合仍超线且 job 在飞 | 跳过派发(单飞行),等收割后再评估 |

---

## 6. 兼容性核对(不破坏现有纪律)

| 现有纪律 | 本方案 |
|---|---|
| 主日志 conversation.jsonl 永不销毁 | 不碰;job 目录是新增的旁路数据 |
| 消息不可变,修改=新实例 | 组装仍走 assemble_semantic_view/replace |
| call_id 恰好配一个结果 | 安检原样跑;收割在回合边界,无半截交互 |
| 横切关注点做成 hook,不改核心循环 | 派发/收割都在 autocompact_hook 内,chat.py 零改动 |
| 压缩状态不进模型上下文 | 派发/收割提示仍只 print 给人看 |
| 视图激活四步安全顺序 | create_compacted_view 原样复用 |
| trim 通道 | 完全不变(仍同步、免费、即时) |

---

## 7. 实施阶段划分(供后续开发参考)

1. **job 协议层**:job.json 状态机 + 目录读写 + 指纹/切分点序列化(纯函数,可独立测试);
2. **子进程入口**:`--compact-job`,复用 compaction_llm 纯函数,本地预验货;
3. **派发**:_auto_summarize 改造 + 单飞行检查 + 节流;
4. **收割**:前缀比对 + 尾部扩展组装 + 三道验货 + 激活;
5. **治理**:超时兜底、/context 显示在飞 job、/compact 冲突处理;
6. **测试**:
   - 集成(mock 模型):派发 → 中间聊两轮 → 收割激活 → 断言视图内容与新增消息接尾正确;
   - 崩溃恢复:派发后杀主进程 → resume → 收割;
   - 冲突:在飞时手动 /compact → job 作废;子进程产出坏笔记 → 验货拒绝 + 节流;
   - 回归:现有 autocompact/compaction 测试改造(同步 _auto_summarize 路径保留给手动命令)。

每阶段一篇 devlog,走既有流程(AGENTS.md 规则 1)。

---

## 8. 风险与开放问题

1. **笔记质量与新鲜度的权衡**:笔记永远落后一个以上回合(派发时的快照)。接尾机制保证不丢消息,但"最新意图"只存在于尾部原文里——这正是七章节笔记里 Next Action 存在的意义,风险可控;
2. **磁盘占用**:每个 job 存一份全量快照(source.jsonl)。长会话约几 MB,收割后可只留 job.json + notes.md、删快照(审计需要的是指纹和笔记,不是原文——原文在主日志里);
3. **ALPHA_AUTO_LLM_COMPACT 默认值**:异步化解决的是体验问题,要不要顺势默认开启?建议先保持默认关,跑一段手动开启的验证再议;
4. **多会话并发**:两个终端开同一工作区的不同会话 → 不同会话目录,天然隔离,无共享状态(与 gptme 的 (logdir, branch) 冷却键同构,但我们更简单:状态本来就分会话目录);
5. **子进程的 stdin/信号**:子进程完全脱离终端运行,需确认模型 SDK 在无 TTY 环境下的行为(现有 subagent 已验证过这条路,风险低);
6. **分块 6+1 次调用的耗时**:最坏情况子进程要跑几分钟,超时上限(建议 10 分钟)要覆盖它;超时值应随分块数放大。

---

## 9. 与 gptme 的最终对照

| 关注点 | gptme | 本方案 |
|---|---|---|
| LLM 压缩执行 | TURN_POST 同步 `_resume_via_llm` | 子进程异步;turn.post 派发 + 收割 |
| 过期处理 | 快照比对,过期丢弃 | 前缀快照不可变 + 新增消息接尾(改造点) |
| 完成通知 | loop.continue/step.pre 钩子排空队列 | turn.post 收割(同属回合边界) |
| 取消 | control.jsonl + STEP_PRE 检查点 | 单飞行 + 冲突作废(MVP 不做主动取消) |
| 重启恢复 | logdir 元数据重建 | job 状态全磁盘,resume 后首个 turn.post 收割 |
| 并发限制 | SubagentBudget + 信号量 | 同会话单飞行 |
| 派生即返回 | subagent() 立即返回 | Popen 不等待,一行提示后回提示符 |
