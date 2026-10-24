# 上下文压缩机制 · 移植开发文档

> **用途**：把 gptme 的 autocompact 机制移植到自己的项目。本文档是开发的唯一依据——所有接口、算法、阈值、消息格式、边界条件都已写死在文中，开发时**不需要**回头读 gptme 源码。
> **来源**：`gptme/tools/autocompact/`（9 文件 ≈1800 行）+ 依赖 `util/master_context.py`、`util/output_storage.py`、`util/reduce.py`、`context/compress.py::strip_reasoning`。行为规格忠实还原自 2026-10 仓库状态（master@c10397da4）。
> **配套阅读**：模块总览笔记 [autocompact.md](autocompact.md)（理解设计动机用，开发以本文档为准）
> **编写日期**：2026-10-22

---

## 目录

- [§1 移植目标与范围](#1-移植目标与范围)
- [§2 总体架构（移植版）](#2-总体架构移植版)
- [§3 核心数据契约](#3-核心数据契约)
- [§4 适配器接口规格（目标项目实现）](#4-适配器接口规格目标项目实现)
- [§5 核心算法规格（直接照抄实现）](#5-核心算法规格直接照抄实现)
- [§6 辅助机制规格](#6-辅助机制规格)
- [§7 参数总表](#7-参数总表)
- [§8 不变量与边界条件清单](#8-不变量与边界条件清单)
- [§9 测试规格](#9-测试规格)
- [§10 开发任务分解（M0–M7）](#10-开发任务分解m0m7)
- [§11 未决问题（开发前需确认）](#11-未决问题开发前需确认)

---

## 1. 移植目标与范围

**移植（本文档规格覆盖，≈可移植核心）**

| 组件 | gptme 位置 | 说明 |
|---|---|---|
| 压缩决策 | decision.py | 三值决策：不压 / 规则压 / LLM 摘要 |
| 三阶段引擎 | engine.py | 剥推理 → 截工具结果 → 抽取式压缩 |
| 句子打分 | scoring.py | 纯函数，零依赖 |
| LLM 摘要 | resume.py | RESUME.md 生成 + 上下文文件提取 |
| Provider 抽象 | context_provider.py | ABC + 注册表 + 结果元数据 |
| 推理剥离 | context/compress.py::strip_reasoning | 纯函数，零依赖 |
| Master context | util/master_context.py | 字节区间索引 + 恢复引用 |
| 截断摘要 | util/output_storage.py | 落盘 + 替换消息格式 |
| 兜底压缩器 | util/reduce.py::reduce_log | 引擎压不干净时的最后手段 |

**不移植（用适配器接口替代，见 §4）**

| gptme 依赖 | 替代策略 |
|---|---|
| LogManager 视图分支（create_view/switch_view/双写） | `ConversationStore` 适配器：最小实现可以只是"快照文件" |
| hooks 系统（TURN_POST / CACHE_INVALIDATED） | `on_turn_end` 触发点 + 可选 `on_compacted` 回调 |
| ToolSpec 工具注册、`/compact` 命令 | 移植后按目标项目的方式挂命令，可后置 |
| llm.models 模型注册表（get_default_model） | `ModelInfoProvider` 适配器 |
| tiktoken 计数 | `TokenCounter` 适配器（可降级为 chars/4） |
| llm.reply | `LLMClient` 适配器 |

---

## 2. 总体架构（移植版）

```
目标项目对话循环
   │  每个 turn 结束（所有工具步骤跑完）
   ▼
┌─ 触发层（节流 60s）─────────────────────────────┐
│  should_auto_compact(log)  → none / rule_based / summarize │
└──────┬──────────────────────┬──────────────────┘
       │ rule_based           │ summarize
       ▼                      ▼
┌─ Provider 接口 ──┐   ┌─ LLM 摘要 ─────────────┐
│ DefaultProvider  │   │ 生成 RESUME.md          │
│  → auto_compact  │   │ 提取 Context Files      │
│    三阶段引擎     │   │ 重建精简日志             │
│  → 仍超限则       │   └────────────────────────┘
│    reduce_log    │
└──────┬───────────┘
       ▼
┌─ 落地（ViewBranchStore，§4，已确认方案）─────────┐
│ 压缩结果 → create_view("compacted-00N") → 切换    │
│ 之后每条新消息双写 master + view；master 永不压缩  │
└──────┬────────────────────────────────────────┘
       ▼
  on_compacted 回调（可选：通知缓存失效）
```

压缩期间产生的所有"被删内容"有三条恢复通路：**master 日志字节引用**（精确还原）、**全量落盘文件**（grep 友好）、**未动的原始日志**（快照/视图方案下天然存在）。

---

## 3. 核心数据契约

### 3.1 Message

```python
@dataclass(frozen=True)  # frozen 重要：消息不可变，压缩用 replace 产生新实例
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    pinned: bool = False  # 钉住的消息任何压缩阶段都跳过
    # replace(content=..., pinned=...) → 返回修改后的副本
```

**关键约定：工具结果以 `role="system"` 落日志**（gptme 的做法）。决策层"巨型工具结果"判定、引擎 Phase 2 的候选筛选都匹配 `role == "system"`。
移植决策点：若目标项目已有 `role="tool"` 之类的显式角色，把 §5 中所有 `role == "system"` 的谓词换成 `is_tool_result(msg)` 一个函数，**集中定义、全局唯一**，不要散落比较。

注意：system 角色同时承担"系统提示词"。系统提示词靠 §5.3 的 `keep_head` 头部保护（默认前 2 条），而不是靠角色区分。

### 3.2 ModelInfo

```python
@dataclass
class ModelInfo:
    name: str  # 供 token 计数器选 tokenizer
    context: int  # 上下文窗口大小（token 数）
```

引擎在拿不到默认模型时需要一个兜底 ModelInfo（gptme 用 gpt-4 的元数据）。移植时定义为常量即可，`context` 取目标项目的主力模型窗口。

---

## 4. 适配器接口规格（目标项目实现）

核心算法只依赖下面 6 个接口。M0 阶段先用 fake 实现跑通骨架，真实实现按目标项目情况替换。

```python
class TokenCounter(Protocol):
    def len_tokens(self, obj: str | Message | list[Message]) -> int: ...
    # 字符串、单条消息、整个日志都要能数。
    # 降级实现：sum(len(str(content))) // 4（gptme 的兜底就是 chars/4）。
    # 一致性要求：同一输入必须稳定返回同一值（决策层与引擎层各自调用，漂移会导致决策误判）。


class ModelInfoProvider(Protocol):
    def get_default_model(self) -> ModelInfo | None: ...
    # None 时核心用 §3.2 的兜底常量。


class ToolUseDetector(Protocol):
    def contains_tool_use(self, msg: Message) -> bool: ...
    # True = 这条 assistant 消息含工具调用，压缩会破坏 tool_call/tool_result 配对。
    # gptme 实现：对 tool 和 xml 两种格式各跑一遍解析，任一解析出调用即 True。
    # 移植按目标项目的工具调用表示实现；没有工具系统则恒返回 False。


class OutputStore(Protocol):
    def save(self, content: str, meta: dict) -> Path: ...
    # 把被截断的全量内容写入文件，返回路径。文件名建议含序号/时间戳。
    # 目录约定：每会话一个目录（gptme: logdir/outputs/autocompact-*.txt）。


class LLMClient(Protocol):
    def reply(self, messages: list[Message], model: str) -> str: ...
    # 非流式一次调用即可，供 §5.5 生成 RESUME。


class ConversationStore(Protocol):
    """视图分支落地（✅ 已确认方案）。gptme 语义：master 永不压缩，
    压缩结果放视图；切到视图后新消息双写两处。"""

    # 读取：log 访问器。current_view 非空 → 返回视图内容；否则返回 master。
    # ⚠ 触发器/引擎拿去压缩的 messages 永远来自 log 访问器（当前生效上下文）。

    def create_view(self, name: str, messages: list[Message]) -> None: ...
        # 注册视图（内存）+ 持久化 views/{name}.jsonl
    def switch_view(self, name: str) -> None: ...
        # 先落盘当前状态，再 current_view = name；此后读 log 即读视图
    def switch_to_master(self) -> None: ...
        # current_view = None，随时切回完整历史
    def append(self, msg: Message) -> None: ...
        # 双写：在视图上 → master 与当前视图都追加；不在视图 → 只写当前日志
    def persist(self) -> None: ...
        # 在视图上 → master 新消息 append 进 conversation.jsonl（永远全量历史）
        #           + 视图写自己的文件；不在视图 → 当前日志写 conversation.jsonl
    def next_view_name(self) -> str: ...
        # 扫描已有 "compacted-N" 视图名取 max+1，3 位零填充 → "compacted-001"…

**索引对齐不变量（视图方案下 master 引用正确性的根基）**：压缩是 1:1 原位替换
（不增删消息），视图创建时与 master 等长；双写又保证之后两边同步增长 →
**任意时刻 len(当前视图) == len(master)**。因此再次压缩视图时，消息下标 i 仍对应
conversation.jsonl 第 i 行，§6.2 字节引用持续有效。任何破坏等长的实现（如在视图上
增删消息）都是 bug，§9 触发器组要为此专设断言。
```

**触发点**：在目标项目对话循环的"一个 turn 完全结束"处调用 §6.4 的触发函数。
**可选回调**：`on_compacted(reason, tokens_before, tokens_after)`——压缩必然使 prompt cache 失效，给上层留通知口。

---

## 5. 核心算法规格（直接照抄实现）

以下伪代码即实现标准。`CTX` 指当前模型的 `context` 窗口值。

### 5.1 strip_reasoning（Phase 1 的实现，纯函数）

```
输入: content, model
输出: (stripped_content, tokens_saved)

stripped = re.sub(r"<think>.*?</think>",      "", content, flags=DOTALL)
stripped = re.sub(r"<thinking>.*?</thinking>", "", stripped, flags=DOTALL)
stripped = re.sub(r"\n\n\n+", "\n\n", stripped)   # 多个连续空行压成两个
stripped = stripped.strip()
tokens_saved = len_tokens(content) - len_tokens(stripped)
```

### 5.2 决策层

```
should_auto_compact(log) -> "none" | "rule_based" | "summarize":

    limit = int(0.9 * CTX)                          # 触发线
    total  = len_tokens(log)
    close  = total >= int(0.5 * CTX)                # 前瞻线（故意激进）
    massive = 存在 msg: ¬pinned ∧ role=="system" ∧ len_tokens(msg.content) > 2000

    if not (total > limit or (close and massive)):
        return "none"

    (_, savings, _) = estimate_compaction_savings(log, limit)
    ratio = savings / total                          # total>0 由上面保证
    return "rule_based" if ratio >= 0.10 else "summarize"
```

```
estimate_compaction_savings(log, limit) -> (total, savings, reasoning_savings):
    # 干跑：不真正压缩，按启发式估三个阶段的节省量

    limit = limit or int(0.9 * CTX)
    would_remove = total > limit or total >= int(0.8 * CTX)   # 注意这条线是 0.8

    savings_tool = savings_reason = savings_compress = 0
    for i, msg in enumerate(log):
        if msg.pinned: continue
        tok = len_tokens(msg.content); dist = len(log) - i - 1

        # P1：距末尾≥5 且含 <think>/<thinking> → 按内容 30% 估
        if dist >= 5 and ("<think>" in msg.content or "<thinking>" in msg.content):
            savings_reason += int(tok * 0.3)
        # P2：会触发截断 且 system 且 >2000 tok → 按 原量-200 估
        if would_remove and msg.role == "system" and tok > 2000:
            savings_tool += tok - 200
        # P3：会触发截断 且 dist≥3 且 assistant 且 >1000 tok → 按 30% 估
        if would_remove and dist >= 3 and msg.role == "assistant" and tok > 1000:
            savings_compress += int(tok * 0.3)

    return (total, savings_tool+savings_reason+savings_compress, savings_reason)
```

### 5.3 引擎 auto_compact_log（三阶段）

```
auto_compact_log(log, limit=None, max_tool_result_tokens=2000,
                 reasoning_strip_age=5, store=None, keep_head=0) -> list[Message]

# ── 准备 ─────────────────────────────────────────────
master_index = build_byte_index(主日志)        # §6.2；无主日志则为空表
limit  = limit ?? int(0.8 * CTX)               # 工作上限 0.8
tokens = len_tokens(log)
close  = tokens >= int(0.7 * CTX)              # 提前返回线 0.7
n = len(log)

needs_p1 = ∃i: (n-i-1) >= 5 ∧ ("<think>"∈c ∨ "<thinking>"∈c)
needs_p2 = tokens > limit ∨ close
needs_p3 = ∃i: (n-i-1) >= 3 ∧ role=="assistant" ∧ ¬tool_use ∧ len_tokens(c) > 1000

if ¬needs_p1 ∧ ¬needs_p2 ∧ ¬needs_p3:
    return [前 keep_head 条标 pinned=True，其余原样]     # 早退也要钉头部

# ── Phase 1：按年龄剥推理 ─────────────────────────────
for i, msg in enumerate(log):
    if msg.pinned or i < keep_head: continue
    if (n - i - 1) >= reasoning_strip_age:
        (c, saved) = strip_reasoning(msg.content)
        if saved > 0: log[i] = msg.replace(content=c)

# ── Phase 2：最大优先截断工具结果 ─────────────────────
current = len_tokens(log); target = int(0.8 * CTX)
if current > target:
    candidates = [(i, tok, msg) for i,msg in enumerate(log)
                  if ¬msg.pinned ∧ i >= keep_head ∧ msg.role=="system"
                  ∧ (tok=len_tokens(msg.content)) > max_tool_result_tokens]
    candidates.sort(by=tok, desc)
    for (i, tok, msg) in candidates:
        if current <= target: break                      # 达标即停
        summary = create_tool_result_summary(msg, tok, store)   # §6.1 格式
        if master_index 且 i < len(master_index):
            summary += "\n\n" + master_reference(i, tok, msg)    # §6.2 格式
        log[i] = msg.replace(content=summary)
        current -= tok - len_tokens(summary)

# ── Phase 3：抽取式压缩长 assistant 消息 ──────────────
# ⚠ 忠实还原：Phase 3 没有 token 压力门槛——只要没走早退，
#   所有合格消息都会被压缩（哪怕当前远低于 target）。
for i, msg in enumerate(log):
    if msg.pinned or i < keep_head: continue
    if (len(log)-i-1) >= 3 ∧ msg.role=="assistant"
       ∧ ¬tool_use(msg) ∧ len_tokens(msg.content) > 1000:
        c = compress_content(msg.content, target_ratio=0.7)      # §5.4
        if len_tokens(c) < len_tokens(msg.content):              # 压完反而更大则跳过
            if master_index 且 i < len(master_index):
                c += "\n\n" + master_reference(i, tok, msg)
            log[i] = msg.replace(content=c)

# ── 出口 ─────────────────────────────────────────────
final = len_tokens(log)
if final <= limit:
    return [前 keep_head 条标 pinned=True，其余原样]

# 兜底：reduce_log（§6.3）
if 0 < keep_head < len(log):
    head = log[:keep_head] 全部 pinned=True
    return head + reduce_log(log[keep_head:], limit - len_tokens(head))
elif keep_head >= len(log):
    return log 全部 pinned=True        # 头部独占超限：宁可超预算也保留任务上下文
else:
    return reduce_log(log, limit)
```

**索引对齐前提**（master 引用正确性的根基）：三阶段全部是**原位替换**（1:1），消息条数与顺序不变，所以压缩后日志的第 i 条 ↔ 主日志第 i 行。Phase 2/3 里所有 `idx` 直接可用于字节索引。**移植时严禁在任何阶段增删消息**。

**pinned 传播的原因**：下游（发给 LLM 前的最终裁剪）只认 `pinned` 标志不认索引，头部不钉住会被最终裁剪丢掉。

### 5.4 打分与抽取式压缩（scoring）

**单句打分** `score_sentence(sentence, position, total) -> float`，各项累加：

| 维度 | 条件 | 分值 |
|---|---|---|
| 位置 | position == 0（首句） | +2.0 |
| | position == total-1（末句） | +1.5 |
| | position < 3 | +1.0 |
| 关键词 | 命中以下任一词各 +0.5（可叠加）：error, fail, success, complete, implement, fix, bug, issue, result, output, TODO, FIXME, NOTE, WARNING（大小写不敏感） | 每词 +0.5 |
| 长度 | len < 10 字符 | −1.0 |
| | len < 50 | +0.3 |
| | len > 200 | −0.2 |
| 语义·决策 | 命中任一（只加一次）：we'll use / we will use / decided to / going with / choosing / solution is / approach is / we chose | +2.0 |
| 语义·结论 | therefore / in summary / the result is / this means / confirmed that / in conclusion / key finding | +1.5 |
| 语义·承诺 | I'll；i will (implement\|create\|fix\|add\|update\|write\|build)；next steps?；action items?；todo:；will implement；plan to；going to (implement\|create\|fix\|add\|update\|write\|build) | +1.5 |
| 语义·动作结果 | created file / fixed / updated / implemented / completed / merged | +1.0 |
| 引用·文件路径 | 命中路径正则（见下） | +1.0 |
| 引用·URL | 命中 URL 正则 | +0.5 |
| 引用·错误 | error\|exception\|traceback / failed / failure（任一） | +1.5 |

正则（大小写不敏感；语义/错误类每类内命中一次即停，不叠加）：

```
文件路径: (?:[/~][a-zA-Z0-9_\-./]+(?:\.[a-zA-Z0-9]+)?|[A-Za-z]:[/\\][a-zA-Z0-9_\-./\\]+(?:\.[a-zA-Z0-9]+)?)
URL:      https?://[^\s<>"\')]+
错误:     \b(error|exception|traceback)\b | \bfailed\b | \bfailure\b
```

**压缩流程** `compress_content(content, target_ratio=0.7) -> str`：

```
1. 抽代码块：re.sub(r"```[\s\S]*?```", 占位符) —— 每个 ``` 围栏块换成 __CODE_BLOCK_i__
   （代码块无条件保留，永不参与打分与删除）
2. 切句：re.split(r"(?<=[.!?])\s+", 去块后的文本)
3. 句数 <= 3 → 原样返回（含代码块还原）
4. 分两组：含占位符的句子（标记句，全保）/ 其余（可打分句）
5. 可打分句逐个 score_sentence(句子, 原始位置i, 总句数)   # ⚠ position 用它在
   完整句列表中的原始下标，不是组内下标；total 含标记句
6. 按分数降序取前 target_count = max(2, int(可打分句数 × 0.7)) 句
7. 选中句 + 标记句 合并，按原始位置升序排序，" ".join 拼接
8. 占位符还原为代码块，返回
```

### 5.5 LLM 摘要（resume）

```
resume_via_llm(manager, msgs, use_snapshot) -> 新日志:

1. prepared = 预处理(msgs)          # 目标项目发送前的等价处理；<3 条 → 拒绝并返回提示
2. 发给 LLM（模型用默认模型；禁用工具调用）：
   消息 = prepared + [user(RESUME_PROMPT)]     # 原始消息在前，prompt 摘要在尾，利于缓存
3. RESUME_PROMPT 原文（输出格式的契约，不可改）：
```

```
Please create a comprehensive resume of this conversation that includes:

1. **Conversation Summary**: Key topics, decisions made, and progress achieved
2. **Technical Context**: Important code changes, configurations, or technical details
3. **Current State**: What was accomplished and what remains to be done
4. **Context Files**: List the specific files that should be included in future context

For the Context Files section, use this format:
## Context Files

List each file on its own line with a bullet point and backticks:
- `path/to/file.py` - Brief rationale for including this file
- `docs/spec.md` - Contains the specification being implemented

Include files such as:
- Specs, PRDs, or design documents being referenced
- Source files being actively modified
- Configuration files relevant to the work
- Task or plan files tracking progress

Format the response as a structured document that could serve as a RESUME.md file.
```

```
4. 摘要全文写入 会话目录/RESUME.md（失败仅告警，不中断）
5. 解析上下文文件清单 parse_context_files(摘要)：
   - 找小节：r"(?:#{1,4}\s*(?:Context Files|Files to Include|Recommended Files|Key Files)[^\n]*\n)([\s\S]*?)(?=\n#{1,4}\s|\Z)"（IGNORECASE）
     找不到小节 → 扫全文
   - 小节内按序尝试三种模式提取路径：
       r"[-*]\s*`([^`]+)`"                                    # 反引号包裹
       r"[-*]\s*([/~.][\w./\-]+(?:\.\w+)?)"                   # / ~/ ./ 开头
       r"[-*]\s*((?:src|docs|tests|config|scripts|tasks|journal|knowledge|lessons)/[\w./\-]+(?:\.\w+)?)"   # 常见相对路径
   - 过滤：http 开头、# 开头；去掉 ./ 前缀；去重保序
6. 加载文件 load_context_files(paths, workspace, 每文件上限2000 token)：
   - 绝对路径直用；~ 展开；其余相对 workspace
   - 只读存在且是文件的；读失败仅告警
   - 超限：逐行累加 token，到 (上限-50) 停，追加 "\n... (truncated, N tokens remaining)"
7. 组装新日志（顺序固定）：
   [头部连续 system 消息（遇首个非 system 停）]      # 保住系统提示词
   + [每个文件一条 system: "Context file `路径`:\n" + 代码块包裹的内容]
   + [system: "Previous conversation resumed from {RESUME.md路径|LLM-generated summary}( with N context files):"]
   + [assistant: 摘要全文]                          # 摘要伪装成 assistant 消息
   其余历史全部丢弃。
8. 落地：自动触发 → store.create_view(store.next_view_name(), 新日志) + switch_view；
   手动命令 → 直接替换当前日志
```

### 5.6 Provider 接口与注册表

```python
class ContextProvider(ABC):
    name: str                                            # 注册与查找键
    def should_compress(messages, config) -> bool
    def compress(messages, config) -> CompactionResult

@dataclass
class CompactionResult:
    messages: list[Message]        # 压缩后的消息流
    source_digest: str             # 源消息的 SHA-256（见下），用于检测压缩后源是否又变
    covered_through: int = -1      # 覆盖到源日志第几条（0-based；全量压缩 = len-1，空输入 = -1）
    limitations: list[str] = field(default_factory=list)   # 人类可读的损失说明

@dataclass
class CompressionConfig:
    limit: int | None = None
    max_tool_result_tokens: int = 2000
    store_dir: Path | None = None        # 落盘/主日志目录
    reasoning_strip_age_threshold: int = 5
    keep_head: int = 0
```

**source_digest 算法**：对每条消息依序 `sha256.update(role.utf8); update(b"\x00"); update(content.utf8); update(b"\x00")`，取最终 hex。

**注册表语义**：
- `register_provider(name, cls)`：运行时注册，后写覆盖（同名直接替换）；
- 初始化时**先**加载 entry-point 插件组（gptme 用 `gptme.context_providers`，移植可改名或砍掉），**再** `setdefault("default", DefaultContextProvider)`——即插件可以用名为 `default` 的实现**整体替换内置算法**，先注册者优先；
- `get_context_provider(name)` 实例化；`list_providers()` 列名。

**DefaultContextProvider**：`compress` = 计算 digest → 调 §5.3 引擎 → 包成 CompactionResult（covered_through = len-1，limitations = ["rule-based truncation of tool results"]）。**引擎要在 compress 方法体内延迟导入**（import 放函数里），避免不用压缩也拖起整条依赖链。

---

## 6. 辅助机制规格

### 6.1 截断摘要消息（create_tool_result_summary）

替换 Phase 2 被截消息的文本，格式：

```
无落盘目录：
  [Large tool output removed - {N} tokens]. Output was automatically removed
  due to size to allow conversation continuation.

有落盘目录：
  [Large tool output removed - {N} tokens]. Full output saved to: {文件路径}
  You can read or grep this file if needed.
```

可选前缀（移植可砍）：若内容前 10 行里有以 `Ran command:` / `Executed:` / `Command:` 开头的行，在 `[...]` 后追加 ` ({该行原文})`。

### 6.2 Master context 字节引用

**索引构建**：主日志为 append-only JSONL（每行一条消息）。顺序读文件按字节累计，第 i 行 → `(message_idx=i, byte_start, byte_end)`。

**引用文本**（追加在被截消息尾部，前空一行）：

```
[Content truncated - {N} tokens]
Master context: {主日志路径} (bytes {start}-{end})
Preview: {首行前200字符}{超200时加...}
To recover: grep or read the master context file at the byte range above.
```

**恢复**：`seek(byte_start)` → `read(byte_end - byte_start)` → 按行解析 JSON 取 content。

### 6.3 兜底压缩器 reduce_log

引擎三阶段后仍超 `limit` 时使用。行为（忠实还原，注意 gptme 源码里 "attempt summarize" 注释是遗迹，**实际不调 LLM**）：

```
reduce_log(log, limit):
    if len_tokens(log) <= limit: return log
    candidates = [非pinned ∧ ¬tool_use 的消息]
    if 空: 告警并原样返回（压不动了）
    longest = candidates 中 content token 最大者
    t = truncate_msg(longest):
        - 含 tool_use → None（不可动）
        - 对每个 ``` 代码块：> 21 行时保留前10+后10行，中间换 [...]
          （gptme 还同样处理 HTML <details> 块；目标项目没有可砍）
        - 无任何可截内容 → None
    if t is None: 原样返回（不再有进展）        # truncate-or-bail
    else: 原位替换，重数 token
    达标 → 返回；未达标且 token 数有下降 → 递归；无下降 → 原样返回
```

### 6.4 触发器（含节流）

```
on_turn_end(manager):                       # 每 turn 末尾调用
    if now - last_attempt < 60s: return     # 进程级状态；防 压缩→写消息→又触发 死循环
    action = should_auto_compact(manager.messages)
    if action == "none": return
    last_attempt = now                      # ⚠ 只有真正要动手才刷新时间戳
    try:
        if action == "rule_based":
            result = get_context_provider("default").compress(messages, config)
            view = store.next_view_name()
            store.create_view(view, result.messages)
            store.switch_view(view)
            on_compacted(tokens_before, tokens_after)     # 可选
            写一条系统提示消息告知压缩结果（自动触发时应标记对用户隐藏/不触发新回复）
        else:  # summarize
            resume_via_llm(...) → 落地
    except Exception: 记日志，静默返回       # 压缩失败绝不能打断对话主流程
```

配置解析（keep_head）：优先环境变量 `GPTME_AUTOCOMPACT_KEEP_HEAD`（移植可改名），非法/负值回退默认 2。

---

## 7. 参数总表

| 参数 | 值 | 用在 | 备注 |
|---|---|---|---|
| 触发线 | `0.9 × CTX` | 决策：超限即候选 | |
| 前瞻线 | `0.5 × CTX` | 决策：接近+有巨型结果即候选 | 故意激进 |
| 估算线 | `0.8 × CTX` | 决策：估算 P2/P3 是否会计入 | |
| 工作上限 | `0.8 × CTX` | 引擎：limit 默认值 & P2 target | |
| 提前返回线 | `0.7 × CTX` | 引擎：close_to_limit | |
| MIN_SAVINGS_RATIO | 0.10 | 决策：低于则走 LLM 摘要 | 缓存失效代价 |
| 巨型工具结果阈值 | 2000 token | 决策 + P2 候选 | |
| 推理剥离年龄 | 距末尾 ≥ 5 | P1 | |
| P3 年龄 / 角色 / 最小体量 | ≥ 3 / assistant / > 1000 tok | P3 | 且 ¬tool_use |
| P3 压缩比 | 0.7（保留 70%） | compress_content | |
| 句子最少保留 | max(2, ⌈比例×0.7⌋) | compress_content | |
| keep_head 默认 | 2 | 全阶段 | 环境变量可覆盖 |
| 重入间隔 | 60 秒 | 触发器 | 进程级 |
| 摘要最少消息数 | 3 条（预处理后） | resume | |
| 上下文文件单文件上限 | 2000 token | resume | 截断留 50 token 余量 |
| digest | SHA-256 | CompactionResult | role+\0+content+\0 逐条 |
| 兜底估算 | chars // 4 | TokenCounter 降级 | |
| 截断代码块行数 | 前 10 + 后 10 | reduce_log | 中间 `[...]` |

> 三层阈值（0.5/0.7/0.8/0.9）**故意不一致**：决策层要抢在爆窗前动手，引擎层要留余量。移植时不要"统一参数"。

---

## 8. 不变量与边界条件清单

**不变量（违反即 bug）**

1. 三阶段与 reduce_log 的截断全部**原位替换**：消息条数、顺序、角色不变 → 索引与主日志字节区间始终对齐。
2. `pinned == True` 的消息任何阶段不改动内容。
3. 含工具调用的 assistant 消息永不被 P3 / reduce_log 压缩（保 tool_call/result 配对可解析）。
4. `i < keep_head` 的消息与 pinned 同等待遇；所有出口（含早退、兜底）都以 `pinned=True` 输出头部。
5. 代码块（``` 围栏）在 P3 中无条件保留。
6. 压缩后若某消息 token 不降反升（P3、引用文本引入），保持原消息。

**边界条件（须逐一处理并测试）**

| # | 场景 | 期望行为 |
|---|---|---|
| 1 | 空日志 / 很短日志 | 早退原样返回 |
| 2 | 全部消息 pinned | needs_* 全 False → 早退；即使超限也原样返回 |
| 3 | keep_head ≥ len(log) 且超限 | 全部 pinned 原样输出，**接受超预算**（任务上下文优先） |
| 4 | 头部自身超 limit | 兜底分支给尾部 `limit - head_tokens`（可到 0），头部完整保留 |
| 5 | 消息 ≤ 3 句 / 全是代码块 | compress_content 原样返回 |
| 6 | 主日志不存在 / 行数 < 消息数 | 不加 master 引用，照常压缩（`i < len(index)` 守卫） |
| 7 | 决策说 rule_based 但实际压不下来 | 引擎兜底 reduce_log；再不行原样返回（不抛异常） |
| 8 | LLM 摘要时无默认模型 / 调用失败 | 返回错误提示消息，不破坏原日志 |
| 9 | 建议的上下文文件不存在 | 跳过并告警；路径去重保序 |
| 10 | 压缩过程中任何异常 | 触发层 catch-all 记日志，对话继续 |

---

## 9. 测试规格

移植 gptme 已有测试场景（对应 tests/test_auto_compact.py、test_autocompact_scoring.py、test_context_providers.py）。**每个里程碑合入前对应测试必须全绿。**

**决策层**
- 总量低于 0.5 CTX → none；超 0.9 CTX 无巨型结果 → 仍候选（超限路径）→ 估算后定去留
- 0.5–0.9 CTX + 巨型 system 消息 → rule_based
- 超限但内容压不出 10% 节省（如全是短 user 消息）→ summarize
- pinned 的巨型消息不计入 massive 判定

**引擎**
- 距末尾 ≥5 的消息 `<think>` 被剥、<5 的保留；剥离后空行清理正确
- P2 从最大开始截：给出 3 个不同大小工具结果，断言截断顺序与"达标即停"
- 截断消息含落盘路径 + master 引用，字节区间与主日志实际行偏移一致
- P3 跳过含工具调用的 assistant；压缩后 token 下降
- keep_head=2：前 2 条原样且输出 pinned=True；下游模拟裁剪时头部不被丢
- 人为压低 limit 使三阶段压不到位 → 走 reduce_log 兜底

**打分**
- 代码块永不丢；≤3 句原样；选中句按原顺序重组
- 首句/末句位置加分；决策/承诺/路径句得分高于普通句（构造对比样例断言）

**摘要**
- 三种"Context Files"小节标题变体都能解析；反引号/裸路径/相对路径三模式
- http 链接被过滤、`./` 前缀被去掉、重复路径去重保序
- 单文件超 2000 token 被截断且带剩余提示
- 新日志顺序 = 头部 system + 文件消息 + 引导语 + assistant 摘要
- 预处理后 <3 条 → 拒绝

**Provider**
- digest 对相同输入稳定、内容一变即变
- 注册表 setdefault 语义：后注册的 default 不覆盖先注册者；get/list 正常

**视图分支（ViewBranchStore）**
- create_view 后未切换：log 读取仍返回 master；switch_view 后返回视图内容
- 在视图上 append → master 与视图各多一条，等长不变量保持；conversation.jsonl 持续增长且内容从未被压缩改写
- next_view_name：无视图 → compacted-001；已有 001/003 → 004
- switch_to_master → 立即回到完整历史
- 端到端二次压缩：在视图上继续对话至再次触发，新视图里消息 i 的字节引用仍指向主日志第 i 行（索引对齐不变量）

**触发器**
- 60s 内第二次调用直接返回；action==none 不刷新时间戳
- provider 抛异常 → 静默返回不冒泡

---

## 10. 开发任务分解（M0–M7）

按依赖排序，每步可独立验收。默认按 Python + pytest；若目标项目异构，接口映射到 §11 确认后再开工。

| 里程碑 | 产出 | 依赖 | 验收标准 |
|---|---|---|---|
| **M0 契约与骨架** | §3 数据类 + §4 六个 Protocol + fake 实现（TokenCounter 用 chars/4）+ 包结构 | — | 骨架可 import；fake 计数器单测通过 |
| **M1 纯函数层** | strip_reasoning（§5.1）+ scoring 全部（§5.4） | M0 | §9 打分组全绿；零外部依赖（不碰 I/O） |
| **M2 决策层** | should_auto_compact + estimate（§5.2） | M0 | §9 决策组全绿 |
| **M3 引擎** | auto_compact_log（§5.3）+ master 索引/引用（§6.2）+ 截断摘要（§6.1，OutputStore fake 先行） | M1 M2 | §9 引擎组全绿；不变量 1–6 有断言 |
| **M4 兜底压缩器** | reduce_log + truncate_msg（§6.3） | M3 | 截断代码块格式正确；truncate-or-bail 行为有测 |
| **M5 Provider 层** | ABC/CompactionResult/注册表/DefaultProvider（§5.6） | M3 | §9 Provider 组全绿；default 可被同名插件替换 |
| **M6 LLM 摘要** | resume_via_llm + 解析/加载（§5.5），LLMClient 用 mock | M0 | §9 摘要组全绿（mock 返回固定 RESUME 文本） |
| **M7 集成** | 触发器+节流（§6.4）+ ViewBranchStore 完整实现（create_view / switch_view / 双写 append / persist / next_view_name，§4）+ on_compacted 回调 + 目标项目对话循环接线 | M2–M6 | 端到端：造超限日志跑一轮 turn，断言 views/compacted-001.jsonl 生成并已切换、master conversation.jsonl 未被压缩且随新消息持续增长（双写）、token 降到 0.8 CTX 以下、60s 节流生效；**二次压缩后 master 字节引用仍按索引对齐**（等长断言） |

M7 之后可选增强（不阻塞合入）：entry-points 插件发现、`/compact` 手动命令、telemetry 记录节省量。

---

## 11. 未决问题（开发前需确认）

| # | 问题 | 影响 | 默认方案（不回答则按此开发） |
|---|---|---|---|
| 1 | 目标项目语言/框架？同步还是 async？ | 全部代码形态 | Python 3.11+，同步 |
| 2 | 现有消息模型？工具结果怎么表示？ | §3.1 契约 | 新建 §3.1 的 Message；工具结果用 role="system" |
| 3 | tokenizer 可用性 | §4 TokenCounter | tiktoken，不可用则 chars/4 |
| 4 | 主力模型与上下文窗口 | §3.2 兜底常量、测试造数 | 兜底 context = 128000 |
| 5 | LLM 客户端形态 | §4 LLMClient | OpenAI 兼容 SDK 单例 |
| 6 | ~~落地策略：快照 / 视图分支 / 直接覆写~~ | §4、M7 | ✅ **已确认（2026-10-22）：视图分支 + 双写**，完整规格见 §4 |
| 7 | 是否需要 entry-points 插件注册表 | §5.6、M5 | 砍掉，仅保留编程式注册 |
| 8 | 环境变量名沿用 `GPTME_AUTOCOMPACT_KEEP_HEAD`？ | §6.4 | 改为项目自己的前缀 |
