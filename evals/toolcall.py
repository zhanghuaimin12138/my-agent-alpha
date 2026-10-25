"""工具调用成功率评测:真模型 × 单步调用 × 五级判分。

测的链路(和真机一致的提示词与原生工具调用通道,但只走一步、不执行工具):
    get_prompt(native=True) + 题目 → llm.reply() 完整生成(不做断点截断)
    → ToolUse.iter_from_content() 解析 → 五级判分 → 报告

设计文档:docs/工具调用成功率测试方案.md(出题纪律、漏斗定义、盲区都在里面)。

用法(在仓库根目录;不传 --model 时走 config.effective_model,和真机同一套解析):
    uv run python -m evals.toolcall --dry-run               # 只列题目,不联网
    uv run python -m evals.toolcall                         # 按配置解析模型,跑一轮
    uv run python -m evals.toolcall --model deepseek --repeat 3 --levels D2 --tools save,edit

判分五道门(S0 基础设施失败单独归类,不计入成功率):
    S1 发起调用      回复里有没有工具调用(两种通道任意一种都算发起)
    S2 解析对工具    原生调用的数量、工具名与期望一致;写成 markdown 代码块 = 通道偏移
    S3 参数合法      对照工具自己的 Parameter 表(必填齐全、类型对、无未知参数)
    S4 参数符合预期  对照题目的期望参数(EXACT 逐字符 / REQUIRED 别瞎编 / CONTAINS 子串)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from my_agent_alpha.codeblock import fenced_regions, strip_think
from my_agent_alpha.config import Config, effective_model
from my_agent_alpha.llm import reply
from my_agent_alpha.llm.models import resolve_model
from my_agent_alpha.message import Message
from my_agent_alpha.prompts import get_prompt
from my_agent_alpha.tools import get_tools
from my_agent_alpha.tools.base import (
    _NATIVE_CALL_RE,
    Parameter,
    ToolSpec,
    ToolUse,
    spec_to_openai_tool,
)

# ---------------- 题目定义 ----------------


class Match(Enum):
    """期望参数的匹配方式(一道期望选一种,规则直白好人工复核)。"""

    EXACT = "exact"  # 实际参数和期望逐键逐字符相等,不允许任何多余键
    REQUIRED = "required"  # 列出的键逐字符相等;没列出的键只能为空(考察"别瞎编参数")
    CONTAINS = "contains"  # 列出的键的值包含给定子串;没列出的键不管(自由文本参数用)


@dataclass(frozen=True)
class CallExpect:
    """对一个工具调用的期望:调哪个工具、参数该长什么样、按什么方式比对。

    示例:CallExpect("read", {"path": "a.txt"}) 表示"期望一次 read 调用,path 恰好是 a.txt"。
    """

    tool: str
    want: dict[str, str]
    mode: Match = Match.EXACT


def one(tool: str, want: dict[str, str], mode: Match = Match.EXACT) -> tuple[CallExpect, ...]:
    """包一层"单期望"(绝大多数题只期望一次调用),让题目表写得短。"""
    return (CallExpect(tool, want, mode),)


@dataclass(frozen=True)
class CallCase:
    """一道题:一句用户输入 + 期望的一次(或几次)原生工具调用。

    expects 里的期望按出现顺序和解析出的调用一一配对(D5 并行题考的就是
    一次回复发多个调用,顺序由题面"先……再……"规定)。
    """

    id: str
    level: str  # 难度层 D1-D5
    prompt: str
    expects: tuple[CallExpect, ...]


# 题目集 v1:9 个可运行工具全覆盖、每工具 ≥3 题,共 30 题。
# 出题纪律:期望参数手写死,不从模型输出推导(自己出题自己答 = 没有独立性)。
# D1 单参数直给 | D2 内容参数转义 | D3 可选参数与默认 | D4 工具选择歧义 | D5 并行多调用
CASES: tuple[CallCase, ...] = (
    # ---- D1:单参数直给(基本盘:会不会发起调用)----
    CallCase("tc-shell-001", "D1", "执行这条命令:ls -la", one("shell", {"command": "ls -la"})),
    CallCase("tc-shell-002", "D1", "运行命令 pwd", one("shell", {"command": "pwd"})),
    CallCase("tc-read-001", "D1", "读取 config.toml", one("read", {"path": "config.toml"})),
    CallCase("tc-read-002", "D1", "看一下 src/app.py 的内容", one("read", {"path": "src/app.py"})),
    CallCase(
        "tc-todo-001",
        "D1",
        "看一下当前的待办清单",
        one("todo", {"command": "read"}),
    ),
    CallCase(
        "tc-todo-002",
        "D1",
        "在待办清单里添加一条:梳理需求",
        one("todo", {"command": "add 梳理需求"}),
    ),
    CallCase(
        "tc-subagent-001",
        "D1",
        "派一个子代理调查 /tmp 目录下最大的三个文件,返回名字和大小",
        one("subagent", {"prompt": "/tmp"}, Match.CONTAINS),
    ),
    # ---- D2:内容参数(转义雷区:引号/反斜杠/花括号/emoji,必须逐字符抄对)----
    CallCase(
        "tc-save-001",
        "D2",
        "把下面方括号里的内容逐字符保存到 config/notes.txt(引号、反斜杠都不能改):"
        '[LINE1=包含"双引号"和\\反斜杠\\;LINE2=中文与 emoji 🎉]',
        one(
            "save",
            {
                "path": "config/notes.txt",
                "content": 'LINE1=包含"双引号"和\\反斜杠\\;LINE2=中文与 emoji 🎉',
            },
        ),
    ),
    CallCase(
        "tc-save-002",
        "D2",
        '把 [settings = {"mode": "fast", "retries": 3}] 这一整行原样写入 data/dump.txt'
        "(单行,别改任何字符)",
        one(
            "save",
            {"path": "data/dump.txt", "content": 'settings = {"mode": "fast", "retries": 3}'},
        ),
    ),
    CallCase(
        "tc-shell-003",
        "D2",
        '执行:echo "done: 100%"',
        one("shell", {"command": 'echo "done: 100%"'}),
    ),
    CallCase(
        "tc-append-001",
        "D2",
        "在 log.md 末尾追加这一行:第三步:重试成功 ✅(用时 3s)",
        one("append", {"path": "log.md", "content": "第三步:重试成功 ✅(用时 3s)"}),
    ),
    CallCase(
        "tc-append-003",
        "D2",
        "往 scratch.md 追加一行:hello world",
        one("append", {"path": "scratch.md", "content": "hello world"}),
    ),
    CallCase(
        "tc-edit-001",
        "D2",
        "把 a.py 里的 print('hi') 改成 print(\"hello\")",
        one("edit", {"path": "a.py", "old": "print('hi')", "new": 'print("hello")'}),
    ),
    CallCase(
        "tc-edit-002",
        "D2",
        "把 src/app.py 里的这两行:\ndef hello():\n    print('hi')\n"
        "整体替换为:\ndef hello():\n    print('hello world')",
        one(
            "edit",
            {
                "path": "src/app.py",
                "old": "def hello():\n    print('hi')",
                "new": "def hello():\n    print('hello world')",
            },
        ),
    ),
    CallCase(
        "tc-memory-001",
        "D2",
        "记住一条:生产服务器地址 10.0.0.1,SSH 端口 22",
        one("memory", {"content": "10.0.0.1"}, Match.CONTAINS),
    ),
    # ---- D3:可选参数与默认(没让给的参数不该瞎编)----
    CallCase(
        "tc-grep-001",
        "D3",
        "在整个工作区搜索 TODO",
        one("grep", {"pattern": "TODO"}, Match.REQUIRED),
    ),
    CallCase(
        "tc-grep-002",
        "D3",
        "在 src 目录下的 Python 文件里搜 TODO",
        one("grep", {"pattern": "TODO", "glob": "*.py", "path": "src"}),
    ),
    CallCase(
        "tc-grep-003",
        "D3",
        "搜一下代码里哪里用了 api_version",
        one("grep", {"pattern": "api_version"}, Match.REQUIRED),
    ),
    CallCase(
        "tc-memory-002",
        "D3",
        "把叫 部署流程 的那条记忆删掉",
        one("memory", {"action": "delete", "name": "部署流程"}, Match.REQUIRED),
    ),
    CallCase(
        "tc-memory-003",
        "D3",
        "把这条记到用户层(换项目也在):我正在学 Rust",
        one("memory", {"content": "Rust", "scope": "user"}, Match.CONTAINS),
    ),
    CallCase(
        "tc-subagent-002",
        "D3",
        "派一个只许用 read 和 grep 工具的子代理去数 README.md 有多少行",
        one("subagent", {"tools": "read,grep"}, Match.CONTAINS),
    ),
    CallCase(
        "tc-subagent-003",
        "D3",
        "改用 glm 模型派个子代理,读 README.md 并给我 50 字概要",
        one("subagent", {"model": "glm", "prompt": "README"}, Match.CONTAINS),
    ),
    # ---- D4:工具选择歧义(多个工具都能干,选没选对)----
    CallCase(
        "tc-edit-003",
        "D4",
        "把 config.toml 里的 debug = true 改成 debug = false",
        one("edit", {"path": "config.toml", "old": "debug = true", "new": "debug = false"}),
    ),
    CallCase(
        "tc-edit-004",
        "D4",
        "把 src/app.py 里的字符串 'hi' 换成 'hello'(只改这一处,不要重写整个文件)",
        one("edit", {"path": "src/app.py", "old": "'hi'", "new": "'hello'"}),
    ),
    CallCase(
        "tc-grep-004",
        "D4",
        "找出 src 目录里哪些地方调用了 fetch_data 函数(告诉我文件和行号就行,先不要读文件内容)",
        one("grep", {"pattern": "fetch_data"}, Match.REQUIRED),
    ),
    CallCase(
        "tc-append-002",
        "D4",
        "在 todo.txt 的末尾追加一行:买牛奶(注意是追加,不要覆盖原文件)",
        one("append", {"path": "todo.txt", "content": "买牛奶"}),
    ),
    CallCase(
        "tc-todo-003",
        "D4",
        "把待办清单里第 1 项标记为已完成",
        one("todo", {"command": "update 1 state=completed"}),
    ),
    # ---- D5:并行多调用(一次回复多个调用,数量/顺序/参数各自都要对)----
    CallCase(
        "tc-save-003",
        "D5",
        "先创建 hello.py,内容是 print('hello');再创建 bye.py,内容是 print('bye')",
        (
            CallExpect("save", {"path": "hello.py", "content": "print('hello')"}),
            CallExpect("save", {"path": "bye.py", "content": "print('bye')"}),
        ),
    ),
    CallCase(
        "tc-read-003",
        "D5",
        "先读 a.py,再读 b.py",
        (CallExpect("read", {"path": "a.py"}), CallExpect("read", {"path": "b.py"})),
    ),
    CallCase(
        "tc-todo-004",
        "D5",
        "建两条待办:先加 修复登录页空指针,再加 写单元测试",
        (
            CallExpect("todo", {"command": "add 修复登录页空指针"}),
            CallExpect("todo", {"command": "add 写单元测试"}),
        ),
    ),
)


# ---------------- 判分 ----------------


@dataclass(frozen=True)
class JudgeResult:
    """判分结果:挂在哪道门、为什么,以及解析现场(报告和归因用)。"""

    passed: bool
    fail_level: str  # "S1".."S4";通过时是 "PASS"
    reasons: tuple[str, ...]
    native_calls: tuple[str, ...]  # 每条原生调用的"人话"描述(display_command)
    markdown_calls: int  # 解析出的 markdown 通道调用数
    raw_at_lines: int  # 原文里 @工具名(...) 的行数(围栏与思考区除外)


def judge_call(text: str, case: CallCase) -> JudgeResult:
    """对一条完整模型回复判分:五道门从前往后过,挂在哪道就记到哪道。

    示例:回复 '好的\\n@read(call_1): {"path": "a.py"}' 对期望 read/path=a.py
    判为通过;回复纯文字挂在 S1;写成了 ```read 代码块挂在 S2(通道偏移)。
    """
    parsed = list(ToolUse.iter_from_content(text))
    native = [u for u in parsed if u.kwargs is not None]
    markdown = [u for u in parsed if u.kwargs is None]
    raw = _count_native_lines(text)
    evidence = {
        "native_calls": tuple(u.display_command() for u in native),
        "markdown_calls": len(markdown),
        "raw_at_lines": raw,
    }

    # S1 发起:两种通道任意一种有,或原文里有(哪怕被丢弃)都算发起过
    if not parsed and raw == 0:
        return JudgeResult(False, "S1", ("没发调用:回复里没有任何工具调用",), **evidence)

    # S2 解析对工具:数量、工具名按顺序一致;旁边混了 markdown 调用也算失败
    # (真机循环里 markdown 调用同样会被执行,多发就是真实的失败)
    want_tools = [e.tool for e in case.expects]
    got_tools = [u.tool.name for u in native]
    if got_tools != want_tools:
        return JudgeResult(
            False, "S2", (_s2_reason(native, markdown, raw, want_tools),), **evidence
        )
    if markdown:
        extra = [u.display_command() for u in markdown]
        return JudgeResult(
            False,
            "S2",
            (f"混用通道:原生 {len(native)} 条正确,另有 markdown 调用 {extra}",),
            **evidence,
        )

    # S3 参数合法:对照工具自己的参数声明(Parameter 表)
    s3: list[str] = []
    for use, _expect in zip(native, case.expects, strict=True):
        s3 += [
            f"[{use.tool.name}] {p}"
            for p in _validate_kwargs(use.kwargs or {}, use.tool.parameters)
        ]
    if s3:
        return JudgeResult(False, "S3", ("参数不合法:" + ";".join(s3),), **evidence)

    # S4 参数符合预期:按出现顺序一一配对比对
    s4: list[str] = []
    for use, expect in zip(native, case.expects, strict=True):
        s4 += [
            f"[{use.tool.name}] {p}"
            for p in _kwargs_problems(expect.want, use.kwargs or {}, expect.mode)
        ]
    if s4:
        return JudgeResult(False, "S4", ("参数值不符:" + ";".join(s4),), **evidence)
    return JudgeResult(True, "PASS", (), **evidence)


def _s2_reason(
    native: list[ToolUse], markdown: list[ToolUse], raw: int, want_tools: list[str]
) -> str:
    """S2 挂掉时的归因:没走原生通道的几种情况要能分开看,别混成一团。"""
    if not native:
        names = [u.tool.name for u in markdown]
        if want_tools[0] in names:
            return f"通道偏移:模型用 markdown 代码块通道调用了 {want_tools[0]}"
        if raw > 0:
            return f"静默丢失:原文有 {raw} 行原生调用,一条也没解析出来(拼装/解析层丢弃)"
        return f"工具名不符:期望原生 {want_tools},模型只有 markdown 调用 {names}"
    got = [u.tool.name for u in native]
    return f"调用数量或工具不符:期望 {want_tools},实得 {got}"


def _count_native_lines(text: str) -> int:
    """数原文里 @工具名(编号): 的出现次数(围栏代码块和思考区里的不算)。

    用途:和解析出的调用数对比 —— 原文有、解析没有 = 我们把模型发的调用丢了
    (已知解析层会静默丢弃坏 JSON);两边都没有 = 模型压根没发。
    """
    body = strip_think(text)
    regions = fenced_regions(body)
    count = 0
    for m in _NATIVE_CALL_RE.finditer(body):
        if not any(a <= m.start() < b for a, b in regions):
            count += 1
    return count


_JSON_TYPES: dict[str, tuple[type[object], ...]] = {
    "string": (str,),
    "integer": (int,),
    "boolean": (bool,),
    "number": (int, float),
    "array": (list,),
}


def _type_ok(value: object, declared: str) -> bool:
    """参数值的 Python 类型是否和声明的 JSON 类型匹配。

    注意 Python 里 bool 是 int 的子类,要先排除(模型发 true 给整数参数不该算对)。
    """
    allowed = _JSON_TYPES.get(declared)
    if allowed is None:
        return True  # 没见过的类型声明就跳过(未来扩展用)
    if isinstance(value, bool) and bool not in allowed:
        return False
    return isinstance(value, allowed)


def _validate_kwargs(kwargs: dict[str, object], params: Sequence[Parameter]) -> list[str]:
    """对照工具的参数声明检查一次调用的参数:必填、未知、类型,返回问题列表。

    示例:kwargs={} 对 read(必填 path)→ ["缺必填参数 path"]。
    """
    declared = {p.name: p for p in params}
    problems: list[str] = []
    for p in params:
        if p.required and p.name not in kwargs:
            problems.append(f"缺必填参数 {p.name}")
    for key, value in kwargs.items():
        if key not in declared:
            problems.append(f"未知参数 {key}={value!r}")
        elif not _type_ok(value, declared[key].type):
            problems.append(f"参数 {key} 类型应为 {declared[key].type},实得 {type(value).__name__}")
    return problems


def _norm(value: object) -> str:
    """把参数值统一成字符串再比对:True→"true"、3→"3"、列表→JSON 文本。

    为什么要有它:模型经 JSON 发来的参数可能是数字/布尔,期望表里全是手写字符串。
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _canon_csv(value: str) -> str:
    """逗号分隔列表的规范化:去空格、按字典序排序("read, grep" 和 "grep,read" 视为相等)。"""
    return ",".join(sorted(p.strip() for p in value.split(",") if p.strip()))


# 个别参数要"语义等价"而不是字符串相等才公平(目前只有 subagent 的 tools 白名单)
_KEY_CANON: dict[str, Callable[[str], str]] = {"tools": _canon_csv}


def _canon_key(key: str, value: str) -> str:
    """比对前按键规范化(有特殊规则的参数才处理,其余原样返回)。"""
    fn = _KEY_CANON.get(key)
    return fn(value) if fn else value


def _kwargs_problems(want: dict[str, str], got: dict[str, object], mode: Match) -> list[str]:
    """按匹配模式比对期望参数与实际参数,返回问题列表(空列表 = 完全符合)。"""
    problems: list[str] = []
    for key, expected in want.items():
        if key not in got:
            problems.append(f"缺参数 {key}(期望 {expected!r})")
            continue
        got_s = _canon_key(key, _norm(got[key]))
        want_s = _canon_key(key, expected)
        ok = want_s in got_s if mode is Match.CONTAINS else got_s == want_s
        if not ok:
            problems.append(f"参数 {key} 期望 {expected!r},实得 {got[key]!r}")
    if mode is Match.EXACT:
        problems += [f"多余参数 {k}={v!r}" for k, v in got.items() if k not in want]
    elif mode is Match.REQUIRED:
        problems += [
            f"自造参数 {k}={v!r}(题目没要求)" for k, v in got.items() if k not in want and _norm(v)
        ]
    return problems


def tool_fingerprint(tools: Sequence[ToolSpec]) -> str:
    """给"模型看到的工具说明书"算一个指纹(sha256 前 12 位)。

    两次评测指纹不同 = 工具提示词改过了,成绩不能直接对比,报告里要标出来。
    """
    payload = json.dumps(
        [spec_to_openai_tool(t) for t in sorted(tools, key=lambda t: t.name)],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ---------------- 运行器 ----------------


@dataclass(frozen=True)
class CaseRun:
    """一道题跑一次的完整记录(判分现场 + 原文存档,失败可回放排查)。"""

    case_id: str
    level: str
    repeat_index: int
    duration: float
    judge: JudgeResult | None  # None = 基础设施失败(网络/限流/中断),不计入成功率
    error: str | None
    usage: dict[str, int] | None
    reply_text: str


_INTERRUPT_MARK = "^C 已中断"  # 与 llm.reply 的中断标记保持一致(那边是私有常量,不导出)


def run_once(
    case: CallCase,
    repeat_index: int,
    model: str,
    sys_messages: Sequence[Message],
    *,
    sleep: float = 0.0,
) -> CaseRun:
    """跑一道题一次:发真实请求、完整生成(不截断)、判分,返回带原文的记录。

    网络错误/限流/超时被归为基础设施失败(judge=None),不参与成功率统计。
    """
    messages = [*sys_messages, Message("user", case.prompt)]
    tools = get_tools()
    start = time.monotonic()
    try:
        # break_on_tooluse=False:评测要看"完整回复",断点截断会把后面的调用砍掉
        msg = reply(messages, model, tools=tools, tool_format="tool", break_on_tooluse=False)
    except Exception as e:
        return CaseRun(
            case.id,
            case.level,
            repeat_index,
            time.monotonic() - start,
            None,
            f"{type(e).__name__}: {e}",
            None,
            "",
        )
    if sleep:
        time.sleep(sleep)
    duration = time.monotonic() - start
    text = msg.content
    if _INTERRUPT_MARK in text:
        return CaseRun(
            case.id, case.level, repeat_index, duration, None, "生成被中断", _usage_of(msg), text
        )
    return CaseRun(
        case.id,
        case.level,
        repeat_index,
        duration,
        judge_call(text, case),
        None,
        _usage_of(msg),
        text,
    )


def _usage_of(msg: Message) -> dict[str, int] | None:
    """从回复消息的 metadata 里取 token 用量(没有就 None)。"""
    usage = msg.metadata.get("usage")
    return usage if isinstance(usage, dict) else None


def select_cases(
    cases: Sequence[CallCase],
    *,
    tools: Sequence[str] = (),
    levels: Sequence[str] = (),
    only: Sequence[str] = (),
    limit: int = 0,
) -> list[CallCase]:
    """按命令行过滤条件选题(工具/难度/编号/数量),什么都不传就全选。"""
    picked = list(cases)
    if tools:
        wanted = set(tools)
        picked = [c for c in picked if any(e.tool in wanted for e in c.expects)]
    if levels:
        wanted = set(levels)
        picked = [c for c in picked if c.level in wanted]
    if only:
        wanted = set(only)
        picked = [c for c in picked if c.id in wanted]
    if limit > 0:
        picked = picked[:limit]
    return picked


def _print_progress(done: int, total: int, run: CaseRun) -> None:
    """打一行进度(全量要跑几分钟,终端得有反馈)。"""
    if run.judge is None:
        verdict = f"⚠️ 基础设施失败:{run.error}"
    elif run.judge.passed:
        verdict = "✅ 通过"
    else:
        first = run.judge.reasons[0] if run.judge.reasons else ""
        verdict = f"❌ {run.judge.fail_level} {first[:60]}"
    print(
        f"[{done}/{total}] {run.case_id} #{run.repeat_index} {verdict}({run.duration:.1f}s)",
        flush=True,
    )


def _new_run_dir() -> Path:
    """建本次评测的产物目录 evals/toolcall_results/<时间-随机码>/(一次运行一个)。"""
    stamp = time.strftime("%Y-%m-%d-%H%M%S")
    run_dir = Path(__file__).parent / "toolcall_results" / f"{stamp}-{secrets.token_hex(3)}"
    (run_dir / "cases").mkdir(parents=True)
    return run_dir


def default_model() -> str:
    """评测的默认模型:不传 --model 时走 effective_model(环境变量/配置),和真机同一套解析。

    Config.load 的 workspace 取当前目录:评测在仓库根跑,项目 alpha.toml(若有)也生效。
    """
    return effective_model(None, Config.load(Path.cwd()))


def run_eval(args: argparse.Namespace, cases: Sequence[CallCase], model: str) -> Path:
    """评测主流程:逐题跑 → 汇总 → 落报告,返回本次 run 的目录。"""
    tools = get_tools()
    sys_messages = get_prompt(tools, native=True)
    runs: list[CaseRun] = []
    total = len(cases) * args.repeat
    done = 0
    for case in cases:
        for k in range(1, args.repeat + 1):
            run = run_once(case, k, model, sys_messages, sleep=args.sleep)
            runs.append(run)
            done += 1
            _print_progress(done, total, run)
    run_dir = _new_run_dir()
    fingerprint = tool_fingerprint(tools)
    summary = _build_summary_json(model, args.repeat, runs, fingerprint)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(
        _render_summary_md(model, cases, runs, fingerprint, args.repeat), encoding="utf-8"
    )
    _write_case_files(run_dir / "cases", cases, runs)
    print(f"\n报告:{run_dir / 'summary.md'}")
    return run_dir


# ---------------- 报告 ----------------


def _stats(runs: Sequence[CaseRun]) -> dict[str, Any]:
    """汇总判分统计:漏斗各级通过数、失败分类计数(纯计算,不碰文件)。"""
    judged = [r for r in runs if r.judge is not None]
    fails = {"S1": 0, "S2": 0, "S3": 0, "S4": 0}
    categories: dict[str, int] = {}
    for r in judged:
        assert r.judge is not None  # 列表来源已保证
        if r.judge.passed:
            continue
        fails[r.judge.fail_level] += 1
        cat = r.judge.reasons[0].split(":", 1)[0] if r.judge.reasons else "未知"
        categories[cat] = categories.get(cat, 0) + 1
    at = [len(judged)]  # at[0]=S1 判分总数,at[1..4]=过了 S1..S4 的数量
    for level in ("S1", "S2", "S3", "S4"):
        at.append(at[-1] - fails[level])
    return {
        "judged": len(judged),
        "infra": len(runs) - len(judged),
        "fails": fails,
        "categories": categories,
        "at": at,
    }


def _group_stats(
    cases: Sequence[CallCase], runs: Sequence[CaseRun], key: Callable[[CallCase], str]
) -> list[tuple[str, int, int]]:
    """按工具或难度层聚合成 (名字, 判分次数, 通过次数),按名字排序。"""
    key_of = {c.id: key(c) for c in cases}
    agg: dict[str, list[int]] = {}
    for r in runs:
        entry = agg.setdefault(key_of.get(r.case_id, "?"), [0, 0])
        if r.judge is not None:
            entry[0] += 1
            if r.judge.passed:
                entry[1] += 1
    return sorted((name, vals[0], vals[1]) for name, vals in agg.items())


def _total_usage(runs: Sequence[CaseRun]) -> tuple[int, int]:
    """汇总全部调用的 token 用量,返回 (输入, 输出)。"""
    total_in = total_out = 0
    for r in runs:
        if r.usage:
            total_in += int(r.usage.get("input", 0))
            total_out += int(r.usage.get("output", 0))
    return total_in, total_out


def _render_summary_md(
    model: str,
    cases: Sequence[CallCase],
    runs: Sequence[CaseRun],
    fingerprint: str,
    repeat: int,
) -> str:
    """渲染人看的报告:漏斗表、按工具/难度成绩、失败原因分布。"""
    s = _stats(runs)
    by_case: dict[str, list[CaseRun]] = {}
    for r in runs:
        by_case.setdefault(r.case_id, []).append(r)
    pass_any = sum(1 for rs in by_case.values() if any(r.judge and r.judge.passed for r in rs))
    rate = s["at"][4] / s["judged"] if s["judged"] else 0.0
    tokens_in, tokens_out = _total_usage(runs)

    lines: list[str] = [
        "# 工具调用成功率报告",
        "",
        f"- 模型:{resolve_model(model).full}",
        f"- 时间:{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 题目 {len(by_case)} 道 × 每题 {repeat} 次 = {len(runs)} 次调用",
        f"- token 用量:输入 {tokens_in} / 输出 {tokens_out}",
        f"- 工具提示词指纹:{fingerprint}(两次 run 指纹不同 = 工具说明改过,成绩不可直接对比)",
    ]
    if s["infra"]:
        lines.append(f"- ⚠️ 基础设施失败 {s['infra']} 次(网络/限流/中断,不计入成功率)")

    gates = (
        ("S1", "发起调用", s["at"][1], s["fails"]["S1"]),
        ("S2", "解析出期望工具", s["at"][2], s["fails"]["S2"]),
        ("S3", "参数合法", s["at"][3], s["fails"]["S3"]),
        ("S4", "参数符合预期", s["at"][4], s["fails"]["S4"]),
    )
    lines += [
        "",
        f"## 漏斗(判分 {s['judged']} 次)",
        "",
        "| 门 | 检查 | 通过 | 本级损失 |",
        "|---|---|---:|---:|",
    ]
    lines += [f"| {lv} | {label} | {passed} | {lost} |" for lv, label, passed, lost in gates]
    lines += [
        "",
        f"**总成功率(S4):{s['at'][4]}/{s['judged']} = {rate:.1%}**",
        f"按题「至少一次通过」:{pass_any}/{len(by_case)}",
    ]

    lines += ["", "## 按工具", "", "| 工具 | 判分 | 通过 | 成功率 |", "|---|---:|---:|---:|"]
    for name, judged_n, passed_n in _group_stats(cases, runs, lambda c: c.expects[0].tool):
        ratio = passed_n / judged_n if judged_n else 0.0
        lines.append(f"| {name} | {judged_n} | {passed_n} | {ratio:.0%} |")

    lines += ["", "## 按难度层", "", "| 难度 | 判分 | 通过 | 成功率 |", "|---|---:|---:|---:|"]
    for name, judged_n, passed_n in _group_stats(cases, runs, lambda c: c.level):
        ratio = passed_n / judged_n if judged_n else 0.0
        lines.append(f"| {name} | {judged_n} | {passed_n} | {ratio:.0%} |")

    if s["categories"]:
        lines += ["", "## 失败原因分布", ""]
        for cat, n in sorted(s["categories"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {cat}:{n}")
    return "\n".join(lines) + "\n"


def _write_case_files(dir_path: Path, cases: Sequence[CallCase], runs: Sequence[CaseRun]) -> None:
    """把每次调用的原文和判分明细落盘(失败归因、人工复核的唯一依据)。"""
    case_map = {c.id: c for c in cases}
    for r in runs:
        case = case_map[r.case_id]
        expects_desc = ";".join(
            f"{e.tool}({e.mode.value}) {json.dumps(e.want, ensure_ascii=False)}"
            for e in case.expects
        )
        if r.judge is None:
            verdict = f"⚠️ 基础设施失败:{r.error}"
            judge_desc = "-"
            native_n = markdown_n = raw_n = 0
        else:
            verdict = "✅ 通过" if r.judge.passed else f"❌ 挂在 {r.judge.fail_level}"
            judge_desc = "\n".join(f"- {reason}" for reason in r.judge.reasons) or "-"
            native_n = len(r.judge.native_calls)
            markdown_n = r.judge.markdown_calls
            raw_n = r.judge.raw_at_lines
        text = (
            f"# {r.case_id} · 第 {r.repeat_index} 次 · {verdict}\n\n"
            f"- 难度:{case.level}\n"
            f"- 期望:{expects_desc}\n"
            f"- 耗时:{r.duration:.1f}s\n\n"
            f"## 判分\n\n{judge_desc}\n\n"
            "## 解析现场\n\n"
            f"- 原生调用 {native_n} 条 / markdown 调用 {markdown_n} 条 / 原文 @行 {raw_n} 行\n\n"
            "## 用户输入\n\n"
            f"{case.prompt}\n\n"
            "## 模型完整回复(原文存档)\n\n"
            f"`````\n{r.reply_text}\n`````\n"
        )
        (dir_path / f"{r.case_id}.{r.repeat_index}.md").write_text(text, encoding="utf-8")


def _build_summary_json(
    model: str, repeat: int, runs: Sequence[CaseRun], fingerprint: str
) -> dict[str, Any]:
    """把全部结果装进一个可序列化字典(机器可读,给版本对比用)。"""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "repeat": repeat,
        "tool_prompt_sha256": fingerprint,
        "runs": [
            {
                "id": r.case_id,
                "level": r.level,
                "repeat_index": r.repeat_index,
                "duration": round(r.duration, 2),
                "infra_error": r.error,
                "fail_level": r.judge.fail_level if r.judge else None,
                "passed": bool(r.judge and r.judge.passed),
                "reasons": list(r.judge.reasons) if r.judge else [],
                "native_calls": list(r.judge.native_calls) if r.judge else [],
                "markdown_calls": r.judge.markdown_calls if r.judge else 0,
                "raw_at_lines": r.judge.raw_at_lines if r.judge else 0,
                "usage": r.usage,
            }
            for r in runs
        ],
    }


def _dry_run(cases: Sequence[CallCase]) -> None:
    """不联网,把题目表打出来(出题审查用:题面、期望工具、难度一目了然)。"""
    print(f"共 {len(cases)} 题:")
    for c in cases:
        tools = ",".join(e.tool for e in c.expects)
        print(f"  {c.id} [{c.level}] {tools:<10} {c.prompt}")


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口;`uv run python -m evals.toolcall --help` 看全部参数。"""
    parser = argparse.ArgumentParser(description="工具调用成功率评测(真模型 × 单步调用 × 五级判分)")
    parser.add_argument(
        "--model",
        default=None,
        help="服务商/模型,如 glm-coding、deepseek/deepseek-chat;"
        "不传则走 effective_model(环境变量/配置文件)",
    )
    parser.add_argument("--repeat", type=int, default=3, help="每题独立跑几次(默认 3,报 pass@k)")
    parser.add_argument("--tools", default="", help="只跑期望这些工具的题,逗号分隔(如 save,edit)")
    parser.add_argument("--levels", default="", help="只跑这些难度层,逗号分隔(如 D2,D4)")
    parser.add_argument("--only", default="", help="只跑指定编号的题,逗号分隔")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题(冒烟用,0=不限)")
    parser.add_argument("--sleep", type=float, default=0.0, help="两次调用之间的间隔秒数(限流时用)")
    parser.add_argument("--dry-run", action="store_true", help="只列题目,不联网不判分")
    args = parser.parse_args(argv)
    # 模型解析和真机同一套:--model > 环境变量/配置(见 config.effective_model)
    model = effective_model(args.model, Config.load(Path.cwd()))

    def _csv(raw: str) -> list[str]:
        """把逗号分隔的命令行参数拆成列表(空串 → 空列表)。"""
        return [p.strip() for p in raw.split(",") if p.strip()]

    cases = select_cases(
        CASES,
        tools=_csv(args.tools),
        levels=_csv(args.levels),
        only=_csv(args.only),
        limit=args.limit,
    )
    if not cases:
        print("没有选中任何题目(检查 --tools/--levels/--only 过滤条件)")
        return 1
    if args.dry_run:
        print(f"模型:{resolve_model(model).full}(--dry-run 不联网)\n")
        _dry_run(cases)
        return 0
    run_eval(args, cases, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
