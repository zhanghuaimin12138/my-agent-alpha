"""evals.toolcall 的单元测试:判分器、题目集体检、比对规则 —— 全部用手写回复,
不发任何网络请求(测的是判分逻辑本身;真模型评测用 uv run python -m evals.toolcall)。
"""

from __future__ import annotations

from dataclasses import replace

from evals.toolcall import (
    CASES,
    CallCase,
    CallExpect,
    CaseRun,
    JudgeResult,
    Match,
    _kwargs_problems,
    _render_summary_md,
    _validate_kwargs,
    default_model,
    judge_call,
    select_cases,
    tool_fingerprint,
)
from my_agent_alpha.llm.models import resolve_model
from my_agent_alpha.tools import get_tool, get_tools


def _case(
    tool: str = "read", want: dict[str, str] | None = None, mode: Match = Match.EXACT
) -> CallCase:
    """造一道"期望单次调用"的最小题目(判分器测试的通用底座)。"""
    return CallCase("tc-test", "D1", "测试输入", (CallExpect(tool, want or {}, mode),))


# ---------------- 判分:五道门各自的行为 ----------------


def test_judge_pass_native_exact() -> None:
    """正确发一次原生调用、参数逐字符一致 → 通过。"""
    text = '好的,我来读:\n@read(call_1): {"path": "config.toml"}\n'
    result = judge_call(text, _case("read", {"path": "config.toml"}))
    assert result.passed
    assert result.fail_level == "PASS"
    assert result.native_calls == ("read: path=config.toml",)
    assert result.markdown_calls == 0
    assert result.raw_at_lines == 1


def test_judge_fail_s1_no_call() -> None:
    """纯文字回复(两种通道都没有)→ 挂在 S1「没发调用」。"""
    result = judge_call("我觉得应该先看看配置文件的内容。", _case("read", {"path": "a.txt"}))
    assert not result.passed
    assert result.fail_level == "S1"
    assert "没发调用" in result.reasons[0]


def test_judge_fail_s2_channel_shift() -> None:
    """模型写成 markdown 代码块(期望原生通道)→ S2「通道偏移」,单独归类。"""
    text = "```shell\nls -la\n```"
    result = judge_call(text, _case("shell", {"command": "ls -la"}))
    assert result.fail_level == "S2"
    assert "通道偏移" in result.reasons[0]
    assert result.markdown_calls == 1


def test_judge_fail_s2_silent_drop() -> None:
    """原文里有 @ 行但 JSON 没写完(解析层静默丢弃)→ S2「静默丢失」。

    这是"模型发了、我们丢了"和"模型没发"能分开归因的关键场景。
    """
    text = '@read(call_1): {"path": "config.to'
    result = judge_call(text, _case("read", {"path": "config.toml"}))
    assert result.fail_level == "S2"
    assert "静默丢失" in result.reasons[0]
    assert result.raw_at_lines == 1


def test_judge_fail_s3_missing_required() -> None:
    """缺必填参数(save 没给 path)→ S3「参数不合法」。"""
    text = '@save(call_1): {"content": "你好"}'
    result = judge_call(text, _case("save", {"path": "a.txt", "content": "你好"}))
    assert result.fail_level == "S3"
    assert "缺必填参数 path" in result.reasons[0]


def test_judge_fail_s3_unknown_param() -> None:
    """发了工具没声明的参数 → S3「未知参数」。"""
    text = '@read(call_1): {"path": "a.txt", "mode": "fast"}'
    result = judge_call(text, _case("read", {"path": "a.txt"}))
    assert result.fail_level == "S3"
    assert "未知参数 mode" in result.reasons[0]


def test_judge_fail_s4_wrong_value() -> None:
    """工具对、参数齐,但值抄错了 → S4「参数值不符」。"""
    text = '@read(call_1): {"path": "b.txt"}'
    result = judge_call(text, _case("read", {"path": "a.txt"}))
    assert result.fail_level == "S4"
    assert "参数 path 期望 'a.txt'" in result.reasons[0]


def test_judge_multi_calls_pass_and_order_fail() -> None:
    """D5:一次回复两个调用,全对才过;顺序颠倒判参数不符(顺序由题面规定)。"""
    case = CallCase(
        "tc-test",
        "D5",
        "先读 a.py 再读 b.py",
        (CallExpect("read", {"path": "a.py"}), CallExpect("read", {"path": "b.py"})),
    )
    ok = '@read(call_1): {"path": "a.py"}\n@read(call_2): {"path": "b.py"}\n'
    assert judge_call(ok, case).passed
    reversed_text = '@read(call_1): {"path": "b.py"}\n@read(call_2): {"path": "a.py"}\n'
    wrong = judge_call(reversed_text, case)
    assert wrong.fail_level == "S4"


def test_judge_think_region_draft_ignored() -> None:
    """思考区里的 @ 行是草稿:不算发起,也不计入原文 @ 行数。"""
    text = '<think>@read(call_1): {"path": "a.py"}</think>\n我先想想。'
    result = judge_call(text, _case("read", {"path": "a.py"}))
    assert result.fail_level == "S1"
    assert result.raw_at_lines == 0


def test_judge_extra_markdown_alongside_native_fails() -> None:
    """原生调用正确但旁边又混了 markdown 调用 → S2「混用通道」(真机里同样会执行)。"""
    text = '好的\n@read(call_1): {"path": "a.txt"}\n```shell\necho hi\n```\n'
    result = judge_call(text, _case("read", {"path": "a.txt"}))
    assert result.fail_level == "S2"
    assert "混用通道" in result.reasons[0]


# ---------------- 比对规则与参数校验 ----------------


def test_kwargs_match_modes() -> None:
    """三种匹配模式的行为:EXACT 拒绝多余、REQUIRED 拒绝自造但允许为空、CONTAINS 看子串。"""
    want = {"path": "a.txt"}
    assert _kwargs_problems(want, {"path": "a.txt"}, Match.EXACT) == []
    assert _kwargs_problems(want, {"path": "a.txt", "mode": "fast"}, Match.EXACT) != []
    # REQUIRED:没列出的键允许出现,但值必须为空
    assert _kwargs_problems(want, {"path": "a.txt", "glob": ""}, Match.REQUIRED) == []
    assert _kwargs_problems(want, {"path": "a.txt", "glob": "*.py"}, Match.REQUIRED) != []
    # CONTAINS:值包含子串即可,没列出的键不管
    assert (
        _kwargs_problems(
            {"prompt": "README"}, {"prompt": "读 README.md 并概要", "model": "glm"}, Match.CONTAINS
        )
        == []
    )


def test_kwargs_tools_whitelist_order_insensitive() -> None:
    """tools 白名单"read, grep"和"grep,read"应视为相等(逗号列表规范化)。"""
    want = {"tools": "read,grep"}
    got: dict[str, object] = {"tools": "grep, read", "prompt": "数行数"}
    assert _kwargs_problems(want, got, Match.CONTAINS) == []


def test_validate_kwargs_type_rules() -> None:
    """类型检查:bool 不冒充 integer/string;数字字符串化后另在 S4 比对。"""
    from my_agent_alpha.tools.base import Parameter

    params: tuple[Parameter, ...] = (Parameter("n", "integer", "数量"),)
    assert _validate_kwargs({"n": 3}, params) == []
    assert _validate_kwargs({"n": True}, params) != []  # bool 不是整数
    assert _validate_kwargs({"n": "3"}, params) != []  # 字符串也不是整数


def test_norm_compares_number_as_string() -> None:
    """模型发数字 3、期望写 "3" → 比对应当通过(_norm 归一化)。"""
    got: dict[str, object] = {"path": "a.txt"}
    assert _kwargs_problems({"path": "a.txt"}, got, Match.EXACT) == []


# ---------------- 题目集体检(防题目本身出错) ----------------


def test_taskset_integrity() -> None:
    """题目表体检:编号唯一、难度合法、全覆盖 9 个工具、期望参数名真实存在、
    EXACT/REQUIRED 模式必须覆盖该工具的全部必填参数(否则题目必挂,是坏题)。"""
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)), "题目编号有重复"
    runnable = {t.name for t in get_tools() if t.is_runnable}
    used = {e.tool for c in CASES for e in c.expects}
    assert used == runnable, "题目集没有覆盖全部可运行工具"
    for c in CASES:
        assert c.level in {"D1", "D2", "D3", "D4", "D5"}
        assert len(c.expects) >= 1
        for e in c.expects:
            spec = get_tool(e.tool)
            assert spec is not None, f"{c.id} 期望了不存在的工具 {e.tool}"
            declared = {p.name for p in spec.parameters}
            assert set(e.want) <= declared, f"{c.id} 的期望参数不在 {e.tool} 的参数表里"
            if e.mode is not Match.CONTAINS:
                required = {p.name for p in spec.parameters if p.required}
                missing = required - set(e.want)
                assert not missing, f"{c.id} 是 {e.mode.value} 模式却缺必填参数 {missing}"


def test_taskset_scale() -> None:
    """规模约定:每工具至少 3 题,总量 30(改题目表时这两个数字要同步改)。"""
    counts: dict[str, int] = {}
    for c in CASES:
        for e in c.expects:
            counts[e.tool] = counts.get(e.tool, 0) + 1
    assert all(n >= 3 for n in counts.values()), counts
    assert len(CASES) == 30


def test_select_cases_filters() -> None:
    """选题过滤:按工具、按难度、按编号、限量,各自生效。"""
    by_tool = select_cases(CASES, tools=["save"])
    assert by_tool and all(e.tool == "save" for c in by_tool for e in c.expects)
    by_level = select_cases(CASES, levels=["D5"])
    assert by_level and all(c.level == "D5" for c in by_level)
    by_id = select_cases(CASES, only=["tc-read-001"])
    assert [c.id for c in by_id] == ["tc-read-001"]
    assert len(select_cases(CASES, limit=3)) == 3


def test_tool_fingerprint_changes_with_tool_doc() -> None:
    """指纹:两次计算稳定;工具说明一改就变(报告靠它判断两次成绩能不能比)。"""
    tools = get_tools()
    assert tool_fingerprint(tools) == tool_fingerprint(tools)
    changed = tuple(
        replace(tools[0], desc=tools[0].desc + "改了") if i == 0 else t for i, t in enumerate(tools)
    )
    assert tool_fingerprint(changed) != tool_fingerprint(tools)


def test_default_model_resolves() -> None:
    """默认模型走 effective_model 解析,结果必须是能被 resolve_model 认识的非空名字。"""
    model = default_model()
    assert isinstance(model, str) and model.strip()
    assert resolve_model(model).full  # 能解析出"服务商/模型"完整名


# ---------------- 报告渲染(轻量冒烟) ----------------


def _run(case: CallCase, k: int, judge: JudgeResult | None, error: str | None = None) -> CaseRun:
    """造一条 CaseRun(报告渲染测试用,不联网)。"""
    return CaseRun(case.id, case.level, k, 1.0, judge, error, {"input": 100, "output": 50}, "原文")


def test_render_summary_smoke() -> None:
    """报告要能渲染出:漏斗表、基础设施失败栏、指纹(样例数据含通过/失败/基础设施三种)。"""
    case = next(c for c in CASES if c.id == "tc-shell-001")
    other = next(c for c in CASES if c.id == "tc-read-001")
    runs = [
        _run(case, 1, JudgeResult(True, "PASS", (), ("shell: command=ls -la",), 0, 1)),
        _run(
            case,
            2,
            JudgeResult(
                False, "S2", ("通道偏移:模型用 markdown 代码块通道调用了 shell",), (), 1, 0
            ),
        ),
        _run(other, 1, None, "APIConnectionError: 连不上"),
    ]
    md = _render_summary_md("mock/echo", [case, other], runs, "abc123def456", 2)
    assert "漏斗" in md
    assert "基础设施" in md
    assert "abc123def456" in md
    assert "按工具" in md and "按难度层" in md
    assert "通道偏移" in md  # 失败原因分布里要能看到
