"""系统提示词组装测试:行动纪律段(gptme 同款防"光说不做")与消息结构。

背景:2026-10-05 会话 3b400c 里 glm-5.3 一场"光说不做"5 次(冒号结尾
宣布行动然后停住),016 的提醒 hook 兜住了但每回都浪费一轮生成 ——
本测试锁定提示词层的预防话术(仿 gptme prompts/templates.py)。
"""

from __future__ import annotations

from my_agent_alpha.prompts import get_prompt


def _static(native: bool) -> str:
    """取静态提示词消息的正文(第一条)。"""
    msgs = get_prompt((), native=native)
    return msgs[0].content


def test_prompt_structure_two_messages() -> None:
    """结构:静态(pinned+hide)+ 动态时间,共两条。"""
    msgs = get_prompt(())
    assert len(msgs) == 2
    assert msgs[0].pinned and msgs[0].hide  # 静态:发给模型但不打印、压缩不删
    assert msgs[1].content.startswith("# 当前时间")  # 动态:时间戳
    assert msgs[1].pinned and msgs[1].hide


def test_markdown_identity_has_action_discipline() -> None:
    """markdown 版身份含"行动纪律"段,四句核心话术都要在。"""
    text = _static(native=False)
    assert "# 行动纪律" in text
    assert "立刻" in text  # 有工具立刻调用,不要描述
    assert "同一条回复里" in text  # 决定→代码块必须同一条消息(gptme 同款)
    assert "冒号结尾" in text  # 禁止停在半句话(glm 实测高发形态)
    assert "需要的话我可以" in text  # 禁止空头提议收尾(gptme next_step_guidance)


def test_native_identity_has_action_discipline() -> None:
    """原生通道同样有"行动纪律"段(措辞换成函数调用)。"""
    text = _static(native=True)
    assert "# 行动纪律" in text
    assert "函数调用" in text
    assert "立刻" in text
    assert "同一条回复里" in text


def test_old_rule_merged_not_left_in_protocol() -> None:
    """旧第 6 条已升级为独立段落:协议列表回到 5 条,不残留重复条目。"""
    text = _static(native=False)
    assert "6." not in text  # 协议列表只有 1-5,第 6 条内容并入了行动纪律
    # 行动纪律紧跟在协议之后,而不是躲在列表末尾(更显眼)
    assert text.index("# 工具调用协议") < text.index("# 行动纪律")


# ---- 审查 #14:系统提示词内容 ----


def test_prompt_contains_tool_docs_and_time() -> None:
    from my_agent_alpha.tools import get_tools

    msgs = get_prompt(get_tools())
    assert len(msgs) == 2
    static, dynamic = msgs
    assert static.pinned and static.hide
    assert "工具调用协议" in static.content
    assert "## shell" in static.content
    assert "## save" in static.content
    assert "当前时间" in dynamic.content
    assert "工具调用协议" not in dynamic.content
