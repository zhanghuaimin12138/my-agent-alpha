"""打分与抽取式压缩测试(§9 打分组)。"""

from __future__ import annotations

import pytest

from my_agent_alpha.autocompact.scoring import compress_content, score_sentence

# 打分对照用的"干净基线句":长度 31(偏短 +0.3),不含任何关键词/路径/语义信号
_BASE = "plain filler text without signals"
# 基线句放在位置 4、总共 10 句时的得分:只有长度 +0.3
_BASE_SCORE = 0.3


def test_position_scores() -> None:
    """首句 +2.0 且在前三句再 +1.0;末句 +1.5。"""
    assert score_sentence(_BASE, 0, 10) == pytest.approx(_BASE_SCORE + 2.0 + 1.0)
    assert score_sentence(_BASE, 9, 10) == pytest.approx(_BASE_SCORE + 1.5)
    assert score_sentence(_BASE, 4, 10) == pytest.approx(_BASE_SCORE)  # 中段没有位置分


def test_keyword_scores_stack() -> None:
    """关键词每个 +0.5,多个可叠加(bug 和 todo 互不干扰)。"""
    one = score_sentence("plain filler bug without signals", 4, 10)
    two = score_sentence("plain filler bug todo signals", 4, 10)
    assert one == pytest.approx(_BASE_SCORE + 0.5)
    assert two == pytest.approx(_BASE_SCORE + 1.0)


def test_error_word_hits_keyword_and_error_regex() -> None:
    """ "error" 同时命中关键词表(+0.5)和报错正则(+1.5),共 +2.0。"""
    got = score_sentence("plain filler error out signals", 4, 10)
    assert got == pytest.approx(_BASE_SCORE + 0.5 + 1.5)


def test_short_sentence_penalized() -> None:
    """太短(<10 字符)的句子扣 1.0:长度分只剩 -1.0 + 0.3。"""
    assert score_sentence("ok", 4, 10) == pytest.approx(-1.0 + 0.3)


def test_semantic_beats_plain() -> None:
    """决策/承诺/结论句的得分明显高于同长度的普通句。"""
    plain = score_sentence("plain filler text without signals", 4, 10)
    assert score_sentence("we'll use sqlite for the storage", 4, 10) > plain  # 决策 +2.0
    assert score_sentence("I'll implement the parser next", 4, 10) > plain  # 承诺 +1.5
    assert score_sentence("therefore the result is fine", 4, 10) > plain  # 结论 +1.5


def test_reference_scores() -> None:
    """文件路径 +1.0;URL 本身 +0.5(路径正则会同时吃到 //host 段,再 +1.0)。"""
    path_hit = score_sentence("see the file /src/main.py for it", 4, 10)
    no_hit = score_sentence("see the file and code for it now", 4, 10)
    assert path_hit == pytest.approx(no_hit + 1.0)
    url_hit = score_sentence("docs live at https://example.com/x", 4, 10)
    no_url = score_sentence("docs live at example dot com page", 4, 10)
    assert url_hit == pytest.approx(no_url + 0.5 + 1.0)


def test_compress_content_short_text_unchanged() -> None:
    """不超过 3 句:原样返回,一个字都不动。"""
    text = "One sentence here. Two sentences here. Three done."
    assert compress_content(text) == text


def test_compress_content_keeps_code_blocks() -> None:
    """代码块无条件保留;被抽掉的只是普通句子。

    注意:占位符会跟在它后面的第一个句子粘成一句(算"标记句"整体保留),
    所以紧挨代码块的 filler 0 也会留下;断言丢弃时挑远离代码块的句子。
    """
    block = "```python\nprint('hello')\n```"
    fillers = " ".join(f"filler sentence number {i} for padding." for i in range(12))
    text = f"intro line. {block} {fillers}"
    result = compress_content(text)
    assert block in result  # 代码块完整保留
    # 可打分句 = intro + filler1..11 共 12 句,保留 max(2, int(12*0.7)) = 8 句:
    # intro(位置分最高)、末尾 filler11(末句分)、filler1..6;丢弃 filler7..10
    assert "filler sentence number 0" in result  # 粘在代码块上,整句保留
    assert "filler sentence number 3" in result
    assert "filler sentence number 8" not in result
    assert "filler sentence number 10" not in result


def test_compress_content_keeps_high_score_ordered() -> None:
    """高价值句子优先保留;保留的句子按原文顺序拼接。"""
    parts = [f"plain filler sentence {i} here with words." for i in range(10)]
    parts[5] = "we'll use sqlite for storage because it is simple."  # 决策句,分数最高档
    result = compress_content(" ".join(parts))
    assert "we'll use sqlite" in result  # 决策句活下来了
    # 保留句按原顺序出现(下标递增)
    kept_idx = [i for i, p in enumerate(parts) if p in result]
    assert kept_idx == sorted(kept_idx)
    assert len(kept_idx) < len(parts)  # 确实压缩了


def test_compress_content_target_count() -> None:
    """10 个可打分句 → 保留 max(2, int(10*0.7)) = 7 句。"""
    parts = [f"plain filler sentence {i} here with words." for i in range(10)]
    result = compress_content(" ".join(parts))
    kept = [p for p in parts if p in result]
    assert len(kept) == 7
