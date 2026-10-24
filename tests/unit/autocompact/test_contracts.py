"""契约层测试:token 计数器、消息指纹、is_tool_result、全量落盘。"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha.autocompact.adapters import EstimateTokenCounter
from my_agent_alpha.autocompact.contracts import FALLBACK_MODEL_INFO, is_tool_result, source_digest
from my_agent_alpha.autocompact.output_storage import FileOutputStore
from my_agent_alpha.message import Message


def test_counter_counts_str_message_and_list() -> None:
    """三种输入都能数;列表 = 各条之和;同一输入永远同一结果。"""
    counter = EstimateTokenCounter()
    msg = Message("user", "abcd")  # 4 个 ascii 字符 → 1 token
    assert counter.len_tokens("abcd") == 1
    assert counter.len_tokens(msg) == 1
    assert counter.len_tokens([msg, msg]) == 2
    assert counter.len_tokens([]) == 0
    # 稳定性:连着数两次必须一样(决策层和引擎层各自调用,漂移会误判)
    assert counter.len_tokens([msg]) == counter.len_tokens(msg)


def test_counter_mixed_ascii_cjk() -> None:
    """中文按约 1.5 字符/token 估:'中'×6 → 4 token。"""
    assert EstimateTokenCounter().len_tokens("中" * 6) == 4


def test_fallback_model_info() -> None:
    """兜底模型信息:主力模型 glm-5.3 的 128k 窗口。"""
    assert FALLBACK_MODEL_INFO.context == 128_000


def test_source_digest_stable_and_sensitive() -> None:
    """指纹稳定(同输入同输出)、灵敏(内容或角色一变就变)。"""
    msgs = [Message("user", "你好"), Message("assistant", "回复")]
    again = [Message("user", "你好"), Message("assistant", "回复")]
    assert source_digest(msgs) == source_digest(again)  # 相等消息 → 同指纹

    changed_content = [Message("user", "你好!"), Message("assistant", "回复")]
    assert source_digest(msgs) != source_digest(changed_content)

    changed_role = [Message("assistant", "你好"), Message("assistant", "回复")]
    assert source_digest(msgs) != source_digest(changed_role)


def test_is_tool_result_excludes_hidden_prompts() -> None:
    """工具输出 = 没标 hide 的 system 消息;系统提示词(hide)不算。"""
    assert is_tool_result(Message("system", "$ ls")) is True
    assert is_tool_result(Message("system", "系统提示词", hide=True)) is False
    assert is_tool_result(Message("user", "$ ls")) is False
    assert is_tool_result(Message("assistant", "$ ls")) is False


def test_file_output_store_sequential_files(logdir: Path) -> None:
    """落盘文件按序号递增命名,内容原样写回。"""
    store = FileOutputStore(logdir)
    p1 = store.save("第一段全量输出", {"tokens": 100})
    p2 = store.save("第二段全量输出", {"tokens": 200})
    assert p1.name == "autocompact-001.txt"
    assert p2.name == "autocompact-002.txt"
    assert p1.read_text(encoding="utf-8") == "第一段全量输出"
    assert p2.read_text(encoding="utf-8") == "第二段全量输出"
