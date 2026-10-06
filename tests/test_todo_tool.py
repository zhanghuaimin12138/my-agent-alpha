"""todo 工具执行端单测:子命令文法、双通道取参、状态应用。

不走 chat() 全流程,直接调 execute_todo / 辅助函数验证行为;
门控与重放的端到端测试在 test_todo_gate.py。
"""

from __future__ import annotations

from my_agent_alpha.message import Message
from my_agent_alpha.tools import todo


def _run(
    code: str | None, args: list[str] | None = None, kwargs: dict[str, str] | None = None
) -> str:
    """执行一次 todo 命令,返回结果消息的文本。"""
    results = list(todo.execute_todo(code, args, kwargs))
    assert len(results) == 1
    assert results[0].role == "system"
    return results[0].content


def test_markdown_channel_add() -> None:
    """markdown 通道:围栏体就是子命令。"""
    text = _run("add 修复登录")
    assert "#1 已添加" in text
    assert "修复登录" in text


def test_native_channel_add_equivalent() -> None:
    """原生通道:kwargs["command"] 与围栏体等价。"""
    text = _run(None, kwargs={"command": "add 修复登录"})
    assert "#1 已添加" in text
    assert todo.has_incomplete_todos()


def test_fence_header_args_fallback() -> None:
    """宽容路径:模型把命令写在围栏头上(```todo add xxx)→ 用 args 拼接。"""
    _run("add 第一件事")
    text = _run(None, args=["update", "1", "state=completed"])
    assert "#1 已更新" in text
    assert not todo.has_incomplete_todos()


def test_update_state_completes() -> None:
    """update state=completed 后清单视为已清(门控判定变 False)。"""
    _run("add 修 bug")
    assert todo.has_incomplete_todos()
    text = _run("update 1 state=completed")
    assert "[已完成]" in text
    assert not todo.has_incomplete_todos()


def test_update_combined_keys() -> None:
    """update 支持 state= 和 text= 组合;text 值可含空格。"""
    _run("add 旧描述")
    text = _run("update 1 state=in_progress text=新 描述")
    assert "[进行中]" in text
    assert "新 描述" in text


def test_remove_and_missing_id() -> None:
    """remove 删除已有项;删不存在的编号给错误说明。"""
    _run("add 任务")
    assert "#1 已删除" in _run("remove 1")
    assert not todo.has_incomplete_todos()
    assert "错误" in _run("remove 1")


def test_clear_renumbers_from_one() -> None:
    """clear 清空后编号从 #1 重新开始(重放幂等的前提)。"""
    _run("add 甲")
    _run("add 乙")
    assert "已清空 2 项" in _run("clear")
    assert "#1 已添加" in _run("add 丙")


def test_read_empty_and_filled() -> None:
    """read 输出:空表一句话;非空表给行式清单和未完成计数。"""
    assert "清单为空" in _run("read")
    _run("add 做事一")
    _run("add 做事二")
    _run("update 1 state=completed")
    text = _run("read")
    assert "未完成 1/2" in text
    assert "#1 [已完成] 做事一" in text
    assert "#2 [待办] 做事二" in text


def test_bad_grammar_reports_usage() -> None:
    """空命令、未知子命令、update 缺编号 → 错误消息带正确文法。"""
    assert "文法" in _run("")
    assert "未知子命令" in _run("fly 高飞")
    assert "文法" in _run("update state=completed")  # 缺编号


def test_update_illegal_state_lists_legal_values() -> None:
    """update 非法 state → 错误列出三个合法值。"""
    _run("add 任务")
    text = _run("update 1 state=done")
    assert "pending" in text and "in_progress" in text and "completed" in text


def test_execute_returns_system_message() -> None:
    """execute_todo 产出的消息角色是 system(工具结果约定)。"""
    (msg,) = list(todo.execute_todo("read", None, None))
    assert isinstance(msg, Message)
    assert msg.role == "system"
