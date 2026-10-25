"""tools/memory.py 测试:两条通道的参数、类型校验、scope、覆盖提示、删除。"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha import Message
from my_agent_alpha.memory_store import list_memories
from my_agent_alpha.tools import execute_msg


def test_memory_tool_writes_project_layer(clean_dirs: Path) -> None:
    msg = Message("assistant", "```memory 测试记忆\n这是标题\n这是正文\n```")
    results = execute_msg(msg)  # 无确认钩子 = 自动放行
    assert "已记住" in results[0].content
    # 写到了当前工作区(测试夹具设过 workspace)的 memory/ 目录
    from my_agent_alpha.context import get_workspace

    assert (get_workspace() / "memory" / "测试记忆.md").exists()


def test_memory_tool_type_and_validation(clean_dirs: Path) -> None:
    """工具通道:type/description 参数与围栏头类型都能用;非法 type 报错不落盘。"""
    from my_agent_alpha.context import get_workspace

    msg = Message(
        "assistant",
        '@memory(c1): {"name": "偏好甲", "content": "喜欢简短\\n正文", '
        '"type": "user", "description": "回答要简短"}',
    )
    results = execute_msg(msg)
    assert "已记住" in results[0].content
    e = list_memories(get_workspace())[0]
    assert e.mem_type == "user" and e.description == "回答要简短"
    # 围栏头第二个词当类型("```memory 围栏偏好 user")
    execute_msg(Message("assistant", "```memory 围栏偏好 user\n围栏标题\n```"))
    e2 = next(m for m in list_memories(get_workspace()) if m.name == "围栏偏好")
    assert e2.mem_type == "user"
    # 非法类型:报错且不写文件
    before = set((get_workspace() / "memory").glob("*.md"))
    results2 = execute_msg(
        Message("assistant", '@memory(c2): {"name": "坏记忆", "content": "x", "type": "diary"}')
    )
    assert "错误" in results2[0].content and "diary" in results2[0].content
    assert set((get_workspace() / "memory").glob("*.md")) == before


def test_tool_scope_param(clean_dirs: Path) -> None:
    """工具通道:类型推断与显式 scope 都能落到对的层,非法 scope 报错不落盘。"""
    from my_agent_alpha.context import get_workspace
    from my_agent_alpha.dirs import get_data_dir

    # conftest 的 autouse 夹具已把 workspace 指到 tmp 并进了 agent_context
    execute_msg(
        Message(
            "assistant",
            '@memory(c1): {"name": "偏好乙", "content": "喜欢简短", "type": "user"}',
        )
    )
    assert (get_data_dir() / "memory" / "偏好乙.md").exists()  # user → 用户层
    execute_msg(
        Message(
            "assistant",
            '@memory(c2): {"name": "事实丙", "content": "项目事实", '
            '"type": "project", "scope": "user"}',
        )
    )
    assert (get_data_dir() / "memory" / "事实丙.md").exists()  # 显式盖过推断
    results = execute_msg(
        Message(
            "assistant",
            '@memory(c3): {"name": "坏scope", "content": "x", "scope": "team"}',
        )
    )
    assert "错误" in results[0].content and "scope" in results[0].content
    assert not (get_workspace() / "memory" / "坏scope.md").exists()


def test_overwrite_notice(clean_dirs: Path) -> None:
    """覆盖提示:同名重写要告诉模型旧内容被替换;跨层同名提示"被盖住"(方案 M3.2)。"""
    execute_msg(Message("assistant", "```memory 部署甲\n旧标题\n旧正文\n```"))
    results = execute_msg(Message("assistant", "```memory 部署甲\n新标题\n新正文\n```"))
    assert "已记住" in results[0].content
    assert "覆盖了旧记忆「旧标题」" in results[0].content
    # 跨层同名:用户层已有,项目层再写 → 提示盖住而不是覆盖
    execute_msg(
        Message(
            "assistant",
            '@memory(c1): {"name": "偏好丙", "content": "用户层版本", "type": "user"}',
        )
    )
    results2 = execute_msg(
        Message(
            "assistant",
            '@memory(c2): {"name": "偏好丙", "content": "项目层版本", "scope": "project"}',
        )
    )
    assert "盖住" in results2[0].content


def test_delete_last_memory_removes_index_stub(clean_dirs: Path) -> None:
    """删光一层后不留只剩标题的空壳索引(真机 2026-1007 反馈)。"""
    from my_agent_alpha.context import get_workspace
    from my_agent_alpha.memory_store import render_memory_index

    execute_msg(Message("assistant", "```memory 独苗\n标题\n正文\n```"))
    assert (get_workspace() / "memory" / "MEMORY.md").exists()
    execute_msg(Message("assistant", "```memory delete 独苗\n```"))
    assert not (get_workspace() / "memory" / "MEMORY.md").exists()
    assert render_memory_index(get_workspace()) == ""


def test_delete_memory(clean_dirs: Path) -> None:
    """删除:原生 action=delete 与围栏 delete 都能删;删完索引同步;没有的报不误删。"""
    from my_agent_alpha.context import get_workspace
    from my_agent_alpha.memory_store import render_memory_index

    execute_msg(Message("assistant", "```memory 待删甲\n标题甲\n正文\n```"))
    execute_msg(Message("assistant", "```memory 待删乙\n标题乙\n正文\n```"))
    # 原生通道删除
    results = execute_msg(
        Message("assistant", '@memory(c1): {"action": "delete", "name": "待删甲"}')
    )
    assert "已删除" in results[0].content
    assert not (get_workspace() / "memory" / "待删甲.md").exists()
    # 围栏通道删除
    results2 = execute_msg(Message("assistant", "```memory delete 待删乙\n```"))
    assert "已删除" in results2[0].content
    # 索引同步:两条都没了;层删空 → 索引文件整个移除(空壳清理)
    assert not (get_workspace() / "memory" / "MEMORY.md").exists()
    assert render_memory_index(get_workspace()) == ""  # 全删光 → 不再注入
    # 删不存在的:明确报告,不炸
    results3 = execute_msg(
        Message("assistant", '@memory(c2): {"action": "delete", "name": "不存在"}')
    )
    assert "没有叫" in results3[0].content
