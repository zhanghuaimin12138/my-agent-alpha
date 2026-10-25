"""长期记忆的端到端:索引注入会话、整理代理(真实子进程)合并删除。"""

from __future__ import annotations

from pathlib import Path

import pytest

from my_agent_alpha import LogManager
from my_agent_alpha.chat import chat
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.memory_store import list_memories, save_memory


def test_e2e_memory_injected_into_session(
    clean_dirs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPHA_LOGS_HOME", str(clean_dirs / "logs"))
    ws = clean_dirs / "ws"
    ws.mkdir()
    save_memory("旧约定", "回答要简短", ws, project_level=True)
    set_scripted_replies(["好的"])
    chat(
        ["hi"],
        logdir=clean_dirs / "logs" / "c1",
        workspace=ws,
        model="mock/echo",
        non_interactive=True,
    )
    manager = LogManager.load(clean_dirs / "logs" / "c1")
    # 精确匹配注入的记忆索引(系统提示词里也有"长期记忆"字样,别撞上)
    seeded = [m for m in manager.messages if m.content.startswith("# 长期记忆")]
    assert seeded and "旧约定" in seeded[0].content
    assert seeded[0].pinned and seeded[0].hide  # 提示词类消息


def test_consolidate_merges_and_deletes(clean_dirs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """整理代理端到端:真子进程 + 白名单 + 脚本导演,合并重复、删除过时。

    这是方案 M4 的验收形态:mock 模型扮演整理代理,通过环境变量脚本
    指挥它先同名覆盖(合并两条近似记忆)再删除证伪条目 ——
    断言全部发生在真实文件系统上。
    """
    import json

    from my_agent_alpha import context
    from my_agent_alpha.memory_store import build_consolidate_prompt
    from my_agent_alpha.tools.subagent import run_subagent

    monkeypatch.setenv("ALPHA_LOGS_HOME", str(clean_dirs / "logs"))
    ws = clean_dirs / "ws"
    ws.mkdir()
    # 预置:两条近似重复 + 一条已被证伪的"事实"
    save_memory("部署流程", "旧发布步骤\n1. ssh", ws, project_level=True, description="发布步骤")
    save_memory(
        "服务器部署", "另一份发布步骤\n2. systemctl", ws, project_level=True, description="发布流程"
    )
    save_memory("过时事实", "项目模型是 kimi", ws, project_level=True, description="已证伪")
    # 导演子代理(环境变量脚本随子进程继承):合并 → 删重复+删证伪 → 报告
    monkeypatch.setenv(
        "ALPHA_MOCK_REPLIES",
        json.dumps(
            [
                "合并两条近似记忆:\n```memory 部署流程\n"
                "---\ndescription: 服务器登录与发布步骤(合并版)\ntype: project\n---\n"
                "服务器发布步骤\n1. ssh admin@10.0.0.1\n2. systemctl restart app\n```",
                "删除被并掉的重复和证伪条目(一条回复可以带多个围栏批量调):\n"
                "```memory delete 服务器部署\n```\n```memory delete 过时事实\n```",
                "整理完成:合并 2 条为 1,删除 1 条证伪条目。",
            ]
        ),
    )

    with context.agent_context(ws, clean_dirs / "logs"):
        msgs = list(
            run_subagent(
                build_consolidate_prompt(ws),
                allowed_tools=("read", "grep", "edit", "memory"),
            )
        )
    assert any("✅" in m.content for m in msgs)  # 子代理跑完并带回结论
    names = {e.name for e in list_memories(ws)}
    assert names == {"部署流程"}  # 合并后的还在;重复的、证伪的都没了
    kept = next(e for e in list_memories(ws))
    assert "systemctl" in kept.path.read_text(encoding="utf-8")  # 合并真的写进去了
    # 索引同步:只剩一行记忆
    index = (ws / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "服务器部署" not in index and "过时事实" not in index
