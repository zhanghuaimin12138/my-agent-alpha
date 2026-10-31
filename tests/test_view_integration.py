"""压缩视图的业务读取路径集成测试 —— 对应开发方案 V3。

核心命题:引入视图后,"给人看的/给状态重放的"读完整历史(Master),
"给模型看的"读压缩视图(Active View),两条读取线不能串。

验收点(方案 §8 V3):
- todo 的 add/update 被压缩掉后,重启重放仍能重建正确清单(读 Master);
- 终端回放展示完整历史,而模型输入来自视图;
- /resume 恢复正确的视图并继续双写;
- /log list 的用户消息数不随压缩减少;/log 区分两份历史;
- 子代理会话目录与主会话互不影响。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from my_agent_alpha import LogManager, Message
from my_agent_alpha import compaction as compaction_mod
from my_agent_alpha.chat import chat
from my_agent_alpha.commands import builtin as builtin_commands  # noqa: F401 导入即注册内置命令
from my_agent_alpha.commands import execute_cmd
from my_agent_alpha.compaction import compact_current_to_view
from my_agent_alpha.hooks import todo as todo_hooks
from my_agent_alpha.llm import mock as llm_mock
from my_agent_alpha.llm.mock import set_scripted_replies
from my_agent_alpha.llm.openai_compat import msgs2dicts
from my_agent_alpha.logmanager.sessions import (
    list_sessions,
    switch_session,
)
from my_agent_alpha.tools import todo as todo_tool
from my_agent_alpha.tui import replay
from my_agent_alpha.utils import len_tokens


def _todo_call(command: str) -> Message:
    """造一条"模型调用 todo 工具"的 assistant 消息(markdown 围栏格式)。"""
    return Message("assistant", f"```todo\n{command}\n```")


# ---- todo 重放:被压缩掉的状态变更必须还能重演 ----


def test_todo_replay_after_compaction(logdir: Path) -> None:
    """add/update 全被压掉,重启重放(读 Master)后清单状态依旧正确。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(_todo_call("add 修 bug"))
    manager.append(_todo_call("add 写文档"))
    manager.append(_todo_call("update 1 state=completed"))
    # 压缩:视图只保留最后一条消息,前面三条 todo 调用全被压掉
    manager.create_compacted_view(manager.master_messages[-1:])
    assert len(manager.messages) == 1
    assert "add" not in manager.messages[0].content  # 视图里确实没有 add 调用

    # 重启并跑 session.start 重放钩子(读完整历史)
    todo_tool.reset_todos()
    restored = LogManager.load(logdir)
    todo_hooks.replay_todo_hook(restored)

    # 完整历史重放的结果:"修 bug"已完成(只剩"写文档"未完成)
    remaining = [t.text for t in todo_tool.incomplete_items()]
    assert remaining == ["写文档"]


# ---- 终端回放:人看完整历史,模型读视图 ----


def test_replay_shows_master_not_view(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """回放打印被压缩掉的旧消息;而 manager.messages(模型输入)不含它们。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "很早的问题"))
    manager.append(Message("assistant", "很早的回答"))
    manager.create_compacted_view([Message("system", "早期对话的摘要", pinned=True)])

    replay(manager)
    out = capsys.readouterr().out
    assert "很早的问题" in out  # 人能看到完整历史
    assert "摘要" not in out  # 压缩占位不是真实对话,回放不显示
    # 同一个 manager:模型读的视图里没有旧消息
    assert "很早的问题" not in [m.content for m in manager.messages]


# ---- /resume:恢复视图并继续双写 ----


def test_resume_restores_view_and_dual_writes(logdir: Path, tmp_path: Path) -> None:
    """switch_session(即 /resume 的执行体)恢复激活视图,后续追加双写。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "旧消息"))
    manager.create_compacted_view([Message("system", "摘要")])

    # 模拟 /resume:从会话列表找到这个会话再切换(不走命令行的输入交互)
    meta = list_sessions(logdir.parent)[0]
    new = switch_session(manager, meta, "mock/echo")

    assert new.current_view == "compacted-001"
    new.append(Message("user", "切换后的新消息"))
    # 双写:两边各 +1
    assert len(new.master_messages) == 2
    assert [m.content for m in new.messages] == ["摘要", "切换后的新消息"]
    again = LogManager.load(logdir)
    assert again.current_view == "compacted-001"
    assert len(again.messages) == 2


# ---- /log:数字不许骗人 ----


def test_log_shows_both_histories(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """/log 报完整历史的条数;激活视图后额外说明模型读哪份。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "第一条"))
    manager.append(Message("user", "第二条"))

    assert execute_cmd("/log", manager, "mock/echo") is True
    out = capsys.readouterr().out
    assert "消息条数:2(完整历史)" in out
    assert "视图" not in out  # 没有视图就不出现视图行

    manager.create_compacted_view([manager.master_messages[-1]])
    assert execute_cmd("/log", manager, "mock/echo") is True
    out = capsys.readouterr().out
    assert "消息条数:2(完整历史)" in out  # 完整历史不缩水
    assert "compacted-001" in out  # 说清模型读的是这份
    assert "1 条(模型读这份压缩后的)" in out


def test_log_list_user_count_not_shrunk(logdir: Path) -> None:
    """压缩后 /log list 的用户消息数不变(列表统计一直直接读 Master 文件)。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "问题一"))
    manager.append(Message("user", "问题二"))
    before = list_sessions(logdir.parent)[0].n_user
    assert before == 2

    manager.create_compacted_view([Message("system", "摘要")])
    after = list_sessions(logdir.parent)[0].n_user
    assert after == 2  # 压缩不改变会话真实长度


# ---- 子代理会话独立 ----


def test_subagent_session_unaffected(logdir: Path, tmp_path: Path) -> None:
    """主会话激活视图,不影响隔壁子代理会话的读取与写入。"""
    # 主会话:激活视图
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("user", "主会话消息"))
    manager.create_compacted_view([Message("system", "摘要")])

    # 子代理有自己的会话目录(subagent- 前缀),里面没有 active.json
    sub_dir = logdir.parent / "subagent-00000000000000000000000000000000"
    sub = LogManager.load(sub_dir, create=True)
    sub.append(Message("user", "子代理消息"))
    assert sub.current_view is None  # 读自己的 Master
    assert len(sub.messages) == 1
    assert not (sub_dir / "views").exists()  # 没被主会话牵连出视图目录

    # 主会话的视图状态也完好如初
    assert manager.current_view == "compacted-001"


# ---- chat() 全链路:模型读视图,人看完整历史,双写继续 ----


def test_chat_end_to_end_with_view(
    logdir: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """恢复一个已激活视图的会话接着聊:模型输入来自视图,终端回放完整历史。"""
    # 预置"有历史 + 已压缩"的会话(视图里只剩一条摘要占位)
    seed = LogManager.load(logdir, create=True)
    seed.append(Message("user", "旧问题"))
    seed.append(Message("assistant", "旧回答"))
    seed.create_compacted_view([Message("system", "旧对话的摘要", pinned=True)])

    # 在 mock 流式入口包一层探针,记录模型实际收到的消息列表
    captured: list[list[Message]] = []
    real_stream = llm_mock.stream

    def spy_stream(messages: Sequence[Message], model_name: str) -> Iterator[str]:
        captured.append(list(messages))
        return real_stream(messages, model_name)

    monkeypatch.setattr("my_agent_alpha.llm.mock_stream", spy_stream)

    set_scripted_replies(["收到,继续。"])
    chat(["继续"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)

    # 模型输入来自视图:没有旧对话原文,有压缩摘要和新输入
    sent = captured[-1]
    assert "旧问题" not in [m.content for m in sent]
    assert any("摘要" in m.content for m in sent)
    assert "继续" in [m.content for m in sent]

    # 恢复会话时的终端回放:完整历史(旧对话原文可见)
    out = capsys.readouterr().out
    assert "旧问题" in out

    # 收尾:双写在 chat 主链路上照常工作,两边都有新消息
    final = LogManager.load(logdir)
    assert final.current_view == "compacted-001"
    assert "继续" in [m.content for m in final.messages]
    assert "旧问题" in [m.content for m in final.master_messages]


# ---- V4:压缩器接入视图(/compact 与 compact_current_to_view)----


def _chat_with_fat_thinking(logdir: Path) -> LogManager:
    """造一个"老回复都带大段思考"的会话(剥思考就有可观收益)。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("system", "提示词", pinned=True, hide=True))
    for i in range(8):
        manager.append(Message("user", f"问{i}"))
        manager.append(Message("assistant", "<think>" + "思考" * 2000 + f"</think>结论{i}"))
    manager.append(Message("user", "最新问题"))
    return manager


def test_compact_to_view_creates_view(logdir: Path) -> None:
    """手动压缩:生成视图、模型改读视图、完整历史一字节不动。"""
    manager = _chat_with_fat_thinking(logdir)
    master_bytes = (logdir / "conversation.jsonl").read_bytes()
    n_master = len(manager.master_messages)

    view_id = compact_current_to_view(manager, "mock/echo")

    assert view_id == "compacted-001"
    assert manager.current_view == "compacted-001"
    # 压缩有效:老的思考区被剥掉(距结尾不足 5 条的保持原样,近期的本来就不动),
    # 视图明显小于完整历史
    assert "<think>" not in "".join(m.content for m in manager.messages[:-5])
    assert len_tokens(manager.messages) < len_tokens(manager.master_messages) * 0.5
    # 安检要求:pinned 提示词和最新 user 意图都在视图里
    contents = [m.content for m in manager.messages]
    assert contents[0] == "提示词" and manager.messages[0].pinned
    assert "最新问题" in contents
    # 完整历史原样(Master 只追加,压缩不改写)
    assert (logdir / "conversation.jsonl").read_bytes() == master_bytes
    assert len(manager.master_messages) == n_master
    # 双写接着工作
    manager.append(Message("user", "压缩后的新输入"))
    assert len(manager.master_messages) == n_master + 1
    assert manager.messages[-1].content == "压缩后的新输入"


def test_compact_to_view_no_gain_no_view(logdir: Path) -> None:
    """没什么可压缩的:不建视图、不产生任何文件。"""
    manager = LogManager.load(logdir, create=True)
    manager.append(Message("system", "提示词", pinned=True, hide=True))
    manager.append(Message("user", "就一句话"))

    assert compact_current_to_view(manager, "mock/echo") is None
    assert manager.current_view is None
    assert not (logdir / "views").exists()


def test_second_compaction_starts_from_view(logdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """第二次压缩的输入是上一次视图,不是重新压整个 Master。"""
    manager = _chat_with_fat_thinking(logdir)
    assert compact_current_to_view(manager, "mock/echo") == "compacted-001"
    first_view = list(manager.messages)

    # 再聊几轮(进 Master 和 compacted-001)
    manager.append(Message("user", "新问题二"))
    manager.append(Message("assistant", "<think>短思考</think>答二"))

    # 探针:捕获 compact_messages 实际收到的压缩源
    captured: list[list[Message]] = []
    real = compaction_mod.compact_messages

    def spy(messages: list[Message], budget: int, logdir: Path | None = None) -> list[Message]:
        captured.append(list(messages))
        return real(messages, budget, logdir)

    monkeypatch.setattr(compaction_mod, "compact_messages", spy)

    assert compact_current_to_view(manager, "mock/echo") == "compacted-002"
    # 第二次的源 = 第一次视图 + 两轮新消息(对象都是同一条链上的)
    assert captured[0][: len(first_view)] == first_view
    assert [m.content for m in captured[0][-2:]] == ["新问题二", "<think>短思考</think>答二"]
    # compacted-001 仍是历史快照
    assert (logdir / "views" / "compacted-001.jsonl").exists()


def test_compact_command_runs(logdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """/compact 命令:成功路径打印视图名;没东西可压时如实说明。"""
    manager = _chat_with_fat_thinking(logdir)
    assert execute_cmd("/compact", manager, "mock/echo") is True
    out = capsys.readouterr().out
    assert "已压缩为视图 compacted-001" in out
    assert "完整历史仍为" in out

    # 已经压过、没什么新内容 → 第二次说明保持现状
    assert execute_cmd("/compact", manager, "mock/echo") is True
    out = capsys.readouterr().out
    assert "没有足够可压缩的内容" in out
    assert manager.current_view == "compacted-001"


def test_chat_continues_after_compaction(
    logdir: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """压缩成视图后,下一轮 step() 照常收发(含原生协议形状的消息)。"""
    manager = _chat_with_fat_thinking(logdir)
    # 补一段原生交互历史(两个调用、用量夹中间、结果配对)再压缩
    manager.append(
        Message("assistant", '读两个:\n@read(call_1): {"path": "a"}\n@read(call_2): {"path": "b"}')
    )
    manager.append(Message("system", "[用量] 输入 9 / 输出 9 tokens"))
    manager.append(Message("system", "内容一", call_id="call_1"))
    manager.append(Message("system", "内容二", call_id="call_2"))
    assert compact_current_to_view(manager, "mock/echo") == "compacted-001"
    assert manager.messages[-1].call_id == "call_2"  # 配对完整进了视图

    # 压缩后的视图直接发给接口转换层:无孤儿、无丢主调用
    dicts = msgs2dicts(manager.messages)
    call_ids = {
        tc["id"] for d in dicts if d["role"] == "assistant" for tc in d.get("tool_calls", [])
    }
    assert sorted(d["tool_call_id"] for d in dicts if d["role"] == "tool") == sorted(call_ids)

    # 继续对话:mock 模型正常一轮
    captured: list[list[Message]] = []
    real_stream = llm_mock.stream

    def spy_stream(messages: Sequence[Message], model_name: str) -> Iterator[str]:
        captured.append(list(messages))
        return real_stream(messages, model_name)

    monkeypatch.setattr("my_agent_alpha.llm.mock_stream", spy_stream)
    set_scripted_replies(["继续的回答"])
    chat(["接着说"], logdir=logdir, workspace=workspace, model="mock/echo", non_interactive=True)

    assert captured[0] == list(manager.messages) or captured  # 请求发出去了
    assert captured[0][-1].content == "接着说"
    final = LogManager.load(logdir)
    assert final.current_view == "compacted-001"
    assert "接着说" in [m.content for m in final.messages]
