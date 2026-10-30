"""会话自动命名(第二轮 R4):第一回合结束后让模型给会话起个短名。

治什么:会话目录名全是"2026-1005-3fa9c1"式的时间戳,/log 列表根本
分不清哪个是哪个。解法(round2-plan R4,对照 gptme 的会话标题):

- turn.post 钩子(不改核心循环):回合结束、会话还没有名字、且已有
  成型对话(至少一对 user + assistant)时,拿当前模型给首轮对话
  起一个 ≤20 字的标题;
- 标题写在会话目录的 meta.json(目录名不动 —— 会话进行中改目录名
  会弄断所有还攥着旧路径的句柄);
- 命名请求不进对话日志:它是元数据,不是消息;失败也不影响会话,
  下个回合再试;
- --name 指定的会话:main 入口直接把名字写成标题,钩子看到已有
  名字就跳过。

用法:无需手动调用 —— chat() 启动时经 register_builtin_hooks() 自动
注册;看结果用 /log(当前会话)或 /log list(全部会话)。
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence

from my_agent_alpha.codeblock import strip_think
from my_agent_alpha.context import get_current_model
from my_agent_alpha.hooks import HookType, register_hook
from my_agent_alpha.llm import reply
from my_agent_alpha.logmanager import LogManager
from my_agent_alpha.logmanager.sessions import (
    read_session_title,
    write_session_title,
)
from my_agent_alpha.message import Message

# 标题最长保留多少字(模型偶尔啰嗦,超了硬截)
_MAX_TITLE_CHARS = 30

# 总开关(默认开):测试环境用 ALPHA_AUTO_NAME=0 关掉,防止命名调用
# 消耗 mock 的脚本回复序列、subagent 子进程多跑一次模型
_AUTO_NAME_ENV = "ALPHA_AUTO_NAME"


def _make_title_prompt(messages: Sequence[Message]) -> str:
    """取首条 user + 首条 assistant(各截 600 字),拼命名请求。"""
    first_user = next((m.content for m in messages if m.role == "user"), "")
    first_assistant = next((m.content for m in messages if m.role == "assistant"), "")
    return (
        "给下面这段对话起一个不超过 20 个字的简短中文标题,概括用户在做什么。"
        "直接输出标题本身:不要解释、不要引号、不要句号结尾。\n\n"
        f"用户:{first_user[:600]}\n\n助手:{first_assistant[:600]}"
    )


def _clean_title(raw: str) -> str:
    """把模型给的标题收拾干净:先剥思考区,再取首行、剥引号、限长。

    为什么先剥思考区:有的模型把推理原文混在回复开头(2026-10-26 实测,
    标题成了"<think>The user wants a title"),剥掉思考区剩下的才是真标题;
    思考区没闭合(只写了一半)时 strip_think 返回空串 → 不写标题,下回合再试。
    """
    body = strip_think(raw).strip()
    if not body:
        return ""
    title = body.split("\n")[0].strip()
    title = title.strip("\"'“”‘’「」『』《》〈〉 ")
    return title[:_MAX_TITLE_CHARS]


def _try_name_session(manager: LogManager) -> None:
    """尝试命名一次:条件不满足就安静返回,失败打印一句(下回合再试)。"""
    if os.environ.get(_AUTO_NAME_ENV, "1") != "1":
        return  # 总开关关闭(测试隔离用)
    if read_session_title(manager.logdir) is not None:
        return  # 已有名字(--name 指定或之前生成过)
    msgs = manager.messages
    has_pair = any(m.role == "user" for m in msgs) and any(m.role == "assistant" for m in msgs)
    if not has_pair:
        return  # 对话还没成型,等下一回合
    model = get_current_model()
    if not model:
        return  # 拿不到当前模型(理论上是恢复场景的空档),不硬猜
    try:
        response = reply([Message("user", _make_title_prompt(msgs))], model)
        title = _clean_title(response.content)
        if not title:
            return  # 模型回了空气:不写空标题,下回合再试
        write_session_title(manager.logdir, title)
        print(f"(会话已命名:{title})")
    except Exception as e:  # 命名是锦上添花:任何失败都不许影响会话本身
        print(f"(会话命名失败:{e};下个回合再试)")


def naming_hook(manager: LogManager, **_ctx: object) -> Iterator[Message]:
    """turn.post:回合结束后给会话命名;不产出任何消息(标题是元数据)。"""
    _try_name_session(manager)
    return iter(())


def register_naming_hook() -> None:
    """注册命名钩子(hooks/builtin.py 启动时调用)。"""
    register_hook("builtin.naming", HookType.TURN_POST, naming_hook, priority=0)
