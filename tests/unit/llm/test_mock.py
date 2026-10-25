"""llm/mock.py 内部行为测试:脚本机制的边角(环境变量传递、防呆)。"""

from __future__ import annotations

import json

from my_agent_alpha import Message
from my_agent_alpha.llm import mock as llm_mock
from my_agent_alpha.llm import reply
from my_agent_alpha.llm.mock import set_scripted_replies


def test_mock_env_script_directs_subprocess(monkeypatch) -> None:
    """环境变量脚本(ALPHA_MOCK_REPLIES):让派出去的子进程也能被导演。

    set_scripted_replies 是进程内状态,子代理子进程看不到;
    要导演子代理的行为(如记忆整理代理),脚本只能走环境变量传过去。
    """
    monkeypatch.setenv("ALPHA_MOCK_REPLIES", json.dumps(["第一条脚本", "第二条脚本"]))
    msgs = [Message("user", "hi")]
    assert llm_mock.chat(msgs, "echo")[0] == "第一条脚本"
    assert llm_mock.chat(msgs, "echo")[0] == "第二条脚本"
    # 脚本用完自动回退 echo;坏 JSON 也回退 echo 而不是炸
    assert llm_mock.chat(msgs, "echo")[0].startswith("Echo")
    monkeypatch.setenv("ALPHA_MOCK_REPLIES", "{坏的json")
    llm_mock.clear_scripted()  # 清掉缓存,逼它重读环境变量
    assert llm_mock.chat(msgs, "echo")[0].startswith("Echo")


def test_set_scripted_bare_string_is_one_reply() -> None:
    """防呆:直接传字符串(忘了包列表)也按一条回复处理,不拆成字符。"""
    set_scripted_replies("一条完整的回复")
    assert reply([Message("user", "x")], "mock/echo").content == "一条完整的回复"
