"""补发提醒:模型"本想调工具却没写成"时,提醒它补上,而不是静默结束。

背景:循环是否继续,只看"最后一条模型回复里有没有可执行的工具调用"。
但真机上出现过模型只在思考里"打算调用"、或把工具名写岔(如 ```websearch)
导致解析出 0 个调用 —— 循环照规则正常结束,用户看到的就是全程静默。

本模块负责两件事:
1. 检测(missed_tooluse_reason):最后一条回复"没写成调用"是不是这五种坑;
2. 提醒(make_nudge):构造一条教模型"怎么写对"的 system 消息,由中层循环
   追加进日志再生成一轮(次数上限 MAX_TOOL_NUDGES,循环层控制)。

继续判定的规则本身不在这里 —— 这里只是回合结束前的兜底,治"忘写代码块"。
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

from my_agent_alpha.codeblock import fenced_regions, iter_from_markdown, strip_think
from my_agent_alpha.message import Message
from my_agent_alpha.tools import get_tool_for_langtag, get_tools

# 提醒消息的开头标记:在日志/终端里一眼能认出"这是补发提醒"(也方便测试定位)
NUDGE_PREFIX = "[工具调用提醒]"

# 一个回合里最多"提醒模型补发工具调用"几次(防模型反复空回复把回合拖成拉锯;
# 和 MAX_STEPS 同一层面的保险丝,可用环境变量 ALPHA_MAX_TOOL_NUDGES 调)
MAX_TOOL_NUDGES = int(os.environ.get("ALPHA_MAX_TOOL_NUDGES", "1"))

# "行动宣告"关键词:正文里出现这些词、又没有任何代码块,大概率是模型在
# 说"我要开始干活了"却没写调用(真机实录:"我来帮你查一下上海今天的天气。")。
# 只收"我 + 动作"的词;刻意不收"帮你/帮您/试试"这类太常见的词 ——
# 真机复测里"我可以帮你做这些事:"(能力介绍)、"需要我帮你找找吗?"
# (主动提议)都是正常最终回答,收了必误报(误报代价是多跑一轮生成)
# 行动宣告词(分支 4 用)。"我来/让我/我去"要用带否定预查的正则:
# 紧挨着 想/要/会/可/如/若/你/别 这些字时是条件句("如果你让我……"那类),
# 不是宣告 —— 真机会话 2026-1007-2da08f 误报过一次,模型把完整回答
# 原样重复了一遍。
_ANNOUNCE_RE = re.compile(r"(?<![想要会可如若你别以能])(我来|让我|我去)")
_ANNOUNCE_WORDS = ("这就", "马上", "稍等", "let me", "i'll ")

# 宽松版"疑似原生调用行"探测器:@工具名(随便什么编号): { 开头。
# 比严格解析的 _NATIVE_CALL_RE 宽(不限制编号字符集):严格解析失败、
# 宽松探测却命中 = 回复里明明有调用行只是没被认出来 —— 该提醒而不是装没看见。
# (真机实录:dashscope 的编号带点号,曾让严格正则整行失配,调用静默丢失)
_LOOSE_NATIVE_CALL_RE = re.compile(r"@[A-Za-z0-9_.\-]+\([^)]*\)\s*:\s*\{")


def _resembles_known_tool(langtag: str) -> bool:
    """判断一个围栏语言标记是不是"某个已知工具名写岔了的近亲"。

    示例:工具叫 websearch.search,模型写 ```websearch —— 把工具名按点号
    拆成段(websearch / search),标记恰好是其中一段,说明模型想调这个
    工具、只是名字没写全,值得提醒它重写;
    ```python、```json 这类和任何工具名都不沾边的,只是普通示例代码,不打扰。

    示例:
        _resembles_known_tool("websearch")  → True(工具 websearch.search 的前段)
        _resembles_known_tool("python")     → False(谁也不是)
    """
    tag = langtag.lower()
    return any(tag in {t.name.lower(), *t.name.lower().split(".")} for t in get_tools())


def missed_tooluse_reason(messages: Sequence[Message]) -> str | None:
    """检查"最后一条模型回复":没解析出工具调用,是不是"本想调却没写成"?

    有问题返回一句给人看的原因(会拼进提醒消息),没问题返回 None。
    只在回合即将因此结束时调用 —— 正常的最终回答不该被打扰。

    覆盖五种真实踩过的坑(真机实录见 docs/tmp.jsonl 与踩坑记录条目 4/5):
    1. 正文为空:整个回复只有 <think> 思考(思考里甚至写着"我来调用工具"),
       或干脆一个字都没有 —— 模型把"打算调用"当成了"已经调用";
    2. 围栏写了,语言标记却不认识:如工具叫 websearch.search、模型写
       ```websearch —— 这种块会被解析器静默忽略,不提醒就永远没人发现;
    3. 正文以冒号结尾(如"现在为您查询:"):话说一半戛然而止,
       典型的"要说接下来执行什么,结果代码块忘写了";
    4. 行动宣告(真机实录"我来帮你查一下上海今天的天气。"):说要干活
       但没带冒号、也没写代码块 —— 只要正文一个围栏都没有、又出现
       句首式的"我来/让我/我去"或"这就/马上/稍等",就值得提醒一次;
       条件句里的"让我"不算("如果你让我记住更多,告诉我即可"是收尾
       提议,不是宣告,2026-1007-2da08f 真机误报过);
    5. 疑似原生调用行没被解析出来:回复里有 "@工具名(编号): {" 形态
       (宽松探测),严格解析却一个调用都没得到 —— 说明编号或参数写得
       不合规,该提醒而不是装没看见。

    示例:
        missed_tooluse_reason([Message("assistant", "<think>我想搜索</think>")])
        → "回复没有正文(只有思考过程或完全为空)"
        missed_tooluse_reason([Message("assistant", "任务完成")]) → None
    """
    content = next((m.content for m in reversed(messages) if m.role == "assistant"), None)
    if content is None:
        return None
    body = strip_think(content).strip()
    if not body:
        # 不用括号:这段话会被拼进提醒消息的括号里,嵌套括号看着难受
        return "回复没有正文,只有思考过程或完全为空"
    has_fence = "```" in body  # 有没有围栏代码块(检测器多个分支都要用)
    for cb in iter_from_markdown(body):
        if cb.lang and get_tool_for_langtag(cb.lang) is None and _resembles_known_tool(cb.lang):
            return f"代码块标记 ```{cb.lang} 不是任何已知工具的名字"
    if not has_fence and (
        _ANNOUNCE_RE.search(body) or any(m in body.lower() for m in _ANNOUNCE_WORDS)
    ):
        return '回复在宣布要执行操作(如"我来……"),却没有工具调用代码块'
    # 围栏里的 @xxx 只是文档示例,探测前先把围栏区域盖掉
    if _LOOSE_NATIVE_CALL_RE.search(_blank_fences(body)):
        return "回复里有工具调用行,但没有被解析出来(工具名、调用编号或参数格式不合规)"
    # 两种冒号都要认:ASCII 的 ":" 和全角的 "："(中文模型爱用全角,
    # 真实翻车样本 docs/tmp.jsonl 第 6 条就是全角冒号结尾)
    if body.endswith((":", "：")):
        return '回复以冒号结尾,像是要说"接下来执行什么",代码块却没写'
    return None


def _blank_fences(text: str) -> str:
    """把文本里围栏代码块的内容整体换成空白(保留长度,围栏外原样)。

    用途:检测"疑似原生调用行"时,得跳过围栏里的示例 —— 文档示例里写
    "@save(call_x): {...}" 很正常,那不是真调用。fenced_regions 给出
    围栏区间后,把区间内的字符逐个换成空格即可。

    示例:"a ```x\n@t(i): {}\n``` b" → "a      \n          \n   b"
    """
    regions = fenced_regions(text)
    if not regions:
        return text
    chars = list(text)
    for a, b in regions:
        for i in range(a, min(b, len(chars))):
            if chars[i] != "\n":  # 换行保留,防止把两段挤到"同一行"
                chars[i] = " "
    return "".join(chars)


def make_nudge(reason: str, tool_format: str) -> Message:
    """构造一条提醒模型"补上工具调用"的 system 消息。

    它会进日志、发给模型、也打印到终端 —— 踩坑记录的教训:"该有调用
    却解析出 0 个"这种事绝不能静默,用户和模型都得看见。
    两种通道措辞不同:markdown 通道教围栏写法,原生通道提醒用函数调用。
    """
    if tool_format == "markdown":
        how = (
            "正确写法是输出围栏代码块,语言标记 = 工具名,例如:\n"
            "```shell\nls -la\n```\n"
            "带参数的工具把参数写在工具名后面,例如 ```save path/to/file.txt。"
        )
    else:
        how = "请直接发起对应的函数调用(function calling),不要只用文字描述。"
    return Message(
        "system",
        f"{NUDGE_PREFIX} 你上一条回复没有产生任何工具调用({reason})。"
        '只在思考里"打算调用"、或只用文字描述要做什么,工具都不会执行。'
        f"{how}"
        "如果任务确实无需工具,请直接给出完整的最终回答(不要停在半句话上)。",
    )
