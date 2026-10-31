"""自动压缩钩子:回合结束后看一眼预算,快超窗就选一种方式压成视图。

为什么挂在 turn.post(回合边界):这是唯一安全的压缩时机 —— 回合内
所有工具调用都已收尾、日志里没有半截交互;在流式生成中或调用还没
全部返回时压缩,会拆散"调用 ↔ 结果"配对,视图就废了。

两种压缩方式怎么选(decide_compaction 三选一,报告 §10.1):
- 没超触发线 → none,什么都不做;
- 超了,机械压缩(删思考块/截超长输出)预演能省一成以上 → trim,
  免费快,能用就用;
- 超了,机械压缩没东西可删(干货型长对话)→ summarize,请模型写工作
  笔记。这条通道要花钱,**默认关闭**,环境变量 ALPHA_AUTO_LLM_COMPACT=1
  开启;手动 /compact summarize 永远可用,不受本钩子限制。

设计约束:
- 不满足触发条件:零消息产出、零磁盘动作;
- 压缩成功:只打一行控制台提示,不往会话日志写通知(模型不需要看见
  压缩过程,人看见即可);
- 压缩失败:trim 走回合冷却;请模型走"同指纹节流" —— 同一段历史
  (指纹相同)失败过就不再自动重试,历史有实质变化(指纹变了)才再试,
  不重复花钱;手动命令不受限;
- 原始日志和旧视图永远原样保留 —— 自动机制不许弄丢历史;
- 同一 turn 内的超窗中途救场仍由 step() 的溢出重试兜底(本钩子只在
  回合边界工作,这是有意的分工)。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from typing import Literal

from my_agent_alpha.compaction import (
    compact_current_to_view,
    compress_long_replies,
    strip_old_thinking,
    truncate_tool_results,
)
from my_agent_alpha.compaction_llm import compact_current_with_llm, digest_messages
from my_agent_alpha.context import get_current_model
from my_agent_alpha.hooks import HookType, register_hook
from my_agent_alpha.llm.models import resolve_model
from my_agent_alpha.logmanager import LogManager
from my_agent_alpha.message import Message
from my_agent_alpha.utils import get_context_budget, len_tokens

logger = logging.getLogger(__name__)

# 触发比例:当前有效历史的估算 tokens 达到预算的这个比例就动手。
# 0.8 = 留两成余量,别等真超窗才压 —— 真超窗要靠请求级兜底救,那是最后防线
_AUTO_COMPACT_RATIO = float(os.environ.get("ALPHA_AUTO_COMPACT_RATIO", "0.8"))

# 失败冷却:机械压缩失败后连续几个回合不再尝试(反复失败反复试只会刷屏)
_COOLDOWN_TURNS = 3

# 压缩目标:触发后要压到"触发线 × 这个比例"以下。不设紧目标的话,
# 一段"刚过触发线"的历史会触发压缩、却压不出一成收益(is_meaningful_reduction
# 拒绝建视图),白忙一场 —— 触发线(0.8)和压缩目标之间必须留出这段落差
_COMPACT_TARGET_RATIO = 0.75

# 机械压缩的"够用线":预演能省这么多比例就用机械的,省不到才请模型
# (和 gptme 的 MIN_SAVINGS_RATIO 同值)
_RULE_SAVINGS_RATIO = 0.10

# 剩余冷却回合数 + 上次"请模型压缩"失败时的历史指纹(模块级状态,
# 和 todo 门控的计数器同款管理方式)
_cooldown = 0
_failed_llm_digest: str | None = None


def reset_autocompact() -> None:
    """冷却计数与节流指纹归零(conftest 测试隔离用)。"""
    global _cooldown, _failed_llm_digest
    _cooldown = 0
    _failed_llm_digest = None


def should_auto_compact(messages: Sequence[Message], budget: int) -> bool:
    """当前有效历史是否大到该压缩:估算 tokens 达到预算 × 触发比例。"""
    return len_tokens(messages) >= budget * _AUTO_COMPACT_RATIO


def llm_auto_enabled() -> bool:
    """自动请模型压缩是否开启(默认关;手动 /compact summarize 不看这个开关)。"""
    return os.environ.get("ALPHA_AUTO_LLM_COMPACT", "0") == "1"


def decide_compaction(
    messages: Sequence[Message], budget: int, target: int
) -> Literal["none", "trim", "summarize"]:
    """三选一:不用压 / 机械压缩够用 / 得请模型写笔记。

    判断方式:拿机械压缩的**软压缩部分**(剥思考/截超长输出/压长回复,
    不丢整条消息)预演一遍(纯计算,免费),能省一成以上就用机械;
    省不动(干货型对话,剩下的都是语义密集的内容)才请模型写笔记。
    为什么不含兜底裁剪:裁剪靠丢旧消息省 token,它总能把历史砍到
    任何目标线(用它估算,"请模型"分支永远轮不到);而丢消息恰恰是
    干货场景里最该避免的 —— 那时应选保留语义的总结。
    """
    if not should_auto_compact(messages, budget):
        return "none"
    soft = strip_old_thinking(list(messages))
    soft = truncate_tool_results(soft, target, None)
    soft = compress_long_replies(soft)
    if len_tokens(soft) <= len_tokens(messages) * (1 - _RULE_SAVINGS_RATIO):
        return "trim"
    return "summarize"


def autocompact_hook(manager: LogManager, **_ctx: object) -> Iterator[Message]:
    """turn.post:预算吃紧时按三选一决策压缩(/compact 的自动版)。

    本钩子永远不产出消息(各分支都返回空迭代器):压缩状态只给人看
    (控制台一行),不进模型上下文 —— 模型不需要、也不应该看见"自己被压缩了"。
    """
    global _cooldown, _failed_llm_digest
    if _cooldown > 0:
        _cooldown -= 1
        return iter(())  # 机械压缩的冷却期:安静跳过,耗完额度再试
    model = get_current_model()
    if not model:
        return iter(())  # 拿不到当前模型(理论上是会话装配的空档),不硬猜
    budget = get_context_budget(resolve_model(model).context)
    # 压缩目标压到触发线以下(见 _COMPACT_TARGET_RATIO 注释),保证有真收益
    target = int(budget * _AUTO_COMPACT_RATIO * _COMPACT_TARGET_RATIO)
    action = decide_compaction(manager.messages, budget, target)
    if action == "none":
        return iter(())  # 还很宽裕:什么都不做
    if action == "trim":
        return _auto_trim(manager, model, target)
    return _auto_summarize(manager, model)


def _auto_trim(manager: LogManager, model: str, target: int) -> Iterator[Message]:
    """机械压缩分支:行为与 V5 版完全一致,失败进回合冷却。"""
    global _cooldown
    try:
        view_id = compact_current_to_view(manager, model, budget=target)
    except Exception as e:
        # 安检不过(比如回合实际没收尾干净)或磁盘问题:冷却,保留一切原样
        _cooldown = _COOLDOWN_TURNS
        logger.warning("自动压缩失败,冷却 %d 个回合(%s)", _COOLDOWN_TURNS, e)
        print(f"(自动压缩失败,接下来 {_COOLDOWN_TURNS} 个回合不再尝试:{e})")
        return iter(())
    if view_id is None:
        return iter(())  # 达到阈值但没有有效收益(省不足一成):安静跳过
    print(
        f"(自动压缩:完整历史 {len(manager.master_messages)} 条 → "
        f"视图 {view_id} {len(manager.messages)} 条)"
    )
    return iter(())


def _auto_summarize(manager: LogManager, model: str) -> Iterator[Message]:
    """请模型写笔记分支:默认关;同一段历史失败过就不再自动重试(不重复花钱)。"""
    global _failed_llm_digest
    if not llm_auto_enabled():
        return iter(())  # 通道关闭:安静跳过(想用请开环境变量或手动 /compact summarize)
    digest = digest_messages(manager.messages)
    if _failed_llm_digest == digest:
        return iter(())  # 同一段历史自动压失败过:历史有实质变化前不再花钱重试
    try:
        result = compact_current_with_llm(manager, model=model)
    except ValueError as e:
        # 笔记不合格/安检不过:不是故障,是货不行;重试未必有用,节流
        _failed_llm_digest = digest
        print(f"(自动请模型压缩未成功,历史有实质变化前不再重试:{e})")
        return iter(())
    if result.status == "success":
        _failed_llm_digest = None
        print(
            f"(自动压缩(模型笔记):完整历史 {len(manager.master_messages)} 条 → "
            f"视图 {result.view_id} 约 {result.tokens_after} tokens)"
        )
        return iter(())
    if result.status == "stale":
        return iter(())  # 摘要期间历史变了:指纹自然不同,下回合会再试
    # failed(模型故障)或 skipped(花了钱却压不出收益):都记节流,别反复烧钱
    _failed_llm_digest = digest
    if result.status == "failed":
        print(f"(自动请模型压缩失败:{result.reason};历史有实质变化前不再重试)")
    return iter(())


def register_autocompact_hook() -> None:
    """注册自动压缩钩子(hooks/builtin.py 启动时调用)。"""
    register_hook("autocompact", HookType.TURN_POST, autocompact_hook, priority=0)
