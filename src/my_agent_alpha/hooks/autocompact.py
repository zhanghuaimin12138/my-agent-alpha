"""自动压缩钩子:回合结束后看一眼预算,快超窗就把当前历史压成视图。

为什么挂在 turn.post(回合边界):这是唯一安全的压缩时机 —— 回合内
所有工具调用都已收尾、日志里没有半截交互;在流式生成中或调用还没
全部返回时压缩,会拆散"调用 ↔ 结果"配对,视图就废了。

设计约束(开发方案 §6.6):
- 不满足触发条件:什么都不做,零消息产出、零磁盘动作;
- 压缩成功:只打一行控制台提示,不往会话日志写通知(模型不需要看见
  压缩过程,人看见即可);
- 压缩失败:进入冷却期,期间不再尝试;原始日志和旧视图原样保留
  —— 自动机制永远不许弄丢历史;
- 同一 turn 内的超窗中途救场仍由 step() 的溢出重试兜底(本钩子只在
  回合边界工作,这是有意的分工)。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence

from my_agent_alpha.compaction import compact_current_to_view
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

# 失败冷却:压缩失败后连续几个回合不再尝试(反复失败反复试只会刷屏)
_COOLDOWN_TURNS = 3

# 压缩目标:触发后要压到"触发线 × 这个比例"以下。不设紧目标的话,
# 一段"刚过触发线"的历史会触发压缩、却压不出一成收益(is_meaningful_reduction
# 拒绝建视图),白忙一场 —— 触发线(0.8)和压缩目标之间必须留出这段落差
_COMPACT_TARGET_RATIO = 0.75

# 剩余冷却回合数(模块级,和 todo 门控的计数器同款管理方式)
_cooldown = 0


def reset_autocompact() -> None:
    """冷却计数归零(conftest 测试隔离用)。"""
    global _cooldown
    _cooldown = 0


def should_auto_compact(messages: Sequence[Message], budget: int) -> bool:
    """当前有效历史是否大到该压缩:估算 tokens 达到预算 × 触发比例。"""
    return len_tokens(messages) >= budget * _AUTO_COMPACT_RATIO


def autocompact_hook(manager: LogManager, **_ctx: object) -> Iterator[Message]:
    """turn.post:预算吃紧时把当前有效历史压成视图(/compact 的自动版)。

    本钩子永远不产出消息(各分支都返回空迭代器):压缩状态只给人看
    (控制台一行),不进模型上下文 —— 模型不需要、也不应该看见"自己被压缩了"。
    """
    global _cooldown
    if _cooldown > 0:
        _cooldown -= 1
        return iter(())  # 冷却期:安静跳过,耗完额度再试
    model = get_current_model()
    if not model:
        return iter(())  # 拿不到当前模型(理论上是会话装配的空档),不硬猜
    budget = get_context_budget(resolve_model(model).context)
    if not should_auto_compact(manager.messages, budget):
        return iter(())  # 还很宽裕:什么都不做
    # 压缩目标压到触发线以下(见 _COMPACT_TARGET_RATIO 注释),保证有真收益
    target = int(budget * _AUTO_COMPACT_RATIO * _COMPACT_TARGET_RATIO)
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


def register_autocompact_hook() -> None:
    """注册自动压缩钩子(hooks/builtin.py 启动时调用)。"""
    register_hook("autocompact", HookType.TURN_POST, autocompact_hook, priority=0)
