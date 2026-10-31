# token 估算（无分词器版）· 移植开发文档

> 来源：`gptme/util/tokens.py`（275 行）→ 目标：约 80 行，零第三方依赖
> 前置阅读：[token-counting.md](token-counting.md)（原理报告）
> 写作日期：2026-10-31 · 格式对齐 [autocompact-dev.md](../tools/autocompact-dev.md)

## 1. 移植目标与范围

**要的**：一个不依赖任何第三方库的 token 估算函数 `len_tokens()`，带缓存，能数字符串 / 消息对象 / 消息列表。

**不要的**（全部砍掉）：tiktoken、分词器下载线程、5 秒超时、`GPTME_TIKTOKEN_TIMEOUT` 等环境变量、分词器缓存。

**保留的骨架**：三分支入口（str / 消息 / 列表）→ 内容哈希做 key 的缓存 → 上限淘汰。

## 2. 关键决策（每条对照源码）

| # | 决策 | 对照 gptme | 说明 |
|---|---|---|---|
| D1 | 缓存 key 用内容的 SHA-256，不存原文 | 一致（tokens.py:253） | 原文可能是一条几十 KB 的消息，哈希固定 64 字符 |
| D2 | 缓存上限 1000 条，满了删最早插入的 | 一致（tokens.py:267-269） | Python 3.7+ 的 dict 按插入序遍历，`pop(next(iter(...)))` 即删最早的 |
| D3 | `model` 参数不保留 | **简化** | gptme 保留它是因为不同模型用不同分词表；估算版结果与模型无关，砍掉。将来接真分词器时再加回来 |
| D4 | **估算值写进缓存** | **相反**（tokens.py:262-264 不缓存估算值） | gptme 不缓存是等将来分词器可用了拿精确值；估算版永远只有估算，不缓存没有任何好处 |
| D5 | **系数不用 ÷4，改"CJK 字符每个算 1 个，其余字符 ÷4（向上取整）"** | **改进** | gptme 的 ÷4 是英文经验值，对中文严重低估（见 §3）。低估比高估危险：以为装得下，实际 API 报错 |
| D6 | 不加锁 | 一致 | gptme 的缓存就没锁（tokens.py:15 是裸 dict）。CPython 的 dict 单操作是原子的，最坏情况是两个线程重复算一次，不会出错。目标项目若高并发再考虑 |
| D7 | 只数消息的文字字段，图片/包装不计 | 一致（tokens.py:248-249 只取 `.content`） | 继承这个"偏低"倾向；D5 对中文的高估恰好部分抵消它 |

## 3. 实测依据（D5 为什么必须改）

在 cl100k_base（真分词器）上实测，对比"÷4 估算 / 真实值"：

| 样本 | 字符 | 真实 token | 字符÷token | ÷4 估算是真实值的 |
|---|---|---|---|---|
| 英文短句 | 52 | 10 | 5.20 | **130%**（高估，安全） |
| 纯中文短句 | 17 | 17 | 1.00 | **24%**（低估 4 倍，危险） |
| 中英混合 | 42 | 19 | 2.21 | **53%**（低估 2 倍，危险） |
| Python 代码 | 68 | 15 | 4.53 | 113%（高估，安全） |

结论：错误方向比错误大小更重要。**高估 → 提前压缩，浪费一点；低估 → 以为装得下实际装不下，请求直接失败**。中文项目用 ÷4 是系统性低估，必须改。

改进后的规则（CJK 每字 1 token + 其余 ÷4 向上取整）验证同一批样本：纯中文 17→17（100%）、混合 21→真实 19（110%）、英文 13→10（130%）、代码 17→15（113%）。全部落在"略高估"的安全区。

## 4. 接口与实现（可直接照抄）

```python
"""token 估算工具 —— 无分词器版，移植自 gptme/util/tokens.py。

估算规则（D5）：每个中日韩字符算 1 个 token，其余字符每 4 个算 1 个（向上取整）。
依据：cl100k_base 实测，纯中文 1.0 字符/token，英文 5.2，代码 4.5；
略高估是故意的——高估只会提前压缩，低估会让请求在 API 处失败。
"""

import hashlib

CACHE_MAX = 1000  # 缓存上限，超过删最早插入的（对齐 gptme tokens.py:267）
CHARS_PER_TOKEN = 4  # 非中日韩字符的系数

_token_cache: dict[str, int] = {}  # 内容SHA-256 -> token数，无锁（对齐 gptme，见 D6）


def _is_cjk(ch: str) -> bool:
    """是否中日韩字符（含全角标点、日文假名）。"""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF  # 汉字基本区
        or 0x3400 <= cp <= 0x4DBF  # 汉字扩展A
        or 0x3040 <= cp <= 0x30FF  # 日文假名
        or 0x3000 <= cp <= 0x303F  # 中文标点（，。、）
        or 0xFF00 <= cp <= 0xFFEF  # 全角字符（！？：）
    )


def _estimate(text: str) -> int:
    """纯函数：估算一段文字的 token 数。"""
    cjk = sum(1 for ch in text if _is_cjk(ch))
    rest = len(text) - cjk
    # 向上取整：宁可多算，不少算（D5）
    return cjk + -(-rest // CHARS_PER_TOKEN)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def len_tokens(content) -> int:
    """数 token（估算版）。

    content 可以是：
      - str                 直接数
      - 有 .content 属性    数它的文字字段（消息对象，鸭子类型即可）
      - list                逐条相加
    空/无内容返回 0。结果进缓存（key=内容哈希）。
    """
    if isinstance(content, list):
        return sum(len_tokens(m) for m in content)
    if hasattr(content, "content"):
        content = content.content
    if not content:
        return 0

    key = _hash(content)
    cached = _token_cache.get(key)
    if cached is not None:
        return cached

    count = _estimate(content)
    if len(_token_cache) >= CACHE_MAX:  # 淘汰最早插入的
        _token_cache.pop(next(iter(_token_cache)))
    _token_cache[key] = count
    return count
```

## 5. 配套：阈值怎么用这个数

gptme 的用法（照搬即可）：拿 `len_tokens(全部消息)` 和"模型窗口 × 比例"比，三个档位（reduce.py:71-72、decision.py）：

```python
used = len_tokens(messages)
ratio = used / model_context_window
# ratio > 0.9  → 硬压缩/丢旧消息
# ratio > 0.75（或 0.5 且有大工具输出）→ 软压缩/截断
# ratio > 0.5  → 可以开始给模型发用量提醒
```

因为估算是近似值，阈值宁低勿高（gptme 对不准的分词器也是这么处理的：把 0.9 降到 0.75，reduce.py:68-72 注释）。

## 6. 测试清单

| 用例 | 输入 | 期望 |
|---|---|---|
| 空内容 | `""`、`None`、`.content=""` 的对象 | 0 |
| 纯英文 | `"the"` | 1（3字符 → ceil(3/4)） |
| 纯中文 | 17 个汉字 | 17 |
| 中英混合 | "看一下 gptme 的文件" | CJK数 + ceil(其余/4) |
| 消息分支 | 带 `.content` 的简单对象 | 与直接传字符串相同 |
| 列表分支 | [msg1, msg2] | 两者之和 |
| 缓存命中 | 同一字符串调两次 | 第二次不重算（可临时替换 `_estimate` 计数验证） |
| 上限淘汰 | 灌入 1001 条不同内容 | `len(_token_cache)==1000`，最早那条已不在 |

## 7. 将来接入真分词器的路径（不改动调用方）

接口已经留好了替换点，升级时只动两处：

1. `_estimate(text)` 换成 `tokenizer.encode(text)` 数个数；
2. 缓存 key 从 `_hash(content)` 换成 `(_hash(content), model)`——因为从这一刻起，结果跟模型有关（不同模型不同分词表），`len_tokens` 加回 `model` 参数。

调用方代码一行不用改。

## 8. 任务分解

- **M0**（半小时）：`_is_cjk` / `_estimate` 两个纯函数 + §6 前 5 个用例
- **M1**（半小时）：`len_tokens` 三分支 + 缓存 + 淘汰 + 后 3 个用例
- **M2**：接入调用方——模型窗口常量 + `context_usage()`（返回 used/ratio）
- **M3**（可选，将来）：按 §7 换真分词器

## 9. 未决问题（开工前想一遍）

1. `CHARS_PER_TOKEN=4` 和"CJK=1"是按 cl100k_base 实测定的；如果目标项目主用别家模型（Claude/Gemini），CJK 单价不同，但"CJK ≥ 1 token/字"的方向对所有主流模型都成立，系数不用改。
2. 缓存上限 1000 条：gptme 的经验值。每条只存 64 字符 key + 整数，内存可忽略；若目标项目单会话消息量巨大，可放到 5000。
3. 不加锁的前提是 CPython；如果目标项目用多进程（不是多线程），缓存各进程各一份，只是重复计算，不出错。
4. 消息对象只认 `.content` 属性——如果目标项目的消息还有"工具调用参数""附件"等要计数的字段，在 `len_tokens` 的消息分支里加，别在 `_estimate` 里加。
