# 035 · iter_from_content 补盲测试

## 目标

给 `tools/base.py` 的 `ToolUse.iter_from_content` 补测。该函数在 031 整理时
已有约 20 个用例(两通道解析、think 剥离、混用排序),本次逐行对照实现
找**没被任何用例锁住的分支**,补 6 个,不写重复。

## 决策及理由

- 落位 `tests/unit/tools/test_base.py`(031 规则:镜像 base.py),不另开文件。
- 以"实现里有、用例里没有"为唯一入选标准,六个盲点:
  1. 冒号后没紧跟 `{` 不算调用——包括 `{` 写在下一行的(按行设计的边界);
  2. 冒号和 `{` 之间允许隔空格;
  3. **未闭合**围栏里的 @ 调用也当示例跳过(此前只测了已闭合围栏;
     fenced_regions 对未闭合围栏一路算到文末);
  4. `streaming=True` 透传:围栏闭合后还要空行确认(此前只在 codeblock
     层测过规则本体,没测本函数的透传);
  5. 块头多词参数整段作为 args 单个元素(`["a b"]` 而非 `["a", "b"]`),
     锁住"args 要么没有、恰好一个"的契约;
  6. 工具名带点号(MCP 的 `t.echo` 前缀)在单元层可解析——注册一个
     自定义 ToolSpec 验证,端到端此前只在 integration/test_mcp.py。
- 实现里 `not isinstance(kwargs, dict)` 分支(base.py:348)实际走不到
  (切片永远是配平的大括号,合法 JSON 必是对象),不硬造用例。

## 改动结构

- `tests/unit/tools/test_base.py`:新增 6 个用例(markdown 通道 1 个、
  原生通道 4 个、streaming 1 个),文件头"覆盖点"清单同步更新。
- 无源码改动。

## 验证结果

- `uv run pytest tests/unit/tools/test_base.py -q` → 36 passed(原 30 + 新 6);
- `uv run pytest -q` → **672 passed**。总数比 032 时的 375 多,来源:
  6 个是本次新增,约 287 个是同日 034 的参数化用例集
  (test_openai_compat.py 12 个函数展开成 295 个用例),其余为 033/034
  期间的小幅增改——本次全量含用户新增用例一起全绿,无冲突。
- `uv run ruff check` / `ruff format` 对该文件通过。

## 下一步

- base.py 里 `find_json_end`(引号内大括号计数)目前只被间接覆盖
  (经 native_braces_inside_strings),转义符 `\\"` 场景没测,可再补。
