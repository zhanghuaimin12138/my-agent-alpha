# AGENTS.md —— AI 开发代理协作规范

本文件是给参与本仓库开发的 AI 代理(以及人类协作者)的行为规范,仿 gptme 的 AGENTS.md。

## 项目使命

仿造 gptme 构建个人 AI agent。**架构蓝图是 [docs/gptme-study.md](docs/gptme-study.md),动手前先读它**;
对 gptme 原实现的疑问可直接查源码副本 `/tmp/gptme`(若仍在磁盘)。

## 术语表

| 术语 | 含义 |
|---|---|
| **Message** | 唯一的消息单元(system/user/assistant 三 role;工具输出是带 call_id 的 system 消息) |
| **Log / LogManager** | 消息列表 + 持久化游标 / 每个会话目录的有状态门面 |
| **Step** | 一次"生成 + 工具执行"的 pass(阶段 1 引入) |
| **Turn** | 一个用户输入引发的连续 step 序列,直到 assistant 无可运行工具调用 |
| **ToolUse** | 从 assistant 文本里解析出的工具调用(阶段 1 引入) |

## 硬性规则

1. **开发记录(必须)**:每次开发会话结束前,在 `docs/devlog/` 新增一篇**中文** markdown
   (命名 `NNN-YYYY-MM-DD-标题.md`,NNN 递增),内容至少包含:目标、决策及理由、
   改动结构、验证结果(真实的测试/命令输出)、下一步。并更新 `docs/devlog/README.md` 索引。
2. **Git**:永不 `git add .` / `git commit -a`,显式列出文件;Conventional Commits
   (feat/fix/docs/refactor/test);未经用户明确要求不 commit、不 push。
3. **代码风格**:所有函数完整类型注解;`uv run ruff format` 过、`uv run ruff check` 零告警、
   `uv run mypy src`(strict)零错误;KISS,函数保持短小。
4. **测试**:`uv run pytest` 全绿才能结束会话;优先集成测试而非重度 mock
   (gptme 的教训:mock provider + 真实文件系统行为)。测试代码同样要过 ruff。
5. **忠实报告**:测试失败就展示失败输出;跳过的步骤要说明;禁止粉饰验证结果。

## 架构纪律(防止走样)

- 循环的继续判定只看"最后一条 assistant 是否还有可运行 ToolUse",不引入额外状态。
- 主日志 `conversation.jsonl` 永不销毁;将来压缩写视图副本,不改写主文件。
- 新的横切关注点(确认/压缩/通知……)一律做成 hook,不改核心循环。
- 消息不可变:修改 = 构造新实例。
- call_id 纪律从第一天保持:每个结构化工具调用恰好配一个结果。

## 常用任务

- **跑全部检查**:`uv run pytest && uv run ruff check . && uv run mypy src`
- **加一个模块**:在 `src/my_agent_alpha/` 下建文件 + 对应 `tests/test_*.py`;导出走包 `__init__`
- **对照 gptme 原实现**:读 `/tmp/gptme/gptme/` 下的对应模块,研究报告有 file:line 索引
