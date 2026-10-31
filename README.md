# my-agent-alpha

一个跑在终端里的个人 AI agent:对话、执行 shell/Python、读写文件、多步自主体完成任务。

三条核心设计原则,贯穿全部模块:

1. **一切皆消息** —— 工具调用内嵌在 assistant 文本里,工具输出是带 `call_id` 的 system 消息,append-only JSONL 日志是唯一状态;
2. **LLM 层统一规范格式** —— 所有 provider 归一为 `<think>` + `@tool(call_id): {json}` 纯文本;
3. **一切横切关注点皆 Hook** —— 确认/压缩/转向/取消都是注入消息的 generator hook。

## 功能一览

- **对话循环**:三层结构(输入 → turn → step),流式生成,`Ctrl-C` 中断保留已生成内容;
- **双格式工具调用**:markdown 围栏代码块 与 原生 function calling 共用一套执行端,`call_id` 严格配对;
- **工具箱**:shell(只读命令放行、灾难命令硬拦)、Python、read / save / edit / grep、todo(完成门控)、subagent(白名单)、memory;
- **确认链**:y / n / e / a 终端询问,`-y` 自主模式(安全护栏仍然生效);
- **上下文压缩**:token 预算、三阶段规则压缩、溢出自愈、自动压缩,主日志永不销毁;
- **通用 hook 系统** + slash 命令(`/log`、`/resume`、`/memory` 等)+ 分层 TOML 配置(命令行 > 环境变量 > 项目 > 用户);
- **多 provider**:OpenAI 兼容接口、智谱 GLM、mock 测试通道;
- **MCP 客户端**:接入外部 MCP server,工具提示词自动生成调用说明;
- **长期记忆**:frontmatter 格式、双上限、新鲜度管理、`/memory consolidate` 整理;
- **多会话管理**:workspace 分组 + uuid 会话目录、会话自动命名、历史列表与切换恢复。

## 快速开始

```bash
uv sync                          # 安装依赖(含开发工具)
uv run pytest                    # 跑测试
uv run ruff check .              # lint
uv run mypy src                  # 类型检查
uv run my-agent-alpha            # 进入交互对话
uv run my-agent-alpha -m deepseek/deepseek-chat -y "帮我看看这个项目"
```

常用参数:`-m` 指定模型、`-n` 跑完队列提示词即退出、`--tool-format` 选工具调用通道、`--tools` 工具白名单、`-y` 自动确认。

## 文档

- [docs/devlog/](docs/devlog/) —— 开发记录(中文,每次会话一篇:目标/决策/验证/下一步)
- [AGENTS.md](AGENTS.md) —— AI 开发代理协作规范

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 骨架:Message + JSONL 日志(LogManager、追加游标、原子重写) | ✅ |
| 1 | 脊柱:step() + markdown 工具解析 + 三层循环 CLI | ✅ |
| 2 | 流式生成 + 流中工具检测(break_on_tooluse)+ 重试策略 | ✅ |
| 3 | 原生工具调用(call_id 配对、JSON Schema、双格式共用执行端) | ✅ |
| 4 | 确认链(y/n/e/a 终端询问、只读放行、灾难硬拦、-y 自主模式) | ✅ |
| 5 | 通用 hook 系统 / slash 命令框架 / 分层 TOML 配置 | ✅ |
| 6 | 上下文压缩(token 预算、三阶段规则压缩、溢出自愈) | ✅ |
| 8 | subagent / MCP / 长期记忆 | ✅ |
| 7 | HTTP Server(SSE)+ WebUI | ⏸ 暂缓 |
| — | **长任务执行优化**:提升连续自主执行长任务的稳定性与效率 | ▶ 下一步 |
