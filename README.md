# my-agent-alpha

个人 AI agent —— 仿造 [gptme](https://github.com/gptme/gptme) 的学习型重实现。

运行在任何有终端的地方:终端对话、执行 shell/Python、读写文件、多步自主体完成任务。
核心设计沿用 gptme 的三大支柱:

1. **一切皆消息** —— 工具调用内嵌在 assistant 文本里,工具输出是带 `call_id` 的 system 消息,append-only JSONL 日志是唯一状态;
2. **LLM 层统一规范格式** —— 所有 provider 归一为 `<think>` + `@tool(call_id): {json}` 纯文本;
3. **一切横切关注点皆 Hook** —— 确认/压缩/转向/取消都是注入消息的 generator hook。

## 快速开始

```bash
uv sync                # 安装依赖(含开发工具)
uv run pytest          # 跑测试
uv run ruff check .    # lint
uv run mypy src        # 类型检查
uv run my-agent-alpha  # 控制台入口(阶段 0 占位)
```

## 文档

- [docs/gptme-study.md](docs/gptme-study.md) —— gptme 架构研究报告(仿造蓝图)
- [docs/devlog/](docs/devlog/) —— 开发记录(中文,每阶段一篇)
- [AGENTS.md](AGENTS.md) —— AI 开发代理协作规范

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 骨架:Message + JSONL 日志(LogManager、追加游标、原子重写) | ✅ |
| 1 | 脊柱:step() + markdown 工具解析 + 三层循环 CLI | ✅ |
| 2 | 流式生成 + 流中工具检测(break_on_tooluse)+ 重试策略 | ✅ |
| 3 | 原生工具调用(call_id 配对、JSON Schema、双格式共用执行端) | ✅ |
| 4 | 确认链与 shell 安全梯 | |
| 5 | Hook 体系 / slash 命令 / 分层配置 | |
| 6 | 上下文压缩 | |
| 7 | HTTP Server(SSE)+ WebUI | |
| 8 | subagent / MCP / 长期记忆 | |
