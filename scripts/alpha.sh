#!/usr/bin/env bash
# 在任意目录启动 my-agent-alpha:用本仓库的虚拟环境跑,但工作目录保持在当前所在目录。
# 不带任何参数运行 = 启动程序进入交互对话(程序一直活着,等你一句一句输入)。
# 后面带的参数原样透传给程序,例如:
#   scripts/alpha.sh                     # 启动,进入交互对话
#   scripts/alpha.sh -m glm/glm-5.3      # 启动时临时换模型,同样进入交互对话
#   scripts/alpha.sh --name debug-xx     # 启动时给会话起名,方便以后 /resume 找回
#   scripts/alpha.sh "看看这个目录"        # 可选:带开场提示词 = 跑完这一句就退出(一次性模式)
set -euo pipefail

# 脚本自己住在 <仓库>/scripts/ 下,往上一级就是仓库根目录;
# 这样算出来的路径不写死,仓库整个搬家后脚本依然有效
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec uv run --project "$PROJECT_DIR" my-agent-alpha "$@"
