"""subagent 工具:派一个子代理去干独立任务,拿回它的结论。

设计(仿 gptme 的 subagent,采用其 MVP 路线 —— 子进程模式):

- 隔离:子代理是独立进程(python -m my_agent_alpha -n -y),
  有自己的会话日志(会话目录/subagent-<编号>-<随机>),跑完即退,
  崩了也不影响父会话;
- 自主:子代理带 -n(非交互)和 -y(自动确认)—— 它是干活的,
  不该中途等人;父代理这边照样过确认链(用户决定"要不要派它");
- 通信:提示词从 stdin 传给子代理(防超长命令行),结果从子代理的
  日志文件里取"最后一条模型回复"(这就是把日志当唯一事实来源的红利);
- 模型:默认继承父会话的模型(可用 model 参数覆盖)。

和 gptme 的差异(记录在案):gptme 是 fire-and-forget + 钩子通知,
支持并行派多个;我们是同步等待(一次派一个,等结果)—— 简单正确,
异步版等真有"并行干活"的需求再演进。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator

from my_agent_alpha.context import get_current_model, get_workspace
from my_agent_alpha.dirs import get_logs_dir
from my_agent_alpha.logmanager import generate_conversation_id
from my_agent_alpha.message import Message
from my_agent_alpha.tools.base import Parameter, ToolSpec

_TIMEOUT = int(os.environ.get("ALPHA_SUBAGENT_TIMEOUT", "300"))


def execute_subagent(
    code: str | None, args: list[str] | None, kwargs: dict[str, str] | None
) -> Iterator[Message]:
    """派一个子代理执行任务,等它跑完,取最后一条模型回复作为结果。"""
    prompt = (kwargs.get("prompt") if kwargs else None) or code or ""
    model = (kwargs.get("model") if kwargs else None) or get_current_model() or "mock/echo"
    if not prompt.strip():
        yield Message("system", "错误: subagent 需要 prompt(任务描述)")
        return

    name = f"subagent-{generate_conversation_id()}"
    workspace = get_workspace()
    cmd = [
        sys.executable,
        "-m",
        "my_agent_alpha",
        "-n",
        "-y",  # 非交互 + 自动确认:子代理自主干活
        "--name",
        name,
        "--workspace",
        str(workspace),
        "-m",
        model,
        "-",  # 提示词从 stdin 读
    ]
    yield Message("system", f"▶ 已派出子代理 {name}(模型 {model}),等待完成…")

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            cwd=workspace,
            env={**os.environ, "PYTHONPATH": _pythonpath()},  # 开发目录直跑也要能 import
        )
    except subprocess.TimeoutExpired:
        yield Message("system", f"✗ 子代理 {name} 超时(>{_TIMEOUT}s),已放弃")
        return
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().split("\n")[-3:]
        yield Message(
            "system", f"✗ 子代理 {name} 失败(退出码 {proc.returncode}):\n" + "\n".join(err)
        )
        return

    result = _last_assistant_text(get_logs_dir() / name / "conversation.jsonl")
    if result is None:
        yield Message("system", f"✗ 子代理 {name} 完成,但日志里没有回复")
        return
    yield Message("system", f"✅ 子代理 {name} 的结论:\n\n{result}")


def _pythonpath() -> str:
    """让子进程能 import 本包:优先已安装位置,开发目录时补 src 路径。"""
    package_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )  # .../src/my_agent_alpha
    src_root = os.path.dirname(package_dir)  # .../src
    existing = os.environ.get("PYTHONPATH", "")
    return f"{src_root}{os.pathsep}{existing}" if existing else src_root


def _last_assistant_text(conv_file: object) -> str | None:
    """从子代理的日志里取最后一条 assistant 消息的正文。"""
    from pathlib import Path

    path = Path(conv_file)  # type: ignore[arg-type]
    if not path.exists():
        return None
    last: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("role") == "assistant" and data.get("content", "").strip():
            last = data["content"]
    return last


tool = ToolSpec(
    name="subagent",
    desc=(
        "派一个子代理独立完成一项任务并返回它的结论。"
        "适合可并行的独立调查/信息收集;子代理有自己的会话,看不到当前对话。"
    ),
    instructions=("任务描述要自包含(子代理不知道我们聊过什么),并写清期望它返回什么。"),
    examples="```subagent\n调查 /tmp 目录下最大的三个文件,返回名字和大小\n```",
    block_types=("subagent",),
    parameters=(
        Parameter("prompt", "string", "给子代理的完整任务描述(自包含)"),
        Parameter("model", "string", "覆盖子代理用的模型(默认继承当前会话)", required=False),
    ),
    execute=execute_subagent,
)
