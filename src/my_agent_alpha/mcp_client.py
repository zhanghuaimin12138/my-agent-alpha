"""MCP 客户端:把外部 MCP 服务器的工具接进来,变成我们的 ToolSpec。

背景(见学习文档/研究报告):MCP(Model Context Protocol)是"工具界的
USB 口"—— 服务器用统一协议暴露工具(echo、查天气、操作 Notion…),
客户端连上就能用,不用为每个工具写集成代码。gptme 的做法(我们照搬):
**每个远端工具转成一个本地一等 ToolSpec**,命名 `服务器名.工具名`,
参数从服务器给的 JSON Schema 换算 —— 于是允许确认链、系统提示词、
两种调用通道全部自动适用于远端工具,零特殊处理。

同步调用异步 SDK 的坑(也是 gptme 踩过的):MCP 的 Python SDK 是
asyncio 的,而我们的 agent 循环是同步的;而且 stdio 传输的"取消作用域"
要求进出在同一个任务里。解法:每个客户端起一个专属事件循环线程,
**整个会话的连接/列举/调用/关闭都发生在同一个常驻协程任务里**,
同步侧通过队列投递操作 —— 两边各一条线程,互不越界。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from my_agent_alpha.config import Config, McpServer
from my_agent_alpha.message import Message
from my_agent_alpha.tools import add_tools
from my_agent_alpha.tools.base import Parameter, ToolSpec

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 60.0  # 首次 npx 要下载包,给足时间
# 单次工具调用的超时不再用固定常量:从 McpServer.timeout 读(默认 120 秒,
# 可在 alpha.toml 里按服务器调)—— 搜索/渲染类工具天然偏慢,60 秒一刀切太紧


class SyncMCPClient:
    """一个 MCP 服务器的同步客户端(专属事件循环线程 + 常驻会话协程)。

    创建一个SyncMCPClient并传入配置单,背后会执行：
    1.init函数:
        1.1 开后台异步线程（一个有event_loop的线程）
        1.2 送入常驻协程函数执行
            （负责监听队列，并异步执行响应的操作，
            比如await list_tools()/await call_tool()
            并将结果写入fut回执单）

    主线程处理同步和异步关键函数：_submit()
        此函数将操作送入异步队列，给后台线程监听并执行
    """

    def __init__(self, server: McpServer) -> None:
        """连上一个 MCP 服务器,连不上就报错(构造完成 = 连接可用)。

        示例:
            client = SyncMCPClient(McpServer(name="t", command="python",
                                             args=["server.py"]))
            client.list_tools()   # 构造完就能直接用
        """
        self.server = server  # 要连的服务器配置(名字、启动命令、参数、环境变量)
        # 单次工具调用的超时秒数(来自配置;超时会在"执行侧"取消调用,见下)
        self.call_timeout: float = float(server.timeout)

        # ---- 后台线程:一个专门跑 asyncio 代码的小天地 ----
        # MCP 库的函数都是"会暂停等结果"的风格(async),必须有 asyncio 环境才能跑;
        # 我们的 agent 主线程是普通同步代码,没有这个环境 → 自己开一个。
        # 新建的 asyncio 环境(实际上是创建一个asyncio调度员对象)
        self._loop = asyncio.new_event_loop()
        # 创建后台线程,线程执行函数是 self._loop.run_forever(),
        # 所以这边把 self._loop.run_forever 传进去
        # run_forever 的意思:这个线程不执行你写的业务代码，专职运转 asyncio 调度,直到被叫停。
        # daemon=True:主程序退出时它自动跟着退,不会卡住进程
        # 所以本质就是后台创建一个线程执行asyncio调度员的run_forever函数
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        # 后台线程启动空转（此时还没有任务），并回到主进程继续执行（后台线程也在执行）
        self._thread.start()

        # ---- 同步世界和后台线程之间的两个交接点 ----
        # 主进程submit,后台进程监听并执行await函数
        self._ops: asyncio.Queue[Any] | None = None
        # ↑ 收件箱:主线程往里放工作单("列工具"/"调工具"/"关"),
        #   后台的会话函数从里面取。注意:在后台线程"连接成功后"才创建
        #   (由 _session_task 里赋值),在那之前是 None。
        self._ready = threading.Event()
        # ↑ 就绪门铃:后台连上服务器、收件箱建好后按响;主线程在这等它,
        #   等到了才说明"构造完成 = 可以用了"。
        self._dead: Exception | None = None
        # ↑ 失败遗言:后台连接过程出错时把异常存这,主线程取出来包成 RuntimeError 抛。

        # 把"常驻会话函数"提交到后台线程去跑 —— 连接、之后所有的工具调用、
        # 直到关闭,全部发生在这一个函数里(SDK 要求开和关必须是同一个任务)
        # 本质就是把Coroutine跨线程送到self._loop（调度员）里，变成task执行，跑完返回future
        self._runner = asyncio.run_coroutine_threadsafe(self._session_task(), self._loop)

        # ---- 等连接结果(最多 _CONNECT_TIMEOUT 秒) ----
        if not self._ready.wait(timeout=_CONNECT_TIMEOUT):
            self.close()
            raise RuntimeError(f"MCP 服务器 {server.name!r} 连接超时")
        if self._dead is not None:  # 门铃响了但是失败的铃声
            self.close()
            raise RuntimeError(f"MCP 服务器 {server.name!r} 连接失败:{self._dead}") from self._dead

    async def _session_task(self) -> None:
        """常驻协程:持有会话。

        后台线程异步“监听”常驻函数，负责执行同步侧投递来的操作
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # 初始化客户端和队列
        # 注意:SDK 里 env 一旦给出就会"替换"整个环境变量(连 PATH 都没了),
        # 所以必须把配置里的少量变量叠加到当前环境之上,而不是只传它们。
        # (gptme 也是这么做的:配置 env 合并到 os.environ 上)
        params = StdioServerParameters(
            command=self.server.command,
            args=list(self.server.args),
            env={**os.environ, **(self.server.env or {})},
        )
        try:
            async with (
                stdio_client(params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                self._ops = asyncio.Queue()  # 初始化队列
                self._ready.set()  # 按铃，放开所有在await self._ready的线程
                while True:  # 循环取队列内容
                    # 从队列里取出一条(取不到就等，会放开当前cpu,继续执行调度员空转函数，
                    # 如果不放开，这个线程就永远阻塞在这了，因为调度员不会再得到cpu，执行调度)
                    kind, payload, fut, timeout = await self._ops.get()
                    # 类型，参数，空白回执（接受结果用的），这次操作的超时秒数

                    try:
                        if kind == "stop":
                            fut.set_result(None)
                            return  # 结束mcp循环监听
                        if kind == "list":  # list操作，执行mcp的list_tools函数，结果写入fut
                            # 超时在"执行侧"生效:到点取消这次调用,而不是主线程干等
                            # (否则一次挂起的调用会永远占着本协程,堵死同服务器后续所有调用)
                            fut.set_result(await asyncio.wait_for(session.list_tools(), timeout))
                        elif kind == "call":  # call操作，执行mcp的call_tool函数，结果写入fut
                            fut.set_result(
                                await asyncio.wait_for(session.call_tool(**payload), timeout)
                            )
                    except Exception as e:
                        fut.set_exception(e)
        except Exception as e:
            self._dead = e
            self._ready.set()  # 连接失败也得放开await self._ready的线程

    ### 下述代码供主线程调用

    def _submit(self, kind: str, payload: dict[str, Any] | None, timeout: float) -> Any:
        """往会话协程投递一个操作并等结果。"""
        if self._ops is None:
            raise RuntimeError(f"MCP 服务器 {self.server.name!r} 未连接")
        fut: concurrent.futures.Future[Any] = concurrent.futures.Future()  # 造空白回执
        put = asyncio.run_coroutine_threadsafe(
            self._ops.put((kind, payload, fut, timeout)), self._loop
        )  # 把工作单塞进后台的收件箱
        put.result(timeout=5)  # (确认塞进去了)
        # 真正的超时由协程里的 asyncio.wait_for 负责(到点取消调用);
        # 主线程这里只是兜底:万一协程自己卡住收不了尾,也不至于永远挂在这里。
        # 余量 10 秒 = 取消收尾的时间
        return fut.result(timeout=timeout + 10.0)  # 同步世界停在这里,等后台把结果写进 fut

    def _op_readable_timeout(self, kind: str, payload: dict[str, Any] | None, what: str) -> Any:
        """投递操作,并把超时异常翻译成人能看懂的话。

        为什么必须有这一步:TimeoutError 的 str() 是**空字符串**,
        直接拼进"执行失败: {e}"就成了空白消息 —— 模型和用户都不知道发生了什么
        (2026-10-04 真实会话踩坑,见 docs/踩坑记录.md 条目 2)。
        """
        try:
            return self._submit(kind, payload, self.call_timeout)
        except TimeoutError as e:  # asyncio.wait_for 超时和兜底超时都走这里
            raise RuntimeError(
                f"MCP 服务器 {self.server.name!r} {what}超时"
                f"({self.call_timeout:.0f} 秒无响应;可能网络不通或服务器卡住,"
                f"可在 alpha.toml 里调大该服务器的 timeout)"
            ) from e

    def list_tools(self) -> list[Any]:
        """列出服务器的全部工具(MCP SDK 的 Tool 对象)。"""
        result = self._op_readable_timeout("list", None, "列出工具")
        return list(getattr(result, "tools", []) or [])

    def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用一个工具,把返回的内容块拼成文本。"""
        result = self._op_readable_timeout(
            "call", {"name": tool_name, "arguments": arguments}, f"调用工具 {tool_name!r} "
        )
        parts = [getattr(block, "text", "") for block in getattr(result, "content", []) or []]
        text = "\n".join(p for p in parts if p)
        if getattr(result, "isError", False):
            text = f"(工具报错) {text}"
        return text or "(无输出)"

    def close(self) -> None:
        """停止会话协程和事件循环线程(幂等)。"""
        if self._ops is not None:
            with contextlib.suppress(Exception):  # 关闭时不再计较
                self._submit("stop", None, 5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


# getter API
def create_mcp_tools(
    servers: tuple[McpServer, ...],
) -> tuple[tuple[ToolSpec, ...], list[SyncMCPClient]]:
    """连上每个配置的服务器,把它的工具全部转成本地 ToolSpec。

    一个服务器连不上只告警跳过,不拖垮会话(gptme 的容错姿态)。
    返回 (工具列表, 客户端列表) —— 客户端要留着,会话结束时 close。
    """
    specs: list[ToolSpec] = []
    clients: list[SyncMCPClient] = []
    for server in servers:
        try:
            client = SyncMCPClient(server)
            remote_tools = client.list_tools()
        except Exception as e:
            logger.warning("MCP 服务器 %s 不可用,已跳过:%s", server.name, e)
            continue
        clients.append(client)
        for rt in remote_tools:
            specs.append(_make_spec(client, server, rt))
    return tuple(specs), clients


def _make_spec(client: SyncMCPClient, server: McpServer, remote_tool: object) -> ToolSpec:
    """把一个远端 MCP 工具包成本地 ToolSpec(名字:服务器.工具)。"""
    name = f"{server.name}.{getattr(remote_tool, 'name', 'unknown')}"
    desc = str(getattr(remote_tool, "description", "") or "")
    # SDK 2.x 叫 input_schema,1.x 叫 inputSchema —— 两个名字都试
    schema = (
        getattr(remote_tool, "input_schema", None)
        or getattr(remote_tool, "inputSchema", None)
        or {}
    )
    params = _schema_to_parameters(schema if isinstance(schema, dict) else {})
    annotations = getattr(remote_tool, "annotations", None)
    read_only = bool(getattr(annotations, "readOnlyHint", False)) if annotations else False

    def execute(
        code: str | None,
        args: list[str] | None,
        kwargs: dict[str, str] | None,
        _client: SyncMCPClient = client,
        _tool_name: str = getattr(remote_tool, "name", "unknown"),
    ) -> Iterator[Message]:
        """调用远端工具,结果文本作为工具输出。

        参数来源:原生通道给 kwargs;markdown 通道的围栏内容(code)
        按 gptme 的约定是一个 JSON 参数体,如 {"text": "hi"}。
        """
        arguments = dict(kwargs) if kwargs else {}  # 原生通道参数在kwargs
        if not arguments and code:  # markdown参数在code
            try:
                parsed = json.loads(code)
            except json.JSONDecodeError:
                yield Message(
                    "system",
                    f"错误: markdown 方式调用 {name} 时,围栏内容必须是 JSON 参数体,"
                    f'例如 {{"text": "hi"}},收到的是:{code[:100]!r}',
                )
                return
            if isinstance(parsed, dict):
                arguments = parsed
        yield Message("system", _client.call(_tool_name, arguments))

    return ToolSpec(
        name=name,
        desc=f"[{server.name}] {desc}".strip(),
        # 说明和示例按参数表自动生成:markdown 通道下模型看不到参数表
        instructions=_markdown_instructions(params),
        examples=_markdown_example(name, params),
        # block_types 决定"markdown 围栏怎么调它":模型写围栏代码块时,
        # 语言标记(工具全名)按它匹配到本工具。
        block_types=(name,),
        parameters=params,  # 原生使用
        read_only=read_only,
        execute=execute,
    )


def _markdown_instructions(params: tuple[Parameter, ...]) -> str:
    """生成 markdown 通道的调用说明:围栏内容必须是 JSON 参数体 + 参数名清单。

    示例(参数表有一个必填的 query 时):
        markdown 通道调用:围栏内容写 JSON 参数体(键名必须用下面的参数名):
        - query(string,必填): Search query
    """
    if not params:
        return "markdown 通道调用:本工具没有参数,围栏内容写空 JSON 即可:`{}`"
    lines = ["markdown 通道调用:围栏内容写 JSON 参数体(键名必须用下面的参数名):"]
    for p in params:
        mark = "必填" if p.required else "可选"
        hint = f":{p.description}" if p.description else ""
        lines.append(f"- {p.name}({p.type},{mark}){hint}")
    return "\n".join(lines)


def _markdown_example(name: str, params: tuple[Parameter, ...]) -> str:
    """按参数表生成一个"可直接照抄"的调用示例围栏(必填参数进示例)。

    示例:工具 websearch.search 有必填参数 query →
        ```websearch.search
        {"query": "<query>"}
        ```
    """
    body = {p.name: _placeholder_for(p) for p in params if p.required}
    return f"```{name}\n{json.dumps(body, ensure_ascii=False)}\n```"


def _placeholder_for(param: Parameter) -> object:
    """给示例参数挑一个占位值:按类型来,让模型一眼看懂该填什么风格。"""
    if param.type in ("integer", "number"):
        return 1
    if param.type == "boolean":
        return True
    return f"<{param.name}>"  # 字符串:拿参数名当占位提示,如 "<query>"


# 辅助函数
def _schema_to_parameters(schema: dict[str, Any]) -> tuple[Parameter, ...]:
    """把 MCP 工具的 JSON Schema 参数表换算成我们的 Parameter 列表。

    input_schema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["city"]
    }
    """
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    params = []
    for name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        params.append(
            Parameter(
                name=str(name),
                type=str(spec.get("type", "string")),
                description=str(spec.get("description", "")),
                required=name in required,
            )
        )
    return tuple(params)


""" session.list_tools()返回架构

ListToolsResult(
    tools=[
        Tool(
            name="get_weather",
            title="Weather",
            description="Get current weather for a city",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["city"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "temperature": {"type": "number"},
                    "condition": {"type": "string"}
                },
                "required": ["temperature", "condition"]
            },
            annotations=...,
            icons=...,
            _meta=...
        ),

        Tool(
            name="search_web",
            ...
        )
    ],
    ...
)
"""

"""


"""


def connect_mcp_tools(workspace: Path) -> list[SyncMCPClient]:
    """按工作区配置连 MCP 服务器:远端工具注册进工具表,返回要保管的客户端。

    什么都没配就返回空列表(绝大多数会话都走这条路径)。
    客户端必须留着并由调用方在会话结束时 close —— 这里只负责连接和注册。
    """
    config = Config.load(workspace)
    if not config.mcp_servers:
        return []
    specs, clients = create_mcp_tools(config.mcp_servers)
    if specs:
        add_tools(specs)  # 内置表基础上追加(名字带"服务器."前缀,不会撞)
    return clients
