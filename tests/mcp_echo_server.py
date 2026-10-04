"""测试用最小 MCP 服务器:echo 和 add 两个工具(供 test_phase8 连接)。

用的是 mcp SDK 2.x 的新 API:MCPServer(v1 叫 FastMCP,已改名)。
"""

from mcp.server.mcpserver import MCPServer

server = MCPServer("test-echo")


@server.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    server.run()  # 默认 stdio 传输
