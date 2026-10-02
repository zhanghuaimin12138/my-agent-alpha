"""LLM 层的错误类型。

单独放一个文件(而不是 llm/__init__.py)是为了避免循环导入:
openai_compat 需要导入它,而 llm/__init__ 又导入 openai_compat。
"""


class ContextLengthError(RuntimeError):
    """历史总长超过模型上下文窗口,接口拒绝了这个请求。

    和普通错误分开:调用方(step)捕获它可以"压缩历史后重试",
    而不是直接报错结束回合。
    """
