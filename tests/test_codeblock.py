"""围栏代码块解析测试。"""

from my_agent_alpha.codeblock import iter_from_markdown


def test_simple_block() -> None:
    blocks = list(iter_from_markdown("前言\n```shell\nls -la\n```\n后记"))
    assert len(blocks) == 1
    assert blocks[0].lang == "shell"
    assert blocks[0].args == ""
    assert blocks[0].content == "ls -la"


def test_lang_with_args() -> None:
    blocks = list(iter_from_markdown("```save path/to/file.txt\n内容\n```"))
    assert blocks[0].lang == "save"
    assert blocks[0].args == "path/to/file.txt"
    assert blocks[0].content == "内容"


def test_multiple_blocks_in_order() -> None:
    text = "先看目录:\n```shell\nls\n```\n再看文件:\n```read\na.py\n```"
    blocks = list(iter_from_markdown(text))
    assert [b.lang for b in blocks] == ["shell", "read"]
    assert blocks[0].start < blocks[1].start


def test_unterminated_block_dropped() -> None:
    """未闭合的块被丢弃 —— 绝不执行半截命令(阶段 2 流式安全的根基)。"""
    text = "好的\n```shell\necho hi\n"
    assert list(iter_from_markdown(text)) == []


def test_unterminated_does_not_swallow_earlier_block() -> None:
    text = "```shell\nls\n```\n说明:\n```shell\necho 半截"
    blocks = list(iter_from_markdown(text))
    assert len(blocks) == 1
    assert blocks[0].content == "ls"


def test_variable_length_fences() -> None:
    """四反引号开栏,三反引号行不算闭合(内容里嵌套 ``` 不误伤)。"""
    text = "````markdown\n```shell\nls\n```\n````"
    blocks = list(iter_from_markdown(text))
    assert len(blocks) == 1
    assert "```" in blocks[0].content


def test_indented_fence_tolerated() -> None:
    text = "  ```shell\n  ls\n  ```"
    blocks = list(iter_from_markdown(text))
    assert blocks[0].lang == "shell"
    assert blocks[0].content == "  ls"  # 内容的缩进保留


def test_inline_code_not_a_block() -> None:
    assert list(iter_from_markdown("行内 `code` 不是围栏块")) == []


def test_multiline_content_preserved() -> None:
    text = "```shell\ncd /tmp\nls\npwd\n```"
    blocks = list(iter_from_markdown(text))
    assert blocks[0].content == "cd /tmp\nls\npwd"
