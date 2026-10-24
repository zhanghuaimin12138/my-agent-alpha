"""围栏代码块解析测试。"""

from my_agent_alpha.codeblock import iter_from_markdown, strip_think


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


# ---- 思考区(<think>)剥离:gptme 同款,2026-10-04 kimi 粘连格式的回归 ----


def test_think_block_stripped_before_parsing() -> None:
    """思考区里写的围栏是草稿,不是工具调用,不该被解析。"""
    text = "<think>要不要看看目录?```shell\nrm -rf /\n```</think>我换个思路"
    assert list(iter_from_markdown(text)) == []


def test_fence_glued_to_think_close_tag() -> None:
    """kimi 实际输出:</think> 和开栏挤在同一行 —— 剥掉 think 后围栏落到行首。"""
    text = "<think>需要搜索天气。</think>```websearch.search\n今天 上海市宝山区 天气\n```"
    blocks = list(iter_from_markdown(text))
    assert len(blocks) == 1
    assert blocks[0].lang == "websearch.search"
    assert blocks[0].content == "今天 上海市宝山区 天气"


def test_unclosed_think_parses_nothing() -> None:
    """思考区没闭合 = 还在生成中,整体不解析(防止执行半截内容)。"""
    assert list(iter_from_markdown("<think>想想 ```shell\necho 半截\n```")) == []


def test_thinking_tag_variant() -> None:
    """Gemini 用 <thinking>...</thinking>,同样要剥。"""
    text = "<thinking>草稿</thinking>\n```shell\nls\n```"
    blocks = list(iter_from_markdown(text))
    assert len(blocks) == 1
    assert blocks[0].content == "ls"


# ---- strip_think 专项:思考区剥离作为纯字符串函数的行为 ----
# (上面经 iter_from_markdown 的用例测的是"剥完之后围栏解析对不对",
#  这里直接测函数本身的输出:那些不会进围栏解析器的形态)


def test_strip_basic_and_empty() -> None:
    """完整思考区剥掉、正文一字不动;空思考区也一样。"""
    assert strip_think("<think>推理…</think>正文回答") == "正文回答"
    assert strip_think("<think></think>正文") == "正文"
    # 剥掉的是"标签 + 思考内容",前后的换行不动(正文在下一行就还在下一行)
    assert strip_think("<think>x</think>\n正文") == "\n正文"


def test_strip_no_think_unchanged() -> None:
    """没有思考区:原样返回(包括正文里就有代码块的常见形态)。"""
    assert strip_think("普通回复") == "普通回复"
    assert strip_think("```shell\nls\n```") == "```shell\nls\n```"


def test_strip_glued_fence_lands_at_line_start() -> None:
    """kimi 粘连格式(踩坑记录条目 1):围栏紧跟 </think> 写在同一行,
    剥完正好落到字符串开头(行首),后面的围栏解析器才看得见。"""
    assert strip_think("<think>先看看目录。</think>```shell\nls\n```") == "```shell\nls\n```"


def test_strip_unclosed_returns_empty() -> None:
    """没闭合的思考区 = 还在生成中,后面什么都可能变 → 整体返回空。

    行首缩进的开标签同样算"思考区开始了"(和围栏的缩进容忍同一个道理)。
    """
    assert strip_think("<think>只写了一半") == ""
    assert strip_think("正文\n   <think>缩进的开标签") == ""


def test_strip_thinking_variant_and_mixed_tags() -> None:
    """<thinking>(Gemini 写法)同样剥;开闭标签名混搭也认(正则允许)。"""
    assert strip_think("<thinking>草稿</thinking>正文") == "正文"
    assert strip_think("<think>短开长闭</thinking>正文") == "正文"
    assert strip_think("<thinking>长开短闭</think>正文") == "正文"


def test_strip_multiple_and_multiline_regions() -> None:
    """多个思考区都剥;思考内容跨多行(哪怕里面写了围栏)整段拿掉。"""
    text = (
        "<think>第一段草稿\n```shell\n草稿里的命令\n```\n</think>中间正文<think>第二段</think>结尾"
    )
    assert strip_think(text) == "中间正文结尾"


def test_strip_midline_tag_not_unclosed() -> None:
    """行中间的 <think>(粘在别的文字后面)不算思考区开始(_THINK_OPEN_RE 的设计):
    没闭合也不触发"整体不解析",原样保留;但完整成对的行中思考区照样剥。"""
    assert (
        strip_think("正文里提到 <think> 这个词,后面没有闭合")
        == "正文里提到 <think> 这个词,后面没有闭合"
    )
    assert strip_think("前面<think>行中思考</think>后面") == "前面后面"


def test_strip_stray_close_tag_kept() -> None:
    """只有闭合标签没有开标签:不当思考区,原样保留。"""
    assert strip_think("只有闭合</think>正文") == "只有闭合</think>正文"


# ---- 流式视角(生成中途解析:围栏要等空行确认才算完整)----


def test_streaming_fence_requires_blank_line() -> None:
    """流式规则:闭合围栏后必须有空行,块才算完整。"""
    complete = "```shell\nls\n```\n\n后续"
    assert len(list(iter_from_markdown(complete, streaming=True))) == 1

    no_blank = "```shell\nls\n```"  # 闭合栏后没空行(流式视角:可能还没写完)
    assert list(iter_from_markdown(no_blank, streaming=True)) == []
    # 非流式(最终解析)依然能拿到 —— 两套规则各司其职
    assert len(list(iter_from_markdown(no_blank))) == 1


def test_streaming_unclosed_think_no_early_stop() -> None:
    """思考区里出现的围栏不算完整调用:think 没闭合前不能触发流式早停。"""
    generating = "<think>举例 ```shell\nls\n```\n再想想"
    assert list(iter_from_markdown(generating, streaming=True)) == []
