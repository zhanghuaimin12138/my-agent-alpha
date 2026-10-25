"""memory_store.py 测试:两层存储、索引渲染、frontmatter、双上限、scope 推断。

工具通道(execute_msg 调 memory 工具)的测试在 tests/unit/tools/test_memory.py;
注入会话与整理代理的端到端在 tests/integration/test_memory.py。
"""

from __future__ import annotations

from pathlib import Path

from my_agent_alpha.memory_store import (
    list_memories,
    memory_dirs,
    render_memory_index,
    save_memory,
)

# ---- 基础:保存/列出/索引 ----


def test_save_and_list(clean_dirs: Path) -> None:
    ws = clean_dirs / "ws"
    ws.mkdir()
    save_memory("部署流程", "服务器发布步骤\n1. ssh …", ws, project_level=True)
    entries = list_memories(ws)
    assert [e.name for e in entries] == ["部署流程"]
    assert entries[0].title == "服务器发布步骤"
    # 索引文件生成了,且有指针
    index = (ws / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "[部署流程](部署流程.md)" in index


def test_two_layers_shadowing(clean_dirs: Path) -> None:
    """同名记忆:项目层盖住用户层(gptme 的 nearest-first)。"""
    ws = clean_dirs / "ws"
    ws.mkdir()
    save_memory("偏好", "用户层的偏好", ws, project_level=False)
    save_memory("偏好", "项目层的偏好", ws, project_level=True)
    entries = list_memories(ws)
    assert len(entries) == 1  # 同名只剩一个
    assert "项目层" in entries[0].path.read_text(encoding="utf-8")
    # 两层的目录都真实存在文件,只是索引取近层
    assert len(memory_dirs(ws)) == 2


def test_render_index_for_prompt(clean_dirs: Path) -> None:
    ws = clean_dirs / "ws"
    ws.mkdir()
    save_memory("部署流程", "服务器发布步骤\n1. ssh …", ws, project_level=True)
    text = render_memory_index(ws)
    assert "长期记忆" in text and "部署流程" in text
    # 链接必须带路径,模型才能直接 read(项目层 → memory/xxx.md)
    assert "](memory/部署流程.md)" in text
    # 空目录 → 空文本(会话就不注入)
    assert render_memory_index(clean_dirs / "nope") == ""


# ---- M1/M2:frontmatter、类型、新鲜度、双上限 ----


def test_frontmatter_roundtrip(clean_dirs: Path) -> None:
    """新格式:保存带 type/description,文件头有 frontmatter,索引行带描述和类型。"""
    ws = clean_dirs / "ws"
    ws.mkdir()
    path = save_memory(
        "部署流程",
        "服务器发布步骤\n1. ssh",
        ws,
        project_level=True,
        mem_type="project",
        description="登录与发布步骤,含 systemctl",
    )
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---")  # frontmatter 打头
    assert "type: project" in text and "created:" in text
    entries = list_memories(ws)
    assert entries[0].description == "登录与发布步骤,含 systemctl"
    assert entries[0].mem_type == "project"
    index = (ws / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "登录与发布步骤" in index and "(project)" in index
    rendered = render_memory_index(ws)
    assert "引用前先验证" in rendered  # M1.4:头部行为协议
    assert "(project·" in rendered  # 类型 + 新鲜度进了注入行


def test_old_format_compat(clean_dirs: Path) -> None:
    """旧格式(无 frontmatter)读时兼容:回填默认值,文件本身一个字都不动。"""
    ws = clean_dirs / "ws"
    ws.mkdir()
    d = ws / "memory"
    d.mkdir(parents=True)
    (d / "旧记忆.md").write_text("# 旧标题\n\n旧正文\n\n*记录于 2026-09-01*\n", encoding="utf-8")
    entries = list_memories(ws)
    assert entries[0].title == "旧标题"
    assert entries[0].mem_type == "project"  # 无 frontmatter → 默认 project
    assert entries[0].description == "旧标题"  # 描述回填标题
    text = render_memory_index(ws)
    assert "旧标题" in text and "project" in text
    # "读时兼容":旧文件不被重写,只有再次保存才升级格式
    assert "*记录于 2026-09-01*" in (d / "旧记忆.md").read_text(encoding="utf-8")


def test_content_frontmatter(clean_dirs: Path) -> None:
    """content 开头自带 frontmatter:字段被提取;显式参数优先于它。"""
    ws = clean_dirs / "ws"
    ws.mkdir()
    save_memory(
        "记忆甲",
        "---\ndescription: 描述甲\ntype: feedback\n---\n标题甲\n正文甲",
        ws,
        project_level=True,
    )
    e = list_memories(ws)[0]
    assert e.description == "描述甲"
    assert e.mem_type == "feedback"
    text = e.path.read_text(encoding="utf-8")
    assert "# 标题甲" in text and "描述甲" in text  # 落盘是新格式:头部重写 frontmatter
    # 显式参数 > 内容 frontmatter
    save_memory(
        "记忆乙", "---\ntype: feedback\n---\n标题乙", ws, project_level=True, mem_type="user"
    )
    e2 = next(m for m in list_memories(ws) if m.name == "记忆乙")
    assert e2.mem_type == "user"


def test_freshness_labels(clean_dirs: Path) -> None:
    """注入行按 mtime 标年龄:今天 / N 天前 / 超龄加 ⚠旧(方案 M2.2)。"""
    import os
    import time

    ws = clean_dirs / "ws"
    ws.mkdir()
    fresh = save_memory("新记忆", "新标题", ws, project_level=True)
    days3 = save_memory("三天记忆", "三天标题", ws, project_level=True)
    stale = save_memory("旧记忆", "旧标题", ws, project_level=True)
    os.utime(days3, (time.time() - 3 * 86400,) * 2)
    os.utime(stale, (time.time() - 40 * 86400,) * 2)
    assert fresh  # 只是让 ruff 不报未使用变量
    text = render_memory_index(ws)
    assert "·今天)" in text
    assert "3 天前)" in text
    assert "40 天前⚠旧)" in text


def test_index_line_cap(clean_dirs: Path) -> None:
    """行数上限:105 条 → 注入只保留 100 行,并注明触发了行数上限(方案 M2.1)。"""
    ws = clean_dirs / "ws"
    ws.mkdir()
    for i in range(105):
        save_memory(f"m{i:03d}", f"标题{i}", ws, project_level=True)
    text = render_memory_index(ws)
    assert "行数上限已截断" in text
    rows = [ln for ln in text.split("\n") if ln.startswith("- [")]
    assert len(rows) == 100


def test_index_byte_cap(clean_dirs: Path) -> None:
    """字节上限:行数没超但每行很长(汉字描述)→ 字节上限兜底(方案 M2.1)。"""
    ws = clean_dirs / "ws"
    ws.mkdir()
    d = ws / "memory"
    d.mkdir(parents=True)
    # 60 行 × 每行约 400B(120 个汉字的描述)≈ 24KB > 16KB,而 60 < 100 行
    for i in range(60):
        (d / f"b{i:02d}.md").write_text(f"# {'长' * 130}\n\n正文\n", encoding="utf-8")
    text = render_memory_index(ws)
    assert "字节上限已截断" in text
    # 注入文本确实被压在双上限附近(允许头部说明的少量超出)
    assert len(text.encode("utf-8")) < 20 * 1024


# ---- M3:scope 推断 ----


def test_scope_inference_by_type(clean_dirs: Path) -> None:
    """层级推断:user/feedback → 用户层,project/reference → 项目层(方案 M3.1)。"""
    from my_agent_alpha.dirs import get_data_dir

    ws = clean_dirs / "ws"
    ws.mkdir()
    # 不传 project_level,全靠类型推断
    save_memory("身份甲", "用户是运维", ws, mem_type="user")
    save_memory("纠正甲", "回答要简短", ws, mem_type="feedback")
    save_memory("事实甲", "项目用 uv", ws, mem_type="project")
    save_memory("指针甲", "看板地址", ws, mem_type="reference")
    user_dir = get_data_dir() / "memory"
    assert (user_dir / "身份甲.md").exists() and (user_dir / "纠正甲.md").exists()
    assert (ws / "memory" / "事实甲.md").exists()
    assert (ws / "memory" / "指针甲.md").exists()
    # 都不传:默认 project → 项目层
    save_memory("默认甲", "裸保存", ws)
    assert (ws / "memory" / "默认甲.md").exists()


def test_scope_explicit_overrides(clean_dirs: Path) -> None:
    """显式指定盖过推断:scope 参数 > 内容 frontmatter 的 scope > 类型推断。"""
    from my_agent_alpha.dirs import get_data_dir

    ws = clean_dirs / "ws"
    ws.mkdir()
    # 显式参数:user 类型硬写进项目层
    save_memory("身份乙", "用户是运维", ws, project_level=True, mem_type="user")
    assert (ws / "memory" / "身份乙.md").exists()
    # 内容 frontmatter 的 scope:project 类型写进用户层
    save_memory("事实乙", "---\nscope: user\n---\n项目事实", ws, mem_type="project")
    assert (get_data_dir() / "memory" / "事实乙.md").exists()


# ---- M4:整理提示词 ----


def test_consolidate_prompt_has_manifest(clean_dirs: Path) -> None:
    """整理提示词带记忆清单与两层路径(CC extract 代理的做法:省一轮 ls)。"""
    from my_agent_alpha.memory_store import build_consolidate_prompt

    ws = clean_dirs / "ws"
    ws.mkdir()
    save_memory("部署流程", "服务器发布步骤", ws, project_level=True, description="发布步骤")
    prompt = build_consolidate_prompt(ws)
    assert "四步走" in prompt  # 定向→收集→固化→剪枝
    assert str(ws / "memory") in prompt  # 两层路径都写明
    assert "部署流程" in prompt and "发布步骤" in prompt  # 清单注入
