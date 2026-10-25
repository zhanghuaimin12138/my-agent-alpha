"""让根目录的 evals 包在单元测试里可导入:它不是安装包(只有 src 下的才是)。
conftest 会在本目录所有测试模块导入前执行,插好路径后面就不用各文件自己处理。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/unit/evals → 仓库根
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
