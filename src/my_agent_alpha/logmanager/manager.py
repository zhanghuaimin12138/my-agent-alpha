"""append-only JSONL 会话日志(仿 gptme LogManager 的持久化核心)。

设计要点(研究精华 #9,见 docs/gptme-study.md §11):

- 追加游标:常规写入只落盘"新增消息",O(新增);通过
  (路径 + 文件字节数 + 已落盘前缀的对象身份 + 尾随换行)四重校验,
  一旦发现文件被外部改动、或内存中前缀被改写/截断/乱序,自动降级为原子全量重写。
- 原子全量重写:同目录临时文件 → 无条件 fsync → os.replace → 目录 fsync,
  崩溃在任何时刻都不会毁掉上一份完整日志(重写路径无条件 fsync:重写本就少见,
  且 rename 先于数据落盘是"空文件/垃圾文件"事故的经典成因)。
- conversation.jsonl 永不销毁:主日志是唯一事实来源
  (压缩视图/分支是后续阶段的事,那也是"写副本、切指针",不动主文件)。
- 读取宽容:未知字段由 Message.from_dict 丢弃;末行损坏(崩溃时写了一半,包括
  被切断的多字节 UTF-8 字符)容忍并告警,中间行损坏则报错 —— 损坏不该被静默吞掉。
  跳过了残尾时游标保持"未知",下一次写入强制全量重写把残尾清掉 —— 否则
  "容忍跳过 + 后续增量追加"会把坏行挤成中间行,日志从此打不开。

阶段 0 刻意省略(详见 docs/devlog/001):文件锁(flock)、events.jsonl WAL、
视图/分支机制、附件内容寻址存储。
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from pathlib import Path

from my_agent_alpha.message import Message

logger = logging.getLogger(__name__)

CONVERSATION_FILENAME = "conversation.jsonl"


def generate_conversation_id() -> str:
    """会话目录名:YYYY-MMDD-<随机 6 位十六进制>。

    (gptme 还会从首条消息生成语义 slug,属于自动命名线程,后续阶段再加。)
    """
    return datetime.now().strftime("%Y-%m%d") + "-" + secrets.token_hex(3)


def _jsonl_line(msg: Message) -> bytes:
    # ensure_ascii=False:日志里的中文保持人类可读;
    # default=str:metadata 里混进不可序列化对象时降级为字符串,不让一条消息毁掉整个写入
    return json.dumps(msg.to_dict(), ensure_ascii=False, default=str).encode("utf-8") + b"\n"


def _fsync_directory(path: Path) -> None:  # 目录（os层面：目录=文件）文件强制落盘
    """fsync 目录,让 os.replace 的改名本身持久化(部分平台不支持,容忍失败)。"""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # 例如 EINVAL/ENOTSUP
    finally:
        os.close(fd)


class Log:
    """消息列表 + 持久化游标。

    游标记录"哪些消息以什么身份写到了哪个文件的多少字节"。
    惯用法:append() 追加内存消息,write_jsonl() 落盘并前移游标。
    """

    def __init__(self, messages: Iterable[Message] = ()) -> None:
        self._messages: list[Message] = list(messages)
        # 已落盘前缀(按对象身份比较,而非相等 —— 相等无法区分"同内容的新对象")
        self._persisted: list[Message] = []
        self._persisted_path: Path | None = None
        self._persisted_size: int = -1  # 落盘时的文件字节数;-1 表示未知

    # ---- 只读访问 ----

    @property
    def messages(self) -> Sequence[Message]:
        """消息列表(防御性拷贝;调用方不得绕过 append 修改)。"""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)

    def __getitem__(self, index: int) -> Message:
        return self._messages[index]

    # ---- 修改 ----

    def append(self, msg: Message) -> None:
        self._messages.append(msg)

    # ---- 持久化 ----

    def _can_append(self, path: Path) -> bool:
        """判断能否走增量追加路径(否则必须全量重写)。"""
        if self._persisted_path != path or self._persisted_size < 0:
            return False
        try:
            if path.stat().st_size != self._persisted_size:
                return False  # 文件被外部截断/追加/替换
        except OSError:
            return False  # 文件已不存在
        if len(self._persisted) > len(self._messages):
            return False  # 内存中被删减过
        for persisted, current in zip(self._persisted, self._messages, strict=False):
            # self._persisted（persisted）, self._messages（current）
            # 的消息一一对照，有不同说明被改写过
            # strict=False:取较短的已落盘前缀作对照,后面的新消息不算
            if persisted is not current:
                return False  # 内存前缀被改写或乱序
        if self._persisted_size > 0:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    return False  # 上次写入不完整(缺尾随换行)
        return True

    def _after_write(self, path: Path) -> None:
        """更新游标信息"""
        self._persisted = list(self._messages)
        self._persisted_path = path
        self._persisted_size = path.stat().st_size

    def write_jsonl(self, path: Path, *, sync: bool = False) -> None:
        """把消息写入 path:能追加则只写新增,否则原子全量重写。

        重写路径总是 fsync(文件 + 目录)—— 崩溃安全是它的存在意义;
        sync=True 额外让增量追加路径也 fsync 文件,用于会话收尾等
        需要强持久化的时刻(追加不改目录项,无需 fsync 目录)。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._can_append(path):
            new_messages = self._messages[len(self._persisted) :]
            with open(path, "ab") as f:
                for msg in new_messages:
                    f.write(_jsonl_line(msg))
                if sync:
                    f.flush()
                    os.fsync(f.fileno())
        else:
            self._rewrite_jsonl(path)
        self._after_write(path)

    def _rewrite_jsonl(self, path: Path) -> None:
        """原子全量重写:临时文件 + 无条件 fsync + os.replace + 目录 fsync。

        也就是写临时文件，之后整体替换"""
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        try:
            with open(tmp, "wb") as f:
                for msg in self._messages:
                    f.write(_jsonl_line(msg))  # 写临时文件
                f.flush()  # 进内核缓存
                os.fsync(f.fileno())  # 强制临时文件数据落盘，无条件:数据必须先于 rename 落盘
            os.replace(tmp, path)  # 原子改名(要么成功要么没发生,不会出现半个名字)
            _fsync_directory(path.parent)  #  强制"目录"(目录也是个文件)落盘,让改名本身不丢
        finally:
            # 异常时清掉残留临时文件(正常路径下 os.replace 已把它变成目标文件)
            if tmp.exists():
                tmp.unlink()
        logger.debug("全量重写 %s(%d 条消息)", path, len(self._messages))

    # ---- 装载 ----

    @classmethod
    def read_jsonl(cls, path: Path) -> Log:
        """从 JSONL 装载进内存。

        正常情况下把游标标到文件末尾(后续 append 走增量路径)。
        末行损坏(崩溃时写了一半,包括被切断的多字节字符)容忍并告警,
        但游标保持未知 → 下次写入强制全量重写,把残尾清掉
        (否则后续增量追加会把坏行挤成中间行,日志从此打不开);
        中间行损坏抛 ValueError —— 损坏不该被静默吞掉。

        按字节读、逐行解码:整文件 read_text 会在多字节字符被撕裂时
        直接抛 UnicodeDecodeError,连"容忍末行"的机会都没有。
        """
        messages: list[Message] = []
        skipped_corrupt = False
        raw_lines = path.read_bytes().split(b"\n")
        if raw_lines and raw_lines[-1] == b"":
            raw_lines.pop()  # 文件以换行结尾:末尾空段不是内容行,真正的末行是它前面那条
        last_idx = len(raw_lines) - 1
        for i, raw in enumerate(raw_lines):
            if not raw.strip():
                continue  # 跳过空行
            try:
                text = raw.decode("utf-8")
                messages.append(Message.from_dict(json.loads(text)))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
                if i == last_idx:
                    logger.warning("日志末行损坏,已跳过(%s): %s", path, e)
                    skipped_corrupt = True
                    break
                raise ValueError(f"日志 {path} 第 {i + 1} 行损坏: {e}") from e
        log = cls(messages)  # 构造新log
        if not skipped_corrupt and path.exists():
            log._after_write(path)
        return log


class LogManager:
    """一个会话目录的有状态门面:装载 / 追加 / 写盘。

    目录布局:
        <logs_dir>/<conversation-id>/conversation.jsonl
    """

    def __init__(self, log: Log, logdir: Path) -> None:
        self._log = log
        self.logdir = logdir

    @property
    def log(self) -> Log:
        return self._log

    @property
    def messages(self) -> Sequence[Message]:
        return self._log.messages

    def __len__(self) -> int:
        return len(self._log)

    def __iter__(self) -> Iterator[Message]:
        return iter(self._log)

    @property
    def conversation_file(self) -> Path:
        return self.logdir / CONVERSATION_FILENAME

    @classmethod
    def load(
        cls,
        logdir: Path,
        initial_msgs: Iterable[Message] = (),
        *,
        create: bool = False,
    ) -> LogManager:
        """装载会话;create=True 时允许新建空会话目录。"""
        conversation_file = logdir / CONVERSATION_FILENAME
        if conversation_file.exists():
            log = Log.read_jsonl(conversation_file)
        elif create:
            logdir.mkdir(parents=True, exist_ok=True)
            log = Log(initial_msgs)
        else:
            raise FileNotFoundError(f"会话不存在: {conversation_file}")
        return cls(log, logdir)

    def append(self, msg: Message, *, sync: bool = False) -> None:
        """追加一条消息并立即落盘。

        打印钩子(quiet 语义的消费者)留待 CLI 阶段接入。
        """
        self._log.append(msg)
        self.write(sync=sync)

    def write(self, *, sync: bool = False) -> None:
        """把当前日志写盘(Log 内部自行决定增量/全量路径)。"""
        self._log.write_jsonl(self.conversation_file, sync=sync)
