"""会话日志:把对话保存成 JSONL 文件(每行一条消息的 JSON),只增不改。

LogManager 的持久化设计,核心解决两个问题:

1. 怎么写得快?—— 追加游标(见 Log 类):平时只在文件末尾追加新消息,
   不重写整个文件;一旦发现文件或内存被意外改动,立即退回安全的"全量重写"。
2. 怎么保证崩溃不丢数据?—— 全量重写走"临时文件 → fsync → 原子改名 → 目录 fsync"
   四步,任何时刻断电,磁盘上要么是完整的新文件、要么是完整的旧文件,不会出现半个。

另外:conversation.jsonl 是对话的完整历史,永远不删除
(压缩上下文时也是另存视图副本,不改这个文件,见下方 LogManager 的视图机制);
读文件时,最后一行如果坏了(程序崩溃时只写了一半),跳过并告警;
中间的行坏了则直接报错,因为那意味着日志被外力破坏,不能装作没看见。

阶段 0 刻意省略(详见 docs/devlog/001):文件锁(多进程同时写)、事件日志(WAL)、
视图/分支机制、附件存储。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from my_agent_alpha.logmanager.views import (
    ViewState,
    next_view_id,
    read_active_state,
    remove_active_state,
    view_path,
    write_active_state,
)
from my_agent_alpha.message import Message, clean_bad_unicode

logger = logging.getLogger(__name__)

CONVERSATION_FILENAME = "conversation.jsonl"


def generate_conversation_id() -> str:
    """生成会话目录名:uuid4 的 32 位随机十六进制。

    为什么不含日期:跨天续聊的会话,文件名里的日期会撒谎;
    看时间交给 /log list,认会话靠标题。

    示例:logdir = 会话根目录 / generate_conversation_id()
        logdir = ~/.local/share/my-agent-alpha/projects/<工作区键>/logs/6f0c…
    """
    return uuid.uuid4().hex


def _jsonl_line(msg: Message) -> bytes:
    """把一条消息转成"一行 JSON + 换行符"的字节(写入文件的最小单位)。

    ensure_ascii=False:中文直接写中文,人打开文件就能读;
    default=str:metadata 里万一混进奇怪对象,降级成字符串,不让一条消息毁掉整个写入;
    errors="replace":内容里万一混进编码不了的"落单代理字符"(终端/接口传来的坏字节),
    换成 ? 继续写 —— 日志是唯一状态,宁可一个字符变 ? 也不能让程序崩掉丢整段对话。
    """
    return (
        clean_bad_unicode(json.dumps(msg.to_dict(), ensure_ascii=False, default=str)).encode(
            "utf-8"
        )
        + b"\n"
    )


def _fsync_directory(path: Path) -> None:  # 目录（os层面：目录=文件）文件强制落盘
    """把目录本身的改动强制写入磁盘,让 os.replace 的改名不会因断电而丢失。

    有的操作系统不支持对目录做 fsync,遇到就静默跳过(尽力而为)。
    """
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
    """内存里的消息列表 + 一个"写到哪了"的游标(书签)。

    游标 = 一个书签,记着三件事:上次写到了哪个文件、写到第几字节、写了哪些消息。
    有了它,追加一条消息只需要在文件末尾加一行,而不是重写全部消息。

    惯常用法:
        log = Log.read_jsonl(文件)   # 打开旧会话
        log.append(Message(...))     # 内存里加一条
        log.write_jsonl(文件)        # 落盘(内部自动选择"只加一行"或"全部重写")
    """

    def __init__(self, messages: Iterable[Message] = ()) -> None:
        """新建 Log,可以带一批初始消息;游标为空(还没写过盘)。"""
        self._messages: list[Message] = list(messages)

        self._persisted: list[Message] = []  # 已持久化到日志的消息
        self._persisted_path: Path | None = None  # 持久化日志的路径
        self._persisted_size: int = -1  # 上次写完时文件的大小(字节);-1 表示还没写过

    # ---- 只读访问 ----

    @property
    def messages(self) -> Sequence[Message]:
        """返回所有消息的列表副本(改副本不影响日志内部,修改请走 append)。"""
        return list(self._messages)

    def __len__(self) -> int:
        """消息条数:len(log)。"""
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        """让 Log 可以被 for 循环:for msg in log。"""
        return iter(self._messages)

    def __getitem__(self, index: int) -> Message:
        """按下标取消息:log[0] 是第一条。"""
        return self._messages[index]

    # ---- 内存修改 ----

    def append(self, msg: Message) -> None:
        """在内存里追加一条消息(注意:还没写盘,写盘调用 write_jsonl)。"""
        self._messages.append(msg)

    # ---- 持久化 ----

    def _can_append(self, path: Path) -> bool:
        """检查"只在文件末尾加新消息"是否安全;任何一项不满足就返回 False:

        1. 上次就是写到这个文件,且知道写到第几字节(书签有效);
        2. 文件现在的字节数 == 书签记录的字节数(没被别的程序动过);
        3. 书签记录的已经持久化的那批消息,现在仍然是内存日志(self._messages)
           开头的那批对象(内存没被改过);
        4. 文件最后一个字节是换行符(上次不是写到一半崩的)。
        """
        # 路径对不上或者书签没有使用过（_persisted_size=-1）
        if self._persisted_path != path or self._persisted_size < 0:
            return False
        try:
            # 现在日志的字节数和内存中的书签数对不上
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
        """更新游标信息"""  # 写盘成功后调用:把书签移到"现在"
        self._persisted = list(self._messages)
        self._persisted_path = path
        self._persisted_size = path.stat().st_size

    def write_jsonl(self, path: Path, *, sync: bool = False) -> None:
        """把内存消息写入持久化日志。安全时只追加新消息(快),不安全时全部重写(慢但保险)。

        示例:
            log = Log.read_jsonl(f); log.append(m); log.write_jsonl(f)
            → 文件只在末尾多了一行,前面的字节原样不动。

        sync=True 额外强制刷盘(fsync),用在"会话结束"这类必须确保落盘的时刻;
        平时默认 False,操作系统自己挑时机刷盘,速度快。
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
        """内存日志全量持久化

        先写临时文件,再一步替换旧文件。
        为什么这样安全:os.replace 是"原子"操作 —— 要么成功(新文件顶上),
        要么失败(旧文件原样保留),不存在替换到一半的状态。
        所以哪怕写到一半断电,磁盘上也总有至少一份完整日志。
        """
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
        """从 JSONL 文件读出所有消息,生成一个 Log(游标直接标到文件末尾)。

        对损坏的处理:
        - 最后一行坏了(比如崩溃时只写了一半)→ 跳过它、打警告,会话照常打开;
          同时故意不设游标,下次写盘会触发全量重写,把坏行清理掉;
        - 中间的行坏了 → 抛 ValueError(说明日志被外力破坏,必须让人知道);
        - 先按字节读、再逐行解码:如果一个中文字符被拦腰截断,
          整个文件一次性解码会直接报错,逐行解码才有机会"只坏一行"。
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
    """管理"一个会话目录"的入口:负责打开/新建会话、追加消息并落盘。

    目录结构:
        logdir:./.local/share/my-agent-alpha/projects/<工作区键>/logs/会话id/
        日志根目录/<会话id>/conversation.jsonl        ← Master:全部消息,一行一条
        日志根目录/<会话id>/views/compacted-NNN.jsonl ← 压缩视图(激活后是"当前有效历史")
        日志根目录/<会话id>/views/active.json         ← 记录当前激活哪个视图

    两个读取口,用途不同别混用:
        manager.messages        ← 当前有效历史:激活了视图就是视图,否则是 Master。
                                  发给模型、判断循环是否继续,都用它;
        manager.master_messages ← 完整历史:todo 重放这类"要重放全部状态变更"的
                                  场景用它,不能让压缩吞掉历史事实。

    一般不需要直接用 Log,用 LogManager 即可:
        manager = LogManager.load(目录, create=True)
        manager.append(Message("user", "你好"))   # 追加并立即写盘
        for msg in manager.messages: ...
    """

    def __init__(self, master_log: Log, logdir: Path) -> None:
        """保存主日志、会话目录与(可能激活的)视图。一般用 load() 创建,不直接构造。"""
        self._master_log = master_log
        self._active_view_log: Log | None = None  # 激活的视图 Log;None = 还没有视图
        self._active_view_id: str | None = None  # 激活视图的名字,如 "compacted-001"
        self.logdir = logdir

    # ---- 读取口 ----

    @property
    def master_log(self) -> Log:
        """主日志(Master)的 Log 对象:完整历史的唯一事实来源。"""
        return self._master_log

    @property
    def master_messages(self) -> Sequence[Message]:
        """完整历史的消息列表副本(不管有没有激活视图)。"""
        return self._master_log.messages

    @property
    def current_view(self) -> str | None:
        """当前激活视图的名字;None 表示没有视图,读取的就是 Master。"""
        return self._active_view_id

    @property
    def log(self) -> Log:
        """当前有效历史的 Log:激活了视图就是视图,否则是 Master。"""
        return self._active_view_log if self._active_view_log is not None else self._master_log

    @property
    def messages(self) -> Sequence[Message]:
        """当前有效历史的消息列表副本。"""
        return self.log.messages

    def __len__(self) -> int:
        """当前有效历史的消息条数:len(manager)。"""
        return len(self.log)

    def __iter__(self) -> Iterator[Message]:
        """让 manager 可以被 for 循环(遍历当前有效历史)。"""
        return iter(self.log)

    @property
    def conversation_file(self) -> Path:
        """本会话的 Master 日志文件完整路径。"""
        return self.logdir / CONVERSATION_FILENAME

    # ---- 打开/恢复 ----

    @classmethod
    def load(
        cls,
        logdir: Path,
        initial_msgs: Iterable[Message] = (),
        *,
        create: bool = False,
    ) -> LogManager:
        """打开一个会话;目录不存在时,只有 create=True 才允许新建。

        打开后还会看一眼 views/active.json:有合法的激活视图就连视图一起恢复
        (模型继续读压缩后的历史);状态坏了或视图文件没了就打警告,回退读 Master。

        示例:
            manager = LogManager.load(日志根 / "2026-0929-a3f1c2")          # 恢复旧会话
            manager = LogManager.load(新目录, 初始消息, create=True)         # 新会话
        """
        conversation_file = logdir / CONVERSATION_FILENAME
        if conversation_file.exists():
            log = Log.read_jsonl(conversation_file)
        elif create:
            logdir.mkdir(parents=True, exist_ok=True)
            log = Log(initial_msgs)
        else:
            raise FileNotFoundError(f"会话不存在: {conversation_file}")
        manager = cls(log, logdir)
        manager._restore_active_view()
        return manager

    def _restore_active_view(self) -> None:
        """按 views/active.json 恢复激活的视图;任何不对劲都回退 Master。

        回退只是"读取来源退回完整历史",不删除任何文件 —— 状态文件坏了
        应当让人看见(告警),而不是顺手抹掉证据。
        """
        state = read_active_state(self.logdir)
        if state is None or state.active_view is None:
            return  # 没激活过视图(或状态不可读):正常走 Master
        path = view_path(self.logdir, state.active_view)
        if not path.exists():
            logger.warning("激活视图文件不存在,回退主日志: %s", path)
            return
        self._active_view_log = Log.read_jsonl(path)
        self._active_view_id = state.active_view

    # ---- 视图的创建与停用 ----

    def create_compacted_view(self, messages: Sequence[Message]) -> str:
        """把一份“压缩后的历史”固化成新视图文件并激活,返回视图名(如 compacted-001)。

        安全顺序(任何一步失败,旧状态都不被破坏):
        1. Master 先同步落盘 —— 压缩的源事实先保住;
        2. 新视图整体写入 views/compacted-NNN.jsonl(全新文件,原子替换);
        3. views/active.json 原子更新为新视图;
        4. 最后才在内存里激活 —— 磁盘没成功之前,读取口保持原样。
        若第 3 步失败:Master 完好、旧视图仍是读取来源,新视图文件只是一个
        没被激活的孤立快照(不参与任何读取,留着也无害)。

        注意:传入的 messages 应当是"当前有效历史"压缩的结果(而不是每次
        重新压缩整个 Master),由调用方保证;本方法只负责存储与激活。
        """
        if not messages:
            raise ValueError("不允许创建空视图(那等于把历史清零)")
        # 1. Master 落盘
        self._master_log.write_jsonl(self.conversation_file, sync=True)
        # 2. 写新视图文件(全新 Log,游标为空 → 走原子全量写入)
        view_id = next_view_id(self.logdir)
        view_log = Log(messages)
        view_log.write_jsonl(view_path(self.logdir, view_id), sync=True)
        # 3. 写状态检查点(记录双方此刻各写到第几条,崩溃恢复的依据)
        write_active_state(
            self.logdir,
            ViewState(
                active_view=view_id,
                master_committed_count=len(self._master_log),
                view_committed_count=len(view_log),
            ),
        )
        # 4. 内存激活:从此 manager.messages 读的是新视图
        self._active_view_log = view_log
        self._active_view_id = view_id
        return view_id

    def switch_to_master(self) -> None:
        """停用激活的视图,读取口回到 Master;视图文件原样保留(历史快照)。

        只删 active.json 这个"指针",不动任何消息文件。调试/兜底用:
        Master 装着完整历史,切回来永远安全。
        """
        self._active_view_log = None
        self._active_view_id = None
        remove_active_state(self.logdir)

    # ---- 追加与写盘 ----

    def append(self, msg: Message, *, sync: bool = False) -> None:
        """追加一条消息并立即写盘。

        sync=True 时额外强制刷盘,用于会话结束等关键时刻(平时不用)。
        """
        self._master_log.append(msg)  # 追加内存
        self.write(sync=sync)  # 持久化日志

    def write(self, *, sync: bool = False) -> None:
        """把当前消息写盘(追加还是重写,Log 内部自己判断)。"""
        self._master_log.write_jsonl(self.conversation_file, sync=sync)
