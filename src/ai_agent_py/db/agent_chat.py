"""通用 Agent 的会话记忆：MySQL 持久化（会话 / 消息 / 滚动摘要三张表）。

LangGraph 的 ``InMemorySaver`` 只能把上下文留在进程里，重启就没了；通用 Agent 的
会话要跨刷新、跨重启保留，所以这里把每一轮的消息按 ``thread_id`` 写进 MySQL：

* **会话**（`tb_agent_chat_session`）：一行一个会话，标题取第一句用户输入，
  记录消息条数与最近更新时间，用于左侧会话列表；
* **消息**（`tb_agent_chat_message`）：一行一条消息，``role`` 为
  ``user`` / ``assistant`` / ``tool``；工具行额外带工具名、入参 JSON、执行结果状态与耗时，
  供界面回放「这一轮调用了哪些工具」；
* **摘要**（`tb_agent_chat_summary`）：一行一个会话的滚动摘要，``until_seq`` 表示覆盖到
  哪一条消息；原始消息不删除，只是不再发送给模型；
* **上下文组装**：不再只按条数取历史，`general/context.py` 会按 token 预算选择最近连续
  尾部，`general/summary.py` 把更早部分合并成摘要；工具调用只用于展示，不进提示词。

时间列一律 UTC 往返（库里的 ``DATETIME(6)`` 不带时区），与 ``db/document_jobs.py`` 一致；
建表脚本见 ``docs/mysql/`` 下的三个 ``tb_agent_chat_*.sql``，本模块的 `Table` 定义必须与脚本一致。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Engine,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.exc import SQLAlchemyError

from ai_agent_py.config import Settings
from ai_agent_py.db.health import get_engine
from ai_agent_py.errors import ConfigurationError, DatabaseError

SESSION_TABLE = "tb_agent_chat_session"
MESSAGE_TABLE = "tb_agent_chat_message"
SUMMARY_TABLE = "tb_agent_chat_summary"

#: 默认应用标识：区分同一张表里其他业务的会话（报销审核等可以共用这套记忆）。
DEFAULT_AGENT_ID = "general"
DEFAULT_TITLE = "新会话"
MAX_TITLE_CHARS = 40

ChatRole = Literal["user", "assistant", "tool"]

_metadata = MetaData()

#: ``docs/mysql/tb_agent_chat_session.sql`` 的 Python 镜像（类型/可空性必须一致）。
agent_chat_session = Table(
    SESSION_TABLE,
    _metadata,
    Column("id", String(64), primary_key=True),
    Column("agent_id", String(32), nullable=False, default=DEFAULT_AGENT_ID),
    Column("title", String(120), nullable=False, default=DEFAULT_TITLE),
    Column("message_count", Integer, nullable=False, default=0),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
    comment="通用 Agent 会话（thread_id 一行一个）",
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
    mysql_collate="utf8mb4_general_ci",
)

#: ``docs/mysql/tb_agent_chat_message.sql`` 的 Python 镜像。
agent_chat_message = Table(
    MESSAGE_TABLE,
    _metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("thread_id", String(64), nullable=False),
    Column("seq", Integer, nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text().with_variant(MEDIUMTEXT(), "mysql"), nullable=False),
    Column("tool_name", String(64)),
    Column("tool_call_id", String(128)),
    Column("tool_args", Text().with_variant(MEDIUMTEXT(), "mysql")),
    Column("tool_status", String(16)),
    Column("elapsed_ms", Integer),
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("thread_id", "seq", name="uk_thread_seq"),
    Index("idx_thread_created", "thread_id", "created_at"),
    comment="通用 Agent 会话消息（用户 / 助手 / 工具调用）",
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
    mysql_collate="utf8mb4_general_ci",
)

#: ``docs/mysql/tb_agent_chat_summary.sql`` 的 Python 镜像。
agent_chat_summary = Table(
    SUMMARY_TABLE,
    _metadata,
    Column("thread_id", String(64), primary_key=True),
    Column("summary", Text().with_variant(MEDIUMTEXT(), "mysql"), nullable=False),
    Column("until_seq", Integer, nullable=False),
    Column("source_messages", Integer, nullable=False),
    Column("tokens", Integer),
    Column("model", String(128)),
    Column("updated_at", DateTime, nullable=False),
    comment="通用 Agent 已压缩历史的摘要（thread_id 一行一个）",
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
    mysql_collate="utf8mb4_general_ci",
)

#: 建表所需的三个脚本（报错提示里直接引用文件名）。
SCHEMA_HINT = (
    f"（请先执行 docs/mysql/{SESSION_TABLE}.sql、docs/mysql/{MESSAGE_TABLE}.sql "
    f"与 docs/mysql/{SUMMARY_TABLE}.sql 建表）"
)


# ---- 数据模型 -------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ChatSession:
    """一行 ``tb_agent_chat_session``。"""

    thread_id: str
    agent_id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """一行 ``tb_agent_chat_message``（不含自增主键，``seq`` 由存储分配）。"""

    role: ChatRole
    content: str
    seq: int = 0
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_args: str | None = None
    tool_status: str | None = None
    elapsed_ms: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ChatSummary:
    """一行 ``tb_agent_chat_summary``：覆盖到 ``until_seq`` 的历史摘要。"""

    thread_id: str
    summary: str
    until_seq: int
    source_messages: int
    tokens: int | None = None
    model: str | None = None
    updated_at: datetime | None = None


def session_title(message: str, limit: int = MAX_TITLE_CHARS) -> str:
    """把第一句用户输入压成会话标题（单行、去空白、截断）。"""
    flattened = " ".join(message.split())
    if not flattened:
        return DEFAULT_TITLE
    return flattened if len(flattened) <= limit else f"{flattened[:limit]}…"


def build_history(messages: Sequence[ChatMessage]) -> list[AnyMessage]:
    """把 MySQL 里的历史消息还原成 LangChain 消息（只保留 user / assistant）。

    工具调用记录只用于界面回放：``AIMessage(tool_calls=...)`` 在没有对应
    ``ToolMessage`` 的情况下会让 OpenAI 兼容接口报错，所以历史里一律不还原工具行。
    """
    history: list[AnyMessage] = []
    for message in messages:
        content = message.content
        if not content.strip():
            continue
        if message.role == "user":
            history.append(HumanMessage(content=content))
        elif message.role == "assistant":
            history.append(AIMessage(content=content))
    return history


# ---- 时间转换 -------------------------------------------------------------
def _naive_utc(value: datetime | None) -> datetime | None:
    """带时区的时间 → 入库用的 UTC 朴素时间。"""
    if value is None:
        return None
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


def _aware_utc(value: datetime | None) -> datetime | None:
    """库里的朴素时间 → 带 UTC 时区的时间（与 ``datetime.now(UTC)`` 对齐）。"""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _session_from_row(row: dict[str, Any]) -> ChatSession:
    return ChatSession(
        thread_id=str(row["id"]),
        agent_id=str(row["agent_id"]),
        title=str(row["title"]),
        message_count=int(row["message_count"]),
        created_at=_aware_utc(row["created_at"]) or datetime.now(UTC),
        updated_at=_aware_utc(row["updated_at"]) or datetime.now(UTC),
    )


def _message_from_row(row: dict[str, Any]) -> ChatMessage:
    return ChatMessage(
        role=str(row["role"]),  # type: ignore[arg-type]
        content=str(row["content"]),
        seq=int(row["seq"]),
        tool_name=row["tool_name"] or None,
        tool_call_id=row["tool_call_id"] or None,
        tool_args=row["tool_args"] or None,
        tool_status=row["tool_status"] or None,
        elapsed_ms=None if row["elapsed_ms"] is None else int(row["elapsed_ms"]),
        created_at=_aware_utc(row["created_at"]),
    )


def _summary_from_row(row: dict[str, Any]) -> ChatSummary:
    return ChatSummary(
        thread_id=str(row["thread_id"]),
        summary=str(row["summary"]),
        until_seq=int(row["until_seq"]),
        source_messages=int(row["source_messages"]),
        tokens=None if row["tokens"] is None else int(row["tokens"]),
        model=row["model"] or None,
        updated_at=_aware_utc(row["updated_at"]),
    )


# ---- 存储接口 -------------------------------------------------------------
class ChatMemoryStore(Protocol):
    """会话记忆的读写接口：生产用 MySQL，单元测试用内存实现。"""

    def ensure_session(
        self,
        thread_id: str,
        *,
        agent_id: str = DEFAULT_AGENT_ID,
        title: str | None = None,
    ) -> ChatSession:
        """会话不存在时创建（已存在则原样返回，只刷新 ``updated_at``）。"""
        ...

    def get_session(self, thread_id: str) -> ChatSession | None:
        """按会话 id 读取；不存在时返回 `None`。"""
        ...

    def list_sessions(
        self,
        *,
        agent_id: str | None = DEFAULT_AGENT_ID,
        limit: int = 50,
    ) -> list[ChatSession]:
        """按最近更新时间倒序返回会话。"""
        ...

    def count_sessions(self, *, agent_id: str | None = DEFAULT_AGENT_ID) -> int:
        """会话总数（概览页展示）。"""
        ...

    def delete_session(self, thread_id: str) -> bool:
        """删除会话及其全部消息；不存在时返回 `False`。"""
        ...

    def load_messages(self, thread_id: str, *, limit: int | None = None) -> list[ChatMessage]:
        """按 ``seq`` 正序返回消息（``limit`` 表示只取最近 N 条）。"""
        ...

    def append_messages(
        self,
        thread_id: str,
        messages: Sequence[ChatMessage],
        *,
        agent_id: str = DEFAULT_AGENT_ID,
        title: str | None = None,
    ) -> ChatSession:
        """追加消息并刷新会话摘要；``title`` 只在会话还没有消息时生效。"""
        ...

    def get_summary(self, thread_id: str) -> ChatSummary | None:
        """读取会话的滚动摘要；没有摘要时返回 `None`。"""
        ...

    def save_summary(self, summary: ChatSummary) -> bool:
        """保存/更新摘要；只有 ``until_seq`` 前进时才写入，返回是否成功。"""
        ...

    def summary_ready(self) -> bool:
        """摘要表是否可用（缺表 / 库异常时返回 `False`，降级为纯窗口截断）。"""
        ...


def _db_error(action: str, exc: SQLAlchemyError) -> DatabaseError:
    raw = str(getattr(exc, "orig", exc)).strip().splitlines()
    detail = raw[0][:300] if raw else exc.__class__.__name__
    hints = ""
    lowered = detail.lower()
    if "doesn't exist" in lowered or "does not exist" in lowered or "1146" in lowered:
        hints = SCHEMA_HINT
    return DatabaseError(f"{action}会话记忆表失败：{detail}{hints}")


class MySQLChatMemoryStore:
    """`ChatMemoryStore` 的 MySQL 实现（SQLAlchemy Core + PyMySQL）。"""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> MySQLChatMemoryStore:
        """用配置里的连接串建（或复用）连接池；未配置 `MYSQL_URL` 时抛 `DatabaseError`。"""
        try:
            return cls(get_engine(settings))
        except ConfigurationError as exc:
            raise DatabaseError(str(exc)) from exc

    # ---- 会话 -------------------------------------------------------------
    def ensure_session(
        self,
        thread_id: str,
        *,
        agent_id: str = DEFAULT_AGENT_ID,
        title: str | None = None,
    ) -> ChatSession:
        now = _naive_utc(datetime.now(UTC))
        statement = mysql_insert(agent_chat_session).values(
            id=thread_id,
            agent_id=agent_id,
            title=title or DEFAULT_TITLE,
            message_count=0,
            created_at=now,
            updated_at=now,
        )
        # 已存在时只刷新 updated_at：标题与计数继续保留。
        statement = statement.on_duplicate_key_update(
            agent_id=agent_id,
            updated_at=statement.inserted.updated_at,
        )
        try:
            with self._engine.begin() as connection:
                connection.execute(statement)
        except SQLAlchemyError as exc:
            raise _db_error("创建", exc) from exc
        session = self.get_session(thread_id)
        if session is None:  # pragma: no cover - 写入成功却读不到，属于库异常
            raise DatabaseError(f"会话 {thread_id} 创建后读取失败")
        return session

    def get_session(self, thread_id: str) -> ChatSession | None:
        statement = select(agent_chat_session).where(agent_chat_session.c.id == thread_id)
        try:
            with self._engine.connect() as connection:
                row = connection.execute(statement).mappings().first()
        except SQLAlchemyError as exc:
            raise _db_error("读取", exc) from exc
        return None if row is None else _session_from_row(dict(row))

    def list_sessions(
        self,
        *,
        agent_id: str | None = DEFAULT_AGENT_ID,
        limit: int = 50,
    ) -> list[ChatSession]:
        statement = select(agent_chat_session)
        if agent_id is not None:
            statement = statement.where(agent_chat_session.c.agent_id == agent_id)
        statement = statement.order_by(agent_chat_session.c.updated_at.desc()).limit(max(1, limit))
        try:
            with self._engine.connect() as connection:
                rows = connection.execute(statement).mappings().all()
        except SQLAlchemyError as exc:
            raise _db_error("读取", exc) from exc
        return [_session_from_row(dict(row)) for row in rows]

    def count_sessions(self, *, agent_id: str | None = DEFAULT_AGENT_ID) -> int:
        statement = select(func.count()).select_from(agent_chat_session)
        if agent_id is not None:
            statement = statement.where(agent_chat_session.c.agent_id == agent_id)
        try:
            with self._engine.connect() as connection:
                return int(connection.execute(statement).scalar_one())
        except SQLAlchemyError as exc:
            raise _db_error("统计", exc) from exc

    def delete_session(self, thread_id: str) -> bool:
        try:
            with self._engine.begin() as connection:
                deleted = connection.execute(
                    delete(agent_chat_session).where(agent_chat_session.c.id == thread_id)
                ).rowcount
                connection.execute(
                    delete(agent_chat_message).where(agent_chat_message.c.thread_id == thread_id)
                )
                connection.execute(
                    delete(agent_chat_summary).where(agent_chat_summary.c.thread_id == thread_id)
                )
        except SQLAlchemyError as exc:
            raise _db_error("删除", exc) from exc
        return bool(deleted)

    # ---- 摘要 -------------------------------------------------------------
    def get_summary(self, thread_id: str) -> ChatSummary | None:
        statement = select(agent_chat_summary).where(agent_chat_summary.c.thread_id == thread_id)
        try:
            with self._engine.connect() as connection:
                row = connection.execute(statement).mappings().first()
        except SQLAlchemyError as exc:
            raise _db_error("读取", exc) from exc
        return None if row is None else _summary_from_row(dict(row))

    def save_summary(self, summary: ChatSummary) -> bool:
        now = _naive_utc(summary.updated_at or datetime.now(UTC))
        assert now is not None
        try:
            with self._engine.begin() as connection:
                row = (
                    connection.execute(
                        select(agent_chat_summary)
                        .where(agent_chat_summary.c.thread_id == summary.thread_id)
                        .with_for_update()
                    )
                    .mappings()
                    .first()
                )
                if row is not None and int(row["until_seq"]) >= summary.until_seq:
                    # 另一请求已经压缩到更靠后的位置，别用旧摘要覆盖新摘要。
                    return False
                values = {
                    "thread_id": summary.thread_id,
                    "summary": summary.summary,
                    "until_seq": summary.until_seq,
                    "source_messages": summary.source_messages,
                    "tokens": summary.tokens,
                    "model": summary.model,
                    "updated_at": now,
                }
                if row is None:
                    connection.execute(insert(agent_chat_summary).values(**values))
                else:
                    connection.execute(
                        update(agent_chat_summary)
                        .where(agent_chat_summary.c.thread_id == summary.thread_id)
                        .values(**values)
                    )
        except SQLAlchemyError as exc:
            raise _db_error("写入", exc) from exc
        return True

    def summary_ready(self) -> bool:
        try:
            with self._engine.connect() as connection:
                connection.execute(select(agent_chat_summary.c.thread_id).limit(1)).first()
        except SQLAlchemyError:
            return False
        return True

    # ---- 消息 -------------------------------------------------------------
    def load_messages(self, thread_id: str, *, limit: int | None = None) -> list[ChatMessage]:
        statement = (
            select(agent_chat_message)
            .where(agent_chat_message.c.thread_id == thread_id)
            .order_by(agent_chat_message.c.seq.desc())
        )
        if limit is not None:
            statement = statement.limit(max(1, limit))
        try:
            with self._engine.connect() as connection:
                rows = connection.execute(statement).mappings().all()
        except SQLAlchemyError as exc:
            raise _db_error("读取", exc) from exc
        return [_message_from_row(dict(row)) for row in reversed(rows)]

    def append_messages(
        self,
        thread_id: str,
        messages: Sequence[ChatMessage],
        *,
        agent_id: str = DEFAULT_AGENT_ID,
        title: str | None = None,
    ) -> ChatSession:
        now = datetime.now(UTC)
        now_naive = _naive_utc(now)
        assert now_naive is not None
        items = list(messages)
        try:
            with self._engine.begin() as connection:
                session_row = self._lock_session(
                    connection, thread_id, agent_id=agent_id, now=now_naive
                )
                if items:
                    next_seq = int(
                        connection.execute(
                            select(func.coalesce(func.max(agent_chat_message.c.seq), 0)).where(
                                agent_chat_message.c.thread_id == thread_id
                            )
                        ).scalar_one()
                    )
                    rows = [
                        {
                            "thread_id": thread_id,
                            "seq": next_seq + offset,
                            "role": message.role,
                            "content": message.content,
                            "tool_name": message.tool_name,
                            "tool_call_id": message.tool_call_id,
                            "tool_args": message.tool_args,
                            "tool_status": message.tool_status,
                            "elapsed_ms": message.elapsed_ms,
                            "created_at": _naive_utc(message.created_at) or now_naive,
                        }
                        for offset, message in enumerate(items, start=1)
                    ]
                    connection.execute(insert(agent_chat_message), rows)

                values: dict[str, Any] = {
                    "message_count": int(session_row["message_count"]) + len(items),
                    "updated_at": now_naive,
                }
                # 只用第一轮的用户输入命名会话，之后不再覆盖用户看到的标题。
                if title and int(session_row["message_count"]) == 0:
                    values["title"] = session_title(title)
                connection.execute(
                    update(agent_chat_session)
                    .where(agent_chat_session.c.id == thread_id)
                    .values(**values)
                )
                row = (
                    connection.execute(
                        select(agent_chat_session).where(agent_chat_session.c.id == thread_id)
                    )
                    .mappings()
                    .one()
                )
                return _session_from_row(dict(row))
        except SQLAlchemyError as exc:
            raise _db_error("写入", exc) from exc

    def _lock_session(
        self,
        connection: Any,
        thread_id: str,
        *,
        agent_id: str,
        now: datetime,
    ) -> dict[str, Any]:
        """锁定会话行（不存在时先创建），保证 ``seq`` 在并发下不重复。"""
        statement = (
            select(agent_chat_session).where(agent_chat_session.c.id == thread_id).with_for_update()
        )
        row = connection.execute(statement).mappings().first()
        if row is not None:
            return dict(row)
        connection.execute(
            insert(agent_chat_session)
            .values(
                id=thread_id,
                agent_id=agent_id,
                title=DEFAULT_TITLE,
                message_count=0,
                created_at=now,
                updated_at=now,
            )
            # 并发首次写入同一个 thread_id 时，后到的一个直接忽略，避免主键冲突。
            .prefix_with("IGNORE")
        )
        created = (
            connection.execute(
                select(agent_chat_session)
                .where(agent_chat_session.c.id == thread_id)
                .with_for_update()
            )
            .mappings()
            .one()
        )
        return dict(created)


class InMemoryChatMemoryStore:
    """`ChatMemoryStore` 的内存实现：单元测试与本地无库演示用（重启即丢失）。"""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._messages: dict[str, list[ChatMessage]] = {}
        self._summaries: dict[str, ChatSummary] = {}

    def ensure_session(
        self,
        thread_id: str,
        *,
        agent_id: str = DEFAULT_AGENT_ID,
        title: str | None = None,
    ) -> ChatSession:
        now = datetime.now(UTC)
        existing = self._sessions.get(thread_id)
        if existing is None:
            session = ChatSession(
                thread_id=thread_id,
                agent_id=agent_id,
                title=title or DEFAULT_TITLE,
                message_count=0,
                created_at=now,
                updated_at=now,
            )
        else:
            session = ChatSession(
                thread_id=existing.thread_id,
                agent_id=agent_id,
                title=existing.title,
                message_count=existing.message_count,
                created_at=existing.created_at,
                updated_at=now,
            )
        self._sessions[thread_id] = session
        self._messages.setdefault(thread_id, [])
        return session

    def get_session(self, thread_id: str) -> ChatSession | None:
        return self._sessions.get(thread_id)

    def list_sessions(
        self,
        *,
        agent_id: str | None = DEFAULT_AGENT_ID,
        limit: int = 50,
    ) -> list[ChatSession]:
        sessions = [
            session
            for session in self._sessions.values()
            if agent_id is None or session.agent_id == agent_id
        ]
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return sessions[: max(1, limit)]

    def count_sessions(self, *, agent_id: str | None = DEFAULT_AGENT_ID) -> int:
        return sum(
            1
            for session in self._sessions.values()
            if agent_id is None or session.agent_id == agent_id
        )

    def delete_session(self, thread_id: str) -> bool:
        existed = self._sessions.pop(thread_id, None) is not None
        self._messages.pop(thread_id, None)
        self._summaries.pop(thread_id, None)
        return existed

    def get_summary(self, thread_id: str) -> ChatSummary | None:
        return self._summaries.get(thread_id)

    def save_summary(self, summary: ChatSummary) -> bool:
        existing = self._summaries.get(summary.thread_id)
        if existing is not None and existing.until_seq >= summary.until_seq:
            return False
        self._summaries[summary.thread_id] = ChatSummary(
            thread_id=summary.thread_id,
            summary=summary.summary,
            until_seq=summary.until_seq,
            source_messages=summary.source_messages,
            tokens=summary.tokens,
            model=summary.model,
            updated_at=summary.updated_at or datetime.now(UTC),
        )
        return True

    def summary_ready(self) -> bool:
        return True

    def load_messages(self, thread_id: str, *, limit: int | None = None) -> list[ChatMessage]:
        messages = self._messages.get(thread_id, [])
        return messages[-limit:] if limit is not None else list(messages)

    def append_messages(
        self,
        thread_id: str,
        messages: Sequence[ChatMessage],
        *,
        agent_id: str = DEFAULT_AGENT_ID,
        title: str | None = None,
    ) -> ChatSession:
        now = datetime.now(UTC)
        session = self._sessions.get(thread_id) or self.ensure_session(thread_id, agent_id=agent_id)
        stored = self._messages.setdefault(thread_id, [])
        next_seq = (stored[-1].seq if stored else 0) + 1
        for offset, message in enumerate(messages):
            stored.append(
                ChatMessage(
                    role=message.role,
                    content=message.content,
                    seq=next_seq + offset,
                    tool_name=message.tool_name,
                    tool_call_id=message.tool_call_id,
                    tool_args=message.tool_args,
                    tool_status=message.tool_status,
                    elapsed_ms=message.elapsed_ms,
                    created_at=message.created_at or now,
                )
            )
        updated = ChatSession(
            thread_id=thread_id,
            agent_id=session.agent_id,
            title=session_title(title) if title and session.message_count == 0 else session.title,
            message_count=session.message_count + len(messages),
            created_at=session.created_at,
            updated_at=now,
        )
        self._sessions[thread_id] = updated
        return updated
