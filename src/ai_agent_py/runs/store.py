"""运行状态机的存储层：MySQL ``tb_agent_run``（生产）与内存实现（测试 / 无库演示）。

和会话记忆一样，读写接口是 :class:`RunStore` 协议：生产用 MySQL，单元测试用内存实现，
API 层不关心背后是哪一种。**状态流转必须是原子的**：两个浏览器同时点「通过」时只能有
一个成功，所以 ``transition`` 用条件更新（``WHERE status IN (...)``）而不是先读后写，
``rowcount == 0`` 就表示「状态已经被别人改过」，调用方拿到 `None` 走冲突分支。

时间列一律 UTC 往返（库里的 ``DATETIME(6)`` 不带时区），与 ``db/agent_chat.py`` 一致；
建表脚本见 ``docs/mysql/tb_agent_run.sql``，本模块的 `Table` 定义必须与脚本一致。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import (
    Column,
    DateTime,
    Engine,
    Index,
    MetaData,
    String,
    Table,
    Text,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.exc import SQLAlchemyError

from ai_agent_py.config import Settings
from ai_agent_py.db.health import get_engine
from ai_agent_py.errors import ConfigurationError, DatabaseError
from ai_agent_py.runs.models import RunRecord, RunStatus

RUN_TABLE = "tb_agent_run"

_metadata = MetaData()

#: ``docs/mysql/tb_agent_run.sql`` 的 Python 镜像（类型/可空性必须一致）。
agent_run = Table(
    RUN_TABLE,
    _metadata,
    Column("run_id", String(64), primary_key=True),
    Column("agent_id", String(32), nullable=False),
    Column("thread_id", String(64), nullable=False),
    Column("status", String(16), nullable=False),
    Column("title", String(200), nullable=False, default=""),
    Column("payload", Text().with_variant(MEDIUMTEXT(), "mysql")),
    Column("result", Text().with_variant(MEDIUMTEXT(), "mysql")),
    Column("approver", String(64)),
    Column("comment", String(500)),
    Column("error", String(500)),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
    Column("expires_at", DateTime),
    Column("finished_at", DateTime),
    Index("idx_agent_status_updated", "agent_id", "status", "updated_at"),
    Index("idx_status_expires", "status", "expires_at"),
    comment="Agent 运行与人工审批（HITL）状态机",
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
    mysql_collate="utf8mb4_general_ci",
)

SCHEMA_HINT = f"（请先执行 docs/mysql/{RUN_TABLE}.sql 建表）"


class _Unset:
    """区分「不改这个字段」与「把字段改成 NULL」。"""


_UNSET = _Unset()
MaybeTime = datetime | None | _Unset


# ---- 时间与 JSON 转换 -------------------------------------------------------
def _naive_utc(value: datetime | None) -> datetime | None:
    """带时区的时间 → 入库用的 UTC 朴素时间。"""
    if value is None:
        return None
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


def _aware_utc(value: datetime | None) -> datetime | None:
    """库里的朴素时间 → 带 UTC 时区的时间。"""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _dump_json(value: dict[str, Any] | None) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


def _load_json(raw: Any) -> dict[str, Any] | None:
    """坏 JSON 当成空对象：一条运行记录损坏不该让整个列表接口 500。"""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _record_from_row(row: dict[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=str(row["run_id"]),
        agent_id=str(row["agent_id"]),
        thread_id=str(row["thread_id"]),
        status=str(row["status"]),  # type: ignore[arg-type]
        title=str(row.get("title") or ""),
        payload=_load_json(row.get("payload")) or {},
        result=_load_json(row.get("result")),
        approver=row.get("approver") or None,
        comment=row.get("comment") or None,
        error=row.get("error") or None,
        created_at=_aware_utc(row.get("created_at")) or datetime.now(UTC),
        updated_at=_aware_utc(row.get("updated_at")) or datetime.now(UTC),
        expires_at=_aware_utc(row.get("expires_at")),
        finished_at=_aware_utc(row.get("finished_at")),
    )


def _db_error(action: str, exc: SQLAlchemyError) -> DatabaseError:
    raw = str(getattr(exc, "orig", exc)).strip().splitlines()
    detail = raw[0][:300] if raw else exc.__class__.__name__
    lowered = detail.lower()
    hints = ""
    if "doesn't exist" in lowered or "does not exist" in lowered or "1146" in lowered:
        hints = SCHEMA_HINT
    return DatabaseError(f"{action}运行记录失败：{detail}{hints}")


# ---- 存储接口 ---------------------------------------------------------------
class RunStore(Protocol):
    """运行记录的读写接口：生产用 MySQL，单元测试用内存实现。"""

    def create(self, record: RunRecord) -> RunRecord:
        """新建一条运行记录（run_id 由调用方生成）。"""
        ...

    def get(self, run_id: str) -> RunRecord | None:
        """按 run_id 读取；不存在时返回 `None`。"""
        ...

    def list_runs(
        self,
        *,
        agent_id: str | None = None,
        statuses: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        """按最近更新时间倒序返回运行记录。"""
        ...

    def transition(
        self,
        run_id: str,
        *,
        expected: Sequence[str],
        status: RunStatus,
        approver: str | None = None,
        comment: str | None = None,
        error: str | None = None,
        payload: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        expires_at: MaybeTime = _UNSET,
        finished_at: MaybeTime = _UNSET,
        updated_at: datetime | None = None,
    ) -> RunRecord | None:
        """条件更新状态：只有当前状态在 ``expected`` 里才生效，否则返回 `None`。"""
        ...

    def due(self, now: datetime) -> list[RunRecord]:
        """返回已经过了审批截止时间、还停在等待状态的运行。"""
        ...


class MySQLRunStore:
    """`RunStore` 的 MySQL 实现（SQLAlchemy Core + PyMySQL）。"""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> MySQLRunStore:
        """用配置里的连接串建（或复用）连接池；未配置 `MYSQL_URL` 时抛 `DatabaseError`。"""
        try:
            return cls(get_engine(settings))
        except ConfigurationError as exc:
            raise DatabaseError(str(exc)) from exc

    def create(self, record: RunRecord) -> RunRecord:
        now = _naive_utc(record.created_at) or datetime.now(UTC).replace(tzinfo=None)
        try:
            with self._engine.begin() as connection:
                connection.execute(
                    insert(agent_run).values(
                        run_id=record.run_id,
                        agent_id=record.agent_id,
                        thread_id=record.thread_id,
                        status=record.status,
                        title=record.title,
                        payload=_dump_json(record.payload),
                        result=_dump_json(record.result),
                        approver=record.approver,
                        comment=record.comment,
                        error=record.error,
                        created_at=now,
                        updated_at=_naive_utc(record.updated_at) or now,
                        expires_at=_naive_utc(record.expires_at),
                        finished_at=_naive_utc(record.finished_at),
                    )
                )
        except SQLAlchemyError as exc:
            raise _db_error("创建", exc) from exc
        return record

    def get(self, run_id: str) -> RunRecord | None:
        statement = select(agent_run).where(agent_run.c.run_id == run_id)
        try:
            with self._engine.connect() as connection:
                row = connection.execute(statement).mappings().first()
        except SQLAlchemyError as exc:
            raise _db_error("读取", exc) from exc
        return None if row is None else _record_from_row(dict(row))

    def list_runs(
        self,
        *,
        agent_id: str | None = None,
        statuses: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        statement = select(agent_run)
        if agent_id is not None:
            statement = statement.where(agent_run.c.agent_id == agent_id)
        if statuses:
            statement = statement.where(agent_run.c.status.in_(list(statuses)))
        statement = statement.order_by(
            agent_run.c.updated_at.desc(), agent_run.c.created_at.desc()
        ).limit(max(1, limit))
        try:
            with self._engine.connect() as connection:
                rows = connection.execute(statement).mappings().all()
        except SQLAlchemyError as exc:
            raise _db_error("读取", exc) from exc
        return [_record_from_row(dict(row)) for row in rows]

    def transition(
        self,
        run_id: str,
        *,
        expected: Sequence[str],
        status: RunStatus,
        approver: str | None = None,
        comment: str | None = None,
        error: str | None = None,
        payload: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        expires_at: MaybeTime = _UNSET,
        finished_at: MaybeTime = _UNSET,
        updated_at: datetime | None = None,
    ) -> RunRecord | None:
        values: dict[str, Any] = {
            "status": status,
            "updated_at": _naive_utc(updated_at or datetime.now(UTC)),
        }
        if approver is not None:
            values["approver"] = approver
        if comment is not None:
            values["comment"] = comment
        if error is not None:
            values["error"] = error
        if payload is not None:
            values["payload"] = _dump_json(payload)
        if result is not None:
            values["result"] = _dump_json(result)
        if not isinstance(expires_at, _Unset):
            values["expires_at"] = _naive_utc(expires_at)
        if not isinstance(finished_at, _Unset):
            values["finished_at"] = _naive_utc(finished_at)

        statement = (
            update(agent_run)
            .where(agent_run.c.run_id == run_id, agent_run.c.status.in_(list(expected)))
            .values(**values)
        )
        try:
            with self._engine.begin() as connection:
                changed = connection.execute(statement).rowcount
                if not changed:
                    return None
                row = (
                    connection.execute(select(agent_run).where(agent_run.c.run_id == run_id))
                    .mappings()
                    .one()
                )
                return _record_from_row(dict(row))
        except SQLAlchemyError as exc:
            raise _db_error("更新", exc) from exc

    def due(self, now: datetime) -> list[RunRecord]:
        moment = _naive_utc(now) or datetime.now(UTC).replace(tzinfo=None)
        statement = (
            select(agent_run)
            .where(
                agent_run.c.status.in_(["pending", "escalated"]),
                agent_run.c.expires_at.is_not(None),
                agent_run.c.expires_at <= moment,
            )
            .order_by(agent_run.c.expires_at)
        )
        try:
            with self._engine.connect() as connection:
                rows = connection.execute(statement).mappings().all()
        except SQLAlchemyError as exc:
            raise _db_error("读取", exc) from exc
        return [_record_from_row(dict(row)) for row in rows]

    def count(self, *, agent_id: str | None = None) -> int:
        """运行总数（概览页展示用）。"""
        statement = select(func.count()).select_from(agent_run)
        if agent_id is not None:
            statement = statement.where(agent_run.c.agent_id == agent_id)
        try:
            with self._engine.connect() as connection:
                return int(connection.execute(statement).scalar_one())
        except SQLAlchemyError as exc:
            raise _db_error("统计", exc) from exc


class InMemoryRunStore:
    """`RunStore` 的内存实现：单元测试与本地无库演示用（重启即丢失）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, RunRecord] = {}

    def create(self, record: RunRecord) -> RunRecord:
        with self._lock:
            if record.run_id in self._runs:
                raise DatabaseError(f"运行 {record.run_id} 已存在")
            self._runs[record.run_id] = record
            return record

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(
        self,
        *,
        agent_id: str | None = None,
        statuses: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        with self._lock:
            items = [
                record
                for record in self._runs.values()
                if (agent_id is None or record.agent_id == agent_id)
                and (not statuses or record.status in statuses)
            ]
        items.sort(key=lambda item: (item.updated_at, item.created_at), reverse=True)
        return items[: max(1, limit)]

    def transition(
        self,
        run_id: str,
        *,
        expected: Sequence[str],
        status: RunStatus,
        approver: str | None = None,
        comment: str | None = None,
        error: str | None = None,
        payload: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        expires_at: MaybeTime = _UNSET,
        finished_at: MaybeTime = _UNSET,
        updated_at: datetime | None = None,
    ) -> RunRecord | None:
        with self._lock:
            current = self._runs.get(run_id)
            if current is None or current.status not in expected:
                return None
            changes: dict[str, Any] = {
                "status": status,
                "updated_at": updated_at or datetime.now(UTC),
            }
            if approver is not None:
                changes["approver"] = approver
            if comment is not None:
                changes["comment"] = comment
            if error is not None:
                changes["error"] = error
            if payload is not None:
                changes["payload"] = payload
            if result is not None:
                changes["result"] = result
            if not isinstance(expires_at, _Unset):
                changes["expires_at"] = expires_at
            if not isinstance(finished_at, _Unset):
                changes["finished_at"] = finished_at
            updated = replace(current, **changes)
            self._runs[run_id] = updated
            return updated

    def due(self, now: datetime) -> list[RunRecord]:
        with self._lock:
            items = [
                record
                for record in self._runs.values()
                if record.status in {"pending", "escalated"}
                and record.expires_at is not None
                and record.expires_at <= now
            ]
        return sorted(items, key=lambda item: item.expires_at or now)
