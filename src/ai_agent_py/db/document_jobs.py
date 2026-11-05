"""文档解析任务的 MySQL 持久化（``tb_document_parse_job``）。

原实现把全部任务写进 ``data/parsed/jobs.json``，每次状态变化都重写整份文件
（O(n) 且没有并发保护）；现在改为一行一个任务的 MySQL 表：

* `DocumentService` 仍在内存里保留任务缓存（前端每 2.5s 轮询一次，走内存最快），
  但每次状态变化都按主键 upsert 到数据库，重启后从数据库整体载入；
* 建表脚本见 ``docs/mysql/tb_document_parse_job.sql``，本模块的 `document_parse_job`
  是它的 SQLAlchemy 镜像，两者必须保持一致；
* 时间列一律 UTC 往返（库里的 DATETIME 不带时区），保证 API 输出的 ``Z`` 时间不变。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Engine,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    delete,
    select,
)
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.exc import SQLAlchemyError

from ai_agent_py.config import Settings
from ai_agent_py.db.health import get_engine
from ai_agent_py.errors import ConfigurationError, DatabaseError
from ai_agent_py.mineru.models import (
    JobState,
    ParseJob,
    ParseOptions,
    RagIngestInfo,
)

TABLE_NAME = "tb_document_parse_job"

_metadata = MetaData()

#: ``docs/mysql/tb_document_parse_job.sql`` 的 Python 镜像（类型/可空性必须一致）。
document_parse_job = Table(
    TABLE_NAME,
    _metadata,
    Column("id", String(64), primary_key=True),
    Column("file_name", String(255), nullable=False),
    Column("file_size", BigInteger, nullable=False),
    Column("suffix", String(16), nullable=False),
    Column("state", String(16), nullable=False),
    Column("model_version", String(32), nullable=False),
    Column("language", String(32), nullable=False),
    Column("enable_formula", Boolean, nullable=False),
    Column("enable_table", Boolean, nullable=False),
    Column("is_ocr", Boolean, nullable=False),
    Column("page_ranges", String(64)),
    Column("batch_id", String(64)),
    Column("err_msg", String(500), nullable=False),
    Column("extracted_pages", Integer, nullable=False),
    Column("total_pages", Integer, nullable=False),
    Column("source_file", String(255)),
    Column("markdown_file", String(255)),
    Column("zip_file", String(255)),
    Column("markdown_chars", Integer, nullable=False),
    Column("asset_count", Integer, nullable=False),
    Column("rag_ingest_chunks", Integer),
    Column("rag_ingest_characters", Integer),
    Column("rag_ingest_collection", String(128)),
    Column("rag_ingested_at", DateTime),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
    Column("started_at", DateTime),
    Column("finished_at", DateTime),
    Column("elapsed_seconds", Numeric(10, 2)),
    comment="文档解析任务记录（原 data/parsed/jobs.json）",
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
    mysql_collate="utf8mb4_general_ci",
)


# ---- 时间与类型转换 -------------------------------------------------------
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


def to_params(job: ParseJob) -> dict[str, Any]:
    """`ParseJob` → 一行记录的列值（嵌套的 options / rag_ingest 摊平成列）。"""
    options = job.options
    rag = job.rag_ingest
    return {
        "id": job.id,
        "file_name": job.file_name,
        "file_size": job.file_size,
        "suffix": job.suffix,
        "state": job.state.value,
        "model_version": options.model_version,
        "language": options.language,
        "enable_formula": options.enable_formula,
        "enable_table": options.enable_table,
        "is_ocr": options.is_ocr,
        "page_ranges": options.page_ranges,
        "batch_id": job.batch_id,
        "err_msg": job.err_msg,
        "extracted_pages": job.extracted_pages,
        "total_pages": job.total_pages,
        "source_file": job.source_file,
        "markdown_file": job.markdown_file,
        "zip_file": job.zip_file,
        "markdown_chars": job.markdown_chars,
        "asset_count": job.asset_count,
        "rag_ingest_chunks": rag.chunks if rag else None,
        "rag_ingest_characters": rag.characters if rag else None,
        "rag_ingest_collection": rag.collection if rag else None,
        "rag_ingested_at": _naive_utc(rag.ingested_at) if rag else None,
        "created_at": _naive_utc(job.created_at),
        "updated_at": _naive_utc(job.updated_at),
        "started_at": _naive_utc(job.started_at),
        "finished_at": _naive_utc(job.finished_at),
        "elapsed_seconds": (
            None if job.elapsed_seconds is None else Decimal(str(job.elapsed_seconds))
        ),
    }


def from_row(row: Mapping[str, Any]) -> ParseJob:
    """一行记录 → `ParseJob`（`to_params` 的逆操作）。"""
    rag_ingest: RagIngestInfo | None = None
    if row["rag_ingested_at"] is not None:
        rag_ingest = RagIngestInfo(
            chunks=int(row["rag_ingest_chunks"] or 0),
            characters=int(row["rag_ingest_characters"] or 0),
            collection=str(row["rag_ingest_collection"] or ""),
            ingested_at=_aware_utc(row["rag_ingested_at"]) or datetime.now(UTC),
        )
    elapsed = row["elapsed_seconds"]
    return ParseJob(
        id=str(row["id"]),
        file_name=str(row["file_name"]),
        file_size=int(row["file_size"]),
        suffix=str(row["suffix"]),
        state=JobState(str(row["state"])),
        options=ParseOptions(
            model_version=str(row["model_version"]),
            language=str(row["language"]),
            enable_formula=bool(row["enable_formula"]),
            enable_table=bool(row["enable_table"]),
            is_ocr=bool(row["is_ocr"]),
            page_ranges=row["page_ranges"] or None,
        ),
        created_at=_aware_utc(row["created_at"]) or datetime.now(UTC),
        updated_at=_aware_utc(row["updated_at"]) or datetime.now(UTC),
        started_at=_aware_utc(row["started_at"]),
        finished_at=_aware_utc(row["finished_at"]),
        elapsed_seconds=None if elapsed is None else float(elapsed),
        batch_id=row["batch_id"] or None,
        err_msg=str(row["err_msg"] or ""),
        extracted_pages=int(row["extracted_pages"]),
        total_pages=int(row["total_pages"]),
        source_file=row["source_file"] or None,
        markdown_file=row["markdown_file"] or None,
        zip_file=row["zip_file"] or None,
        markdown_chars=int(row["markdown_chars"]),
        asset_count=int(row["asset_count"]),
        rag_ingest=rag_ingest,
    )


# ---- 存储接口 -------------------------------------------------------------
class JobStore(Protocol):
    """任务记录的读写接口：生产用 MySQL，单元测试用内存实现。"""

    def load_all(self) -> list[ParseJob]:
        """按创建时间倒序返回全部任务。"""
        ...

    def get(self, job_id: str) -> ParseJob | None:
        """按主键读取单个任务；不存在时返回 `None`。"""
        ...

    def save(self, job: ParseJob) -> None:
        """按主键 upsert 一条任务记录。"""
        ...

    def delete(self, job_id: str) -> None:
        """删除一条任务记录（不存在时静默返回）。"""
        ...


def _db_error(action: str, exc: SQLAlchemyError) -> DatabaseError:
    raw = str(getattr(exc, "orig", exc)).strip().splitlines()
    detail = raw[0][:300] if raw else exc.__class__.__name__
    hint = ""
    if "doesn't exist" in detail.lower() or "does not exist" in detail.lower():
        hint = f"（请先执行 docs/mysql/{TABLE_NAME}.sql 建表）"
    return DatabaseError(f"{action} {TABLE_NAME} 失败：{detail}{hint}")


class MySQLJobStore:
    """`tb_document_parse_job` 的读写实现（SQLAlchemy Core + PyMySQL）。"""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> MySQLJobStore:
        """用配置里的连接串建（或复用）连接池。

        未配置 `MYSQL_URL` 时统一抛 `DatabaseError`，让上层只需要处理一种
        「任务记录库不可用」的错误。
        """
        try:
            return cls(get_engine(settings))
        except ConfigurationError as exc:
            raise DatabaseError(str(exc)) from exc

    def load_all(self) -> list[ParseJob]:
        statement = select(document_parse_job).order_by(document_parse_job.c.created_at.desc())
        try:
            with self._engine.connect() as connection:
                rows = connection.execute(statement).mappings().all()
        except SQLAlchemyError as exc:
            raise _db_error("读取", exc) from exc
        return [from_row(dict(row)) for row in rows]

    def get(self, job_id: str) -> ParseJob | None:
        statement = select(document_parse_job).where(document_parse_job.c.id == job_id)
        try:
            with self._engine.connect() as connection:
                row = connection.execute(statement).mappings().first()
        except SQLAlchemyError as exc:
            raise _db_error("读取", exc) from exc
        return None if row is None else from_row(dict(row))

    def load_file_names(self, doc_ids: Sequence[str]) -> dict[str, str]:
        """按主键批量读取文件名（只查两列，供检索结果回填来源）。

        ``rag_chunk`` 里只有 ``doc_id``，检索后拼引用时用这个方法一次查完，
        避免为了几个文件名把整张任务表加载成 `ParseJob`。
        """
        ids = [doc_id for doc_id in dict.fromkeys(doc_ids) if doc_id]
        if not ids:
            return {}
        statement = select(document_parse_job.c.id, document_parse_job.c.file_name).where(
            document_parse_job.c.id.in_(ids)
        )
        try:
            with self._engine.connect() as connection:
                rows = connection.execute(statement).mappings().all()
        except SQLAlchemyError as exc:
            raise _db_error("读取", exc) from exc
        return {str(row["id"]): str(row["file_name"]) for row in rows}

    def save(self, job: ParseJob) -> None:
        params = to_params(job)
        statement = mysql_insert(document_parse_job).values(**params)
        # 主键冲突时覆盖其余列：创建与后续每次状态更新共用一条语句。
        statement = statement.on_duplicate_key_update(
            **{name: statement.inserted[name] for name in params if name != "id"}
        )
        try:
            with self._engine.begin() as connection:
                connection.execute(statement)
        except SQLAlchemyError as exc:
            raise _db_error("写入", exc) from exc

    def delete(self, job_id: str) -> None:
        try:
            with self._engine.begin() as connection:
                connection.execute(
                    delete(document_parse_job).where(document_parse_job.c.id == job_id)
                )
        except SQLAlchemyError as exc:
            raise _db_error("删除", exc) from exc
