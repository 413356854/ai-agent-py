"""MySQL connection pool and a single read-only health check."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.db.dsn import MySQLTarget
from ai_agent_py.errors import ConfigurationError, DatabaseError

logger = logging.getLogger(__name__)

DatabaseStatus = Literal["ok", "down", "not_configured"]

_ENGINES: list[Engine] = []


class DatabaseHealth(BaseModel):
    """What `/api/db/health` and `ai-agent-py db check` report."""

    configured: bool
    ok: bool
    status: DatabaseStatus
    host: str | None = None
    port: int | None = None
    database: str | None = None
    user: str | None = None
    server_version: str | None = None
    session_time_zone: str | None = None
    table_count: int | None = None
    latency_ms: float | None = None
    pool: dict[str, int] | None = None
    error: str | None = None


def _pool_stats(engine: Engine) -> dict[str, int] | None:
    pool = engine.pool
    stats: dict[str, int] = {}
    # `overflow()` is an internal counter (it starts at -pool_size), so it is
    # deliberately not reported; checkedin/checkedout already show utilisation.
    for name in ("size", "checkedin", "checkedout"):
        method = getattr(pool, name, None)
        if not callable(method):
            continue
        try:
            stats[name] = int(method())
        except (TypeError, ValueError):  # pragma: no cover - pool implementation specific
            continue
    return stats or None


def _translate_error(exc: BaseException, target: MySQLTarget) -> str:
    raw = str(exc)
    lowered = raw.lower()
    where = f"{target.host}:{target.port}/{target.database}"

    if "access denied" in lowered:
        return "用户名或密码错误（检查 MYSQL_USERNAME / MYSQL_PASSWORD）"
    if "unknown database" in lowered:
        return f"数据库不存在：{target.database}"
    if "can't connect" in lowered or "connection refused" in lowered or "10061" in lowered:
        return f"无法连接 MySQL {where}，请确认 MySQL 服务已启动、端口未被占用"
    if "timed out" in lowered or "timeout" in lowered:
        return f"连接 MySQL {where} 超时（{target.connect_timeout}s）"
    if "cryptography" in lowered or "caching_sha2" in lowered or "rsa" in lowered:
        return "MySQL 8 认证需要 cryptography（已在项目依赖中；请确认虚拟环境完整）"
    return raw.strip()[:500] or raw.__class__.__name__


@lru_cache(maxsize=4)
def _build_engine(
    dsn: str,
    pool_size: int,
    max_overflow: int,
    pool_recycle: int,
    pool_timeout: float,
) -> Engine:
    engine = create_engine(
        dsn,
        pool_pre_ping=True,
        pool_recycle=pool_recycle,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        future=True,
    )
    _ENGINES.append(engine)
    logger.debug("MySQL pool created (size=%d, overflow=%d)", pool_size, max_overflow)
    return engine


def get_engine(settings: Settings | None = None) -> Engine:
    """Return the process-wide pool for the configured database.

    Raises `ConfigurationError` when `MYSQL_URL` is not set, and `DatabaseError`
    when the URL itself cannot be parsed.
    """
    cfg = settings or get_settings()
    try:
        target = cfg.mysql_target
    except ValueError as exc:
        raise DatabaseError(f"MySQL URL 配置无效：{exc}") from exc
    if target is None:
        raise ConfigurationError(
            "未配置 MySQL。请在 .env 中设置 MYSQL_URL / MYSQL_USERNAME / MYSQL_PASSWORD。"
        )
    return _build_engine(
        target.dsn,
        cfg.mysql_pool_size,
        cfg.mysql_max_overflow,
        cfg.mysql_pool_recycle,
        float(cfg.mysql_connect_timeout),
    )


def check_database(settings: Settings | None = None) -> DatabaseHealth:
    """Open one pooled connection and report what the server says.

    Never raises: the caller is a health endpoint, and a down database must be
    reported as data rather than by breaking the whole application.
    """
    cfg = settings or get_settings()
    try:
        target = cfg.mysql_target
    except ValueError as exc:
        return DatabaseHealth(
            configured=True,
            ok=False,
            status="down",
            error=f"MySQL URL 配置无效：{exc}",
        )
    if target is None:
        return DatabaseHealth(configured=False, ok=False, status="not_configured")

    started = time.perf_counter()
    try:
        engine = get_engine(cfg)
        with engine.connect() as connection:
            version = connection.execute(text("SELECT VERSION()")).scalar_one()
            database = connection.execute(text("SELECT DATABASE()")).scalar_one()
            user = connection.execute(text("SELECT CURRENT_USER()")).scalar_one()
            zone = connection.execute(text("SELECT @@session.time_zone")).scalar_one()
            table_count = connection.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE()"
                )
            ).scalar_one()
    except ConfigurationError as exc:
        return DatabaseHealth(configured=False, ok=False, status="not_configured", error=str(exc))
    except DatabaseError as exc:
        return DatabaseHealth(
            configured=True,
            ok=False,
            status="down",
            host=target.host,
            port=target.port,
            database=target.database,
            user=target.username,
            error=str(exc),
        )
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("MySQL 健康检查失败: %s", exc)
        return DatabaseHealth(
            configured=True,
            ok=False,
            status="down",
            host=target.host,
            port=target.port,
            database=target.database,
            user=target.username,
            error=_translate_error(exc, target),
        )

    return DatabaseHealth(
        configured=True,
        ok=True,
        status="ok",
        host=target.host,
        port=target.port,
        database=str(database or target.database),
        user=str(user or target.username),
        server_version=str(version),
        session_time_zone=str(zone),
        table_count=int(table_count),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        pool=_pool_stats(engine),
    )


def dispose_engines() -> None:
    """Close every pooled connection (tests and graceful shutdown)."""
    for engine in _ENGINES:
        engine.dispose()
    _ENGINES.clear()
    _build_engine.cache_clear()
