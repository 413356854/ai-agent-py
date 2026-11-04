"""Translate a JDBC style MySQL URL into a SQLAlchemy/PyMySQL URL.

The Java project configures the pool with::

    jdbc:mysql://localhost:3306/db_py_agent?useUnicode=true&characterEncoding=utf8
        &connectionTimeZone=Asia/Shanghai&allowPublicKeyRetrieval=true&useSSL=false

Only the options that matter to PyMySQL are carried over:

* ``characterEncoding`` -> ``charset`` (``utf8`` becomes ``utf8mb4``),
* ``connectionTimeZone`` -> ``init_command=SET time_zone='+08:00'``.

The JDBC ``connectTimeout`` is in milliseconds and is not read here; the
project uses ``MYSQL_CONNECT_TIMEOUT`` (seconds) for both connecting and
waiting for a pooled connection. The JDBC-only switches (``useUnicode``,
``allowPublicKeyRetrieval``, ``useSSL=false``) are accepted and ignored:
PyMySQL decodes text as UTF-8 by default, retrieves the server's RSA key
automatically for ``caching_sha2_password``, and does not enable TLS unless
asked to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import parse_qsl, unquote, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.engine import URL

DEFAULT_PORT = 3306
DEFAULT_CONNECT_TIMEOUT = 5

_ACCEPTED_SCHEMES = frozenset({"mysql", "mysql+pymysql"})
_OFFSET = re.compile(r"^[+-]\d{2}:\d{2}$")


@dataclass(frozen=True, slots=True)
class MySQLTarget:
    """Everything needed to open (and describe) the MySQL pool."""

    host: str
    port: int
    database: str
    username: str
    password: str = field(repr=False)
    dsn: str = field(repr=False)
    safe_dsn: str
    charset: str
    timezone: str | None
    connect_timeout: int


def _normalise_charset(name: str) -> str:
    value = name.strip().lower().replace("-", "")
    if value in {"utf8", "utf8mb3", "utf8mb4", "unicode"}:
        return "utf8mb4"
    return value or "utf8mb4"


def _timezone_offset(name: str) -> str | None:
    """Return ``+08:00`` for ``Asia/Shanghai``; ``None`` when not determinable."""
    text = name.strip()
    if not text or text.upper() in {"LOCAL", "SERVER"}:
        return None
    if _OFFSET.match(text):
        return text
    try:
        offset = datetime.now(ZoneInfo(text)).utcoffset()
    except (ZoneInfoNotFoundError, ValueError):
        return None
    if offset is None:
        return None
    seconds = int(offset.total_seconds())
    sign = "+" if seconds >= 0 else "-"
    seconds = abs(seconds)
    return f"{sign}{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"


def _query_options(raw_query: str) -> dict[str, str]:
    return {key.lower(): value for key, value in parse_qsl(raw_query, keep_blank_values=True)}


def parse_mysql_url(
    raw: str,
    *,
    username: str | None = None,
    password: str | None = None,
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
) -> MySQLTarget:
    """Parse a JDBC / SQLAlchemy MySQL URL into a validated target.

    Raises `ValueError` with a Chinese message when a mandatory part is
    missing, so `Settings` can fail fast instead of surfacing a broken pool
    later.
    """
    value = raw.strip()
    if not value:
        raise ValueError("MySQL URL 不能为空")
    if value.lower().startswith("jdbc:"):
        value = value[5:]

    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    if scheme not in _ACCEPTED_SCHEMES:
        raise ValueError(
            f"不支持的 MySQL URL 协议: {parsed.scheme or '(空)'}，"
            "请使用 jdbc:mysql:// 或 mysql+pymysql://"
        )

    host = parsed.hostname
    if not host:
        raise ValueError("MySQL URL 缺少主机名")
    try:
        port = parsed.port or DEFAULT_PORT
    except ValueError as exc:
        raise ValueError(f"MySQL URL 端口不合法: {parsed.port}") from exc

    database = unquote(parsed.path.lstrip("/"))
    if not database:
        raise ValueError("MySQL URL 缺少数据库名（例如 /db_py_agent）")

    user = username or (unquote(parsed.username) if parsed.username else None)
    if not user:
        raise ValueError("缺少 MYSQL_USERNAME（URL 里也没有用户名）")
    secret = password
    if secret is None and parsed.password is not None:
        secret = unquote(parsed.password)
    secret = secret or ""

    options = _query_options(parsed.query)
    charset = _normalise_charset(options.get("characterencoding", "utf8mb4"))
    timezone_name = options.get("connectiontimezone") or options.get("servertimezone")
    offset = _timezone_offset(timezone_name) if timezone_name else None

    query: dict[str, str] = {
        "charset": charset,
        "connect_timeout": str(connect_timeout),
    }
    if offset:
        query["init_command"] = f"SET time_zone = '{offset}'"

    url = URL.create(
        drivername="mysql+pymysql",
        username=user,
        password=secret,
        host=host,
        port=port,
        database=database,
        query=query,
    )
    return MySQLTarget(
        host=host,
        port=port,
        database=database,
        username=user,
        password=secret,
        dsn=url.render_as_string(hide_password=False),
        safe_dsn=url.render_as_string(hide_password=True),
        charset=charset,
        timezone=timezone_name or None,
        connect_timeout=connect_timeout,
    )
