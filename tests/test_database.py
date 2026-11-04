"""MySQL URL translation, health reporting and the diagnostic entry points."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from typer.testing import CliRunner

import ai_agent_py.api.app as api_app
import ai_agent_py.api.database as db_api
import ai_agent_py.cli as cli
import ai_agent_py.db.health as health
from ai_agent_py.config import Settings
from ai_agent_py.db.dsn import parse_mysql_url
from ai_agent_py.db.health import DatabaseHealth, check_database, dispose_engines

JDBC_URL = (
    "jdbc:mysql://localhost:3306/db_py_agent"
    "?useUnicode=true&characterEncoding=utf8&connectionTimeZone=Asia/Shanghai"
    "&allowPublicKeyRetrieval=true&useSSL=false"
)


@pytest.fixture(autouse=True)
def _dispose_pools() -> Any:
    dispose_engines()
    yield
    dispose_engines()


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "mysql_url": JDBC_URL,
        "mysql_username": "admin",
        "mysql_password": "admin",
    }
    values.update(overrides)
    return Settings(**values)


def test_jdbc_url_is_translated_to_pymysql() -> None:
    cfg = _settings()
    target = cfg.mysql_target
    assert target is not None
    assert (target.host, target.port, target.database) == ("localhost", 3306, "db_py_agent")
    assert target.username == "admin"
    assert target.password == "admin"

    parsed = make_url(cfg.mysql_dsn or "")
    assert parsed.drivername == "mysql+pymysql"
    query = parsed.query
    assert query["charset"] == "utf8mb4"
    assert query["connect_timeout"] == "5"
    assert query["init_command"] == "SET time_zone = '+08:00'"


def test_sqlalchemy_url_and_special_password() -> None:
    cfg = Settings(
        _env_file=None,
        mysql_url="mysql://admin:p%40ss@localhost:3306/db_py_agent",
    )
    target = cfg.mysql_target
    assert target is not None
    assert target.password == "p@ss"
    assert "p@ss" not in repr(target)
    assert "p%40ss" not in repr(target)
    assert "p%40ss" in (cfg.mysql_dsn or "")
    assert "p@ss" not in (cfg.mysql_safe_dsn or "")
    assert "***" in (cfg.mysql_safe_dsn or "")


def test_explicit_credentials_override_url_user() -> None:
    cfg = Settings(
        _env_file=None,
        mysql_url="mysql://someone:secret@localhost:3306/db_py_agent",
        mysql_username="admin",
        mysql_password="admin",
    )
    target = cfg.mysql_target
    assert target is not None
    assert (target.username, target.password) == ("admin", "admin")


def test_fixed_offset_timezone_is_kept() -> None:
    cfg = Settings(
        _env_file=None,
        mysql_url="mysql://admin:admin@localhost:3306/db_py_agent?connectionTimeZone=%2B09:00",
    )
    assert "init_command" in (cfg.mysql_dsn or "")
    assert "%2B09%3A00" in (cfg.mysql_dsn or "")


def test_blank_mysql_is_not_configured() -> None:
    cfg = Settings(_env_file=None)
    assert cfg.mysql_target is None
    assert cfg.mysql_safe_dsn is None
    assert cfg.describe()["mysql"] == "未配置"

    result = check_database(cfg)
    assert result.configured is False
    assert result.ok is False
    assert result.status == "not_configured"


def test_missing_username_and_unsupported_scheme_fail_fast() -> None:
    with pytest.raises(ValueError, match="MYSQL_USERNAME"):
        parse_mysql_url("jdbc:mysql://localhost:3306/db_py_agent")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, mysql_url="postgresql://localhost/db_py_agent")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, mysql_url="jdbc:mysql://localhost:3306/db_py_agent")


def test_describe_masks_mysql_password() -> None:
    described = _settings().describe()
    assert described["mysql"] == "admin@localhost:3306/db_py_agent"
    assert described["mysql_password"].startswith("set")
    assert "admin" not in described["mysql_password"]


def test_mysql_pool_settings_are_validated() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, mysql_pool_size=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, mysql_connect_timeout=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, mysql_max_overflow=-2)


class _Result:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one(self) -> Any:
        return self._value


class _Connection:
    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, statement: Any) -> _Result:
        sql = str(statement)
        if "information_schema.tables" in sql:
            return _Result(23)
        if "VERSION()" in sql:
            return _Result("8.0.17")
        if "DATABASE()" in sql:
            return _Result("db_py_agent")
        if "CURRENT_USER()" in sql:
            return _Result("admin@localhost")
        if "@@session.time_zone" in sql:
            return _Result("+08:00")
        raise AssertionError(f"unexpected SQL: {sql}")


class _Pool:
    def size(self) -> int:
        return 5

    def checkedin(self) -> int:
        return 2

    def checkedout(self) -> int:
        return 1


class _Engine:
    pool = _Pool()

    def connect(self) -> _Connection:
        return _Connection()


def test_health_reports_server_information(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "get_engine", lambda cfg: _Engine())
    result = check_database(_settings())

    assert result.ok is True
    assert result.status == "ok"
    assert result.server_version == "8.0.17"
    assert result.database == "db_py_agent"
    assert result.user == "admin@localhost"
    assert result.session_time_zone == "+08:00"
    assert result.table_count == 23
    assert result.pool == {"size": 5, "checkedin": 2, "checkedout": 1}
    assert result.latency_ms is not None and result.latency_ms >= 0


def test_health_translates_access_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = OperationalError(
        "SELECT 1",
        {},
        Exception("Access denied for user 'admin'@'localhost' (using password: YES)"),
    )
    monkeypatch.setattr(health, "get_engine", lambda cfg: (_ for _ in ()).throw(failure))
    result = check_database(_settings())
    assert result.ok is False
    assert result.status == "down"
    assert "用户名或密码错误" in (result.error or "")


def test_api_database_health(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    fake = DatabaseHealth(
        configured=True,
        ok=True,
        status="ok",
        host="localhost",
        port=3306,
        database="db_py_agent",
        user="admin@localhost",
        server_version="8.0.17",
        table_count=23,
        latency_ms=3.2,
    )
    monkeypatch.setattr(db_api, "check_database", lambda cfg: fake)
    body = TestClient(api_app.app).get("/api/db/health").json()
    assert body["status"] == "ok"
    assert body["server_version"] == "8.0.17"
    assert body["table_count"] == 23


def test_health_endpoint_includes_database(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    fake = DatabaseHealth(configured=False, ok=False, status="not_configured")
    monkeypatch.setattr(api_app, "get_settings", lambda: settings)
    monkeypatch.setattr(api_app, "check_database", lambda cfg: fake)
    body = TestClient(api_app.app).get("/health").json()
    assert body["database"]["status"] == "not_configured"


def test_cli_db_check(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    fake = DatabaseHealth(
        configured=True,
        ok=True,
        status="ok",
        host="localhost",
        port=3306,
        database="db_py_agent",
        user="admin@localhost",
        server_version="8.0.17",
        table_count=23,
        latency_ms=1.5,
        pool={"size": 5, "checkedin": 1, "checkedout": 0},
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "check_database", lambda cfg: fake)

    result = CliRunner().invoke(cli.app, ["db", "check"])
    assert result.exit_code == 0, result.output
    assert "8.0.17" in result.output
    assert "db_py_agent" in result.output
    assert "MySQL 连接正常" in result.output


def test_cli_db_check_fails_when_down(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    fake = DatabaseHealth(
        configured=True,
        ok=False,
        status="down",
        error="无法连接 MySQL",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "check_database", lambda cfg: fake)

    result = CliRunner().invoke(cli.app, ["db", "check"])
    assert result.exit_code == 1
    assert "无法连接 MySQL" in result.output
