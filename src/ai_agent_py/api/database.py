"""Database diagnostics endpoint (MySQL connection only)."""

from __future__ import annotations

from fastapi import APIRouter

from ai_agent_py.config import get_settings
from ai_agent_py.db.health import DatabaseHealth, check_database

router = APIRouter(prefix="/api/db", tags=["database"])


@router.get("/health", response_model=DatabaseHealth, summary="MySQL 连接与库信息")
def database_health() -> DatabaseHealth:
    """实际建立一次连接，返回服务端版本、库名、表数与耗时。"""
    return check_database(get_settings())
