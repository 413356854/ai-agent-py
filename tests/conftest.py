"""Shared pytest fixtures: isolated settings, offline embeddings, memory searcher."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from ai_agent_py.config import Settings, reset_settings_cache
from ai_agent_py.offline import LexicalHashEmbeddings, OfflineChunkSearcher

_ENV_KEYS = (
    "APP_ENV",
    "LLM_API_KEY",
    "LLM_MODEL",
    "GENERAL_AGENT_MODEL",
    "LLM_BASE_URL",
    "OPENAI_API_KEY",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_BATCH_SIZE",
    "DASHSCOPE_API_KEY",
    "DATA_DIR",
    "MILVUS_URI",
    "MILVUS_TOKEN",
    "MILVUS_DATABASE",
    "MILVUS_COLLECTION",
    "MILVUS_DIM",
    "MILVUS_METRIC",
    "MILVUS_ANALYZER",
    "MILVUS_BATCH_SIZE",
    "RERANK_ENABLED",
    "RERANK_MODEL",
    "RERANK_API_KEY",
    "RERANK_ENDPOINT",
    "RERANK_CANDIDATES",
    "RERANK_MIN_SCORE",
    "RERANK_INSTRUCT",
    "RERANK_TIMEOUT",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVER_K",
    "AGENT_WORKSPACE",
    "AGENT_SHELL_TIMEOUT",
    "AGENT_SHELL_MAX_OUTPUT",
    "AGENT_MEMORY_SESSIONS",
    "AGENT_MODEL_CONTEXT_TOKENS",
    "AGENT_CONTEXT_MAX_TOKENS",
    "AGENT_OUTPUT_RESERVE_TOKENS",
    "AGENT_CONTEXT_SAFETY_TOKENS",
    "AGENT_TOKEN_CHARS",
    "AGENT_MEMORY_CANDIDATES",
    "AGENT_HISTORY_KEEP_MESSAGES",
    "AGENT_MESSAGE_MAX_CHARS",
    "AGENT_TOOL_RESULT_MAX_CHARS",
    "AGENT_TOOL_RESULT_HEAD_CHARS",
    "AGENT_TOOL_RESULT_TAIL_CHARS",
    "AGENT_TOOL_TURN_MAX_CHARS",
    "AGENT_SUMMARY_ENABLED",
    "AGENT_SUMMARY_TRIGGER_TOKENS",
    "AGENT_SUMMARY_KEEP_MESSAGES",
    "AGENT_SUMMARY_MAX_CHARS",
    "AGENT_SUMMARY_MAX_TOKENS",
    "AGENT_SUMMARY_INPUT_MAX_CHARS",
    "AGENT_CONTEXT_OVERFLOW_RETRIES",
    "WEB_SEARCH_API_KEY",
    "WEB_SEARCH_BASE_URL",
    "WEB_SEARCH_MODEL",
    "WEB_SEARCH_STRATEGY",
    "WEB_SEARCH_FORCED",
    "WEB_SEARCH_ENABLE_CITATION",
    "WEB_SEARCH_TIMEOUT",
    "WEB_EXTRACTOR_BASE_URL",
    "WEB_EXTRACTOR_API_KEY",
    "WEB_EXTRACTOR_MODEL",
    "WEB_EXTRACTOR_TIMEOUT",
    "WEB_EXTRACTOR_MAX_CHARS",
    "MYSQL_URL",
    "MYSQL_USERNAME",
    "MYSQL_PASSWORD",
    "MYSQL_POOL_SIZE",
    "MYSQL_MAX_OVERFLOW",
    "MYSQL_POOL_RECYCLE",
    "MYSQL_CONNECT_TIMEOUT",
    "RAG_DELETE_PENDING_DIR",
    "OSS_BUCKET",
    "OSS_ENDPOINT",
    "OSS_REGION",
    "OSS_ACCESS_KEY_ID",
    "OSS_ACCESS_KEY_SECRET",
    "OSS_PREFIX",
    "OSS_ACCESS_TOKEN",
    "OSS_URL_EXPIRES",
    "OSS_MAX_FILE_MB",
    "OSS_ALLOWED_SUFFIXES",
    "OSS_TIMEOUT",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the developer's real .env / shell variables out of the tests."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path / "raw",
    )


@pytest.fixture
def embeddings() -> Embeddings:
    return LexicalHashEmbeddings()


@pytest.fixture
def searcher(embeddings: Embeddings) -> OfflineChunkSearcher:
    """离线检索器：同时是 `ingest` 的写入器，行为对齐 Milvus 那条链路。"""
    return OfflineChunkSearcher(embeddings)


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "uv.md").write_text(
        "# 依赖管理\n\n本项目使用 uv 管理依赖和虚拟环境，命令是 `uv sync` 与 `uv add`。",
        encoding="utf-8",
    )
    (root / "langgraph.md").write_text(
        "# 流程编排\n\nLangGraph 用状态图编排智能体，节点之间通过条件边跳转。",
        encoding="utf-8",
    )
    (root / "notes.txt").write_bytes("报销流程：先在系统提交申请，再由主管审批。".encode("gbk"))
    (root / "ignored.rst").write_text("unsupported file", encoding="utf-8")
    return root
