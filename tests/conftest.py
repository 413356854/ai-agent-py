"""Shared pytest fixtures: isolated settings, offline embeddings, memory store."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ai_agent_py.config import Settings, reset_settings_cache
from ai_agent_py.offline import LexicalHashEmbeddings
from ai_agent_py.rag.store import build_vector_store

_ENV_KEYS = (
    "APP_ENV",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "OPENAI_API_KEY",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_BASE_URL",
    "VECTOR_STORE",
    "COLLECTION_NAME",
    "CHROMA_DIR",
    "DATA_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVER_K",
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
        vector_store="memory",
        chroma_dir=tmp_path / "chroma",
        data_dir=tmp_path / "raw",
    )


@pytest.fixture
def embeddings() -> Embeddings:
    return LexicalHashEmbeddings()


@pytest.fixture
def store(settings: Settings, embeddings: Embeddings) -> VectorStore:
    return build_vector_store(embeddings=embeddings, settings=settings)


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
