"""Vector store construction, reset and inspection.

Two backends are supported out of the box:

* ``chroma`` - persistent local index (default, survives restarts),
* ``memory`` - in-process store, handy for tests and demos.

Both are plain LangChain `VectorStore`s, so anything else (FAISS, pgvector,
Milvus, ...) can be dropped in by extending `build_vector_store`.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.llm import get_embeddings

logger = logging.getLogger(__name__)


def build_vector_store(
    *,
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
) -> VectorStore:
    """Create (or open) the configured vector store."""
    cfg = settings or get_settings()
    embedding = embeddings or get_embeddings(cfg)

    if cfg.vector_store == "chroma":
        from langchain_chroma import Chroma

        cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Opening Chroma at %s (collection=%s)", cfg.chroma_dir, cfg.collection_name)
        return Chroma(
            collection_name=cfg.collection_name,
            embedding_function=embedding,
            persist_directory=str(cfg.chroma_dir),
        )

    return InMemoryVectorStore(embedding=embedding)


def clear_vector_store(store: VectorStore) -> None:
    """Remove every document from the store (best effort, backend agnostic)."""
    reset = getattr(store, "reset_collection", None)
    if callable(reset):
        reset()
        return None

    inner: Any = getattr(store, "store", None)
    if isinstance(inner, dict):
        inner.clear()
        return None

    logger.warning("Don't know how to clear %s; continuing without reset", type(store).__name__)
    return None


def store_size(store: VectorStore) -> int | None:
    """Number of indexed chunks, when the backend can report it."""
    collection = getattr(store, "_collection", None)
    count = getattr(collection, "count", None)
    if callable(count):
        try:
            return int(count())
        except Exception:  # pragma: no cover - backend specific
            return None

    inner = getattr(store, "store", None)
    if isinstance(inner, dict):
        return len(inner)
    return None


def store_kind(store: VectorStore) -> str:
    """Human readable backend name for reports."""
    return type(store).__name__
