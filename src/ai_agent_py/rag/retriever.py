"""Retriever factory (similarity / MMR / score-threshold search)."""

from __future__ import annotations

from typing import Any

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from ai_agent_py.config import SearchType, Settings, get_settings
from ai_agent_py.rag.store import build_vector_store


def get_retriever(
    *,
    store: VectorStore | None = None,
    settings: Settings | None = None,
    k: int | None = None,
    search_type: SearchType | None = None,
) -> BaseRetriever:
    """Build a retriever over the configured (or given) vector store."""
    cfg = settings or get_settings()
    vector_store = store or build_vector_store(settings=cfg)
    kind: SearchType = search_type or cfg.retriever_search_type
    limit = k or cfg.retriever_k

    search_kwargs: dict[str, Any] = {"k": limit}
    if kind == "mmr":
        # MMR needs a larger candidate pool than the final k.
        search_kwargs["fetch_k"] = max(cfg.retriever_fetch_k, limit * 4)
        search_kwargs["lambda_mult"] = 0.5
    elif kind == "similarity_score_threshold":
        search_kwargs["score_threshold"] = cfg.retriever_score_threshold

    return vector_store.as_retriever(search_type=kind, search_kwargs=search_kwargs)
