"""Ingestion pipeline: files -> documents -> chunks -> embeddings -> store."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import AiAgentError
from ai_agent_py.rag.loaders import load_documents
from ai_agent_py.rag.splitter import split_documents
from ai_agent_py.rag.store import build_vector_store, clear_vector_store, store_kind

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
"""Called as ``on_progress(done_chunks, total_chunks)``."""


@dataclass(slots=True)
class IngestReport:
    """What an ingestion run did."""

    files: int
    documents: int
    chunks: int
    characters: int
    batches: int
    store: str
    collection: str
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "documents": self.documents,
            "chunks": self.chunks,
            "characters": self.characters,
            "batches": self.batches,
            "store": self.store,
            "collection": self.collection,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


def chunk_id(document: Document) -> str:
    """Stable id so re-ingesting the same chunk updates instead of duplicating."""
    key = "|".join(
        str(document.metadata.get(field, "")) for field in ("source", "page", "chunk_index")
    )
    digest = hashlib.sha1(f"{key}|{document.page_content}".encode()).hexdigest()
    return f"chunk-{digest[:32]}"


def ingest(
    paths: Sequence[str | Path] | None = None,
    *,
    store: VectorStore | None = None,
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
    recreate: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    batch_size: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> IngestReport:
    """Load, split and index everything below `paths` (defaults to `DATA_DIR`)."""
    started = time.perf_counter()
    cfg = settings or get_settings()
    targets: list[str | Path] = list(paths) if paths else [cfg.data_dir]
    batch = batch_size or cfg.embedding_batch_size
    if batch <= 0:
        raise ValueError("batch_size must be positive")

    documents = load_documents(targets)
    chunks = split_documents(
        documents,
        settings=cfg,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.info("Loaded %d documents -> %d chunks", len(documents), len(chunks))

    active_store = store or build_vector_store(embeddings=embeddings, settings=cfg)
    if recreate:
        clear_vector_store(active_store)
        logger.info("Vector store cleared before ingestion")

    batches = _add_in_batches(
        active_store,
        chunks,
        batch_size=batch,
        on_progress=on_progress,
    )

    return IngestReport(
        files=len({str(doc.metadata.get("source")) for doc in documents}),
        documents=len(documents),
        chunks=len(chunks),
        characters=sum(len(chunk.page_content) for chunk in chunks),
        batches=batches,
        store=store_kind(active_store),
        collection=cfg.collection_name,
        elapsed_seconds=time.perf_counter() - started,
    )


def _add_in_batches(
    store: VectorStore,
    chunks: Sequence[Document],
    *,
    batch_size: int,
    on_progress: ProgressCallback | None,
) -> int:
    total = len(chunks)
    if not total:
        return 0

    if on_progress is not None:
        on_progress(0, total)

    done = 0
    batches = 0
    for start in range(0, total, batch_size):
        window = list(chunks[start : start + batch_size])
        ids = [chunk_id(chunk) for chunk in window]
        try:
            store.add_documents(window, ids=ids)
        except Exception as exc:  # surface a actionable message
            raise AiAgentError(
                f"写入向量库失败。若是因为重复 id，请加 `--recreate` 重建索引。原始错误: {exc}"
            ) from exc
        batches += 1
        done += len(window)
        if on_progress is not None:
            on_progress(done, total)
    return batches
