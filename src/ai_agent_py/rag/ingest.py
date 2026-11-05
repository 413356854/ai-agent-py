"""Ingestion pipeline: files -> documents -> chunks -> embeddings -> Milvus ``rag_chunk``.

这是「本地目录建索引」的入口（``ai-agent-py ingest``）。它与「文档解析 → 入库 RAG」
共用同一个写入器 `rag.milvus.MilvusRagWriter`，所以两条入口最终都写进 ``rag_chunk``
一张表，检索侧（`rag.milvus_search.MilvusChunkSearcher`）不再区分数据来自哪条链路。

* 一个源文件 = 一个 ``doc_id``（``doc-`` + 相对路径 sha1 的前 16 位）；同一路径重复入库
  时按 ``doc_id`` 先删后写，换切分参数也不会留下孤儿块；
* 写入器通过 ``writer=`` 注入：离线演示与单元测试用 `offline.OfflineChunkSearcher`，
  既不需要 Milvus，也不需要 embedding Key。
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.rag.loaders import load_documents
from ai_agent_py.rag.splitter import split_documents

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
"""Called as ``on_progress(done_chunks, total_chunks)``."""


class ChunkWriter(Protocol):
    """写入器协议：`MilvusRagWriter` 与离线用的内存写入器都满足它。"""

    def index_document(self, doc_id: str, chunks: Sequence[Document]) -> Any:
        """覆盖写入一篇文档的全部切片（先删同 ``doc_id`` 的旧切片）。"""
        ...


@dataclass(slots=True)
class IngestReport:
    """What an ingestion run did."""

    files: int
    documents: int
    chunks: int
    characters: int
    collection: str
    store: str
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "documents": self.documents,
            "chunks": self.chunks,
            "characters": self.characters,
            "collection": self.collection,
            "store": self.store,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


def document_id(source: str) -> str:
    """源文件 → 稳定的 ``doc_id``：同一个文件重新入库时覆盖旧切片。"""
    return f"doc-{hashlib.sha1(source.encode('utf-8')).hexdigest()[:16]}"


def ingest(
    paths: Sequence[str | Path] | None = None,
    *,
    writer: ChunkWriter | None = None,
    settings: Settings | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> IngestReport:
    """Load, split and index everything below `paths` (defaults to `DATA_DIR`)."""
    started = time.perf_counter()
    cfg = settings or get_settings()
    targets: list[str | Path] = list(paths) if paths else [cfg.data_dir]

    documents = load_documents(targets)
    chunks = split_documents(
        documents,
        settings=cfg,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.info("Loaded %d documents -> %d chunks", len(documents), len(chunks))

    # 注意不能用 `writer or ...`：离线检索器实现了 __len__，空库时布尔值为 False。
    active = writer if writer is not None else _default_writer(cfg)
    groups = group_by_source(chunks)
    total = len(chunks)
    done = 0
    if on_progress is not None:
        on_progress(0, total)
    for source, window in groups.items():
        active.index_document(document_id(source), window)
        done += len(window)
        if on_progress is not None:
            on_progress(done, total)

    return IngestReport(
        files=len(groups),
        documents=len(documents),
        chunks=total,
        characters=sum(len(chunk.page_content) for chunk in chunks),
        collection=str(getattr(active, "collection", "") or ""),
        store=type(active).__name__,
        elapsed_seconds=time.perf_counter() - started,
    )


def group_by_source(chunks: Sequence[Document]) -> dict[str, list[Document]]:
    """按 ``metadata['source']`` 分组：一个源文件写一次，便于按 ``doc_id`` 覆盖。"""
    groups: dict[str, list[Document]] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source") or "unknown")
        groups.setdefault(source, []).append(chunk)
    return groups


def _default_writer(settings: Settings) -> ChunkWriter:
    # 延迟导入：只有真正写 Milvus 时才需要 pymilvus / embedding 依赖。
    from ai_agent_py.rag.milvus import MilvusRagWriter

    return MilvusRagWriter(settings=settings)
