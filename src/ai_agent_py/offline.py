"""Offline building blocks: a stub chat model and a lexical embedding model.

They keep `ai-agent-py demo` and the test suite runnable without an API key or
network access, while exercising exactly the same code paths as the real models.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ai_agent_py.rag.filters import SearchFilter


class StaticChatModel(BaseChatModel):
    """A chat model that always replies with the same text (offline demo)."""

    response: str = "(offline demo)"

    @property
    def _llm_type(self) -> str:
        return "static"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])


class LexicalHashEmbeddings(Embeddings):
    """Hashing bag-of-words embeddings: deterministic, offline, dependency free.

    Obviously not semantic, but queries that share surface forms (words or
    character bigrams) with a chunk do score higher, which is enough to
    smoke-test ingestion, retrieval and the whole chain without an API key.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _tokens(self, text: str) -> list[str]:
        lowered = text.lower()
        words = re.findall(r"[a-z0-9_]+", lowered)
        chars = [ch for ch in lowered if not ch.isspace()]
        bigrams = [first + second for first, second in itertools.pairwise(chars)]
        return words + bigrams

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OfflineChunkSearcher:
    """内存检索器：`ai-agent-py demo` 与单测用，不需要 Milvus 或 API Key。

    同时实现两个协议，行为和 Milvus 那条链路对齐：

    * `rag.ingest.ChunkWriter`：``index_document(doc_id, chunks)`` 按 ``doc_id`` 覆盖写入；
    * `rag.milvus_search.DocumentSearcher`：``search(query, k=..., filter=...)`` 按余弦
      相似度取前 k 条，``filter`` 按 ``doc_ids`` 过滤（与 Milvus 链路同一套
      `SearchFilter`），返回的 ``Document.metadata`` 带 ``doc_id`` / ``chunk_id`` /
      ``chunk_index`` / ``source``。
    """

    collection = "memory"

    def __init__(self, embeddings: Embeddings | None = None, *, default_k: int = 4) -> None:
        self.embeddings: Embeddings = embeddings or LexicalHashEmbeddings()
        self.default_k = default_k
        self._items: dict[str, tuple[Document, list[float]]] = {}

    def index_document(self, doc_id: str, chunks: Sequence[Document]) -> int:
        """覆盖写入一篇文档的全部切片，返回写入条数。"""
        self.delete_document(doc_id)
        contents = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(contents) if contents else []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            key = f"{doc_id}:{index}"
            metadata = {
                **chunk.metadata,
                "doc_id": doc_id,
                "chunk_id": key,
                "chunk_index": index,
            }
            self._items[key] = (Document(page_content=chunk.page_content, metadata=metadata), vector)
        return len(contents)

    def delete_document(self, doc_id: str) -> None:
        """删除该文档的全部切片（重新入库前调用）。"""
        prefix = f"{doc_id}:"
        for key in [key for key in self._items if key.startswith(prefix)]:
            self._items.pop(key, None)

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: SearchFilter | None = None,
    ) -> list[Document]:
        """按余弦相似度返回前 ``k`` 条（向量已归一化，点积即余弦）。"""
        text = (query or "").strip()
        if not text or not self._items:
            return []
        vector = self.embeddings.embed_query(text)
        wanted = frozenset(filter.doc_ids) if filter is not None else frozenset()
        scored = [
            (sum(left * right for left, right in zip(vector, stored, strict=True)), document)
            for document, stored in self._items.values()
            if not wanted or str(document.metadata.get("doc_id") or "") in wanted
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        limit = max(1, k or self.default_k)
        return [document for _, document in scored[:limit]]

    def __len__(self) -> int:
        return len(self._items)
