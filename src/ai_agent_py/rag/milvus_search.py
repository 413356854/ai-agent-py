"""``rag_chunk`` 的混合检索：稠密向量 + BM25 稀疏向量 + RRF（+ 可选精排）。

集合结构见 ``docs/milvus/create_rag_chunk.py``：一列 ``embedding``（语义相似）+
一列 ``sparse``（BM25，由 Milvus 根据 ``content`` 自动生成）。两路召回各取
``MILVUS_CANDIDATES`` 条，用 ``RRFRanker(MILVUS_RRF_K)`` 融合后取前 ``k`` 条：

    query ──embedding──▶ 稠密检索 ─┐
          └─原文──────▶ BM25 检索 ─┴─RRF─▶ 候选 ─(可选精排)─▶ documents

配了精排（``RERANK_ENABLED`` 且有 Key）时，RRF 会保留 ``RERANK_CANDIDATES`` 条
候选，交给 `rag.rerank.DashScopeReranker`（qwen3.7-text-rerank）重新打分后再截到
``k`` 条；精排失败只记 warning，退回 RRF 顺序，不影响问答可用性。

降级策略（本地没配 embedding Key 时也能用）：

* 有 Embedding Key：稠密 + BM25 混合检索；
* 没有 Key：只用 BM25 关键词检索（集合就是按 BM25 建的，``content`` 已开 analyzer）；
* 查询向量生成失败（网络抖动等）但集合支持 BM25：记一条 warning 后改用 BM25；
* 两者都不可用：抛 `ConfigurationError`，前端提示去配 Key。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import ConfigurationError, MilvusError, MilvusSchemaError
from ai_agent_py.llm import get_embeddings
from ai_agent_py.rag.milvus import create_milvus_client
from ai_agent_py.rag.rerank import Reranker, get_reranker

logger = logging.getLogger(__name__)

#: 检索结果里要回填的字段（不含两个向量字段）。
SEARCH_FIELDS = ("chunk_id", "doc_id", "chunk_index", "page", "content")

#: 集合信息（是否存在 / 是否有 BM25 / 是否已加载）的缓存时间。Milvus 重启后集合
#: 会变成 released，短暂缓存既省 RPC，也能在重启后自动重新加载。
PREPARE_TTL_SECONDS = 30.0


class DocumentSearcher(Protocol):
    """检索接口：`graph.rag_chat` 只依赖这个方法，测试可以注入假实现。"""

    def search(self, query: str, *, k: int | None = None) -> list[Document]:
        """返回与 ``query`` 最相关的片段（``metadata`` 里带 chunk_id / doc_id / page）。"""
        ...


class MilvusChunkSearcher:
    """在 ``rag_chunk`` 上做混合检索；``client`` / ``embeddings`` 可注入。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        embeddings: Embeddings | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._embeddings = embeddings
        self._prepared_at = 0.0
        self._has_sparse = False
        self._embedding_error: str | None = None
        # 构造时解析一次：没配 Key / 显式关闭时是 None，检索路径就完全不用判空。
        self.reranker: Reranker | None = (
            reranker if reranker is not None else get_reranker(self.settings)
        )

    # ---- 连接与集合信息 -------------------------------------------------
    @property
    def database(self) -> str:
        return self.settings.milvus_database

    @property
    def collection(self) -> str:
        return self.settings.milvus_collection

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = create_milvus_client(self.settings)
        return self._client

    def _prepare(self) -> None:
        """集合是否存在、是否有 BM25 字段、是否已加载。"""
        now = time.monotonic()
        if self._prepared_at and now - self._prepared_at < PREPARE_TTL_SECONDS:
            return
        try:
            exists = self.client.has_collection(self.collection)
        except Exception as exc:
            raise MilvusError(f"检查 Milvus 集合失败：{exc}") from exc
        if not exists:
            raise MilvusSchemaError(
                f"Milvus 集合 {self.database}.{self.collection} 不存在。"
                "请先执行 `uv run python docs/milvus/create_rag_chunk.py --dim "
                f"{self.settings.milvus_dim}` 并把文档入库"
            )
        try:
            description = self.client.describe_collection(self.collection)
        except Exception as exc:
            raise MilvusError(f"读取 Milvus 集合结构失败：{exc}") from exc
        fields = {str(field.get("name")): field for field in description.get("fields", []) if field}
        self._has_sparse = "sparse" in fields
        if "embedding" not in fields:
            raise MilvusSchemaError(f"Milvus 集合 {self.collection} 缺少 embedding 字段")
        try:
            self.client.load_collection(self.collection)
        except Exception as exc:
            raise MilvusError(f"加载 Milvus 集合 {self.collection} 失败：{exc}") from exc
        self._prepared_at = now

    # ---- 对外接口 -------------------------------------------------------
    def search(self, query: str, *, k: int | None = None) -> list[Document]:
        """混合检索（+ 可选精排）；返回的 ``Document.metadata`` 带 ``chunk_id`` 等字段。"""
        text = (query or "").strip()
        if not text:
            return []
        self._prepare()
        limit = max(1, k or self.settings.retriever_k)
        # 精排需要比最终 k 更多的候选：只把 RRF 的前 k 条交给交叉编码器，
        # 等于在最坏情况下也只能「在已经选错的 4 条里换顺序」。
        fetch = limit
        if self.reranker is not None:
            fetch = max(limit, self.settings.rerank_candidates)
        candidates = max(self.settings.milvus_candidates, fetch * 3)

        vector = self._query_vector(text)
        if vector is None and not self._has_sparse:
            raise ConfigurationError(
                "RAG 问答需要向量模型或 BM25 索引：请配置 EMBEDDING_API_KEY"
                "（切分并入库的集合必须带 sparse 字段）"
            )

        if vector is not None and self._has_sparse:
            hits = self._hybrid(text, vector, fetch, candidates)
            mode = "hybrid"
        elif vector is not None:
            hits = self._dense(vector, fetch)
            mode = "dense"
        else:
            hits = self._bm25(text, fetch, candidates)
            mode = "bm25"

        documents = [self._document(hit, mode) for hit in hits]
        documents = [document for document in documents if document.page_content.strip()]
        if self.reranker is not None and documents:
            return self._rerank(text, documents, limit)
        return documents[:limit]

    # ---- 精排 -----------------------------------------------------------
    def _rerank(self, query: str, documents: Sequence[Document], limit: int) -> list[Document]:
        """RRF → 精排 → 前 ``limit`` 条；精排失败退回 RRF 顺序。

        这里故意捕获所有异常：精排是可选增强，超时 / 401 / 返回结构变了都只应
        降精度，不该让一次问答直接 502。失败原因写 warning，方便定位 Key 或配额问题。
        """
        reranker = self.reranker
        if reranker is None:  # pragma: no cover - 调用点已经判空
            return list(documents[:limit])
        try:
            ranked = reranker.rerank(query, documents, top_n=limit)
        except Exception as exc:
            logger.warning("精排失败，退回 RRF 排序：%s", exc)
            return list(documents[:limit])
        return ranked[:limit] if ranked else list(documents[:limit])

    # ---- 三种检索 -------------------------------------------------------
    def _hybrid(
        self,
        text: str,
        vector: Sequence[float],
        limit: int,
        candidates: int,
    ) -> list[Any]:
        from pymilvus import AnnSearchRequest, RRFRanker

        requests = [
            AnnSearchRequest(
                data=[list(vector)],
                anns_field="embedding",
                param={
                    "metric_type": self.settings.milvus_metric,
                    "params": {"ef": self.settings.milvus_ef},
                },
                limit=candidates,
            ),
            AnnSearchRequest(
                data=[text],
                anns_field="sparse",
                param={"metric_type": "BM25", "params": {}},
                limit=candidates,
            ),
        ]
        try:
            result = self.client.hybrid_search(
                self.collection,
                requests,
                ranker=RRFRanker(self.settings.milvus_rrf_k),
                limit=limit,
                output_fields=list(SEARCH_FIELDS),
            )
        except Exception as exc:
            raise MilvusError(f"Milvus 混合检索失败：{exc}") from exc
        return list(result[0]) if result else []

    def _dense(self, vector: Sequence[float], limit: int) -> list[Any]:
        try:
            result = self.client.search(
                self.collection,
                data=[list(vector)],
                anns_field="embedding",
                search_params={
                    "metric_type": self.settings.milvus_metric,
                    "params": {"ef": self.settings.milvus_ef},
                },
                limit=limit,
                output_fields=list(SEARCH_FIELDS),
            )
        except Exception as exc:
            raise MilvusError(f"Milvus 向量检索失败：{exc}") from exc
        return list(result[0]) if result else []

    def _bm25(self, text: str, limit: int, candidates: int) -> list[Any]:
        try:
            result = self.client.search(
                self.collection,
                data=[text],
                anns_field="sparse",
                search_params={"metric_type": "BM25", "params": {}},
                limit=min(limit, candidates),
                output_fields=list(SEARCH_FIELDS),
            )
        except Exception as exc:
            raise MilvusError(f"Milvus 关键词检索失败：{exc}") from exc
        return list(result[0]) if result else []

    # ---- 查询向量 -------------------------------------------------------
    def _query_vector(self, query: str) -> list[float] | None:
        """生成查询向量；没有配置 Key 时返回 ``None``（改用 BM25）。"""
        if self._embeddings is None and self._embedding_error is None:
            try:
                self._embeddings = get_embeddings(self.settings)
            except ConfigurationError as exc:
                self._embedding_error = str(exc)
                self._embeddings = None
                logger.info("未配置向量模型，RAG 检索改用 BM25：%s", exc)
        if self._embeddings is None:
            return None

        try:
            vector = list(self._embeddings.embed_query(query))
        except Exception as exc:
            if self._has_sparse:
                logger.warning("生成查询向量失败，退回 BM25 检索：%s", exc)
                return None
            raise MilvusError(f"生成查询向量失败：{exc}") from exc
        if len(vector) != self.settings.milvus_dim:
            raise MilvusSchemaError(
                f"查询向量维度是 {len(vector)}，MILVUS_DIM 是 {self.settings.milvus_dim}；"
                "换 embedding 模型后必须新建集合并重新入库"
            )
        return vector

    # ---- 结果转换 -------------------------------------------------------
    def _document(self, hit: Any, mode: str) -> Document:
        entity = hit.get("entity") if isinstance(hit, dict) else None
        entity = entity if isinstance(entity, dict) else {}
        chunk_id = str(entity.get("chunk_id") or _hit_id(hit) or "")
        doc_id = str(entity.get("doc_id") or "")
        page = _optional_int(entity.get("page"))
        metadata: dict[str, Any] = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk_index": _optional_int(entity.get("chunk_index")) or 0,
            "page": page,
            # 统一成 RAG 链的引用字段：没有 MySQL 文档名时就退回 doc_id。
            "source": doc_id or chunk_id or "unknown",
            "score": _optional_float(_hit_distance(hit)),
            "retrieval": mode,
        }
        return Document(
            page_content=str(entity.get("content") or ""),
            metadata=metadata,
            id=chunk_id or None,
        )


def get_milvus_searcher(
    settings: Settings | None = None,
    *,
    client: Any | None = None,
    embeddings: Embeddings | None = None,
    reranker: Reranker | None = None,
) -> MilvusChunkSearcher:
    """工厂函数：便于在 API / CLI 里替换成注入连接的实现。"""
    return MilvusChunkSearcher(
        settings,
        client=client,
        embeddings=embeddings,
        reranker=reranker,
    )


def _hit_id(hit: Any) -> str:
    identifier = getattr(hit, "id", None)
    if identifier is None and isinstance(hit, dict):
        identifier = hit.get("id") or hit.get("chunk_id")
    return str(identifier or "")


def _hit_distance(hit: Any) -> Any:
    distance = getattr(hit, "distance", None)
    if distance is None and isinstance(hit, dict):
        distance = hit.get("distance")
    return distance


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None
