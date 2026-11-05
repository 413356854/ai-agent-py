"""`rag_chunk` 混合检索：RRF、BM25 降级、缺 Key、维度与集合校验。"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings

import ai_agent_py.rag.milvus_search as milvus_search
from ai_agent_py.config import Settings
from ai_agent_py.errors import ConfigurationError, MilvusSchemaError, RerankError
from ai_agent_py.offline import LexicalHashEmbeddings
from ai_agent_py.rag.filters import SearchFilter
from ai_agent_py.rag.milvus_search import MilvusChunkSearcher

HITS: list[dict[str, Any]] = [
    {
        "chunk_id": "chunk-1",
        "distance": 0.032,
        "entity": {
            "chunk_id": "chunk-1",
            "doc_id": "doc-a",
            "chunk_index": 0,
            "page": 1,
            "content": "企业所得税的应纳税所得额，是指企业每一纳税年度的收入总额减除不征税收入。",
        },
    },
    {
        "chunk_id": "chunk-2",
        "distance": 0.021,
        "entity": {
            "chunk_id": "chunk-2",
            "doc_id": "doc-a",
            "chunk_index": 1,
            "page": None,
            "content": "业务招待费支出按照发生额的 60% 扣除，但最高不得超过当年销售收入的 5‰。",
        },
    },
]


class FakeSearchClient:
    def __init__(
        self,
        *,
        exists: bool = True,
        has_sparse: bool = True,
        doc_id_field: bool = True,
    ) -> None:
        self.exists = exists
        self.has_sparse = has_sparse
        self.doc_id_field = doc_id_field
        self.loaded = False
        self.hybrid_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []

    def has_collection(self, name: str) -> bool:
        return self.exists

    def describe_collection(self, name: str) -> dict[str, Any]:
        fields: list[dict[str, Any]] = [
            {"name": "chunk_id"},
            {"name": "chunk_index"},
            {"name": "page"},
            {"name": "content"},
            {"name": "embedding"},
        ]
        if self.doc_id_field:
            fields.append({"name": "doc_id"})
        if self.has_sparse:
            fields.append({"name": "sparse"})
        return {"fields": fields}

    def load_collection(self, name: str) -> None:
        self.loaded = True

    def hybrid_search(
        self,
        collection: str,
        reqs: list[Any],
        ranker: Any = None,
        limit: int = 10,
        output_fields: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        self.hybrid_calls.append(
            {
                "requests": reqs,
                "ranker": ranker,
                "limit": limit,
                "output_fields": output_fields,
                # hybrid_search 没有顶层 filter：过滤条件挂在每个子请求的 expr 上。
                "filters": [getattr(request, "expr", None) for request in reqs],
            }
        )
        return [HITS[:limit]]

    def search(
        self,
        collection: str,
        data: Any = None,
        anns_field: str = "",
        search_params: dict[str, Any] | None = None,
        limit: int = 10,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        self.search_calls.append(
            {
                "data": data,
                "anns_field": anns_field,
                "params": search_params,
                "limit": limit,
                "filter": kwargs.get("filter"),
            }
        )
        return [HITS[:limit]]


class BrokenEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding endpoint down")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("embedding endpoint down")


class FakeReranker:
    """把候选倒序返回，并记录每次调用（验证 RRF 之后确实又排了一次）。"""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def rerank(
        self,
        query: str,
        documents: Any,
        *,
        top_n: int | None = None,
    ) -> list[Any]:
        docs = list(documents)
        self.calls.append(
            {
                "query": query,
                "chunk_ids": [document.metadata["chunk_id"] for document in docs],
                "top_n": top_n,
            }
        )
        if self.error is not None:
            raise self.error
        ranked = docs[::-1]
        return ranked[:top_n] if top_n is not None else ranked


def make_searcher(
    *,
    dim: int = 8,
    exists: bool = True,
    has_sparse: bool = True,
    doc_id_field: bool = True,
    embeddings: Embeddings | object | None = "auto",
    reranker: FakeReranker | None = None,
) -> tuple[MilvusChunkSearcher, FakeSearchClient]:
    settings = Settings(
        _env_file=None,
        milvus_database="db_tax_rag",
        milvus_collection="rag_chunk",
        milvus_dim=dim,
        milvus_candidates=30,
        milvus_rrf_k=60,
        retriever_k=2,
        rerank_candidates=30,
    )
    client = FakeSearchClient(exists=exists, has_sparse=has_sparse, doc_id_field=doc_id_field)
    if embeddings == "auto":
        embeddings = LexicalHashEmbeddings(dimensions=dim)
    searcher = MilvusChunkSearcher(
        settings,
        client=client,
        embeddings=embeddings,  # type: ignore[arg-type]
        reranker=reranker,
    )
    return searcher, client


def test_hybrid_search_uses_dense_and_sparse_with_rrf() -> None:
    searcher, client = make_searcher()

    documents = searcher.search("企业所得税的扣除项目", k=2)

    assert len(client.hybrid_calls) == 1
    call = client.hybrid_calls[0]
    assert [request.anns_field for request in call["requests"]] == ["embedding", "sparse"]
    assert call["requests"][0].param["metric_type"] == "COSINE"
    assert call["requests"][1].param["metric_type"] == "BM25"
    assert call["requests"][1].data == ["企业所得税的扣除项目"]
    assert call["ranker"].dict()["strategy"] == "rrf"
    assert call["limit"] == 2
    assert client.loaded is True

    assert [document.metadata["chunk_id"] for document in documents] == ["chunk-1", "chunk-2"]
    first = documents[0]
    assert first.metadata["doc_id"] == "doc-a"
    assert first.metadata["page"] == 1
    assert first.metadata["source"] == "doc-a"
    assert first.metadata["retrieval"] == "hybrid"
    assert first.metadata["score"] == 0.032
    assert documents[1].metadata["page"] is None


def test_missing_embedding_key_falls_back_to_bm25(monkeypatch) -> None:
    searcher, client = make_searcher(embeddings=None)

    def no_key(settings: Settings | None = None) -> Any:
        raise ConfigurationError("Missing embedding API key")

    monkeypatch.setattr(milvus_search, "get_embeddings", no_key)

    documents = searcher.search("业务招待费", k=2)

    assert client.hybrid_calls == []
    assert [call["anns_field"] for call in client.search_calls] == ["sparse"]
    assert client.search_calls[0]["params"]["metric_type"] == "BM25"
    assert [document.metadata["retrieval"] for document in documents] == ["bm25", "bm25"]


def test_embedding_failure_falls_back_to_bm25() -> None:
    searcher, client = make_searcher(embeddings=BrokenEmbeddings())

    documents = searcher.search("业务招待费", k=1)

    assert client.hybrid_calls == []
    assert [call["anns_field"] for call in client.search_calls] == ["sparse"]
    assert len(documents) == 1


def test_no_retrieval_channel_raises_configuration_error(monkeypatch) -> None:
    searcher, _ = make_searcher(has_sparse=False, embeddings=None)

    def no_key(settings: Settings | None = None) -> Any:
        raise ConfigurationError("Missing embedding API key")

    monkeypatch.setattr(milvus_search, "get_embeddings", no_key)

    with pytest.raises(ConfigurationError, match="EMBEDDING_API_KEY"):
        searcher.search("企业所得税")


def test_dimension_mismatch_is_reported() -> None:
    searcher, _ = make_searcher(dim=8, embeddings=LexicalHashEmbeddings(dimensions=4))

    with pytest.raises(MilvusSchemaError, match="查询向量维度"):
        searcher.search("企业所得税")


def test_missing_collection_gives_actionable_error() -> None:
    searcher, _ = make_searcher(exists=False)

    with pytest.raises(MilvusSchemaError, match=r"create_rag_chunk\.py"):
        searcher.search("企业所得税")


def test_empty_query_returns_no_documents() -> None:
    searcher, client = make_searcher()

    assert searcher.search("   ") == []
    assert client.hybrid_calls == []
    assert client.search_calls == []


def test_rerank_fetches_more_candidates_then_truncates_to_k() -> None:
    reranker = FakeReranker()
    searcher, client = make_searcher(reranker=reranker)

    documents = searcher.search("业务招待费", k=2)

    # 开了精排：RRF 保留 RERANK_CANDIDATES=30 条候选，而不是直接取 k=2 条。
    assert client.hybrid_calls[0]["limit"] == 30
    assert reranker.calls == [
        {"query": "业务招待费", "chunk_ids": ["chunk-1", "chunk-2"], "top_n": 2}
    ]
    assert [document.metadata["chunk_id"] for document in documents] == ["chunk-2", "chunk-1"]


def test_rerank_failure_falls_back_to_rrf_order() -> None:
    searcher, _ = make_searcher(reranker=FakeReranker(error=RerankError("quota exceeded")))

    documents = searcher.search("业务招待费", k=2)

    assert [document.metadata["chunk_id"] for document in documents] == ["chunk-1", "chunk-2"]


# ---- 文档级过滤（R-01）----------------------------------------------------
def test_doc_ids_filter_is_passed_to_hybrid_search() -> None:
    searcher, client = make_searcher()

    searcher.search("业务招待费", k=2, filter=SearchFilter(doc_ids=("doc-a", "doc-b")))

    expr = 'doc_id in ["doc-a", "doc-b"]'
    # 稠密与 BM25 两路都要带过滤条件，否则没过滤的那一路会把范围外的块带回来。
    assert client.hybrid_calls[0]["filters"] == [expr, expr]


def test_filter_is_passed_to_bm25_fallback() -> None:
    searcher, client = make_searcher(embeddings=BrokenEmbeddings())

    searcher.search("业务招待费", k=2, filter=SearchFilter(doc_ids=("doc-a",)))

    # 退化为 BM25 时过滤条件不能丢
    assert [call["anns_field"] for call in client.search_calls] == ["sparse"]
    assert client.search_calls[0]["filter"] == 'doc_id in ["doc-a"]'


def test_filter_is_passed_to_dense_search() -> None:
    searcher, client = make_searcher(has_sparse=False)

    searcher.search("业务招待费", k=2, filter=SearchFilter(doc_ids=("doc-a",)))

    assert client.hybrid_calls == []
    assert [call["anns_field"] for call in client.search_calls] == ["embedding"]
    assert client.search_calls[0]["filter"] == 'doc_id in ["doc-a"]'


def test_no_filter_keeps_search_request_unchanged() -> None:
    searcher, client = make_searcher()

    searcher.search("业务招待费", k=2)
    searcher.search("业务招待费", k=2, filter=SearchFilter())
    searcher.search("业务招待费", k=2, filter=SearchFilter(doc_ids=("  ",)))

    # 空 doc_ids 等价于不过滤：子请求不带 expr，检索行为与改造前一致。
    assert [call["filters"] for call in client.hybrid_calls] == [
        [None, None],
        [None, None],
        [None, None],
    ]


def test_filter_literal_is_escaped_before_reaching_milvus() -> None:
    searcher, client = make_searcher()

    searcher.search("业务招待费", filter=SearchFilter(doc_ids=('x" or doc_id != ""',)))

    expr = 'doc_id in ["x\\" or doc_id != \\"\\""]'
    assert client.hybrid_calls[0]["filters"] == [expr, expr]


def test_filter_without_doc_id_field_gives_actionable_error() -> None:
    searcher, client = make_searcher(doc_id_field=False)

    with pytest.raises(MilvusSchemaError, match="doc_id"):
        searcher.search("业务招待费", filter=SearchFilter(doc_ids=("doc-a",)))

    assert client.hybrid_calls == []
