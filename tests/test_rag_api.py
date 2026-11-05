"""RAG 记录 / RAG 问答 HTTP 接口：注入假 reader、假图与假任务表，不连外部服务。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import ai_agent_py.api.rag as api_rag
from ai_agent_py.api.app import app as fastapi_app
from ai_agent_py.config import Settings
from ai_agent_py.errors import ConfigurationError, MilvusSchemaError
from ai_agent_py.graph.rag_chat import build_rag_chat_graph
from ai_agent_py.mineru.models import (
    JobState,
    ParseJob,
    ParseOptions,
    RagIngestInfo,
)
from ai_agent_py.rag.filters import MAX_FILTER_DOC_IDS, SearchFilter
from ai_agent_py.rag.records import (
    RagChunkPage,
    RagChunkRecord,
    RagCollectionStats,
    RagDocStat,
)
from fakes import FakeDocumentNameResolver, ScriptedChatModel, StreamingScriptedChatModel

_FILE_NAME = "中华人民共和国企业所得税法实施条例.docx"

_SETTINGS = Settings(
    _env_file=None,
    milvus_uri="http://localhost:19530",
    milvus_database="db_tax_rag",
    milvus_collection="rag_chunk",
    milvus_dim=1536,
)


# ---- 假实现 ---------------------------------------------------------------
class FakeReader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def overview(self) -> RagCollectionStats:
        return RagCollectionStats(
            total_chunks=6,
            docs=[RagDocStat(doc_id="doc-a", chunks=4), RagDocStat(doc_id="doc-b", chunks=2)],
        )

    def list_chunks(self, **kwargs: Any) -> RagChunkPage:
        self.calls.append(kwargs)
        if kwargs.get("doc_id") == "doc-missing":
            raise MilvusSchemaError("Milvus 集合 db_tax_rag.rag_chunk 不存在")
        return RagChunkPage(
            total=4,
            items=[
                RagChunkRecord(
                    chunk_id="chunk-1",
                    doc_id="doc-a",
                    chunk_index=0,
                    page=3,
                    content="企业所得税的扣除项目包括成本、费用、税金和损失。",
                )
            ],
        )


class BrokenReader:
    def overview(self) -> RagCollectionStats:
        raise MilvusSchemaError("Milvus 集合 db_tax_rag.rag_chunk 不存在")


class FakeSearcher:
    def __init__(self, *, reranked: bool = False) -> None:
        self.reranked = reranked
        self.filters: list[SearchFilter | None] = []

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: SearchFilter | None = None,
    ) -> list[Document]:
        self.filters.append(filter)
        metadata: dict[str, Any] = {
            "chunk_id": "chunk-1",
            "doc_id": "doc-a",
            "chunk_index": 0,
            "page": 3,
            "source": "doc-a",
            "score": 0.032,
            "retrieval": "hybrid",
        }
        if self.reranked:
            metadata.update(
                {"retrieval_score": 0.032, "score": 0.93, "rerank_score": 0.93, "reranked": True}
            )
        return [
            Document(
                page_content="企业实际发生的与取得收入有关的、合理的支出，准予在计算应纳税所得额时扣除。",
                metadata=metadata,
            )
        ]


class EmptyFirstSearcher(FakeSearcher):
    """第一条是空片段：answer 节点过滤后，[1] 应该对应真正有正文的那条。"""

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: SearchFilter | None = None,
    ) -> list[Document]:
        documents = super().search(query, k=k, filter=filter)
        return [
            Document(
                page_content="",
                metadata={
                    "chunk_id": "chunk-empty",
                    "doc_id": "doc-a",
                    "chunk_index": 0,
                    "page": 0,
                    "source": "doc-a",
                },
            ),
            *documents,
        ]


def _job() -> ParseJob:
    ingested = datetime(2026, 11, 3, 7, 41, 37, tzinfo=UTC)
    return ParseJob(
        id="doc-a",
        file_name=_FILE_NAME,
        file_size=23_997,
        suffix="docx",
        state=JobState.done,
        options=ParseOptions(),
        created_at=ingested,
        updated_at=ingested,
        rag_ingest=RagIngestInfo(
            chunks=4,
            characters=3_000,
            collection="db_tax_rag.rag_chunk",
            ingested_at=ingested,
        ),
    )


def make_client(
    monkeypatch: Any,
    *,
    reader: Any | None = None,
    graph: Any | None = None,
    jobs: Mapping[str, ParseJob] | None = None,
    settings: Settings | None = None,
) -> TestClient:
    cfg = settings or _SETTINGS
    monkeypatch.setattr(api_rag, "get_settings", lambda: cfg)
    monkeypatch.setattr(api_rag, "_document_jobs", lambda settings: dict(jobs or {}))
    if reader is not None:
        monkeypatch.setattr(api_rag, "_reader", lambda: reader)
    if graph is not None:
        monkeypatch.setattr(api_rag, "_chat_graph", lambda: graph)
    return TestClient(fastapi_app)


def make_graph(model: Any, *, searcher: Any | None = None) -> Any:
    return build_rag_chat_graph(
        llm=model,
        searcher=searcher if searcher is not None else FakeSearcher(),
        resolver=FakeDocumentNameResolver({"doc-a": _FILE_NAME}),
        settings=_SETTINGS,
    )


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = "message"
        data = "{}"
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        events.append((event, json.loads(data)))
    return events


# ---- /api/rag/overview ----------------------------------------------------
def test_overview_reports_collection_docs_and_config(monkeypatch) -> None:
    client = make_client(
        monkeypatch,
        reader=FakeReader(),
        jobs={"doc-a": _job()},
    )

    response = client.get("/api/rag/overview")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["database"] == "db_tax_rag"
    assert body["collection"] == "rag_chunk"
    assert body["dim"] == 1536
    assert body["total_chunks"] == 6
    assert body["total_docs"] == 2
    assert body["doc_names_available"] is True
    assert body["llm_configured"] is False
    assert body["embedding_configured"] is False
    assert body["rerank_model"] == "qwen3.7-text-rerank"
    assert body["rerank_configured"] is False
    assert body["rerank_candidates"] == 30
    assert body["retriever_k"] == 4
    assert body["docs"][0] == {
        "doc_id": "doc-a",
        "chunks": 4,
        "file_name": "中华人民共和国企业所得税法实施条例.docx",
        "state": "done",
        "ingested_at": "2026-11-03T07:41:37+00:00",
    }
    assert body["docs"][1]["file_name"] is None


def test_overview_reports_rerank_configured(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        milvus_uri="http://localhost:19530",
        milvus_database="db_tax_rag",
        milvus_collection="rag_chunk",
        milvus_dim=1536,
        rerank_api_key="sk-test",
    )
    client = make_client(monkeypatch, reader=FakeReader(), settings=settings)

    body = client.get("/api/rag/overview").json()

    assert body["rerank_configured"] is True
    assert body["rerank_model"] == "qwen3.7-text-rerank"


def test_overview_reports_unavailable_collection_without_http_error(monkeypatch) -> None:
    client = make_client(monkeypatch, reader=BrokenReader())

    response = client.get("/api/rag/overview")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert "不存在" in body["error"]
    assert body["total_chunks"] is None
    assert body["docs"] == []


# ---- /api/rag/chunks ------------------------------------------------------
def test_chunks_passes_filters_and_adds_document_name(monkeypatch) -> None:
    reader = FakeReader()
    client = make_client(monkeypatch, reader=reader, jobs={"doc-a": _job()})

    response = client.get(
        "/api/rag/chunks",
        params={"doc_id": "doc-a", "keyword": "扣除", "page": 2, "page_size": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert body["page"] == 2
    assert body["page_size"] == 5
    assert body["doc_id"] == "doc-a"
    assert body["keyword"] == "扣除"
    assert reader.calls == [{"doc_id": "doc-a", "keyword": "扣除", "page": 2, "page_size": 5}]
    item = body["items"][0]
    assert item["chunk_id"] == "chunk-1"
    assert item["doc_name"] == "中华人民共和国企业所得税法实施条例.docx"
    assert item["page"] == 3
    assert item["characters"] == len(item["content"])


def test_chunks_returns_503_when_collection_missing(monkeypatch) -> None:
    client = make_client(monkeypatch, reader=FakeReader())

    response = client.get("/api/rag/chunks", params={"doc_id": "doc-missing"})

    assert response.status_code == 503
    assert "RAG 集合未就绪" in response.json()["detail"]


def test_chunks_rejects_invalid_pagination(monkeypatch) -> None:
    client = make_client(monkeypatch, reader=FakeReader())

    assert client.get("/api/rag/chunks", params={"page": 0}).status_code == 422
    assert client.get("/api/rag/chunks", params={"page_size": 500}).status_code == 422


# ---- /api/rag/chat --------------------------------------------------------
def test_chat_returns_answer_and_sources(monkeypatch) -> None:
    model = ScriptedChatModel(
        responses=[AIMessage(content="扣除项目包括成本、费用、税金和损失 [1]。")]
    )
    client = make_client(monkeypatch, graph=make_graph(model))

    response = client.post(
        "/api/rag/chat",
        json={"message": "企业所得税的扣除项目有哪些？", "thread_id": "t-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "扣除项目包括成本、费用、税金和损失 [1]。"
    assert body["thread_id"] == "t-1"
    assert body["query"] == "企业所得税的扣除项目有哪些？"
    source = body["sources"][0]
    assert source["index"] == 1
    assert source["chunk_id"] == "chunk-1"
    assert source["doc_id"] == "doc-a"
    assert source["source"] == _FILE_NAME
    assert source["page"] == 3
    assert source["score"] == 0.032
    assert source["reranked"] is False
    assert source["retrieval"] == "hybrid"
    assert source["snippet"].startswith("企业实际发生")

    citation = body["citations"][0]
    assert citation["index"] == 1
    assert citation["chunk_id"] == "chunk-1"
    assert citation["status"] == "weak"
    assert citation["removed"] is False
    assert citation["start"] is not None
    assert body["citation_stats"] == {
        "total": 1,
        "ok": 0,
        "weak": 1,
        "removed": 0,
        "invalid": 0,
        "unsupported": 0,
    }


def test_chat_removes_out_of_range_citation_and_reports_stats(monkeypatch) -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="扣除项目包括成本 [3]。")])
    client = make_client(monkeypatch, graph=make_graph(model))

    response = client.post(
        "/api/rag/chat",
        json={"message": "企业所得税的扣除项目有哪些？", "thread_id": "t-3"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "扣除项目包括成本。"
    assert body["citations"][0]["status"] == "invalid"
    assert body["citations"][0]["reason"] == "out_of_range"
    assert body["citations"][0]["removed"] is True
    assert body["citation_stats"]["removed"] == 1
    assert body["citation_stats"]["invalid"] == 1


def test_chat_marks_reranked_sources(monkeypatch) -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="扣除项目包括成本、费用 [1]。")])
    graph = build_rag_chat_graph(
        llm=model,
        searcher=FakeSearcher(reranked=True),
        resolver=FakeDocumentNameResolver({"doc-a": _FILE_NAME}),
        settings=_SETTINGS,
    )
    client = make_client(monkeypatch, graph=graph)

    response = client.post("/api/rag/chat", json={"message": "扣除项目有哪些？"})

    source = response.json()["sources"][0]
    assert source["reranked"] is True
    assert source["retrieval"] == "hybrid"
    # 展示的 score 是精排分（0~1），不再是 RRF 融合分。
    assert source["score"] == 0.93


def test_chat_returns_503_without_model_key(monkeypatch) -> None:
    def no_graph() -> Any:
        raise ConfigurationError("Missing chat model API key")

    client = make_client(monkeypatch)
    monkeypatch.setattr(api_rag, "_chat_graph", no_graph)

    response = client.post("/api/rag/chat", json={"message": "你好"})

    assert response.status_code == 503
    assert "RAG 问答未就绪" in response.json()["detail"]


def test_chat_validates_empty_message(monkeypatch) -> None:
    client = make_client(monkeypatch, graph=make_graph(ScriptedChatModel(responses=[])))

    assert client.post("/api/rag/chat", json={"message": ""}).status_code == 422


def test_chat_stream_emits_stage_sources_tokens_and_done(monkeypatch) -> None:
    model = StreamingScriptedChatModel(
        responses=[AIMessage(content="企业所得税的答案")],
        chunk_size=2,
    )
    client = make_client(monkeypatch, graph=make_graph(model))

    response = client.post(
        "/api/rag/chat/stream",
        json={"message": "企业所得税怎么算？", "thread_id": "t-2"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [event for event, _ in events]
    assert names[0] == "stage"
    assert names[1] == "sources"
    assert names[-1] == "done"

    tokens = "".join(payload["text"] for event, payload in events if event == "token")
    assert tokens == "企业所得税的答案"

    sources_event = events[1][1]
    assert sources_event["sources"][0]["chunk_id"] == "chunk-1"
    assert sources_event["sources"][0]["source"] == _FILE_NAME

    done = events[-1][1]
    assert done["answer"] == "企业所得税的答案"
    assert done["query"] == "企业所得税怎么算？"
    assert done["sources"][0]["doc_id"] == "doc-a"
    assert done["citations"] == []
    assert done["citation_stats"]["total"] == 0


def test_chat_stream_done_carries_citations(monkeypatch) -> None:
    model = StreamingScriptedChatModel(
        responses=[AIMessage(content="扣除项目包括成本 [1]。")],
        chunk_size=3,
    )
    client = make_client(monkeypatch, graph=make_graph(model))

    response = client.post(
        "/api/rag/chat/stream",
        json={"message": "企业所得税的扣除项目有哪些？", "thread_id": "t-4"},
    )

    done = parse_sse(response.text)[-1][1]
    assert done["answer"] == "扣除项目包括成本 [1]。"
    assert done["citations"][0]["index"] == 1
    assert done["citations"][0]["start"] is not None
    # 流式 token 里还是模型原文；done 才是校验后的权威答案。
    assert done["answer"].count("[1]") == 1
    assert done["citation_stats"]["total"] == 1


def test_chat_stream_done_aligns_sources_after_filtering_empty_chunks(monkeypatch) -> None:
    model = StreamingScriptedChatModel(
        responses=[AIMessage(content="扣除项目包括成本 [1]。")],
        chunk_size=4,
    )
    client = make_client(monkeypatch, graph=make_graph(model, searcher=EmptyFirstSearcher()))

    response = client.post(
        "/api/rag/chat/stream",
        json={"message": "企业所得税的扣除项目有哪些？", "thread_id": "t-5"},
    )

    done = parse_sse(response.text)[-1][1]
    assert [source["chunk_id"] for source in done["sources"]] == ["chunk-1"]
    assert done["citations"][0]["chunk_id"] == "chunk-1"


# ---- 文档级过滤（R-01）----------------------------------------------------
def test_chat_scopes_search_to_requested_documents(monkeypatch) -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="扣除项目包括成本 [1]。")])
    searcher = FakeSearcher()
    client = make_client(monkeypatch, graph=make_graph(model, searcher=searcher))

    response = client.post(
        "/api/rag/chat",
        json={
            "message": "企业所得税的扣除项目有哪些？",
            "thread_id": "t-filter",
            "doc_ids": [" doc-a ", "doc-b", "doc-a"],
        },
    )

    assert response.status_code == 200
    # 请求里的 doc_ids 去空白 / 去重后变成类型化过滤条件
    assert searcher.filters == [SearchFilter(doc_ids=("doc-a", "doc-b"))]


def test_chat_without_doc_ids_does_not_filter(monkeypatch) -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="答案 [1]。")])
    searcher = FakeSearcher()
    client = make_client(monkeypatch, graph=make_graph(model, searcher=searcher))

    response = client.post("/api/rag/chat", json={"message": "扣除项目有哪些？"})

    assert response.status_code == 200
    assert searcher.filters == [None]


def test_chat_stream_scopes_search_to_requested_documents(monkeypatch) -> None:
    model = StreamingScriptedChatModel(responses=[AIMessage(content="答案")], chunk_size=2)
    searcher = FakeSearcher()
    client = make_client(monkeypatch, graph=make_graph(model, searcher=searcher))

    response = client.post(
        "/api/rag/chat/stream",
        json={"message": "扣除项目有哪些？", "doc_ids": ["doc-a"]},
    )

    assert response.status_code == 200
    assert searcher.filters == [SearchFilter(doc_ids=("doc-a",))]


def test_chat_rejects_too_many_doc_ids(monkeypatch) -> None:
    client = make_client(monkeypatch, graph=make_graph(ScriptedChatModel(responses=[])))

    response = client.post(
        "/api/rag/chat",
        json={
            "message": "你好",
            "doc_ids": [f"doc-{index}" for index in range(MAX_FILTER_DOC_IDS + 1)],
        },
    )

    assert response.status_code == 422


def test_chat_rejects_overlong_doc_id(monkeypatch) -> None:
    client = make_client(monkeypatch, graph=make_graph(ScriptedChatModel(responses=[])))

    response = client.post(
        "/api/rag/chat",
        json={"message": "你好", "doc_ids": ["x" * 65]},
    )

    assert response.status_code == 422
