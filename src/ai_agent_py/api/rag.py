"""RAG 管理接口：``rag_chunk`` 记录查询（RAG 记录）+ 聊天式问答（RAG 问答）。

路由前缀 ``/api/rag``，与前端「RAG 管理 → RAG 记录 / RAG 问答」两个栏目对应：

* ``GET /overview``：Milvus 连接参数、模型配置、总块数与文档分布；
* ``GET /chunks``：分页查询 ``rag_chunk``（按文档 / 正文关键词过滤）；
* ``POST /chat``：LangGraph 多轮问答（同步返回答案 + 引用；来源已按 ``doc_id``
  回查 MySQL 文件名）；
* ``POST /chat/stream``：同上，用 SSE 逐 token 推送，供聊天界面使用。

Milvus / 模型不可用时返回 502/503 与中文原因，前端直接展示，不用猜。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, HumanMessage

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.db.document_jobs import MySQLJobStore
from ai_agent_py.errors import AiAgentError, ConfigurationError, MilvusError, MilvusSchemaError
from ai_agent_py.graph.agent import message_text
from ai_agent_py.graph.rag_chat import ask_rag_chat, build_rag_chat_graph, rag_chat_config
from ai_agent_py.llm import get_chat_model
from ai_agent_py.mineru.models import ParseJob
from ai_agent_py.rag.doc_names import DOC_NAME_KEY, SOURCE_KEY
from ai_agent_py.rag.milvus_search import MilvusChunkSearcher
from ai_agent_py.rag.models import (
    RagChatRequest,
    RagChatResponse,
    RagChunkListOut,
    RagChunkOut,
    RagDocOut,
    RagOverviewOut,
    RagSourceOut,
)
from ai_agent_py.rag.records import MilvusRagReader

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])

#: 引用片段在接口里展示的字符数（完整正文在「RAG 记录」里看）。
SNIPPET_CHARACTERS = 200


# ---- 进程级单例 -----------------------------------------------------------
@lru_cache(maxsize=1)
def _reader() -> MilvusRagReader:
    return MilvusRagReader(get_settings())


@lru_cache(maxsize=1)
def _chat_graph() -> CompiledStateGraph:
    """LangGraph RAG 图：流式模型 + Milvus 混合检索（RRF + 精排）+ MySQL 文件名回查 + 进程内多轮记忆。"""
    from langgraph.checkpoint.memory import InMemorySaver

    cfg = get_settings()
    llm = get_chat_model(cfg, streaming=True)
    searcher = MilvusChunkSearcher(cfg)
    return build_rag_chat_graph(
        llm=llm,
        searcher=searcher,
        checkpointer=InMemorySaver(),
        settings=cfg,
    )


def _document_jobs(settings: Settings) -> dict[str, ParseJob]:
    """``doc_id`` → 解析任务（文件名等）；MySQL 不可用时返回空字典。

    ``rag_chunk`` 里只有 ``doc_id``，文件名留在 MySQL ``tb_document_parse_job``。
    这里只做展示用的**尽力而为**关联，MySQL 挂了不影响记录 / 问答本身。
    """
    try:
        store = MySQLJobStore.from_settings(settings)
        return {job.id: job for job in store.load_all()}
    except AiAgentError as exc:
        logger.warning("读取文档任务表失败，RAG 记录只显示 doc_id：%s", exc)
        return {}


# ---- 接口 -----------------------------------------------------------------
@router.get("/overview", response_model=RagOverviewOut, summary="RAG 集合概览与文档分布")
def overview() -> RagOverviewOut:
    cfg = get_settings()
    base: dict[str, Any] = {
        "milvus_uri": cfg.milvus_uri,
        "database": cfg.milvus_database,
        "collection": cfg.milvus_collection,
        "dim": cfg.milvus_dim,
        "metric": cfg.milvus_metric,
        "analyzer": cfg.milvus_analyzer,
        "embedding_model": cfg.embedding_model,
        "embedding_configured": bool(cfg.resolved_embedding_api_key),
        "llm_model": cfg.llm_model,
        "llm_configured": bool(cfg.resolved_llm_api_key),
        # 精排状态：enabled 但没 Key 时等于关闭，前端按 configured 展示。
        "rerank_model": cfg.rerank_model,
        "rerank_configured": bool(cfg.rerank_enabled and cfg.resolved_rerank_api_key),
        "rerank_candidates": cfg.rerank_candidates,
        "retriever_k": cfg.retriever_k,
    }
    try:
        stats = _reader().overview()
    except AiAgentError as exc:
        logger.warning("读取 RAG 集合概览失败：%s", exc)
        return RagOverviewOut(
            ready=False,
            error=str(exc),
            total_chunks=None,
            total_docs=None,
            docs=[],
            doc_names_available=False,
            **base,
        )

    jobs = _document_jobs(cfg)
    docs = [_doc_out(stat.doc_id, stat.chunks, jobs) for stat in stats.docs]
    return RagOverviewOut(
        ready=True,
        total_chunks=stats.total_chunks,
        total_docs=len(docs),
        docs=docs,
        docs_truncated=stats.truncated,
        doc_names_available=bool(jobs),
        **base,
    )


@router.get("/chunks", response_model=RagChunkListOut, summary="rag_chunk 记录列表")
def list_chunks(
    doc_id: Annotated[str | None, Query(max_length=64, description="按文档 id 过滤")] = None,
    keyword: Annotated[
        str | None, Query(max_length=100, description="正文关键词（中缀匹配）")
    ] = None,
    page: Annotated[int, Query(ge=1, le=10_000, description="页码，从 1 开始")] = 1,
    page_size: Annotated[int, Query(ge=1, le=200, description="每页条数")] = 20,
) -> RagChunkListOut:
    cfg = get_settings()
    clean_doc = (doc_id or "").strip() or None
    clean_keyword = (keyword or "").strip() or None
    try:
        result = _reader().list_chunks(
            doc_id=clean_doc,
            keyword=clean_keyword,
            page=page,
            page_size=page_size,
        )
    except AiAgentError as exc:
        raise _rag_error(exc) from exc

    jobs = _document_jobs(cfg) if result.items else {}
    items = [
        RagChunkOut(
            chunk_id=record.chunk_id,
            doc_id=record.doc_id,
            doc_name=jobs[record.doc_id].file_name if record.doc_id in jobs else None,
            chunk_index=record.chunk_index,
            page=record.page,
            content=record.content,
            characters=record.characters,
        )
        for record in result.items
    ]
    return RagChunkListOut(
        total=result.total,
        page=page,
        page_size=page_size,
        doc_id=clean_doc,
        keyword=clean_keyword,
        truncated=result.truncated,
        items=items,
    )


@router.post("/chat", response_model=RagChatResponse, summary="RAG 问答（LangGraph，多轮）")
def chat(payload: RagChatRequest) -> RagChatResponse:
    graph = _graph_or_503()
    try:
        turn = ask_rag_chat(graph, payload.message, thread_id=payload.thread_id)
    except AiAgentError as exc:
        raise _rag_error(exc) from exc
    except Exception as exc:  # 上游模型错误
        logger.exception("RAG 问答失败")
        raise HTTPException(status_code=502, detail=f"问答失败：{exc}") from exc
    return RagChatResponse(
        answer=turn.answer,
        thread_id=payload.thread_id,
        query=turn.query,
        sources=_sources(turn.documents),
    )


@router.post("/chat/stream", summary="RAG 问答（SSE 流式输出）")
def chat_stream(payload: RagChatRequest) -> StreamingResponse:
    """逐 token 推送答案；事件类型见 ``_stream_events``。

    先返回 ``sources``（引用片段，文件名已按 doc_id 从 MySQL 回查），再连续返回
    ``token``，最后 ``done`` 带完整答案。出错时返回 ``error`` 事件
    （HTTP 状态码此时已是 200，前端按事件处理）。
    """
    graph = _graph_or_503()
    return StreamingResponse(
        _stream_events(graph, payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---- SSE ------------------------------------------------------------------
def _stream_events(
    graph: CompiledStateGraph,
    payload: RagChatRequest,
) -> Iterator[str]:
    """把 LangGraph 的 messages / updates 流翻译成前端消费的 SSE 事件。"""
    config = rag_chat_config(thread_id=payload.thread_id)
    query = ""
    answer = ""
    documents: list[Document] = []
    yield _sse("stage", {"stage": "retrieve", "message": "正在检索知识库…"})

    try:
        # LangGraph 的多模式重载对 mypy 不友好（各种 stream_mode 组合会推出不同
        # 类型），这里统一按 (mode, payload) 处理，运行时是稳定的两种结构。
        stream = cast(
            "Iterator[tuple[str, Any]]",
            graph.stream(
                {"messages": [HumanMessage(content=payload.message)]},
                config,
                stream_mode=["messages", "updates"],
            ),
        )
        for mode, chunk in stream:
            if mode == "messages":
                message, metadata = chunk
                # 只转发 answer 节点里真正的流式 token；condense 的改写过程不展示，
                # 非流式模型产出的完整 AIMessage 也不转发（否则会和 done 重复）。
                if metadata.get("langgraph_node") != "answer":
                    continue
                if not isinstance(message, AIMessageChunk):
                    continue
                text = message_text(message)
                if text:
                    yield _sse("token", {"text": text})
            elif mode == "updates":
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    if node == "condense":
                        query = str(update.get("query") or query)
                    elif node == "retrieve":
                        documents = list(update.get("documents") or [])
                        query = str(update.get("query") or query)
                    elif node == "resolve":
                        # retrieve 之后才拿到文件名：引用事件放在 resolve 完成时发，
                        # 前端展示的 source 与提示词里的来源保持一致。
                        documents = list(update.get("documents") or [])
                        yield _sse(
                            "sources",
                            {
                                "query": query,
                                "sources": [source.model_dump() for source in _sources(documents)],
                            },
                        )
                    elif node == "answer":
                        answer = str(update.get("answer") or answer)
    except GeneratorExit:  # 客户端断开：让 LangGraph 正常收尾
        raise
    except Exception as exc:
        logger.exception("RAG 问答流式输出失败")
        yield _sse("error", {"message": _error_message(exc)})
        return

    yield _sse(
        "done",
        {
            "answer": answer,
            "query": query,
            "sources": [source.model_dump() for source in _sources(documents)],
        },
    )


def _sse(event: str, payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {data}\n\n"


# ---- 小工具 ---------------------------------------------------------------
def _graph_or_503() -> CompiledStateGraph:
    try:
        return _chat_graph()
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"RAG 问答未就绪：{exc}") from exc


def _rag_error(exc: AiAgentError) -> HTTPException:
    """项目错误 → HTTP 状态码（集合缺失 503、Milvus 访问失败 502）。"""
    if isinstance(exc, MilvusSchemaError):
        return HTTPException(status_code=503, detail=f"RAG 集合未就绪：{exc}")
    if isinstance(exc, ConfigurationError):
        return HTTPException(status_code=503, detail=f"RAG 能力未配置：{exc}")
    if isinstance(exc, MilvusError):
        return HTTPException(status_code=502, detail=f"访问 Milvus 失败：{exc}")
    return HTTPException(status_code=500, detail=str(exc))


def _error_message(exc: Exception) -> str:
    """SSE 里的错误文案（与 HTTP 接口的 detail 保持一致）。"""
    if isinstance(exc, AiAgentError):
        return str(_rag_error(exc).detail or exc)
    return f"问答失败：{exc}"


def _doc_out(doc_id: str, chunks: int, jobs: Mapping[str, ParseJob]) -> RagDocOut:
    job = jobs.get(doc_id)
    ingested_at = job.rag_ingest.ingested_at if job and job.rag_ingest else None
    return RagDocOut(
        doc_id=doc_id,
        chunks=chunks,
        file_name=job.file_name if job else None,
        state=job.state.value if job else None,
        ingested_at=ingested_at.isoformat() if ingested_at else None,
    )


def _sources(documents: Sequence[Document]) -> list[RagSourceOut]:
    sources: list[RagSourceOut] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        doc_id = str(metadata.get("doc_id") or "")
        sources.append(
            RagSourceOut(
                index=index,
                chunk_id=str(metadata.get("chunk_id") or document.id or ""),
                doc_id=doc_id,
                # resolve 节点已经把 source 换成 MySQL 文件名；查不到时退回 doc_id。
                source=str(
                    metadata.get(DOC_NAME_KEY) or metadata.get(SOURCE_KEY) or doc_id or "unknown"
                ),
                page=_as_int(metadata.get("page")) or None,
                chunk_index=_as_int(metadata.get("chunk_index")),
                score=_as_float(metadata.get("score")),
                reranked=bool(metadata.get("reranked")),
                retrieval=str(metadata.get("retrieval") or ""),
                snippet=_snippet(document.page_content),
            )
        )
    return sources


def _snippet(text: str, limit: int = SNIPPET_CHARACTERS) -> str:
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else f"{flattened[:limit]}…"


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None
