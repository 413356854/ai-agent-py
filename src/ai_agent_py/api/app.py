"""FastAPI service: /query (RAG chain), /chat (agent), /health, /info."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_agent_py import __version__
from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import AiAgentError, ConfigurationError
from ai_agent_py.graph.agent import ask_agent, build_agent_graph
from ai_agent_py.llm import get_chat_model
from ai_agent_py.rag.chain import answer_question
from ai_agent_py.rag.retriever import get_retriever
from ai_agent_py.rag.store import build_vector_store, store_size

logger = logging.getLogger(__name__)

Resources = tuple[Settings, Any, Any, Any]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")
    k: int | None = Field(default=None, ge=1, le=20, description="召回条数，默认取 RETRIEVER_K")


class SourceOut(BaseModel):
    index: int
    source: str
    page: int | None = None
    snippet: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="用户消息")
    thread_id: str = Field(
        default="default", min_length=1, description="会话 ID，用于保留多轮上下文"
    )


class ChatResponse(BaseModel):
    answer: str
    thread_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    llm: str
    vector_store: str
    collection: str
    chunks: int | None = None


@lru_cache(maxsize=1)
def _resources() -> Resources:
    """Process wide singletons: settings, chat model, vector store, retriever."""
    cfg = get_settings()
    llm = get_chat_model(cfg)
    store = build_vector_store(settings=cfg)
    retriever = get_retriever(store=store, settings=cfg)
    return cfg, llm, store, retriever


@lru_cache(maxsize=1)
def _agent_graph() -> Any:
    from langgraph.checkpoint.memory import InMemorySaver

    _, llm, _, retriever = _resources()
    return build_agent_graph(llm=llm, retriever=retriever, checkpointer=InMemorySaver())


def _safe_resources() -> Resources:
    try:
        return _resources()
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _retriever_for(k: int | None) -> Any:
    cfg, _, store, retriever = _safe_resources()
    if k is None:
        return retriever
    return get_retriever(store=store, settings=cfg, k=k)


def create_app() -> FastAPI:
    """Application factory (uvicorn: `ai_agent_py.api.app:app`)."""
    app = FastAPI(
        title="ai-agent-py",
        version=__version__,
        description="RAG（LangChain）+ Agent（LangGraph）服务",
    )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        cfg = get_settings()
        chunks: int | None = None
        try:
            chunks = store_size(_safe_resources()[2])
        except HTTPException:
            chunks = None
        return HealthResponse(
            status="ok",
            version=__version__,
            llm=cfg.llm_model,
            vector_store=cfg.vector_store,
            collection=cfg.collection_name,
            chunks=chunks,
        )

    @app.get("/info", tags=["ops"])
    def info() -> dict[str, Any]:
        return get_settings().describe()

    @app.post("/query", response_model=QueryResponse, tags=["rag"])
    def query(payload: QueryRequest) -> QueryResponse:
        _, llm, _, _ = _safe_resources()
        try:
            result = answer_question(payload.question, llm=llm, retriever=_retriever_for(payload.k))
        except AiAgentError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # upstream provider errors
            logger.exception("query failed")
            raise HTTPException(status_code=502, detail=f"生成失败: {exc}") from exc
        return QueryResponse(
            answer=result.answer,
            sources=[SourceOut(**source.as_dict()) for source in result.sources],
        )

    @app.post("/chat", response_model=ChatResponse, tags=["agent"])
    def chat(payload: ChatRequest) -> ChatResponse:
        _safe_resources()
        try:
            graph = _agent_graph()
            answer = ask_agent(graph, payload.message, thread_id=payload.thread_id)
        except HTTPException:
            raise
        except Exception as exc:  # upstream provider errors
            logger.exception("chat failed")
            raise HTTPException(status_code=502, detail=f"智能体执行失败: {exc}") from exc
        return ChatResponse(answer=answer, thread_id=payload.thread_id)

    return app


app = create_app()
