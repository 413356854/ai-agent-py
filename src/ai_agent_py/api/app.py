"""FastAPI service: /query (RAG chain), /chat (agent), /documents (MinerU), web UI."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_agent_py import __version__
from ai_agent_py.api.agents import router as agents_router
from ai_agent_py.api.database import router as database_router
from ai_agent_py.api.documents import asset_router
from ai_agent_py.api.documents import router as documents_router
from ai_agent_py.api.general_agent import router as general_agent_router
from ai_agent_py.api.rag import router as rag_router
from ai_agent_py.config import PROJECT_ROOT, Settings, get_settings
from ai_agent_py.db.health import DatabaseHealth, check_database
from ai_agent_py.errors import AiAgentError, ConfigurationError
from ai_agent_py.graph.agent import ask_agent, build_agent_graph
from ai_agent_py.llm import get_chat_model
from ai_agent_py.rag.chain import answer_question
from ai_agent_py.rag.milvus_search import get_milvus_searcher
from ai_agent_py.rag.records import MilvusRagReader

logger = logging.getLogger(__name__)

WEB_DIST = PROJECT_ROOT / "web" / "dist"

Resources = tuple[Settings, Any, Any]


class SPAStaticFiles(StaticFiles):
    """静态托管前端构建产物；未命中的路径回退到 index.html（前端路由）。"""

    async def get_response(self, path: str, scope: Any) -> Any:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and not path.startswith("api/"):
                return await super().get_response("index.html", scope)
            raise


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
    database: DatabaseHealth


@lru_cache(maxsize=1)
def _resources() -> Resources:
    """Process wide singletons: settings, chat model, Milvus searcher."""
    cfg = get_settings()
    llm = get_chat_model(cfg)
    searcher = get_milvus_searcher(settings=cfg)
    return cfg, llm, searcher


@lru_cache(maxsize=1)
def _agent_graph() -> Any:
    from langgraph.checkpoint.memory import InMemorySaver

    _, llm, searcher = _resources()
    return build_agent_graph(llm=llm, searcher=searcher, checkpointer=InMemorySaver())


def _safe_resources() -> Resources:
    try:
        return _resources()
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
            chunks = MilvusRagReader(cfg).count()
        except AiAgentError:
            chunks = None
        return HealthResponse(
            status="ok",
            version=__version__,
            llm=cfg.llm_model,
            vector_store="milvus",
            collection=cfg.milvus_collection,
            chunks=chunks,
            database=check_database(cfg),
        )

    @app.get("/info", tags=["ops"])
    def info() -> dict[str, Any]:
        return get_settings().describe()

    @app.post("/query", response_model=QueryResponse, tags=["rag"])
    def query(payload: QueryRequest) -> QueryResponse:
        _, llm, searcher = _safe_resources()
        try:
            result = answer_question(payload.question, llm=llm, searcher=searcher, k=payload.k)
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

    # ---- 文档解析（MinerU 精准解析 API）---------------------------------
    app.include_router(documents_router)
    app.include_router(asset_router)

    # ---- RAG 记录 / RAG 问答（Milvus rag_chunk + LangGraph）--------------
    app.include_router(rag_router)

    # ---- Agent 应用（报销审核等，LangGraph 图导出）-----------------------
    app.include_router(agents_router)

    # ---- 通用 Agent（全部工具 + MySQL 会话记忆，SSE 流式）----------------
    app.include_router(general_agent_router)

    # ---- MySQL 连接诊断 -------------------------------------------------
    app.include_router(database_router)

    # ---- 静态资源：管理后台前端 ------------------------------------------
    if (WEB_DIST / "index.html").is_file():
        app.mount("/", SPAStaticFiles(directory=WEB_DIST, html=True), name="web")
    else:

        @app.get("/", include_in_schema=False, response_class=HTMLResponse)
        def web_placeholder() -> str:
            return _placeholder_page(WEB_DIST)

    return app


def _placeholder_page(web_dist: Path) -> str:
    """前端还没构建时的提示页（`cd web && pnpm install && pnpm build`）。"""
    relative = "web"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>ai-agent-py</title>
<style>
  body{{font-family:system-ui,"Segoe UI","Microsoft YaHei",sans-serif;background:#f5f7fa;
       margin:0;display:flex;min-height:100vh;align-items:center;justify-content:center}}
  .card{{background:#fff;border-radius:12px;padding:32px 40px;box-shadow:0 6px 24px rgba(0,0,0,.08);max-width:640px}}
  h1{{font-size:20px;margin:0 0 12px}} code{{background:#f0f2f5;padding:2px 6px;border-radius:4px}}
  pre{{background:#0d1117;color:#e6edf3;padding:14px 16px;border-radius:8px;overflow:auto}}
  a{{color:#1677ff}}
</style></head>
<body><div class="card">
  <h1>后端已启动 ✅</h1>
  <p>前端还未构建。先安装依赖并打包：</p>
  <pre>cd {relative}
pnpm install
pnpm build</pre>
  <p>然后刷新本页即可进入「RAG 管理 → 文档解析」。</p>
  <p>API 文档：<a href="/docs">/docs</a> · 解析配置：<a href="/api/documents/config">/api/documents/config</a></p>
  <p>构建产物目录：<code>{web_dist}</code></p>
</div></body></html>"""


app = create_app()
