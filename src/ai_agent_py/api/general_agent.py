"""通用 Agent 接口：概览、MySQL 会话、对话（同步 + SSE 流式）。

路由前缀 ``/api/general-agent``，对应前端「Agent 应用 → 通用 Agent」：

* ``GET /overview``：对话模型、工作目录、全部工具、上下文预算与 MySQL 会话记忆状态；
* ``GET /sessions`` / ``GET /sessions/{id}/messages`` / ``DELETE /sessions/{id}``：
  会话列表、历史回放（含已压缩标记）与删除；
* ``POST /sessions/{id}/compact``：手动把更早历史压缩成摘要；
* ``POST /chat``：同步跑一轮，返回答案、新消息与上下文统计；
* ``POST /chat/stream``：SSE 逐 token 推送，工具调用与压缩阶段实时可见。

上下文压缩分三层（详见 :mod:`ai_agent_py.general.context`）：

1. 工具结果单次「头 + 标记 + 尾」修剪；单轮累计超预算时把较早结果换成占位；
2. 历史按 token 预算保留最近连续尾部；
3. 更早历史折叠成 MySQL 摘要（``tb_agent_chat_summary``），原始消息不删除。

上下文溢出（provider 报 context length）时，同步/流式接口都会强制再压缩一次并重试。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Any, cast
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

from ai_agent_py.alioss.service import OssImageService
from ai_agent_py.config import Settings, get_settings
from ai_agent_py.db.agent_chat import (
    DEFAULT_AGENT_ID,
    MESSAGE_TABLE,
    SESSION_TABLE,
    SUMMARY_TABLE,
    ChatMemoryStore,
    ChatMessage,
    ChatSession,
    ChatSummary,
    MySQLChatMemoryStore,
)
from ai_agent_py.errors import AiAgentError, ConfigurationError, OssError
from ai_agent_py.general.agent import (
    GENERAL_AGENT_SYSTEM_PROMPT,
    build_general_agent_graph,
    build_general_tools,
    describe_tools,
)
from ai_agent_py.general.context import (
    build_history_budget,
    estimate_records_tokens,
    estimate_text_tokens,
    select_history,
)
from ai_agent_py.general.summary import SummaryError, summarize_history
from ai_agent_py.graph.agent import agent_config, message_text
from ai_agent_py.llm import get_chat_model
from ai_agent_py.tools.policy import ToolPolicies
from ai_agent_py.tools.shell import SHELL_TOOL_NAME

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/general-agent", tags=["general-agent"])

#: 工具结果落库 / 推给前端的最大字符数（read 等工具单次可能返回上百万字符）。
MAX_TOOL_RESULT_CHARS = 20_000
#: 工具入参 JSON 的最大字符数（shell 命令可能很长）。
MAX_TOOL_ARGS_CHARS = 4_000
#: provider 上下文超限错误里常见的关键词（OpenAI 兼容接口措辞不完全一致）。
_CONTEXT_OVERFLOW_HINTS = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "context window",
    "too many tokens",
    "reduce the length",
    "prompt is too long",
    "input is too long",
)


# ---- 进程级单例 -----------------------------------------------------------
@lru_cache(maxsize=1)
def _tools() -> list[BaseTool]:
    return build_general_tools(get_settings())


@lru_cache(maxsize=1)
def _chat_model() -> BaseChatModel:
    # streaming=True：ChatOpenAI 走流式，SSE 才能逐 token 输出；摘要调用也用同一个模型。
    cfg = get_settings()
    return get_chat_model(cfg, streaming=True, model=cfg.general_agent_model)


@lru_cache(maxsize=1)
def _graph() -> CompiledStateGraph:
    cfg = get_settings()
    return build_general_agent_graph(llm=_chat_model(), tools=_tools(), settings=cfg)


@lru_cache(maxsize=1)
def _store() -> ChatMemoryStore:
    return MySQLChatMemoryStore.from_settings(get_settings())


def _graph_or_503() -> CompiledStateGraph:
    try:
        return _graph()
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"通用 Agent 未就绪：{exc}") from exc


def _store_or_503() -> ChatMemoryStore:
    try:
        return _store()
    except AiAgentError as exc:
        raise _memory_error(exc) from exc


# ---- 接口模型 -------------------------------------------------------------
class ToolOut(BaseModel):
    name: str
    group: str
    group_label: str
    description: str
    timeout_seconds: float | None = Field(
        default=None,
        description="单次调用超时（秒）；null 表示由工具自身控制（shell 的命令超时）",
    )
    retryable: bool = Field(default=False, description="临时失败是否自动重试")
    max_retries: int = Field(default=0, description="最多重试次数")
    risk_level: str = Field(default="low", description="low / medium / high")
    concurrent: bool = Field(default=True, description="同一轮是否与其他工具并行执行")


class ContextConfigOut(BaseModel):
    """概览页展示的上下文预算配置。"""

    model_context_tokens: int
    history_budget_tokens: int
    output_reserve_tokens: int
    token_chars: float
    history_keep_messages: int
    tool_result_max_chars: int
    tool_turn_max_chars: int
    summary_enabled: bool
    summary_trigger_tokens: int
    summary_keep_messages: int
    summary_max_chars: int
    overflow_retries: int


class ContextOut(BaseModel):
    """一轮对话实际使用的上下文统计。"""

    estimated_tokens: int = Field(description="摘要 + 历史消息的估算 token")
    history_budget_tokens: int
    model_context_tokens: int
    messages_used: int
    messages_dropped: int
    summary_used: bool
    summary_chars: int
    compressed_messages: int = Field(description="摘要累计覆盖的消息条数")
    summarized_messages: int = Field(default=0, description="本轮新压缩的消息条数")
    summary_error: str | None = None


class MemoryStatusOut(BaseModel):
    configured: bool = Field(description="是否配置了 MYSQL_URL")
    ok: bool = Field(description="会话记忆表是否可读写")
    tables: list[str]
    agent_id: str
    candidate_messages: int
    session_count: int | None = None
    summary_table_ready: bool | None = None
    error: str | None = None


class OverviewOut(BaseModel):
    ready: bool
    llm_model: str
    llm_configured: bool
    max_turns: int
    workspace: str
    shell_enabled: bool
    tools: list[ToolOut]
    memory: MemoryStatusOut
    context: ContextConfigOut
    warnings: list[str] = Field(default_factory=list)


class SessionOut(BaseModel):
    thread_id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str


class SummaryOut(BaseModel):
    summary: str
    until_seq: int
    source_messages: int
    tokens: int | None = None
    model: str | None = None
    updated_at: str | None = None


class MessageOut(BaseModel):
    seq: int
    role: str
    content: str
    compressed: bool = Field(default=False, description="是否已被摘要覆盖，不再发送给模型")
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_args: str | None = None
    tool_status: str | None = None
    elapsed_ms: int | None = None
    created_at: str | None = None


class MessageListOut(BaseModel):
    thread_id: str
    total: int
    items: list[MessageOut]
    summary: SummaryOut | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000, description="用户消息")
    thread_id: str = Field(
        default="",
        max_length=64,
        description="会话 ID；留空由服务端生成（响应里返回）",
    )
    images: list[str] = Field(
        default_factory=list,
        max_length=9,
        description="已上传到私域 OSS 的图片直链（最多 9 张）；会以 Markdown 图片链接写入会话",
    )


MAX_CHAT_IMAGES = 9


def _image_urls(payload: ChatRequest) -> list[str]:
    """校验图片直链（本站 Bucket + OSS_PREFIX），返回去重后的 URL。"""
    if not payload.images:
        return []

    cfg = get_settings()
    service = OssImageService(cfg)
    urls: list[str] = []
    for raw in payload.images:
        value = (raw or "").strip()
        if not value:
            continue
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise HTTPException(status_code=400, detail="图片地址必须是 http(s) URL")
        if not cfg.oss_host or parsed.hostname != cfg.oss_host:
            raise HTTPException(status_code=400, detail="图片地址必须来自本站私域 OSS 的 Bucket")
        key = unquote(parsed.path).lstrip("/")
        try:
            service.normalize_key(key)
        except OssError as exc:
            raise HTTPException(status_code=400, detail=f"图片地址不合法：{exc}") from exc
        if value not in urls:
            urls.append(value)

    if len(urls) > MAX_CHAT_IMAGES:
        raise HTTPException(status_code=400, detail=f"一次最多发送 {MAX_CHAT_IMAGES} 张图片")
    return urls


def _user_text(message: str, urls: list[str]) -> str:
    """落库 / 界面回放用的文本：文字 + Markdown 图片链接。"""
    if not urls:
        return message
    gallery = "\n".join(f"![用户上传的图片]({url})" for url in urls)
    return f"{message}\n\n{gallery}"


def _human_content(message: str, urls: list[str]) -> str | list[Any]:
    """发给模型的内容：有图片时按 LangChain 标准 content block 组装。

    标准格式来自官方文档：``{"type": "image", "url": "..."}``；
    ``ChatOpenAI`` 等集成会自动转换成 provider 原生的 ``image_url`` 结构。
    """
    if not urls:
        return message
    blocks: list[Any] = [{"type": "text", "text": message}]
    blocks.extend({"type": "image", "url": url} for url in urls)
    return blocks


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    session: SessionOut
    tool_calls: int
    elapsed_ms: float
    context: ContextOut
    summary: SummaryOut | None = None
    messages: list[MessageOut] = Field(
        default_factory=list, description="本轮新增的助手 / 工具消息"
    )


class CompactRequest(BaseModel):
    keep: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="手动压缩时保留最近多少条消息（默认 AGENT_SUMMARY_KEEP_MESSAGES）",
    )


class CompactOut(BaseModel):
    compacted: bool
    summary: str = ""
    until_seq: int = 0
    source_messages: int = 0
    tokens: int = 0
    summarized: int = 0
    message: str = ""


# ---- 概览 -----------------------------------------------------------------
@router.get(
    "/overview",
    response_model=OverviewOut,
    summary="通用 Agent 概览（模型 / 工具 / 上下文 / MySQL）",
)
def overview() -> OverviewOut:
    cfg = get_settings()
    tools = _tools()
    warnings: list[str] = []
    llm_configured = bool(cfg.resolved_llm_api_key)
    if not llm_configured:
        warnings.append(
            "未配置对话模型 Key（LLM_API_KEY / DASHSCOPE_API_KEY），对话接口会返回 503。"
        )
    if not cfg.resolved_web_search_api_key:
        warnings.append("未配置联网搜索 Key，web_search 调用只会返回配置提示。")
    memory = _memory_status(cfg)
    if memory.error:
        warnings.append(f"MySQL 会话记忆不可用：{memory.error}")
    if memory.summary_table_ready is False:
        warnings.append(
            f"摘要表 {SUMMARY_TABLE} 不可用，上下文压缩会降级为纯窗口截断；"
            f"请执行 docs/mysql/{SUMMARY_TABLE}.sql 建表。"
        )

    budget = build_history_budget(
        cfg,
        tools,
        system_prompt=GENERAL_AGENT_SYSTEM_PROMPT,
    )
    context = ContextConfigOut(
        model_context_tokens=cfg.agent_model_context_tokens,
        history_budget_tokens=budget,
        output_reserve_tokens=cfg.agent_output_reserve_tokens,
        token_chars=cfg.agent_token_chars,
        history_keep_messages=cfg.agent_history_keep_messages,
        tool_result_max_chars=cfg.agent_tool_result_max_chars,
        tool_turn_max_chars=cfg.agent_tool_turn_max_chars,
        summary_enabled=cfg.agent_summary_enabled,
        summary_trigger_tokens=cfg.agent_summary_trigger_tokens,
        summary_keep_messages=cfg.agent_summary_keep_messages,
        summary_max_chars=cfg.agent_summary_max_chars,
        overflow_retries=cfg.agent_context_overflow_retries,
    )
    policies = ToolPolicies.from_settings(cfg)
    metas = describe_tools(tools)
    return OverviewOut(
        ready=llm_configured and memory.ok,
        llm_model=cfg.general_agent_model or cfg.llm_model,
        llm_configured=llm_configured,
        max_turns=cfg.agent_max_turns,
        workspace=str(cfg.agent_workspace),
        shell_enabled=any(tool.name == SHELL_TOOL_NAME for tool in tools),
        tools=[
            ToolOut(
                name=item.name,
                group=item.group,
                group_label=item.group_label,
                description=item.description,
                **policies.describe(item.name),
            )
            for item in metas
        ],
        memory=memory,
        context=context,
        warnings=warnings,
    )


def _memory_status(cfg: Settings) -> MemoryStatusOut:
    base: dict[str, Any] = {
        "configured": bool(cfg.mysql_url),
        "tables": [SESSION_TABLE, MESSAGE_TABLE, SUMMARY_TABLE],
        "agent_id": DEFAULT_AGENT_ID,
        "candidate_messages": cfg.agent_memory_candidates,
    }
    if not cfg.mysql_url:
        return MemoryStatusOut(
            **base,
            ok=False,
            error="未配置 MySQL（MYSQL_URL / MYSQL_USERNAME / MYSQL_PASSWORD）",
        )
    try:
        store = _store()
        count = store.count_sessions(agent_id=DEFAULT_AGENT_ID)
        summary_ready = _summary_ready(store)
    except AiAgentError as exc:
        return MemoryStatusOut(**base, ok=False, error=str(exc))
    except Exception as exc:  # 概览页不该因为库异常整页报错
        logger.exception("读取会话记忆状态失败")
        return MemoryStatusOut(**base, ok=False, error=f"会话记忆不可用：{exc}")
    return MemoryStatusOut(
        **base,
        ok=True,
        session_count=count,
        summary_table_ready=summary_ready,
    )


# ---- 会话（MySQL）---------------------------------------------------------
@router.get("/sessions", response_model=list[SessionOut], summary="会话列表（MySQL）")
def list_sessions(
    limit: Annotated[
        int | None,
        Query(ge=1, le=200, description="返回条数，默认取 AGENT_MEMORY_SESSIONS"),
    ] = None,
) -> list[SessionOut]:
    cfg = get_settings()
    try:
        sessions = _store().list_sessions(
            agent_id=DEFAULT_AGENT_ID,
            limit=limit or cfg.agent_memory_sessions,
        )
    except AiAgentError as exc:
        raise _memory_error(exc) from exc
    return [_session_out(session) for session in sessions]


@router.get(
    "/sessions/{thread_id}/messages",
    response_model=MessageListOut,
    summary="会话消息（MySQL，含工具调用与压缩标记）",
)
def list_messages(
    thread_id: str,
    limit: Annotated[int, Query(ge=1, le=1_000, description="最近多少条消息")] = 500,
) -> MessageListOut:
    store = _store_or_503()
    try:
        session = store.get_session(thread_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"会话不存在：{thread_id}")
        messages = store.load_messages(thread_id, limit=limit)
    except AiAgentError as exc:
        raise _memory_error(exc) from exc
    summary_row, summary_error = _safe_summary(store, thread_id)
    if summary_error:
        logger.warning("读取会话摘要失败（不影响消息回放）：%s", summary_error)
    until_seq = summary_row.until_seq if summary_row else 0
    return MessageListOut(
        thread_id=thread_id,
        total=session.message_count,
        items=[_message_out(message, compressed=message.seq <= until_seq) for message in messages],
        summary=_summary_out(summary_row),
    )


@router.post(
    "/sessions/{thread_id}/compact",
    response_model=CompactOut,
    summary="手动压缩会话历史（生成/更新摘要）",
)
def compact_session(thread_id: str, payload: CompactRequest | None = None) -> CompactOut:
    """把更早的历史合并进摘要，保留最近 ``keep`` 条消息。

    即使未到自动触发阈值也可以调用；原始消息不会删除，只影响后续发给模型的上下文。
    """
    cfg = get_settings()
    store = _store_or_503()
    if not _summary_ready(store):
        raise HTTPException(
            status_code=503,
            detail=f"摘要表 {SUMMARY_TABLE} 不可用，请先执行 docs/mysql/{SUMMARY_TABLE}.sql",
        )
    try:
        session = store.get_session(thread_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"会话不存在：{thread_id}")
        records = store.load_messages(thread_id, limit=cfg.agent_memory_candidates)
    except AiAgentError as exc:
        raise _memory_error(exc) from exc

    summary_row, summary_error = _safe_summary(store, thread_id)
    if summary_error:
        raise HTTPException(status_code=503, detail=f"读取摘要失败：{summary_error}")
    if summary_row is not None:
        records = [record for record in records if record.seq > summary_row.until_seq]
    history = _history_records(records)
    if len(history) <= 1:
        return CompactOut(compacted=False, message="没有可压缩的历史消息")

    keep = payload.keep if payload is not None and payload.keep else cfg.agent_summary_keep_messages
    keep = max(1, min(keep, len(history) - 1))
    to_summarize = history[:-keep]
    try:
        new_row = _summarize_and_save(
            cfg,
            store,
            thread_id,
            to_summarize,
            summary_row,
        )
    except SummaryError as exc:
        raise HTTPException(status_code=502, detail=f"压缩失败：{exc}") from exc
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"通用 Agent 未就绪：{exc}") from exc
    except AiAgentError as exc:
        raise _memory_error(exc) from exc
    return CompactOut(
        compacted=True,
        summary=new_row.summary,
        until_seq=new_row.until_seq,
        source_messages=new_row.source_messages,
        tokens=new_row.tokens or 0,
        summarized=len(to_summarize),
        message=f"已把 {len(to_summarize)} 条消息压缩成摘要",
    )


@router.delete("/sessions/{thread_id}", status_code=204, summary="删除会话及其消息")
def delete_session(thread_id: str) -> Response:
    store = _store_or_503()
    try:
        deleted = store.delete_session(thread_id)
    except AiAgentError as exc:
        raise _memory_error(exc) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"会话不存在：{thread_id}")
    return Response(status_code=204)


# ---- 同步对话 -------------------------------------------------------------
@router.post("/chat", response_model=ChatResponse, summary="通用 Agent 对话（同步）")
def chat(payload: ChatRequest) -> ChatResponse:
    cfg = get_settings()
    graph = _graph_or_503()
    store = _store_or_503()
    thread_id = _thread_id(payload.thread_id)
    image_urls = _image_urls(payload)
    content = _user_text(payload.message, image_urls)
    model_content = _human_content(payload.message, image_urls)
    started = time.perf_counter()

    try:
        store.ensure_session(thread_id, title=payload.message)
        context = _load_context(cfg, store, thread_id)
        user_seq = context.last_seq + 1
        session = store.append_messages(
            thread_id,
            [ChatMessage(role="user", content=content)],
            title=payload.message,
        )
    except AiAgentError as exc:
        raise _memory_error(exc) from exc

    prompt_messages: list[AnyMessage] = [
        *context.messages,
        HumanMessage(content=model_content),
    ]
    try:
        result = _invoke_graph(graph, prompt_messages, context.summary, cfg)
    except Exception as exc:  # 上游模型 / 图执行错误
        if _is_context_overflow(exc) and cfg.agent_context_overflow_retries > 0:
            logger.warning("通用 Agent 上下文超限，强制压缩后重试：%s", exc)
            try:
                context = _load_context(
                    cfg,
                    store,
                    thread_id,
                    force_summary=True,
                    keep_override=2,
                    max_seq=user_seq - 1,
                )
                prompt_messages = [
                    *context.messages,
                    HumanMessage(content=model_content),
                ]
                result = _invoke_graph(graph, prompt_messages, context.summary, cfg)
            except Exception as retry_exc:
                logger.exception("通用 Agent 压缩后重试失败")
                raise HTTPException(
                    status_code=502,
                    detail=f"上下文超限，压缩后重试仍失败：{retry_exc}",
                ) from retry_exc
        else:
            logger.exception("通用 Agent 执行失败")
            raise HTTPException(status_code=502, detail=f"通用 Agent 执行失败：{exc}") from exc

    new_messages = list(result.get("messages") or [])[len(prompt_messages) :]
    answer, rows = _collect_turn(new_messages)
    try:
        session = store.append_messages(thread_id, rows)
        persisted = store.load_messages(thread_id, limit=len(rows)) if rows else []
    except AiAgentError as exc:
        raise _memory_error(exc) from exc

    return ChatResponse(
        answer=answer,
        thread_id=thread_id,
        session=_session_out(session),
        tool_calls=sum(1 for row in rows if row.role == "tool"),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        context=context.stats,
        summary=_summary_out(context.summary_row),
        messages=[_message_out(message) for message in persisted],
    )


# ---- SSE 流式对话 ---------------------------------------------------------
@router.post("/chat/stream", summary="通用 Agent 对话（SSE 流式，工具调用实时展示）")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """事件顺序：``session`` →（``tool_call`` → ``tool_result``）… → ``token``… → ``done``。

    压缩过历史时在 ``session`` 前额外发送 ``stage: compact``；上下文超限重试时
    发送 ``stage: overflow-retry``。出错时发 ``error``（HTTP 状态码此时已是 200）。
    """
    cfg = get_settings()
    graph = _graph_or_503()
    store = _store_or_503()
    return StreamingResponse(
        _stream_events(graph=graph, store=store, payload=payload, settings=cfg),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _stream_events(
    *,
    graph: CompiledStateGraph,
    store: ChatMemoryStore,
    payload: ChatRequest,
    settings: Settings,
) -> Iterator[str]:
    """把 LangGraph 的 messages / updates 流翻译成前端消费的 SSE 事件。"""
    started = time.perf_counter()
    thread_id = _thread_id(payload.thread_id)
    image_urls = _image_urls(payload)
    content = _user_text(payload.message, image_urls)
    model_content = _human_content(payload.message, image_urls)

    # 先把用户消息落库：即使模型报错，刷新后这一句也还在会话里。
    try:
        store.ensure_session(thread_id, title=payload.message)
        context = _load_context(settings, store, thread_id)
        user_seq = context.last_seq + 1
        session = store.append_messages(
            thread_id,
            [ChatMessage(role="user", content=content)],
            title=payload.message,
        )
    except AiAgentError as exc:
        yield _sse("error", {"message": f"MySQL 会话记忆不可用：{exc}"})
        return

    if context.stats.summarized_messages:
        yield _sse("stage", {"stage": "compact", "message": _compact_message(context.stats)})
    yield _sse("session", _session_payload(session))

    prompt_messages: list[AnyMessage] = [
        *context.messages,
        HumanMessage(content=model_content),
    ]
    call_meta: dict[str, dict[str, Any]] = {}
    tool_started: dict[str, float] = {}
    tool_rows: list[ChatMessage] = []
    answer = ""
    streamed = ""
    emitted = False
    attempts = 0

    while True:
        try:
            # LangGraph 多模式 stream 的重载对 mypy 不友好，这里统一按 (mode, payload) 处理。
            stream = cast(
                "Iterator[tuple[str, Any]]",
                graph.stream(
                    {"messages": prompt_messages, "summary": context.summary},
                    agent_config(max_turns=settings.agent_max_turns),
                    stream_mode=["messages", "updates"],
                ),
            )
            for mode, chunk in stream:
                if mode == "messages":
                    message, metadata = chunk
                    # 只转发 agent 节点真正生成的 token；工具节点的输出走 tool_result。
                    if metadata.get("langgraph_node") != "agent":
                        continue
                    if not isinstance(message, AIMessageChunk):
                        continue
                    text = message_text(message)
                    if text:
                        streamed += text
                        emitted = True
                        yield _sse("token", {"text": text})
                elif mode == "updates":
                    for node, update in chunk.items():
                        if not isinstance(update, dict):
                            continue
                        messages = [
                            item
                            for item in (update.get("messages") or [])
                            if isinstance(item, BaseMessage)
                        ]
                        if node == "agent":
                            for item in messages:
                                if isinstance(item, AIMessage):
                                    if item.tool_calls:
                                        emitted = True
                                    yield from _agent_update_events(item, call_meta, tool_started)
                                    text = message_text(item).strip()
                                    if text and not item.tool_calls:
                                        answer = text
                        elif node == "tools":
                            for item in messages:
                                if not isinstance(item, ToolMessage):
                                    continue
                                emitted = True
                                info = call_meta.get(item.tool_call_id, {})
                                content = _clip_tool_result(_tool_output(item))
                                status = _tool_status(content, item)
                                elapsed = _tool_elapsed(item)
                                if elapsed is None:
                                    elapsed = round(
                                        (
                                            time.perf_counter()
                                            - tool_started.get(item.tool_call_id, started)
                                        )
                                        * 1000
                                    )
                                name = str(info.get("name") or item.name or "tool")
                                tool_rows.append(
                                    ChatMessage(
                                        role="tool",
                                        content=content,
                                        tool_name=name,
                                        tool_call_id=item.tool_call_id,
                                        tool_args=_dump_args(info.get("args")),
                                        tool_status=status,
                                        elapsed_ms=elapsed,
                                    )
                                )
                                yield _sse(
                                    "tool_result",
                                    {
                                        "id": item.tool_call_id,
                                        "name": name,
                                        "status": status,
                                        "elapsed_ms": elapsed,
                                        "content": content,
                                    },
                                )
            break
        except GeneratorExit:
            raise
        except Exception as exc:  # 上游模型 / 工具执行错误
            if (
                _is_context_overflow(exc)
                and not emitted
                and attempts < settings.agent_context_overflow_retries
            ):
                attempts += 1
                logger.warning("通用 Agent 流式上下文超限，强制压缩后重试：%s", exc)
                yield _sse(
                    "stage",
                    {
                        "stage": "overflow-retry",
                        "message": "上下文超出模型窗口，正在压缩历史后重试…",
                    },
                )
                try:
                    context = _load_context(
                        settings,
                        store,
                        thread_id,
                        force_summary=True,
                        keep_override=2,
                        max_seq=user_seq - 1,
                    )
                except Exception as prep_exc:
                    logger.exception("压缩后重建上下文失败")
                    yield _sse("error", {"message": f"上下文压缩失败：{prep_exc}"})
                    return
                prompt_messages = [
                    *context.messages,
                    HumanMessage(content=model_content),
                ]
                if context.stats.summarized_messages:
                    yield _sse(
                        "stage",
                        {"stage": "compact", "message": _compact_message(context.stats)},
                    )
                continue
            logger.exception("通用 Agent 流式输出失败")
            _persist_best_effort(store, thread_id, tool_rows, answer or streamed)
            yield _sse("error", {"message": _error_message(exc)})
            return

    rows = [*tool_rows]
    if not answer:
        answer = streamed.strip()
    if answer:
        rows.append(ChatMessage(role="assistant", content=answer))

    try:
        session = store.append_messages(thread_id, rows)
        persisted = store.load_messages(thread_id, limit=len(rows)) if rows else []
    except AiAgentError as exc:
        yield _sse(
            "error",
            {"message": f"回答已生成，但写入 MySQL 会话记忆失败：{exc}"},
        )
        return

    summary_payload = _summary_out(context.summary_row)
    yield _sse(
        "done",
        {
            "answer": answer,
            "thread_id": thread_id,
            "session": _session_payload(session),
            "tool_calls": len(tool_rows),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "context": context.stats.model_dump(),
            "summary": summary_payload.model_dump() if summary_payload is not None else None,
            "messages": [_message_out(message).model_dump() for message in persisted],
        },
    )


# ---- 上下文组装 -----------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _HistoryContext:
    """一轮对话开始前的历史上下文（不含本轮用户消息）。"""

    messages: list[AnyMessage]
    summary: str
    summary_row: ChatSummary | None
    stats: ContextOut
    last_seq: int


def _load_context(
    cfg: Settings,
    store: ChatMemoryStore,
    thread_id: str,
    *,
    force_summary: bool = False,
    keep_override: int | None = None,
    max_seq: int | None = None,
) -> _HistoryContext:
    """读取历史 → 必要时生成摘要 → 按 token 预算选出上下文。"""
    summary_row, summary_error = _safe_summary(store, thread_id)
    records = store.load_messages(thread_id, limit=cfg.agent_memory_candidates)
    if max_seq is not None:
        records = [record for record in records if record.seq <= max_seq]
    last_seq = max((record.seq for record in records), default=0)
    if summary_row is not None:
        records = [record for record in records if record.seq > summary_row.until_seq]

    summary_text = summary_row.summary if summary_row is not None else ""
    summarized_now = 0
    if cfg.agent_summary_enabled and not summary_error:
        if not _summary_ready(store):
            summary_error = (
                f"摘要表 {SUMMARY_TABLE} 不可用，已降级为窗口截断；"
                f"请执行 docs/mysql/{SUMMARY_TABLE}.sql"
            )
        else:
            history = _history_records(records)
            if history:
                trigger_budget = build_history_budget(
                    cfg,
                    _tools(),
                    system_prompt=GENERAL_AGENT_SYSTEM_PROMPT,
                    summary=summary_text,
                )
                history_tokens = estimate_records_tokens(history, settings=cfg)
                trigger = min(cfg.agent_summary_trigger_tokens, trigger_budget)
                if (force_summary or history_tokens > trigger) and len(history) > 1:
                    keep = (
                        keep_override
                        if keep_override is not None
                        else cfg.agent_summary_keep_messages
                    )
                    keep = max(1, min(keep, len(history) - 1))
                    to_summarize = history[:-keep]
                    try:
                        summary_row = _summarize_and_save(
                            cfg, store, thread_id, to_summarize, summary_row
                        )
                    except SummaryError as exc:
                        summary_error = str(exc)
                        logger.warning("自动压缩历史失败，降级为窗口截断：%s", exc)
                    except AiAgentError as exc:
                        summary_error = str(exc)
                        logger.warning("写入摘要失败，降级为窗口截断：%s", exc)
                    else:
                        summary_text = summary_row.summary
                        summarized_now = len(to_summarize)
                        records = [
                            record for record in records if record.seq > summary_row.until_seq
                        ]

    budget = build_history_budget(
        cfg,
        _tools(),
        system_prompt=GENERAL_AGENT_SYSTEM_PROMPT,
        summary=summary_text,
    )
    history = _history_records(records)
    selection = select_history(history, settings=cfg, budget_tokens=budget)
    summary_tokens = estimate_text_tokens(summary_text, chars_per_token=cfg.agent_token_chars)
    stats = ContextOut(
        estimated_tokens=selection.estimated_tokens + summary_tokens,
        history_budget_tokens=budget,
        model_context_tokens=cfg.agent_model_context_tokens,
        messages_used=selection.used,
        messages_dropped=selection.dropped,
        summary_used=bool(summary_text),
        summary_chars=len(summary_text),
        compressed_messages=summary_row.source_messages if summary_row is not None else 0,
        summarized_messages=summarized_now,
        summary_error=summary_error,
    )
    return _HistoryContext(
        messages=selection.messages,
        summary=summary_text,
        summary_row=summary_row,
        stats=stats,
        last_seq=last_seq,
    )


def _summarize_and_save(
    cfg: Settings,
    store: ChatMemoryStore,
    thread_id: str,
    records: list[ChatMessage],
    previous: ChatSummary | None,
) -> ChatSummary:
    """生成摘要并校验「必须比原文小」，然后落库。"""
    if not records:
        raise SummaryError("没有可压缩的历史消息")
    outcome = summarize_history(
        _chat_model(),
        records,
        settings=cfg,
        previous_summary=previous.summary if previous is not None else "",
    )
    source_tokens = estimate_records_tokens(records, settings=cfg)
    if outcome.tokens >= source_tokens:
        raise SummaryError(
            f"摘要没有比原文更小（{outcome.tokens} >= {source_tokens} tokens），拒绝替换"
        )
    row = ChatSummary(
        thread_id=thread_id,
        summary=outcome.summary,
        until_seq=records[-1].seq,
        source_messages=(previous.source_messages if previous is not None else 0) + len(records),
        tokens=outcome.tokens,
        model=outcome.model,
        updated_at=datetime.now(UTC),
    )
    if not store.save_summary(row):
        raise SummaryError("另一个请求已经压缩到更靠后的位置，请刷新后重试")
    logger.info(
        "通用 Agent 摘要 %s：压缩 %d 条消息（%d → %d tokens），覆盖到 seq %d",
        thread_id,
        len(records),
        source_tokens,
        outcome.tokens,
        row.until_seq,
    )
    return row


def _history_records(records: Sequence[ChatMessage]) -> list[ChatMessage]:
    """只保留可进上下文的历史消息（user / assistant，非空）。"""
    return [
        record
        for record in records
        if record.role in {"user", "assistant"} and record.content.strip()
    ]


def _safe_summary(
    store: ChatMemoryStore,
    thread_id: str,
) -> tuple[ChatSummary | None, str | None]:
    getter = getattr(store, "get_summary", None)
    if not callable(getter):
        return None, "会话存储不支持摘要"
    try:
        return getter(thread_id), None
    except AiAgentError as exc:
        return None, str(exc)
    except Exception as exc:  # 摘要读取失败不应阻塞对话
        logger.exception("读取摘要失败")
        return None, f"读取摘要失败：{exc}"


def _summary_ready(store: ChatMemoryStore) -> bool:
    ready = getattr(store, "summary_ready", None)
    if not callable(ready):
        return False
    try:
        return bool(ready())
    except Exception:
        logger.exception("检查摘要表失败")
        return False


def _invoke_graph(
    graph: CompiledStateGraph,
    messages: list[AnyMessage],
    summary: str,
    cfg: Settings,
) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        graph.invoke(
            {"messages": messages, "summary": summary},
            agent_config(max_turns=cfg.agent_max_turns),
        ),
    )


def _is_context_overflow(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return any(hint in text for hint in _CONTEXT_OVERFLOW_HINTS)


def _compact_message(stats: ContextOut) -> str:
    return (
        f"已把更早的 {stats.summarized_messages} 条消息压缩成摘要"
        f"（累计覆盖 {stats.compressed_messages} 条）"
    )


# ---- 消息整理 -------------------------------------------------------------
def _agent_update_events(
    message: AIMessage,
    call_meta: dict[str, dict[str, Any]],
    tool_started: dict[str, float],
) -> Iterator[str]:
    """把 agent 节点里的工具调用请求翻译成 ``tool_call`` 事件。"""
    now = time.perf_counter()
    for index, call in enumerate(message.tool_calls or []):
        call_id = str(call.get("id") or f"call-{len(call_meta) + 1}")
        name = str(call.get("name") or "")
        args = call.get("args") or {}
        call_meta[call_id] = {"name": name, "args": args}
        tool_started[call_id] = now
        yield _sse(
            "tool_call",
            {
                "id": call_id,
                "name": name,
                "args": _dump_args(args),
                "index": index,
            },
        )


def _collect_turn(messages: Sequence[AnyMessage]) -> tuple[str, list[ChatMessage]]:
    """从图返回的新消息里整理出「助手回答 + 工具调用记录」。"""
    call_meta: dict[str, dict[str, Any]] = {}
    rows: list[ChatMessage] = []
    answer = ""

    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                call_id = str(call.get("id") or f"call-{len(call_meta) + 1}")
                call_meta[call_id] = {
                    "name": str(call.get("name") or ""),
                    "args": call.get("args") or {},
                }
            text = message_text(message).strip()
            if text and not message.tool_calls:
                answer = text
        elif isinstance(message, ToolMessage):
            info = call_meta.get(message.tool_call_id, {})
            content = _clip_tool_result(_tool_output(message))
            rows.append(
                ChatMessage(
                    role="tool",
                    content=content,
                    tool_name=str(info.get("name") or message.name or "tool"),
                    tool_call_id=message.tool_call_id,
                    tool_args=_dump_args(info.get("args")),
                    tool_status=_tool_status(content, message),
                    elapsed_ms=_tool_elapsed(message),
                )
            )

    if not answer and messages:
        answer = message_text(messages[-1]).strip()
    if answer:
        rows.append(ChatMessage(role="assistant", content=answer))
    return answer, rows


def _persist_best_effort(
    store: ChatMemoryStore,
    thread_id: str,
    tool_rows: Sequence[ChatMessage],
    answer: str,
) -> None:
    """流式失败时尽力保存已经跑完的工具调用与半截回答，不覆盖原始异常。"""
    rows = list(tool_rows)
    if answer:
        rows.append(ChatMessage(role="assistant", content=answer))
    if not rows:
        return
    try:
        store.append_messages(thread_id, rows)
    except AiAgentError as exc:
        logger.warning("写入会话记忆失败：%s", exc)


def _tool_output(message: ToolMessage) -> str:
    """工具结果的完整文本：优先取 artifact 里的原始输出（模型侧可能已被裁剪）。"""
    artifact = getattr(message, "artifact", None)
    if isinstance(artifact, dict):
        raw = artifact.get("raw")
        if isinstance(raw, str):
            return raw
    return message_text(message)


def _clip_tool_result(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n…（已截断，原文共 {len(text)} 字）"


def _dump_args(args: Any) -> str | None:
    if args is None:
        return None
    if isinstance(args, str):
        text = args
    else:
        try:
            text = json.dumps(args, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(args)
    if not text.strip():
        return None
    return text if len(text) <= MAX_TOOL_ARGS_CHARS else f"{text[:MAX_TOOL_ARGS_CHARS]}…"


def _tool_elapsed(message: ToolMessage) -> int | None:
    """工具节点写入的真实耗时（毫秒）；老消息没有 artifact 时返回 None。"""
    artifact = getattr(message, "artifact", None)
    if isinstance(artifact, dict):
        value = artifact.get("elapsed_ms")
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _tool_status(content: str, message: ToolMessage | None = None) -> str:
    """工具状态：优先取工具节点的 artifact（ok / error / timeout / invalid_args）。"""
    if message is not None:
        artifact = getattr(message, "artifact", None)
        if isinstance(artifact, dict):
            status = artifact.get("status")
            if isinstance(status, str) and status:
                return status
    head = content.lstrip()
    if head.startswith(
        (
            "错误：",
            "工具执行失败",
            "未知工具",
            "工具执行超时",
            "工具参数不合法",
        )
    ):
        return "error"
    return "ok"


# ---- 小工具 ---------------------------------------------------------------
def _thread_id(raw: str) -> str:
    value = (raw or "").strip()
    return value or uuid.uuid4().hex


def _session_out(session: ChatSession) -> SessionOut:
    return SessionOut(
        thread_id=session.thread_id,
        title=session.title,
        message_count=session.message_count,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )


def _session_payload(session: ChatSession) -> dict[str, Any]:
    return _session_out(session).model_dump()


def _summary_out(summary: ChatSummary | None) -> SummaryOut | None:
    if summary is None:
        return None
    return SummaryOut(
        summary=summary.summary,
        until_seq=summary.until_seq,
        source_messages=summary.source_messages,
        tokens=summary.tokens,
        model=summary.model,
        updated_at=summary.updated_at.isoformat() if summary.updated_at else None,
    )


def _message_out(message: ChatMessage, *, compressed: bool = False) -> MessageOut:
    return MessageOut(
        seq=message.seq,
        role=message.role,
        content=message.content,
        compressed=compressed,
        tool_name=message.tool_name,
        tool_call_id=message.tool_call_id,
        tool_args=message.tool_args,
        tool_status=message.tool_status,
        elapsed_ms=message.elapsed_ms,
        created_at=message.created_at.isoformat() if message.created_at else None,
    )


def _memory_error(exc: AiAgentError) -> HTTPException:
    return HTTPException(status_code=503, detail=f"MySQL 会话记忆不可用：{exc}")


def _error_message(exc: Exception) -> str:
    if isinstance(exc, AiAgentError):
        return str(exc)
    return f"通用 Agent 执行失败：{exc}"


def _sse(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {data}\n\n"
