"""切分预览、RAG 入库，以及「RAG 记录 / RAG 问答」的接口数据模型。"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from ai_agent_py.rag.filters import MAX_DOC_ID_LENGTH, MAX_FILTER_DOC_IDS

#: 单次入库最多接受的切片数：防止误传超大数组把 embedding 请求撑爆。
MAX_INGEST_CHUNKS = 20_000


class ChunkOut(BaseModel):
    """一个切分片段（预览用，不含向量）。"""

    index: int
    content: str
    characters: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkIn(BaseModel):
    """前端查阅后回传的切片；服务端按数组顺序重新编号后原样入库。"""

    content: str = Field(min_length=1, max_length=100_000, description="切片正文")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="标题层级 / 页码等，仅保留白名单字段"
    )


class SplitPreviewOut(BaseModel):
    """切分预览响应：缓存标识、来源文件、策略、统计信息与片段列表。"""

    job_id: str
    file_name: str
    suffix: str
    source_file: str = Field(default="", description="实际用于切分的 markdown_file")
    split_id: str = Field(description="本次切分结果在服务端缓存里的 id，入库时回传")
    strategy: str
    strategy_label: str
    chunk_size: int
    chunk_overlap: int
    total: int
    characters: int
    truncated: bool
    chunks: list[ChunkOut]


class RagIngestRequest(BaseModel):
    """入库参数，三种模式按优先级二选一/缺省：

    * ``split_id``：推荐。前端把切分预览拿到的 ``split_id`` 原样传回，服务端
      从进程内缓存读取**完整**切片（预览截断也不影响），不再重新切分、也不
      需要在请求体里携带几千条切片。
    * ``chunks``：无状态调用（多实例 / 脚本）时直接回传切片内容，服务端按数组
      顺序重新编号后入库。
    * 两者都不传：退回服务端读取 ``markdown_file`` 现切。

    ``chunk_size`` / ``chunk_overlap`` 只在「两者都不传」重新切分时生效。
    """

    split_id: str | None = Field(
        default=None, max_length=64, description="切分预览返回的缓存 id（推荐）"
    )
    chunks: list[ChunkIn] | None = Field(
        default=None,
        max_length=MAX_INGEST_CHUNKS,
        description="前端查阅过的切片（无状态调用）；与 split_id 二选一",
    )
    total: int | None = Field(
        default=None,
        ge=0,
        description="切分总数；与实际入库数量不一致时拒绝，避免只写回一部分切片",
    )
    chunk_size: int | None = Field(default=None, ge=20, le=4000, description="块大小（字符）")
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000, description="块重叠（字符）")


class RagIngestResultOut(BaseModel):
    """写入 Milvus ``rag_chunk`` 的结果。"""

    job_id: str
    chunks: int
    characters: int
    database: str
    collection: str
    store: str
    elapsed_seconds: float
    mode: Literal["cached", "reviewed", "resplit"] = Field(
        default="resplit",
        description="cached=按 split_id 读缓存；reviewed=按回传切片；resplit=服务端重新切分",
    )


# ---- RAG 记录（rag_chunk 列表查询）----------------------------------------
class RagChunkOut(BaseModel):
    """一条 ``rag_chunk`` 记录（不回传向量字段）。"""

    chunk_id: str
    doc_id: str
    doc_name: str | None = Field(default=None, description="MySQL 里的文件名；查不到时为 null")
    chunk_index: int
    page: int | None = None
    content: str
    characters: int


class RagChunkListOut(BaseModel):
    """一页 ``rag_chunk`` 记录；``total`` 是过滤后的总数。"""

    total: int
    page: int
    page_size: int
    doc_id: str | None = None
    keyword: str | None = None
    truncated: bool = Field(
        default=False, description="按文档浏览且块数超过扫描上限时，分页只覆盖部分块"
    )
    items: list[RagChunkOut] = Field(default_factory=list)


class RagDocOut(BaseModel):
    """集合里的一篇文档：块数 + MySQL 任务信息（可选）。"""

    doc_id: str
    chunks: int
    file_name: str | None = None
    state: str | None = None
    ingested_at: str | None = Field(default=None, description="ISO 8601，UTC")


class RagOverviewOut(BaseModel):
    """RAG 记录页的概览：连接参数、配置状态、总块数与文档分布。"""

    ready: bool = Field(description="Milvus 可连接且集合存在时为 true")
    error: str | None = None
    milvus_uri: str
    database: str
    collection: str
    dim: int
    metric: str
    analyzer: str
    embedding_model: str
    embedding_configured: bool
    llm_model: str
    llm_configured: bool
    rerank_model: str = Field(default="", description="RRF 之后的精排模型")
    rerank_configured: bool = Field(
        default=False, description="精排已启用且有 Key；false 表示只用 RRF 排序"
    )
    rerank_candidates: int = Field(default=0, description="送进精排的候选条数")
    retriever_k: int = Field(default=0, description="最终返回给模型的片段数")
    total_chunks: int | None = None
    total_docs: int | None = None
    docs: list[RagDocOut] = Field(default_factory=list)
    docs_truncated: bool = Field(default=False, description="文档分布超过扫描上限，统计不完整")
    doc_names_available: bool = Field(default=False, description="MySQL 任务表可读，能显示文件名")


# ---- RAG 问答 -------------------------------------------------------------
class RagChatRequest(BaseModel):
    """聊天式问答请求。"""

    message: str = Field(min_length=1, max_length=8000, description="用户消息")
    thread_id: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        description="会话 ID：相同 id 复用 LangGraph checkpointer 里的多轮上下文",
    )
    doc_ids: list[Annotated[str, Field(max_length=MAX_DOC_ID_LENGTH)]] = Field(
        default_factory=list,
        max_length=MAX_FILTER_DOC_IDS,
        description=(
            "只在指定文档内检索（rag_chunk.doc_id）；空数组表示全库。"
            "多轮对话每轮都要带上；未带的轮次按全库检索，不会沿用上一轮的范围"
        ),
    )


class RagSourceOut(BaseModel):
    """答案引用的一条检索片段。"""

    index: int
    chunk_id: str
    doc_id: str
    source: str = Field(description="文件名（检索后按 doc_id 回查 MySQL）；查不到时退回 doc_id")
    page: int | None = None
    chunk_index: int = 0
    score: float | None = Field(default=None, description="精排分（0~1）；未精排时是 RRF 融合分")
    reranked: bool = Field(default=False, description="这一条是否经过精排重排")
    retrieval: str = Field(default="", description="召回方式：hybrid / dense / bm25")
    snippet: str = ""


class RagCitationOut(BaseModel):
    """答案里的一条引用及其校验结果。"""

    index: int = Field(description="对应的来源序号（1 开始）；越界时保留原编号")
    chunk_id: str = ""
    doc_id: str = ""
    quote: str = Field(default="", description="答案里原来的方括号文本，例如 [2]")
    start: int | None = Field(default=None, description="答案内字符起点；被移除时为 null")
    end: int | None = Field(default=None, description="答案内字符终点（不含）；被移除时为 null")
    status: Literal["ok", "weak", "invalid", "unsupported"] = Field(
        default="ok",
        description="ok=通过；weak=支持度弱但保留；invalid=编号越界/空片段；unsupported=数字冲突",
    )
    reason: str = Field(default="", description="判定原因：out_of_range / numeric_mismatch 等")
    removed: bool = Field(default=False, description="是否已从答案正文中删除")


class RagCitationStatsOut(BaseModel):
    """一次回答的引用统计；``total == ok + weak + removed``。"""

    total: int = 0
    ok: int = 0
    weak: int = 0
    removed: int = 0
    invalid: int = 0
    unsupported: int = 0


class RagChatResponse(BaseModel):
    """非流式问答响应。"""

    answer: str
    thread_id: str
    query: str = Field(default="", description="实际用于检索的问题（多轮追问会被改写）")
    sources: list[RagSourceOut] = Field(default_factory=list)
    citations: list[RagCitationOut] = Field(
        default_factory=list, description="答案里的引用与校验结果（含已移除的条目）"
    )
    citation_stats: RagCitationStatsOut = Field(default_factory=RagCitationStatsOut)
