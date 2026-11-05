"""``rag_chunk`` 的只读查询（管理后台「RAG 记录」页）。

和 `rag.milvus.MilvusRagWriter` 相对：这里只读，用于把 Milvus 里的检索块列出来
给用户核对（切分是否合理、某篇文档入了多少块、关键词命中哪些内容）。

设计要点：

* 只取 ``chunk_id`` / ``doc_id`` / ``chunk_index`` / ``page`` / ``content``，
  **不返回** ``embedding`` / ``sparse`` 两个向量字段——1536 维向量对前端毫无意义，
  还会把响应撑到几百 KB；
* 条数用 Milvus 的 ``count(*)`` 聚合，过滤后的总数与分页严格一致；
* 关键词用 ``content like "%kw%"`` 中缀匹配（Milvus 2.5+ 支持，且 ``content``
  正好开了 analyzer，与 BM25 用的是同一份文本）；
* Milvus 的 ``query`` **不支持 ORDER BY**（``sort`` / ``order_by`` 会被静默忽略）。
  所以不带关键词时先只取「主键 + 文档 + 序号 + 页码」这类轻量字段扫一遍，在 Python
  侧按 ``(doc_id, chunk_index)`` 排好序，再按本页的 ``chunk_id`` 回查正文——这样
  翻页时块序号是连续的（同一篇文档按原文顺序阅读），而不是 Milvus 的存储顺序；
* 关键词查询与超大集合（超过 ``MAX_SCAN_ROWS``）退回服务端 ``offset/limit`` 分页，
  只保证「页内」有序，并在响应里用 ``truncated`` 标明。
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import MilvusError, MilvusSchemaError
from ai_agent_py.rag.filters import escape_literal
from ai_agent_py.rag.milvus import create_milvus_client

#: 列表接口返回的字段（不含两个向量字段）。
OUTPUT_FIELDS = ("chunk_id", "doc_id", "chunk_index", "page", "content")

#: 全局排序只需要这几个轻量字段，正文在排好序后再按主键回查。
META_FIELDS = ("chunk_id", "doc_id", "chunk_index", "page")

#: Milvus ``query`` 单次最多返回 16384 行，文档分布统计也按这个上限扫描。
MAX_SCAN_ROWS = 16_384

#: 集合「存在 + 已加载」检查的缓存时间：Milvus 重启后集合会变成 released，
#: 短暂缓存既避免每个请求都多两次 RPC，也能在重启后自动恢复。
OPEN_TTL_SECONDS = 30.0


@dataclass(slots=True)
class RagChunkRecord:
    """一个检索块（列表展示用，不含向量）。"""

    chunk_id: str
    doc_id: str
    chunk_index: int
    page: int | None
    content: str

    @property
    def characters(self) -> int:
        return len(self.content)


@dataclass(slots=True)
class RagChunkPage:
    """一页检索块：``total`` 是**过滤后**的总数，用于前端分页。"""

    total: int
    items: list[RagChunkRecord]
    truncated: bool = False


@dataclass(slots=True)
class RagDocStat:
    """一篇文档在集合里的块数。"""

    doc_id: str
    chunks: int


@dataclass(slots=True)
class RagCollectionStats:
    """集合概览：总块数 + 文档分布（超过扫描上限时 ``truncated``）。"""

    total_chunks: int
    docs: list[RagDocStat]
    truncated: bool = False


class MilvusRagReader:
    """``rag_chunk`` 只读访问器；``client`` 可注入，单元测试不连真实 Milvus。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._opened_at = 0.0

    # ---- 连接与校验 -----------------------------------------------------
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

    def _open(self) -> None:
        """确认集合存在并加载（Milvus 重启后集合是 released 状态）。"""
        now = time.monotonic()
        if self._opened_at and now - self._opened_at < OPEN_TTL_SECONDS:
            return
        try:
            exists = self.client.has_collection(self.collection)
        except Exception as exc:
            raise MilvusError(f"检查 Milvus 集合失败：{exc}") from exc
        if not exists:
            raise MilvusSchemaError(
                f"Milvus 集合 {self.database}.{self.collection} 不存在。"
                "请先执行 `uv run python docs/milvus/create_rag_chunk.py --dim "
                f"{self.settings.milvus_dim}`，入库后再回来查看记录"
            )
        try:
            self.client.load_collection(self.collection)
        except Exception as exc:
            raise MilvusError(f"加载 Milvus 集合 {self.collection} 失败：{exc}") from exc
        self._opened_at = now

    # ---- 查询 -----------------------------------------------------------
    def count(self, *, doc_id: str | None = None, keyword: str | None = None) -> int:
        """过滤后的精确条数（``count(*)`` 聚合）。"""
        self._open()
        try:
            rows = self.client.query(
                self.collection,
                filter=build_filter(doc_id=doc_id, keyword=keyword),
                output_fields=["count(*)"],
            )
        except Exception as exc:
            raise MilvusError(f"统计 rag_chunk 失败：{exc}") from exc
        for row in rows:
            value = row.get("count(*)") if isinstance(row, dict) else None
            if value is not None:
                return int(value)
        return 0

    def list_chunks(
        self,
        *,
        doc_id: str | None = None,
        keyword: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> RagChunkPage:
        """按文档 / 关键词分页查询检索块。"""
        self._open()
        page = max(1, page)
        page_size = max(1, page_size)
        total = self.count(doc_id=doc_id, keyword=keyword)
        offset = (page - 1) * page_size
        if total == 0 or offset >= total:
            return RagChunkPage(total=total, items=[])

        expression = build_filter(doc_id=doc_id, keyword=keyword)
        if keyword or total > MAX_SCAN_ROWS:
            # 关键词 / 超大集合：服务端分页（Milvus 不支持排序，只保证页内有序）。
            rows = self._query(
                filter=expression,
                limit=min(page_size, total - offset),
                offset=offset,
            )
            records = sorted(
                (record_from_row(row) for row in rows),
                key=_record_order,
            )
            return RagChunkPage(total=total, items=records, truncated=not keyword)

        # 常规浏览：先取轻量元数据全局排序，再只回查本页的正文。
        rows = self._query(filter=expression, fields=META_FIELDS, limit=MAX_SCAN_ROWS)
        ordered = sorted((record_from_row(row) for row in rows), key=_record_order)
        window = ordered[offset : offset + page_size]
        contents = self._fetch_contents([record.chunk_id for record in window])
        return RagChunkPage(
            total=total,
            items=[replace(record, content=contents.get(record.chunk_id, "")) for record in window],
        )

    def overview(self) -> RagCollectionStats:
        """总块数 + 每篇文档的块数（最多扫描 ``MAX_SCAN_ROWS`` 行）。"""
        self._open()
        total = self.count()
        rows = self._query(filter="", fields=("doc_id",), limit=MAX_SCAN_ROWS)

        counter: Counter[str] = Counter()
        for row in rows:
            doc_id = str(row.get("doc_id") or "").strip()
            if doc_id:
                counter[doc_id] += 1
        docs = [
            RagDocStat(doc_id=doc_id, chunks=chunks)
            for doc_id, chunks in counter.items()
            if chunks > 0
        ]
        docs.sort(key=lambda item: (-item.chunks, item.doc_id))
        return RagCollectionStats(
            total_chunks=total,
            docs=docs,
            truncated=total > sum(counter.values()),
        )

    # ---- 内部 -----------------------------------------------------------
    def _query(
        self,
        *,
        filter: str,
        limit: int,
        offset: int = 0,
        fields: Sequence[str] = OUTPUT_FIELDS,
    ) -> list[dict[str, Any]]:
        try:
            rows = self.client.query(
                self.collection,
                filter=filter,
                output_fields=list(fields),
                limit=limit,
                offset=offset,
            )
        except Exception as exc:
            raise MilvusError(f"查询 rag_chunk 失败：{exc}") from exc
        return [dict(row) for row in rows]

    def _fetch_contents(self, chunk_ids: Sequence[str]) -> dict[str, str]:
        """按主键批量回查正文（``client.get`` 默认不返回向量字段）。"""
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return {}
        try:
            rows = self.client.get(self.collection, ids=ids)
        except Exception as exc:
            raise MilvusError(f"读取 rag_chunk 正文失败：{exc}") from exc
        return {
            str(row.get("chunk_id") or ""): str(row.get("content") or "")
            for row in rows
            if isinstance(row, dict)
        }


# ---- 过滤表达式 -----------------------------------------------------------
def build_filter(*, doc_id: str | None = None, keyword: str | None = None) -> str:
    """拼 Milvus 过滤表达式（``doc_id`` 精确匹配 + 正文中缀匹配）。"""
    clauses: list[str] = []
    clean_doc = (doc_id or "").strip()
    if clean_doc:
        clauses.append(f'doc_id == "{escape_literal(clean_doc)}"')
    clean_keyword = (keyword or "").strip()
    if clean_keyword:
        # like 的 % 是通配符：只转义会破坏字符串字面量的反斜杠与引号。
        clauses.append(f'content like "%{escape_literal(clean_keyword)}%"')
    return " and ".join(clauses)


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def record_from_row(row: dict[str, Any]) -> RagChunkRecord:
    """Milvus 行 → `RagChunkRecord`（``page`` 为 NULL 时保持 ``None``）。"""
    return RagChunkRecord(
        chunk_id=str(row.get("chunk_id") or ""),
        doc_id=str(row.get("doc_id") or ""),
        chunk_index=_as_int(row.get("chunk_index")),
        page=_as_int(row.get("page")) or None,
        content=str(row.get("content") or ""),
    )


def _record_order(record: RagChunkRecord) -> tuple[str, int]:
    """展示顺序：先按文档，再按块序号（同一篇文档按原文顺序阅读）。"""
    return record.doc_id, record.chunk_index
