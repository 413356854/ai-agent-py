"""``rag_chunk`` 只读查询：分页、过滤、排序、文档分布与异常提示。"""

from __future__ import annotations

import re
from typing import Any

import pytest

from ai_agent_py.config import Settings
from ai_agent_py.errors import MilvusError, MilvusSchemaError
from ai_agent_py.rag.records import (
    MilvusRagReader,
    build_filter,
)

_DOC_ID = re.compile(r'doc_id == "(.*)"')
_LIKE = re.compile(r'content like "%(.*)%"')
_COUNT = "count(*)"


def _row(chunk_id: str, doc_id: str, index: int, content: str, page: int | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "chunk_index": index,
        "page": page,
        "content": content,
    }


def _matches(row: dict[str, Any], expression: str) -> bool:
    """只实现测试用到的两种过滤：doc_id 相等 + content 中缀匹配。"""
    if not expression:
        return True
    for clause in expression.split(" and "):
        clause = clause.strip()
        doc = _DOC_ID.fullmatch(clause)
        if doc:
            if row["doc_id"] != doc.group(1).replace('\\"', '"'):
                return False
            continue
        like = _LIKE.fullmatch(clause)
        if like:
            needle = like.group(1).replace('\\"', '"')
            if needle not in row["content"]:
                return False
            continue
        raise AssertionError(f"测试未实现的过滤条件：{clause}")
    return True


class FakeMilvusClient:
    """只实现读路径用到的查询方法（模拟 Milvus query 的 limit/offset 语义）。"""

    def __init__(self, rows: list[dict[str, Any]], *, exists: bool = True) -> None:
        self.rows = rows
        self.exists = exists
        self.queries: list[dict[str, Any]] = []
        self.loaded = False

    def has_collection(self, name: str) -> bool:
        return self.exists

    def load_collection(self, name: str) -> None:
        self.loaded = True

    def query(
        self,
        collection: str,
        filter: str = "",
        output_fields: list[str] | None = None,
        timeout: float | None = None,
        ids: Any = None,
        partition_names: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        fields = list(output_fields or [])
        limit = int(kwargs.get("limit") or len(self.rows))
        offset = int(kwargs.get("offset") or 0)
        matched = [row for row in self.rows if _matches(row, filter)]
        self.queries.append({"filter": filter, "fields": fields, "limit": limit, "offset": offset})
        if fields == [_COUNT]:
            return [{_COUNT: len(matched)}]
        return [
            {field: row.get(field) for field in fields} for row in matched[offset : offset + limit]
        ]

    def get(self, collection: str, ids: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        wanted = set(ids)
        return [dict(row) for row in self.rows if row["chunk_id"] in wanted]


def make_reader(
    rows: list[dict[str, Any]], *, exists: bool = True
) -> tuple[MilvusRagReader, FakeMilvusClient]:
    settings = Settings(_env_file=None, milvus_database="db_tax_rag", milvus_collection="rag_chunk")
    client = FakeMilvusClient(rows, exists=exists)
    return MilvusRagReader(settings, client=client), client


def sample_rows() -> list[dict[str, Any]]:
    # 故意打乱存储顺序：按文档浏览时必须按 chunk_index 重新排序。
    return [
        _row("chunk-a2", "doc-a", 2, "业务招待费按发生额 60% 扣除。"),
        _row("chunk-b0", "doc-b", 0, "增值税应纳税额的计算。"),
        _row("chunk-a0", "doc-a", 0, "企业所得税的应纳税所得额。", page=1),
        _row("chunk-a3", "doc-a", 3, "广告费与业务宣传费的扣除限额。"),
        _row("chunk-b1", "doc-b", 1, "小规模纳税人的征收率。"),
        _row("chunk-a1", "doc-a", 1, "税前扣除凭证的要求。", page=2),
    ]


def test_list_all_sorts_current_page_and_reports_total() -> None:
    reader, _ = make_reader(sample_rows())

    result = reader.list_chunks(page=1, page_size=10)

    assert result.total == 6
    assert [item.chunk_id for item in result.items] == [
        "chunk-a0",
        "chunk-a1",
        "chunk-a2",
        "chunk-a3",
        "chunk-b0",
        "chunk-b1",
    ]
    assert result.items[0].page == 1
    assert result.items[1].page == 2
    assert result.truncated is False


def test_list_by_document_sorts_by_chunk_index_across_pages() -> None:
    reader, client = make_reader(sample_rows())

    first = reader.list_chunks(doc_id="doc-a", page=1, page_size=2)
    second = reader.list_chunks(doc_id="doc-a", page=2, page_size=2)

    assert first.total == 4
    assert [item.chunk_index for item in first.items] == [0, 1]
    assert [item.chunk_index for item in second.items] == [2, 3]
    # 正文确实是按主键回查出来的，不是元数据占位
    assert all(item.content for item in first.items + second.items)
    data_queries = [query for query in client.queries if query["fields"] != [_COUNT]]
    assert data_queries
    assert all(query["offset"] == 0 for query in data_queries)
    assert data_queries[0]["fields"] == ["chunk_id", "doc_id", "chunk_index", "page"]


def test_unfiltered_pages_are_globally_ordered() -> None:
    reader, _ = make_reader(sample_rows())

    first = reader.list_chunks(page=1, page_size=2)
    second = reader.list_chunks(page=2, page_size=2)

    assert [item.chunk_index for item in first.items] == [0, 1]
    assert [item.chunk_index for item in second.items] == [2, 3]


def test_large_collection_falls_back_to_server_paging(monkeypatch: pytest.MonkeyPatch) -> None:
    reader, client = make_reader(sample_rows())
    monkeypatch.setattr(reader, "count", lambda **kwargs: 20_000)

    result = reader.list_chunks(page=2, page_size=2)

    assert result.truncated is True
    assert [item.chunk_index for item in result.items] == [0, 3]
    data_queries = [query for query in client.queries if query["fields"] != [_COUNT]]
    assert data_queries[0]["offset"] == 2
    assert data_queries[0]["limit"] == 2


def test_keyword_filters_content_and_counts_matches() -> None:
    reader, _ = make_reader(sample_rows())

    result = reader.list_chunks(keyword="扣除", page=1, page_size=10)

    assert result.total == 3
    assert {item.doc_id for item in result.items} == {"doc-a"}
    assert all("扣除" in item.content for item in result.items)
    assert result.items[0].characters > 0


def test_page_beyond_total_returns_empty_items() -> None:
    reader, _ = make_reader(sample_rows())

    result = reader.list_chunks(page=9, page_size=10)

    assert result.total == 6
    assert result.items == []


def test_overview_reports_document_distribution() -> None:
    reader, client = make_reader(sample_rows())

    stats = reader.overview()

    assert stats.total_chunks == 6
    assert [(doc.doc_id, doc.chunks) for doc in stats.docs] == [("doc-a", 4), ("doc-b", 2)]
    assert stats.truncated is False
    assert client.loaded is True


def test_overview_marks_truncated_when_scan_is_limited() -> None:
    reader, client = make_reader(sample_rows())
    # 模拟 Milvus 返回行数少于总数（超过单次 query 上限的场景）
    original_query = client.query

    def limited_query(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        kwargs["limit"] = min(int(kwargs.get("limit") or 2), 2)
        return original_query(*args, **kwargs)

    client.query = limited_query  # type: ignore[method-assign]

    stats = reader.overview()

    assert stats.total_chunks == 6
    assert sum(doc.chunks for doc in stats.docs) == 2
    assert stats.truncated is True


def test_missing_collection_gives_actionable_error() -> None:
    reader, _ = make_reader([], exists=False)

    with pytest.raises(MilvusSchemaError, match=r"create_rag_chunk\.py"):
        reader.list_chunks()


def test_query_failure_is_wrapped_as_milvus_error() -> None:
    reader, client = make_reader(sample_rows())

    def boom(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("connection reset")

    client.query = boom  # type: ignore[method-assign]

    with pytest.raises(MilvusError, match="统计 rag_chunk 失败"):
        reader.list_chunks()


def test_build_filter_escapes_string_literals() -> None:
    expression = build_filter(doc_id='a"b\\c', keyword="100%")

    assert expression == 'doc_id == "a\\"b\\\\c" and content like "%100%%"'
    assert build_filter() == ""
    assert build_filter(doc_id="  doc-1  ", keyword="  ") == 'doc_id == "doc-1"'
