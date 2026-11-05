"""Milvus `rag_chunk` 写入器：字段、批大小、覆盖写与 schema 校验。"""

from __future__ import annotations

import re
from typing import Any

import pytest
from langchain_core.documents import Document

from ai_agent_py.config import Settings
from ai_agent_py.errors import MilvusError, MilvusSchemaError
from ai_agent_py.offline import LexicalHashEmbeddings
from ai_agent_py.rag.milvus import MilvusRagWriter, chunk_id


class FakeMilvusClient:
    """只实现 MilvusRagWriter 用到的 6 个方法，便于离线验证写入逻辑。"""

    def __init__(self, *, dim: int = 1536, fields: tuple[str, ...] | None = None) -> None:
        self.dim = dim
        self.fields = fields or (
            "chunk_id",
            "doc_id",
            "chunk_index",
            "page",
            "content",
            "embedding",
            "sparse",
        )
        self.rows: dict[str, dict[str, Any]] = {}
        self.batches: list[int] = []
        #: 按调用顺序记录 ("delete", filter) / ("upsert", 行数)，用于验证「先删后写」。
        self.operations: list[tuple[str, Any]] = []
        self.flushes = 0
        self.loaded = False
        self.exists = True

    def has_collection(self, name: str) -> bool:
        return self.exists

    def describe_collection(self, name: str) -> dict[str, Any]:
        params = {"dim": self.dim} if self.dim else {}
        return {"fields": [{"name": field, "params": params} for field in self.fields]}

    def load_collection(self, name: str) -> None:
        self.loaded = True

    def query(
        self,
        collection: str,
        *,
        filter: str,
        output_fields: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        match = re.search(r'doc_id == "(.*)"', filter)
        assert match, filter
        doc_id = match.group(1)
        return [
            {"chunk_id": row["chunk_id"]} for row in self.rows.values() if row["doc_id"] == doc_id
        ][:limit]

    def delete(self, collection: str, *, filter: str) -> dict[str, int]:
        self.operations.append(("delete", filter))
        doc_match = re.search(r'doc_id == "(.*)"', filter)
        if doc_match:
            doc_id = doc_match.group(1)
            before = len(self.rows)
            self.rows = {key: row for key, row in self.rows.items() if row["doc_id"] != doc_id}
            return {"delete_count": before - len(self.rows)}
        removed = 0
        for value in re.findall(r'"(chunk-[^"]+)"', filter):
            removed += 1 if self.rows.pop(value, None) is not None else 0
        return {"delete_count": removed}

    def upsert(self, collection: str, rows: list[dict[str, Any]]) -> None:
        self.operations.append(("upsert", len(rows)))
        self.batches.append(len(rows))
        for row in rows:
            assert "sparse" not in row, "BM25 稀疏向量必须由 Milvus 生成"
            self.rows[row["chunk_id"]] = dict(row)

    def flush(self, collection: str) -> None:
        self.flushes += 1


def make_writer(*, dim: int = 1536, embedding_batch: int = 2, batch: int = 2):
    settings = Settings(
        _env_file=None,
        milvus_database="db_tax_rag",
        milvus_collection="rag_chunk",
        milvus_dim=dim,
        embedding_batch_size=embedding_batch,
        milvus_batch_size=batch,
    )
    client = FakeMilvusClient(dim=dim)
    writer = MilvusRagWriter(
        settings,
        client=client,
        embeddings=LexicalHashEmbeddings(dimensions=dim),
    )
    return writer, client


def test_writer_upserts_rows_without_sparse_vector() -> None:
    writer, client = make_writer()

    chunks = [
        Document(page_content="第一块。", metadata={"chunk_index": 0}),
        Document(page_content="第二块。", metadata={"chunk_index": 1, "page": 2}),
        Document(page_content="第三块。", metadata={"chunk_index": 2}),
    ]
    report = writer.index_document("doc-1", chunks)

    assert report.chunks == 3
    assert report.characters == len("第一块。第二块。第三块。")
    assert report.collection == "rag_chunk"
    assert client.loaded is True
    assert client.flushes == 1
    assert client.batches == [2, 1]
    assert len(client.rows) == 3

    row = client.rows[chunk_id("doc-1", 1, "第二块。")]
    assert row["doc_id"] == "doc-1"
    assert row["chunk_index"] == 1
    assert row["page"] == 2
    assert len(row["embedding"]) == 1536


def test_reindex_removes_stale_chunks_and_is_idempotent() -> None:
    writer, client = make_writer()
    chunks = [
        Document(page_content="第一块。", metadata={}),
        Document(page_content="第二块。", metadata={}),
    ]
    writer.index_document("doc-1", chunks)
    first_ids = {chunk_id("doc-1", 0, "第一块。"), chunk_id("doc-1", 1, "第二块。")}
    assert set(client.rows) == first_ids

    report = writer.index_document("doc-1", chunks[:1])

    assert report.chunks == 1
    assert set(client.rows) == {chunk_id("doc-1", 0, "第一块。")}
    # 内容没变的块主键不变，重复入库不会产生副本。
    writer.index_document("doc-1", chunks[:1])
    assert set(client.rows) == {chunk_id("doc-1", 0, "第一块。")}


def test_reindex_deletes_every_old_row_before_upserting() -> None:
    """重复入库：先删该 doc_id 的全部旧切片，再写新切片（允许换切分参数）。"""
    writer, client = make_writer()
    writer.index_document(
        "doc-1",
        [
            Document(page_content="旧块一", metadata={}),
            Document(page_content="旧块二", metadata={}),
            Document(page_content="旧块三", metadata={}),
        ],
    )
    client.operations.clear()

    new_contents = ["新块一", "新块二", "新块三", "新块四"]
    report = writer.index_document(
        "doc-1",
        [Document(page_content=content, metadata={}) for content in new_contents],
    )

    assert report.chunks == 4
    # 第一次写操作必须是删除旧切片，之后才是 upsert
    assert next(name for name, _ in client.operations) == "delete"
    assert all(name == "upsert" for name, _ in client.operations[1:])
    assert set(client.rows) == {
        chunk_id("doc-1", index, content) for index, content in enumerate(new_contents)
    }
    assert not any(row["content"].startswith("旧块") for row in client.rows.values())


def test_empty_chunks_delete_existing_rows() -> None:
    writer, client = make_writer()
    writer.index_document("doc-1", [Document(page_content="内容", metadata={})])

    report = writer.index_document("doc-1", [])

    assert report.chunks == 0
    assert client.rows == {}
    assert client.flushes == 2


def test_delete_document_removes_every_chunk_and_flushes() -> None:
    """删除对称：公开的 delete_document 一次清空该文档，别的文档不受影响。"""
    writer, client = make_writer()
    writer.index_document(
        "doc-1",
        [
            Document(page_content="第一块。", metadata={}),
            Document(page_content="第二块。", metadata={}),
        ],
    )
    writer.index_document("doc-2", [Document(page_content="别的文档。", metadata={})])
    client.operations.clear()
    client.flushes = 0

    assert writer.delete_document("doc-1") == 2
    assert {row["doc_id"] for row in client.rows.values()} == {"doc-2"}
    assert client.operations == [("delete", 'doc_id == "doc-1"')]
    assert client.flushes == 1


def test_delete_document_skips_missing_collection() -> None:
    """集合还没建 / 换库后重试补删：没有东西可删，视为删除成功。"""
    writer, client = make_writer()
    client.exists = False

    assert writer.delete_document("doc-1") == 0
    assert client.operations == []
    assert client.loaded is False


def test_delete_document_wraps_milvus_errors() -> None:
    writer, client = make_writer()

    def boom(collection: str, *, filter: str) -> dict[str, int]:
        raise RuntimeError("connection refused")

    client.delete = boom  # type: ignore[method-assign]

    with pytest.raises(MilvusError, match="删除 Milvus 切片失败"):
        writer.delete_document("doc-1")


def test_dimension_mismatch_is_rejected() -> None:
    writer, client = make_writer(dim=1024)
    client.dim = 1536  # 线上集合仍是 1536 维，配置却改成了 1024

    with pytest.raises(MilvusSchemaError, match="向量维度"):
        writer.index_document("doc-1", [Document(page_content="内容", metadata={})])


def test_missing_field_is_rejected() -> None:
    writer, _ = make_writer()
    writer._client.fields = ("chunk_id", "doc_id", "chunk_index", "content", "embedding")

    with pytest.raises(MilvusSchemaError, match="缺少字段"):
        writer.index_document("doc-1", [Document(page_content="内容", metadata={})])
