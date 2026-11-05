"""检索后按 doc_id 回查 MySQL 文件名：批量、缓存、去重与降级。"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document

from ai_agent_py.config import Settings
from ai_agent_py.errors import DatabaseError
from ai_agent_py.rag.doc_names import MySQLDocumentNameResolver, apply_document_names


class FakeStore:
    """记录调用参数的 ``DocumentNameStore`` 内存实现。"""

    def __init__(self, names: dict[str, str] | None = None) -> None:
        self.names = dict(names or {})
        self.calls: list[list[str]] = []

    def load_file_names(self, doc_ids: Sequence[str]) -> dict[str, str]:
        self.calls.append(list(doc_ids))
        return {doc_id: self.names[doc_id] for doc_id in doc_ids if doc_id in self.names}


class BrokenStore:
    def __init__(self) -> None:
        self.calls = 0

    def load_file_names(self, doc_ids: Sequence[str]) -> dict[str, str]:
        self.calls += 1
        raise DatabaseError("MySQL 未启动")


class StubResolver:
    def __init__(self, names: dict[str, str] | None = None) -> None:
        self.names = dict(names or {})
        self.calls: list[list[str]] = []

    def resolve(self, doc_ids: Sequence[str]) -> dict[str, str]:
        self.calls.append(list(doc_ids))
        return {doc_id: self.names[doc_id] for doc_id in doc_ids if doc_id in self.names}


def _document(doc_id: str = "doc-a", source: str = "doc-a") -> Document:
    return Document(
        page_content="业务招待费按发生额的 60% 扣除。",
        metadata={"doc_id": doc_id, "chunk_id": f"chunk-{doc_id}", "source": source},
    )


def test_resolve_batches_unique_ids_and_caches_results() -> None:
    store = FakeStore({"doc-a": "企业所得税法.docx", "doc-b": "增值税法.docx"})
    resolver = MySQLDocumentNameResolver(Settings(_env_file=None), store=store)

    assert resolver.resolve(["doc-a", "doc-a", "doc-b"]) == {
        "doc-a": "企业所得税法.docx",
        "doc-b": "增值税法.docx",
    }
    # 重复 id 只查一次，且两条一次查完
    assert store.calls == [["doc-a", "doc-b"]]

    # 已缓存的 id 不再查库，只补查没见过的
    assert resolver.resolve(["doc-b", "doc-missing"]) == {"doc-b": "增值税法.docx"}
    assert store.calls == [["doc-a", "doc-b"], ["doc-missing"]]


def test_resolve_degrades_and_cools_down_when_mysql_fails() -> None:
    store = BrokenStore()
    resolver = MySQLDocumentNameResolver(Settings(_env_file=None), store=store)

    assert resolver.resolve(["doc-a"]) == {}
    # 冷却期内不再重试，避免每轮问答都撞一次连接超时
    assert resolver.resolve(["doc-b"]) == {}
    assert store.calls == 1


def test_unconfigured_mysql_is_skipped_silently() -> None:
    resolver = MySQLDocumentNameResolver(Settings(_env_file=None))

    assert resolver.resolve(["doc-a"]) == {}


def test_apply_document_names_rewrites_source_for_each_chunk() -> None:
    documents = [_document(), _document(), _document("doc-b", "doc-b")]
    resolver = StubResolver({"doc-a": "企业所得税法.docx", "doc-b": "增值税法.docx"})

    result = apply_document_names(documents, resolver)

    assert result == documents
    assert [document.metadata["source"] for document in result] == [
        "企业所得税法.docx",
        "企业所得税法.docx",
        "增值税法.docx",
    ]
    assert [document.metadata["doc_name"] for document in result] == [
        "企业所得税法.docx",
        "企业所得税法.docx",
        "增值税法.docx",
    ]
    # 同一篇文档的多个片段只解析一次
    assert resolver.calls == [["doc-a", "doc-b"]]


def test_apply_keeps_doc_id_when_name_is_unknown() -> None:
    document = _document()
    resolver = StubResolver()

    apply_document_names([document], resolver)

    assert document.metadata["source"] == "doc-a"
    assert "doc_name" not in document.metadata


def test_apply_swallows_resolver_errors() -> None:
    class BrokenResolver:
        def resolve(self, doc_ids: Sequence[str]) -> dict[str, str]:
            raise RuntimeError("解析器自身出错")

    document = _document()

    apply_document_names([document], BrokenResolver())

    assert document.metadata["source"] == "doc-a"


def test_apply_without_resolver_is_a_noop() -> None:
    document = _document()

    assert apply_document_names([document], None) == [document]
    assert document.metadata["source"] == "doc-a"
