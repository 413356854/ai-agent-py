"""`search_knowledge_base` 工具：文档级过滤（doc_ids）透传与默认范围。"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from ai_agent_py.rag.filters import SearchFilter
from ai_agent_py.tools import build_knowledge_tool


class RecordingSearcher:
    """记录每次检索的 query / k / filter，返回一条固定资料。"""

    def __init__(self, documents: list[Document] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.documents = (
            documents
            if documents is not None
            else [Document(page_content="住宿费标准：每晚 500 元", metadata={"source": "制度.docx"})]
        )

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: SearchFilter | None = None,
    ) -> list[Document]:
        self.calls.append({"query": query, "k": k, "filter": filter})
        return list(self.documents)


def test_tool_without_doc_ids_does_not_filter() -> None:
    searcher = RecordingSearcher()
    tool = build_knowledge_tool(searcher)

    text = tool.invoke({"query": "住宿费标准"})

    assert searcher.calls == [{"query": "住宿费标准", "k": None, "filter": None}]
    assert text.startswith("[1] 来源：制度.docx")


def test_tool_forwards_doc_ids_as_typed_filter() -> None:
    searcher = RecordingSearcher()
    tool = build_knowledge_tool(searcher)

    tool.invoke({"query": "住宿费标准", "doc_ids": [" doc-a ", "doc-b", "doc-a"]})

    assert searcher.calls[0]["filter"] == SearchFilter(doc_ids=("doc-a", "doc-b"))


def test_tool_default_doc_ids_apply_when_model_omits_them() -> None:
    searcher = RecordingSearcher()
    tool = build_knowledge_tool(searcher, default_doc_ids=["doc-a"])

    tool.invoke({"query": "住宿费标准"})

    assert searcher.calls[0]["filter"] == SearchFilter(doc_ids=("doc-a",))


def test_tool_doc_ids_override_default_scope() -> None:
    searcher = RecordingSearcher()
    tool = build_knowledge_tool(searcher, default_doc_ids=["doc-a"])

    tool.invoke({"query": "住宿费标准", "doc_ids": ["doc-b"]})

    assert searcher.calls[0]["filter"] == SearchFilter(doc_ids=("doc-b",))


def test_empty_doc_ids_fall_back_to_default_scope() -> None:
    searcher = RecordingSearcher()
    tool = build_knowledge_tool(searcher, default_doc_ids=["doc-a"])

    tool.invoke({"query": "住宿费标准", "doc_ids": []})
    tool.invoke({"query": "住宿费标准", "doc_ids": ["  "]})

    assert [call["filter"] for call in searcher.calls] == [
        SearchFilter(doc_ids=("doc-a",)),
        SearchFilter(doc_ids=("doc-a",)),
    ]


def test_tool_reports_empty_result() -> None:
    tool = build_knowledge_tool(RecordingSearcher(documents=[]))

    assert tool.invoke({"query": "不存在的问题"}) == "未检索到相关资料。"
