"""Text splitting: chunk numbering, size bounds, validation."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from ai_agent_py.rag.splitter import split_documents


def test_chunks_are_numbered_and_bounded(settings) -> None:
    document = Document(
        page_content="第一条规则。第二条规则。第三条规则。" * 40,
        metadata={"source": "rules.md"},
    )

    chunks = split_documents([document], settings=settings, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(len(chunk.page_content) <= 120 for chunk in chunks)
    assert all(chunk.metadata["source"] == "rules.md" for chunk in chunks)
    assert all(chunk.page_content.strip() == chunk.page_content for chunk in chunks)


def test_empty_input_returns_nothing(settings) -> None:
    assert split_documents([], settings=settings) == []


def test_invalid_overlap_is_rejected(settings) -> None:
    document = Document(page_content="内容", metadata={})
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_documents([document], settings=settings, chunk_size=50, chunk_overlap=50)
