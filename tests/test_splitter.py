"""Text splitting: 后缀路由、结构化切分器、块大小与编号。"""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from ai_agent_py.rag.splitter import (
    split_documents,
    split_text,
    splitter_for_suffix,
    splitter_label,
)


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
    assert split_text("   \n", suffix=".md", settings=settings) == []


def test_invalid_overlap_is_rejected(settings) -> None:
    document = Document(page_content="内容", metadata={})
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_documents([document], settings=settings, chunk_size=50, chunk_overlap=50)
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_text("内容", suffix=".md", settings=settings, chunk_size=50, chunk_overlap=50)


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".md", "markdown"),
        ("MD", "markdown"),
        ("notes.markdown", "markdown"),
        (".html", "html"),
        (".HTM", "html"),
        ("page.html", "html"),
        (".json", "json"),
        (".pdf", "recursive"),
        (".docx", "recursive"),
        ("", "recursive"),
        (None, "recursive"),
    ],
)
def test_suffix_selects_splitter(suffix: str | None, expected: str) -> None:
    assert splitter_for_suffix(suffix) == expected


def test_markdown_splitter_keeps_header_metadata(settings) -> None:
    text = "# 税法\n\n企业所得税税率 25%。\n\n## 增值税\n\n标准税率 13%。\n"

    chunks = split_text(text, suffix=".md", settings=settings, chunk_size=200, chunk_overlap=0)

    assert [chunk.metadata["h1"] for chunk in chunks] == ["税法", "税法"]
    assert chunks[0].metadata["splitter"] == "markdown"
    assert chunks[1].metadata["h2"] == "增值税"
    assert "标准税率" in chunks[1].page_content


def test_html_splitter_merges_heading_only_chunks(settings) -> None:
    html = "<h1>税法</h1><p>企业所得税。</p><h2>税率</h2><p>25%。</p>"

    chunks = split_text(html, suffix=".html", settings=settings, chunk_size=200, chunk_overlap=0)

    assert len(chunks) == 2
    assert chunks[0].page_content.startswith("税法")
    assert "企业所得税" in chunks[0].page_content
    assert chunks[1].metadata["h2"] == "税率"
    assert chunks[0].metadata["splitter"] == "html"


def test_json_splitter_keeps_valid_json_chunks(settings) -> None:
    payload = {
        "公司": {"名称": "示例", "税种": ["企业所得税", "增值税"] * 6},
        "备注": "x" * 120,
    }

    chunks = split_text(
        json.dumps(payload, ensure_ascii=False),
        suffix=".json",
        settings=settings,
        chunk_size=100,
        chunk_overlap=0,
    )

    assert len(chunks) > 1
    assert all(json.loads(chunk.page_content) is not None for chunk in chunks)
    assert all(chunk.metadata["splitter"] == "json" for chunk in chunks)
    # 单个超长字符串值允许略超 chunk_size（再切就不是合法 JSON），但不应无限膨胀。
    assert max(len(chunk.page_content) for chunk in chunks) <= 140


def test_invalid_json_falls_back_to_text(settings) -> None:
    chunks = split_text("{broken", suffix=".json", settings=settings, chunk_size=200)

    assert len(chunks) == 1
    assert "json_error" in chunks[0].metadata
    assert chunks[0].metadata["splitter"] == "json"


def test_oversized_structured_section_is_split_and_keeps_headers(settings) -> None:
    text = "# 长章节\n\n" + "这是一段很长的正文。" * 30

    chunks = split_text(text, suffix=".md", settings=settings, chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 80 for chunk in chunks)
    assert all(chunk.metadata["h1"] == "长章节" for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_split_documents_uses_each_source_suffix(settings) -> None:
    documents = [
        Document(page_content="# 标题\n\n正文。" * 20, metadata={"source": "a.md"}),
        Document(page_content="第一段。第二段。" * 20, metadata={"source": "b.pdf", "page": 3}),
    ]

    chunks = split_documents(documents, settings=settings, chunk_size=80, chunk_overlap=10)

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    markdown_chunks = [chunk for chunk in chunks if chunk.metadata["source"] == "a.md"]
    pdf_chunks = [chunk for chunk in chunks if chunk.metadata["source"] == "b.pdf"]
    assert all(chunk.metadata["splitter"] == "markdown" for chunk in markdown_chunks)
    assert all(chunk.metadata["splitter"] == "recursive" for chunk in pdf_chunks)
    assert all(chunk.metadata["page"] == 3 for chunk in pdf_chunks)


def test_splitter_label_is_chinese() -> None:
    assert splitter_label("markdown") == "Markdown 标题切分"
    assert splitter_label("unknown") == "unknown"
