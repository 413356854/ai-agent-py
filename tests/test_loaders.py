"""Document loading: encodings, metadata, directory walking, errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_agent_py.errors import DocumentLoadError
from ai_agent_py.rag.loaders import iter_files, load_documents, load_file


def test_loads_utf8_and_gbk_text(tmp_path: Path) -> None:
    markdown = tmp_path / "a.md"
    markdown.write_text("# 标题\n\n正文内容。", encoding="utf-8")
    legacy = tmp_path / "b.txt"
    legacy.write_bytes("中文内容：报销流程。".encode("gbk"))

    documents = load_documents([markdown, legacy])

    assert len(documents) == 2
    assert "标题" in documents[0].page_content
    assert "报销流程" in documents[1].page_content
    assert documents[0].metadata["source"].endswith("a.md")
    assert documents[0].metadata["file_type"] == "md"


def test_directory_walk_filters_unsupported_files(corpus_dir: Path) -> None:
    documents = load_documents([corpus_dir])
    names = {document.metadata["file_name"] for document in documents}
    assert names == {"uv.md", "langgraph.md", "notes.txt"}


def test_iter_files_is_sorted_and_deduplicated(corpus_dir: Path) -> None:
    files = iter_files([corpus_dir, corpus_dir / "uv.md"])
    assert len(files) == len({path.resolve() for path in files})
    assert [path.name for path in files] == sorted(path.name for path in files)


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="路径不存在"):
        load_documents([tmp_path / "nope"])


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    bad = tmp_path / "data.rst"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="不支持"):
        load_file(bad)


def test_empty_file_produces_no_documents(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text("   \n\n", encoding="utf-8")
    assert load_file(empty) == []
