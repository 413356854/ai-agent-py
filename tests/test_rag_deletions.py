"""RAG 补删队列：落盘、重试次数、路径安全与损坏标记容错（全部离线）。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_agent_py.rag.deletions import PendingRagDelete, RagDeleteQueue


def _marker(doc_id: str, *, created_at: datetime) -> PendingRagDelete:
    return PendingRagDelete(doc_id=doc_id, created_at=created_at, updated_at=created_at)


def test_record_and_read_roundtrip(tmp_path: Path) -> None:
    queue = RagDeleteQueue(tmp_path / "pending")
    queue.record(
        "job-1",
        file_name="税法.pdf",
        collection="db_tax_rag.rag_chunk",
        error="Milvus 连接失败",
    )

    assert len(queue) == 1
    item = queue.items()[0]
    assert item.doc_id == "job-1"
    assert item.file_name == "税法.pdf"
    assert item.collection == "db_tax_rag.rag_chunk"
    assert item.attempts == 1
    assert "Milvus" in item.last_error
    assert item.created_at.tzinfo is not None


def test_record_accumulates_attempts(tmp_path: Path) -> None:
    queue = RagDeleteQueue(tmp_path / "pending")
    first = queue.record("job-1", file_name="a.pdf", error="第一次")
    second = queue.record("job-1", error="第二次")

    assert second.attempts == 2
    assert second.created_at == first.created_at
    # 后续重试没传文件名 / 集合时保留旧值，排障不会丢信息
    assert second.file_name == "a.pdf"
    assert second.last_error == "第二次"
    assert len(queue.items()) == 1


def test_discard_removes_marker_and_is_idempotent(tmp_path: Path) -> None:
    queue = RagDeleteQueue(tmp_path / "pending")
    queue.record("job-1")

    queue.discard("job-1")
    assert queue.items() == []

    queue.discard("job-1")  # 已经不存在时静默返回
    assert queue.items() == []


def test_items_are_ordered_by_creation_time(tmp_path: Path) -> None:
    directory = tmp_path / "pending"
    directory.mkdir()
    later = _marker("job-later", created_at=datetime(2026, 1, 2, tzinfo=UTC))
    earlier = _marker("job-earlier", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    for item in (later, earlier):
        (directory / f"{item.doc_id}.json").write_text(json.dumps(item.as_dict()), encoding="utf-8")

    assert [item.doc_id for item in RagDeleteQueue(directory).items()] == [
        "job-earlier",
        "job-later",
    ]


def test_doc_id_cannot_escape_directory(tmp_path: Path) -> None:
    """doc_id 来自 API 路径参数：拼文件名前必须清洗，不能删到目录外面。"""
    queue = RagDeleteQueue(tmp_path / "pending")
    queue.record("../../etc/passwd", error="x")

    files = list((tmp_path / "pending").glob("*.json"))
    assert len(files) == 1
    assert files[0].parent == tmp_path / "pending"
    assert queue.items()[0].doc_id == "../../etc/passwd"


def test_corrupt_marker_is_ignored(tmp_path: Path) -> None:
    directory = tmp_path / "pending"
    directory.mkdir()
    (directory / "broken.json").write_text("{not json", encoding="utf-8")
    (directory / "empty.json").write_text("{}", encoding="utf-8")
    (directory / "list.json").write_text(json.dumps([1, 2]), encoding="utf-8")

    queue = RagDeleteQueue(directory)
    queue.record("job-1")

    assert [item.doc_id for item in queue.items()] == ["job-1"]


def test_tmp_file_is_not_left_behind(tmp_path: Path) -> None:
    queue = RagDeleteQueue(tmp_path / "pending")
    queue.record("job-1")

    assert list((tmp_path / "pending").glob("*.tmp")) == []


def test_from_dict_defaults_missing_fields() -> None:
    item = PendingRagDelete.from_dict({"doc_id": "job-1"})

    assert item.doc_id == "job-1"
    assert item.file_name is None
    assert item.collection is None
    assert item.attempts == 0
    assert item.last_error == ""
    assert item.created_at.tzinfo is not None
