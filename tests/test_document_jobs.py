"""`tb_document_parse_job` 的行映射与建表脚本一致性测试（不连数据库）。"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone

from ai_agent_py.config import PROJECT_ROOT
from ai_agent_py.db.document_jobs import (
    TABLE_NAME,
    document_parse_job,
    from_row,
    to_params,
)
from ai_agent_py.mineru.models import (
    IngestInfo,
    JobState,
    ParseJob,
    ParseOptions,
    RagIngestInfo,
)
from fakes import InMemoryJobStore

_CREATED = datetime(2026, 11, 2, 13, 41, 42, 328626, tzinfo=UTC)

#: 建表脚本里「缩进 + 反引号列名 + 类型」的列定义行。
_COLUMN_DEF = re.compile(
    r"^\s+`(?P<name>[a-z_]+)`\s+(?P<type>VARCHAR|BIGINT|INT|TINYINT|DATETIME|DECIMAL)\b",
    re.MULTILINE,
)


def _full_job() -> ParseJob:
    return ParseJob(
        id="e7e290545dab450785d6aa9dc8c3b629",
        file_name="中华人民共和国发票管理办法_2023-07-20.docx",
        file_size=23997,
        suffix="docx",
        state=JobState.done,
        options=ParseOptions(
            model_version="vlm",
            language="chinese_cht",
            enable_formula=True,
            enable_table=True,
            is_ocr=False,
            page_ranges="1-10,12",
        ),
        created_at=_CREATED,
        updated_at=_CREATED + timedelta(seconds=9),
        started_at=_CREATED + timedelta(milliseconds=33),
        finished_at=_CREATED + timedelta(seconds=9, milliseconds=310),
        elapsed_seconds=9.28,
        batch_id="7d0e53d3-4427-46e3-a063-573be797a74c",
        err_msg="",
        extracted_pages=1,
        total_pages=1,
        source_file="source.docx",
        markdown_file="full.md",
        zip_file="mineru.zip",
        markdown_chars=4884,
        asset_count=0,
        ingest=IngestInfo(
            chunks=12,
            characters=4884,
            collection="rag_docs",
            store="Chroma",
            source="/srv/data/parsed/full.md",
            ingested_at=_CREATED + timedelta(minutes=1),
        ),
        rag_ingest=RagIngestInfo(
            chunks=12,
            characters=4884,
            collection="db_tax_rag.rag_chunk",
            ingested_at=_CREATED + timedelta(minutes=2),
        ),
    )


def test_table_has_every_persisted_column() -> None:
    assert set(to_params(_full_job())) == {column.name for column in document_parse_job.c}


def test_sql_script_and_python_table_agree() -> None:
    script = (PROJECT_ROOT / "docs" / "mysql" / f"{TABLE_NAME}.sql").read_text(encoding="utf-8")
    declared = {match["name"] for match in _COLUMN_DEF.finditer(script)}
    assert declared == {column.name for column in document_parse_job.c}


def test_roundtrip_keeps_every_field() -> None:
    job = _full_job()
    restored = from_row(to_params(job))
    assert restored.model_dump(mode="python") == job.model_dump(mode="python")


def test_roundtrip_without_ingest() -> None:
    job = _full_job().model_copy(
        update={
            "ingest": None,
            "rag_ingest": None,
            "started_at": None,
            "finished_at": None,
        }
    )
    restored = from_row(to_params(job))
    assert restored.ingest is None
    assert restored.rag_ingest is None
    assert restored.rag_ingested is False
    assert restored.started_at is None
    assert restored.finished_at is None
    assert restored.model_dump(mode="python") == job.model_dump(mode="python")


def test_timestamps_are_stored_as_naive_utc_and_read_back_aware() -> None:
    # 数据库会话时区是 +08:00，库里存的必须仍是 UTC，否则时间会整体漂移 8 小时。
    job = _full_job().model_copy(
        update={"created_at": datetime(2026, 11, 3, 8, 0, tzinfo=timezone(timedelta(hours=8)))}
    )
    params = to_params(job)
    assert params["created_at"] == datetime(2026, 11, 3, 0, 0)  # 朴素 UTC
    assert params["created_at"].tzinfo is None

    restored = from_row(params)
    assert restored.created_at == datetime(2026, 11, 3, 0, 0, tzinfo=UTC)
    assert restored.created_at.utcoffset() == timedelta(0)


def test_elapsed_seconds_keeps_two_decimals() -> None:
    params = to_params(_full_job())
    assert str(params["elapsed_seconds"]) == "9.28"
    assert from_row(params).elapsed_seconds == 9.28


def test_in_memory_store_is_a_deep_copy() -> None:
    store = InMemoryJobStore()
    job = _full_job()
    store.save(job)

    job.state = JobState.failed  # 改内存对象不应影响「库」里的记录
    loaded = store.load_all()[0]
    assert loaded.state is JobState.done
    assert loaded.id == job.id
    assert len(store) == 1

    loaded.err_msg = "本地修改"
    assert store.load_all()[0].err_msg == ""

    store.delete(job.id)
    assert len(store) == 0
    store.delete("不存在的 id")  # 静默返回
