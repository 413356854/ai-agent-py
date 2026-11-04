"""文档解析 HTTP 接口测试（TestClient + 假 MinerU 客户端，不联网）。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ai_agent_py.api.app import create_app
from ai_agent_py.api.documents import _PAGE_RANGES as PAGE_RANGES
from ai_agent_py.config import Settings
from ai_agent_py.errors import ConfigurationError
from ai_agent_py.mineru.client import MineruClient
from ai_agent_py.mineru.service import DocumentService
from ai_agent_py.rag.ingest import IngestReport
from ai_agent_py.rag.milvus import RagIngestReport
from fakes import InMemoryJobStore
from test_mineru import FakeMineru

TERMINAL = {"done", "failed"}


def _app(tmp_path: Path, fake: FakeMineru, **overrides: Any) -> tuple[TestClient, DocumentService]:
    settings = Settings(
        _env_file=None,
        app_env="test",
        mineru_api_key="test-token",
        mineru_output_dir=tmp_path / "parsed",
        mineru_poll_interval=0,
        mineru_timeout=10,
        **overrides,
    )
    service = DocumentService(
        settings,
        client_factory=lambda: MineruClient(
            "test-token", settings.mineru_base_url, client=fake.client()
        ),
        store=InMemoryJobStore(),
    )
    app = create_app()
    app.state.documents = service
    return TestClient(app), service


def _wait(client: TestClient, job_id: str, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/documents/{job_id}").json()
        if body["state"] in TERMINAL:
            return body
        time.sleep(0.02)
    raise AssertionError(f"任务 {job_id} 在 {timeout}s 内没有结束")


def test_config_reports_capabilities(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, FakeMineru())
    with client:
        body = client.get("/api/documents/config").json()
    assert body["configured"] is True
    assert body["model_version"] == "vlm"
    assert set(body["model_versions"]) == {"vlm", "pipeline", "MinerU-HTML"}
    assert ".pdf" in body["accepted_suffixes"]
    assert {".md", ".json", ".html"} <= set(body["accepted_suffixes"])
    assert body["max_file_mb"] == 200
    assert body["rag_configured"] is False
    assert body["rag_database"] == "db_tax_rag"
    assert body["rag_collection"] == "rag_chunk"


def test_upload_parse_preview_download_delete(tmp_path: Path) -> None:
    fake = FakeMineru()
    client, _ = _app(tmp_path, fake)
    with client:
        response = client.post(
            "/api/documents/parse",
            files={"file": ("测试 文档.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={
                "model_version": "vlm",
                "language": "en",
                "enable_table": "false",
                "page_ranges": "1-2",
            },
        )
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["file_name"] == "测试 文档.pdf"
        assert job["file_size"] == len(b"%PDF-1.4 fake")
        assert job["suffix"] == "pdf"
        assert job["state_label"]

        done = _wait(client, job["id"])
        assert done["state"] == "done", done["err_msg"]
        assert done["markdown_file"] == "full.md"
        assert done["total_pages"] == 2
        assert done["progress"] == 1.0
        assert done["assets_base"] == f"/parsed/{job['id']}/"

        listing = client.get("/api/documents").json()
        assert listing[0]["id"] == job["id"]

        preview = client.get(f"/api/documents/{job['id']}/markdown")
        assert preview.status_code == 200
        assert "测试文档" in preview.json()["markdown"]
        assert preview.json()["assets_base"] == f"/parsed/{job['id']}/"

        markdown_file = client.get(
            f"/api/documents/{job['id']}/download", params={"kind": "markdown"}
        )
        assert markdown_file.status_code == 200
        assert "测试文档" in markdown_file.text

        archive = client.get(f"/api/documents/{job['id']}/download", params={"kind": "zip"})
        assert archive.status_code == 200
        assert archive.content[:2] == b"PK"

        asset = client.get(f"/parsed/{job['id']}/images/img0.jpg")
        assert asset.status_code == 200

        assert client.delete(f"/api/documents/{job['id']}").status_code == 204
        assert client.get(f"/api/documents/{job['id']}").status_code == 404
        assert client.get("/api/documents").json() == []

    # 解析参数确实透传给了 MinerU
    payload = fake.batch_requests[0]
    assert payload["language"] == "en"
    assert payload["enable_table"] is False
    assert payload["files"][0]["name"] == "测试 文档.pdf"
    assert payload["files"][0]["page_ranges"] == "1-2"
    assert fake.uploaded == b"%PDF-1.4 fake"


def test_uploaded_service_renders_parsed_assets(tmp_path: Path) -> None:
    """`/parsed` 静态目录与任务目录使用同一份配置。"""
    client, service = _app(tmp_path, FakeMineru())
    with client:
        job = client.post(
            "/api/documents/parse", files={"file": ("a.pdf", b"%PDF", "application/pdf")}
        ).json()
        _wait(client, job["id"])
        assert (service.job_dir(job["id"]) / "images" / "img0.jpg").is_file()


def test_upload_rejects_unsupported_type(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, FakeMineru())
    with client:
        response = client.post(
            "/api/documents/parse", files={"file": ("evil.exe", b"MZ", "application/octet-stream")}
        )
    assert response.status_code == 400
    assert "不支持" in response.json()["detail"]


def test_upload_validates_model_version(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, FakeMineru())
    with client:
        response = client.post(
            "/api/documents/parse",
            files={"file": ("a.pdf", b"%PDF", "application/pdf")},
            data={"model_version": "gpt-9"},
        )
    assert response.status_code == 422


@pytest.mark.parametrize("value", ["7", "1-2", "2,4-6", "2--2", "-2", "1-10,12"])
def test_page_ranges_accepts_documented_formats(value: str) -> None:
    assert PAGE_RANGES.match(value)


@pytest.mark.parametrize("value", ["", "abc", "1-", "1;2", "1--", "1-2-3", "1-2a"])
def test_page_ranges_rejects_garbage(value: str) -> None:
    assert not PAGE_RANGES.match(value)


def test_unknown_job_returns_404(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, FakeMineru())
    with client:
        assert client.get("/api/documents/nope").status_code == 404
        assert client.get("/api/documents/nope/markdown").status_code == 404
        assert client.delete("/api/documents/nope").status_code == 404


def test_ingest_requires_embeddings_configuration(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, FakeMineru())
    with client:
        job = client.post(
            "/api/documents/parse", files={"file": ("a.pdf", b"%PDF", "application/pdf")}
        ).json()
        _wait(client, job["id"])
        response = client.post(f"/api/documents/{job['id']}/ingest")
    assert response.status_code == 503
    assert "EMBEDDING_API_KEY" in response.json()["detail"]


@pytest.mark.asyncio
async def test_ingest_writes_to_vector_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, service = _app(tmp_path, FakeMineru())

    def fake_ingest(path: Path) -> IngestReport:
        assert path.name == "full.md"
        return IngestReport(
            files=1,
            documents=1,
            chunks=3,
            characters=42,
            batches=1,
            store="InMemoryVectorStore",
            collection="rag_docs",
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(DocumentService, "_ingest_sync", lambda self, path: fake_ingest(path))

    with client:
        job = client.post(
            "/api/documents/parse", files={"file": ("a.pdf", b"%PDF", "application/pdf")}
        ).json()
        _wait(client, job["id"])
        response = client.post(f"/api/documents/{job['id']}/ingest")
        assert response.status_code == 200, response.text
        assert response.json()["chunks"] == 3
        detail = client.get(f"/api/documents/{job['id']}").json()
        assert detail["ingest"]["chunks"] == 3
    assert await service.get(job["id"]) is not None


def test_markdown_upload_is_directly_splittable(tmp_path: Path) -> None:
    """`.md` 不走 MinerU：上传后立即完成，切分器选 Markdown。"""
    client, _ = _app(tmp_path, FakeMineru())
    content = "# 税法\n\n企业所得税税率 25%。\n\n## 增值税\n\n标准税率 13%。\n"

    with client:
        response = client.post(
            "/api/documents/parse",
            files={"file": ("notes.md", content.encode(), "text/markdown")},
        )
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["state"] == "done"
        assert job["suffix"] == "md"
        assert job["options"]["model_version"] == "direct"
        assert job["markdown_file"] == "source.md"

        chunks = client.get(f"/api/documents/{job['id']}/chunks").json()
        assert chunks["source_file"] == "source.md"
        assert chunks["split_id"]
        assert chunks["strategy"] == "markdown"
        assert chunks["strategy_label"] == "Markdown 标题切分"
        assert chunks["total"] >= 2
        assert chunks["chunks"][0]["metadata"]["h1"] == "税法"
        assert all(chunk["content"] for chunk in chunks["chunks"])

        # limit 只影响回传的片段数，用于前端提示「未全部展示」。
        limited = client.get(f"/api/documents/{job['id']}/chunks", params={"limit": 1}).json()
        assert limited["truncated"] is True
        assert len(limited["chunks"]) == 1

        # 预览 / 下载仍可读
        preview = client.get(f"/api/documents/{job['id']}/markdown")
        assert preview.status_code == 200
        assert "增值税" in preview.json()["markdown"]


def test_json_upload_is_validated_and_split_by_structure(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, FakeMineru())
    payload = {"公司": {"名称": "示例", "税种": ["企业所得税", "增值税"] * 6}}

    with client:
        response = client.post(
            "/api/documents/parse",
            files={
                "file": (
                    "tax.json",
                    json.dumps(payload, ensure_ascii=False).encode(),
                    "application/json",
                )
            },
        )
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["state"] == "done"
        assert job["markdown_file"] == "full.md"

        markdown = client.get(f"/api/documents/{job['id']}/markdown").json()["markdown"]
        assert "```json" in markdown

        chunks = client.get(f"/api/documents/{job['id']}/chunks", params={"chunk_size": 100}).json()
        assert chunks["source_file"] == "full.md"
        assert chunks["strategy"] == "json"
        assert chunks["total"] > 1
        for chunk in chunks["chunks"]:
            json.loads(chunk["content"])


def test_invalid_json_upload_is_rejected(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, FakeMineru())

    with client:
        response = client.post(
            "/api/documents/parse",
            files={"file": ("broken.json", b"{broken", "application/json")},
        )
        assert response.status_code == 400
        assert "JSON" in response.json()["detail"]
        assert client.get("/api/documents").json() == []


def test_split_reads_markdown_file_not_original_source(tmp_path: Path) -> None:
    """切分只读 markdown_file：原始 HTML 删掉也不影响，策略是 Markdown。"""
    fake = FakeMineru()
    client, service = _app(tmp_path, fake)
    html = "<h1>税法</h1><p>企业所得税。</p><h2>税率</h2><p>25%。</p>"

    with client:
        job = client.post(
            "/api/documents/parse", files={"file": ("page.html", html.encode(), "text/html")}
        ).json()
        _wait(client, job["id"])
        job_dir = service.job_dir(job["id"])
        assert (job_dir / "source.html").is_file()
        # 切分只认 full.md：原始 HTML 不再参与切分
        (job_dir / "source.html").unlink()

        response = client.get(f"/api/documents/{job['id']}/chunks")
        assert response.status_code == 200, response.text
        chunks = response.json()
        assert chunks["source_file"] == "full.md"
        assert chunks["strategy"] == "markdown"
        assert chunks["strategy_label"] == "Markdown 标题切分"
        assert chunks["chunks"][0]["metadata"]["h1"] == "测试文档"

    # 没显式指定模型时，HTML 自动走 MinerU-HTML。
    assert fake.batch_requests[0]["model_version"] == "MinerU-HTML"


def test_split_preview_requires_markdown_result(tmp_path: Path) -> None:
    client, service = _app(tmp_path, FakeMineru())
    with client:
        job = client.post(
            "/api/documents/parse", files={"file": ("a.pdf", b"%PDF", "application/pdf")}
        ).json()
        done = _wait(client, job["id"])
        # 模拟解析结果被移走 / 尚未生成
        (service.job_dir(job["id"]) / str(done["markdown_file"])).unlink()
        response = client.get(f"/api/documents/{job['id']}/chunks")
        assert response.status_code == 409
        assert client.get("/api/documents/nope/chunks").status_code == 404


def test_rag_ingest_reports_missing_embedding_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = _app(tmp_path, FakeMineru())

    def fail(self: DocumentService, doc_id: str, chunks: list[Any]) -> RagIngestReport:
        raise ConfigurationError("Missing embedding API key. set DASHSCOPE_API_KEY")

    monkeypatch.setattr(DocumentService, "_write_rag_sync", fail)

    with client:
        job = client.post(
            "/api/documents/parse", files={"file": ("a.md", b"# A\n\nB", "text/markdown")}
        ).json()
        response = client.post(f"/api/documents/{job['id']}/rag", json={})
    assert response.status_code == 503
    assert "DASHSCOPE_API_KEY" in response.json()["detail"]


def test_rag_ingest_writes_milvus_and_updates_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = _app(tmp_path, FakeMineru())

    def fake_write(self: DocumentService, doc_id: str, chunks: list[Any]) -> RagIngestReport:
        assert doc_id
        assert chunks
        return RagIngestReport(
            chunks=len(chunks),
            characters=sum(len(chunk.page_content) for chunk in chunks),
            database="db_tax_rag",
            collection="rag_chunk",
            elapsed_seconds=0.2,
        )

    monkeypatch.setattr(DocumentService, "_write_rag_sync", fake_write)

    with client:
        job = client.post(
            "/api/documents/parse",
            files={
                "file": ("a.md", "# A\n\n第一段。\n\n# B\n\n第二段。".encode(), "text/markdown")
            },
        ).json()
        response = client.post(
            f"/api/documents/{job['id']}/rag",
            json={"chunk_size": 100, "chunk_overlap": 10},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["database"] == "db_tax_rag"
        assert body["collection"] == "rag_chunk"
        assert body["store"] == "Milvus"
        assert body["chunks"] == 2
        assert body["mode"] == "resplit"

        detail = client.get(f"/api/documents/{job['id']}").json()
        assert detail["rag_ingested"] is True
        assert detail["rag_ingest"]["collection"] == "db_tax_rag.rag_chunk"
        assert detail["rag_ingest"]["chunks"] == 2
        # Chroma 的入库摘要不会被 RAG 入库覆盖
        assert detail["ingest"] is None

        # 允许重复入库：第二次仍然成功，摘要按最新一次刷新
        again = client.post(
            f"/api/documents/{job['id']}/rag",
            json={"chunk_size": 100, "chunk_overlap": 10},
        )
        assert again.status_code == 200, again.text
        assert client.get(f"/api/documents/{job['id']}").json()["rag_ingest"]["chunks"] == 2


def test_rag_ingest_writes_reviewed_chunks_without_resplitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """无状态模式：直接回传 ``chunks`` 时按原样入库，不再读 markdown_file 重新切分。"""
    client, _ = _app(tmp_path, FakeMineru())
    captured: dict[str, Any] = {}

    def fake_write(self: DocumentService, doc_id: str, chunks: list[Any]) -> RagIngestReport:
        captured["doc_id"] = doc_id
        captured["chunks"] = [(chunk.page_content, dict(chunk.metadata)) for chunk in chunks]
        return RagIngestReport(
            chunks=len(chunks),
            characters=sum(len(chunk.page_content) for chunk in chunks),
            database="db_tax_rag",
            collection="rag_chunk",
            elapsed_seconds=0.05,
        )

    monkeypatch.setattr(DocumentService, "_write_rag_sync", fake_write)

    with client:
        job = client.post(
            "/api/documents/parse",
            files={"file": ("a.md", "# A\n\n原文只有一段。".encode(), "text/markdown")},
        ).json()
        # 模拟用户在弹窗里查阅后调整过的切片：内容与 Markdown 原文不同
        payload = {
            "total": 2,
            "chunks": [
                {
                    "content": "用户改过的第一段",
                    "metadata": {"h1": "A", "page": 3, "evil": "不应入库"},
                },
                {"content": "补充的第二段", "metadata": {"h2": "B"}},
            ],
        }
        response = client.post(f"/api/documents/{job['id']}/rag", json=payload)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["mode"] == "reviewed"
        assert body["chunks"] == 2

        detail = client.get(f"/api/documents/{job['id']}").json()
        assert detail["rag_ingested"] is True
        assert detail["rag_ingest"]["chunks"] == 2

        # 回传切片数与切分总数不一致：拒绝入库，避免静默丢掉未回传的切片
        mismatch = client.post(
            f"/api/documents/{job['id']}/rag",
            json={"chunks": [{"content": "只有一段"}], "total": 3},
        )
        assert mismatch.status_code == 422
        assert "切片数量不一致" in mismatch.json()["detail"]

    assert captured["doc_id"] == job["id"]
    contents = [content for content, _ in captured["chunks"]]
    assert contents == ["用户改过的第一段", "补充的第二段"]
    first, second = captured["chunks"][0][1], captured["chunks"][1][1]
    assert first["h1"] == "A"
    assert first["page"] == 3
    assert first["chunk_index"] == 0
    assert second["chunk_index"] == 1
    # 白名单之外的 metadata 不会写进向量库
    assert "evil" not in first


def test_rag_ingest_reads_full_split_from_cache_by_split_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """预览只回 1 条，入库按 split_id 从缓存读到完整切片，且不再重新切分。"""
    client, service = _app(tmp_path, FakeMineru())
    captured: dict[str, Any] = {}

    def fake_write(self: DocumentService, doc_id: str, chunks: list[Any]) -> RagIngestReport:
        captured["chunks"] = [chunk.page_content for chunk in chunks]
        return RagIngestReport(
            chunks=len(chunks),
            characters=sum(len(chunk.page_content) for chunk in chunks),
            database="db_tax_rag",
            collection="rag_chunk",
            elapsed_seconds=0.05,
        )

    monkeypatch.setattr(DocumentService, "_write_rag_sync", fake_write)

    with client:
        job = client.post(
            "/api/documents/parse",
            files={
                "file": ("a.md", "# A\n\n第一段。\n\n# B\n\n第二段。".encode(), "text/markdown")
            },
        ).json()
        full = client.get(f"/api/documents/{job['id']}/chunks").json()
        assert full["total"] == 2

        # 模拟前端只展示 1 条（truncated），此时请求体里只有 split_id + total
        limited = client.get(f"/api/documents/{job['id']}/chunks", params={"limit": 1}).json()
        assert limited["truncated"] is True
        assert len(limited["chunks"]) == 1

        response = client.post(
            f"/api/documents/{job['id']}/rag",
            json={"split_id": limited["split_id"], "total": limited["total"]},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["mode"] == "cached"
        assert body["chunks"] == 2

        detail = client.get(f"/api/documents/{job['id']}").json()
        assert detail["rag_ingested"] is True
        assert detail["rag_ingest"]["chunks"] == 2

    # 缓存里是完整切片：与不截断时的预览结果逐条一致
    assert captured["chunks"] == [chunk["content"] for chunk in full["chunks"]]
    assert len(service.chunk_cache) >= 1


def test_rag_ingest_rejects_stale_or_foreign_split_id(tmp_path: Path) -> None:
    """split_id 只对「本任务的缓存」有效；缓存失效 / 串任务 / 双传参数都要拒绝。"""
    client, service = _app(tmp_path, FakeMineru())

    with client:
        first = client.post(
            "/api/documents/parse",
            files={"file": ("a.md", "# A\n\n甲".encode(), "text/markdown")},
        ).json()
        preview = client.get(f"/api/documents/{first['id']}/chunks").json()

        second = client.post(
            "/api/documents/parse",
            files={"file": ("b.md", "# B\n\n乙".encode(), "text/markdown")},
        ).json()
        foreign = client.post(
            f"/api/documents/{second['id']}/rag",
            json={"split_id": preview["split_id"]},
        )
        assert foreign.status_code == 409
        assert "不属于当前文档" in foreign.json()["detail"]

        both = client.post(
            f"/api/documents/{first['id']}/rag",
            json={"split_id": preview["split_id"], "chunks": [{"content": "x"}]},
        )
        assert both.status_code == 422
        assert "二选一" in both.json()["detail"]

        # 模拟服务重启 / TTL 到期：缓存清空后必须重新切分
        service.chunk_cache.clear()
        stale = client.post(
            f"/api/documents/{first['id']}/rag",
            json={"split_id": preview["split_id"]},
        )
        assert stale.status_code == 409
        assert "缓存已失效" in stale.json()["detail"]


def test_delete_job_drops_cached_splits(tmp_path: Path) -> None:
    """删除任务时顺手清掉它的切分缓存，避免内存里留下孤儿切片。"""
    client, service = _app(tmp_path, FakeMineru())
    with client:
        job = client.post(
            "/api/documents/parse",
            files={"file": ("a.md", b"# A\n\nB", "text/markdown")},
        ).json()
        client.get(f"/api/documents/{job['id']}/chunks")
        assert len(service.chunk_cache) == 1

        assert client.delete(f"/api/documents/{job['id']}").status_code == 204
        assert len(service.chunk_cache) == 0


def test_rag_ingest_rejects_empty_reviewed_chunks(tmp_path: Path) -> None:
    """空切片 / 空列表都要在入库前拦截，不要静默删掉旧块。"""
    client, _ = _app(tmp_path, FakeMineru())
    with client:
        job = client.post(
            "/api/documents/parse",
            files={"file": ("a.md", b"# A\n\nB", "text/markdown")},
        ).json()

        blank = client.post(
            f"/api/documents/{job['id']}/rag",
            json={"chunks": [{"content": "   ", "metadata": {}}]},
        )
        assert blank.status_code == 422
        assert "切片内容为空" in blank.json()["detail"]

        empty = client.post(f"/api/documents/{job['id']}/rag", json={"chunks": []})
        assert empty.status_code == 400
        assert "没有可入库的切片" in empty.json()["detail"]
