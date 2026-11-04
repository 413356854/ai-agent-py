"""MinerU 客户端 / 解包 / 调度器测试（httpx.MockTransport，全程离线）。"""

from __future__ import annotations

import asyncio
import io
import json
import ssl
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_agent_py.config import Settings
from ai_agent_py.errors import DocumentLoadError, MineruAuthError, MineruError, MineruTaskFailed
from ai_agent_py.mineru.archive import extract_result_zip
from ai_agent_py.mineru.client import MineruClient
from ai_agent_py.mineru.models import JobState, ParseJob, ParseOptions
from ai_agent_py.mineru.service import DocumentService, safe_file_name
from fakes import InMemoryJobStore

MARKDOWN = "# 测试文档\n\n这是一段由 MinerU 解析出来的正文。\n\n![图](images/img0.jpg)\n"


def make_zip(
    markdown: str = MARKDOWN, *, images: int = 1, extra: dict[str, bytes] | None = None
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("full.md", markdown)
        archive.writestr(
            "content_list.json",
            json.dumps(
                [
                    {"type": "text", "text": "第一页", "page_idx": 0},
                    {"type": "text", "text": "第二页", "page_idx": 1},
                ],
                ensure_ascii=False,
            ),
        )
        for index in range(images):
            archive.writestr(f"images/img{index}.jpg", b"\xff\xd8jpeg")
        for name, payload in (extra or {}).items():
            archive.writestr(name, payload)
    return buffer.getvalue()


class FakeMineru:
    """按需返回 running / done / failed 的假 MinerU 服务。"""

    def __init__(
        self,
        *,
        states: tuple[str, ...] = ("running", "done"),
        err_msg: str = "",
        batch_code: int = 0,
        auth_failure: bool = False,
        payload: bytes | None = None,
    ) -> None:
        self.states = states
        self.err_msg = err_msg
        self.batch_code = batch_code
        self.auth_failure = auth_failure
        self.payload = payload if payload is not None else make_zip()
        self.batch_requests: list[dict[str, Any]] = []
        self.uploaded: bytes | None = None
        self.polls = 0
        self.auth_headers: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.auth_headers.append(request.headers.get("authorization", ""))
        url = str(request.url)
        if url.endswith("/file-urls/batch") and request.method == "POST":
            self.batch_requests.append(json.loads(request.content))
            if self.auth_failure:
                return httpx.Response(
                    401,
                    json={"success": False, "msgCode": "A0202", "msg": "user authenticate failed"},
                )
            return httpx.Response(
                200,
                json={
                    "code": self.batch_code,
                    "msg": "ok" if self.batch_code == 0 else "参数错误",
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": ["https://upload.test/file-1"],
                    },
                },
            )
        if url == "https://upload.test/file-1" and request.method == "PUT":
            self.uploaded = request.content
            return httpx.Response(200)
        if "/extract-results/batch/batch-1" in url:
            index = min(self.polls, len(self.states) - 1)
            state = self.states[index] if self.states else "done"
            self.polls += 1
            item: dict[str, Any] = {
                "file_name": "demo.pdf",
                "data_id": "job",
                "state": state,
                "err_msg": self.err_msg if state == "failed" else "",
                "extract_progress": {"extracted_pages": 1, "total_pages": 2},
            }
            if state == "done":
                item["full_zip_url"] = "https://cdn.test/result.zip"
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {"batch_id": "batch-1", "extract_result": [item]},
                },
            )
        if url == "https://cdn.test/result.zip":
            return httpx.Response(200, content=self.payload)
        return httpx.Response(404, json={"code": -1, "msg": f"unexpected {url}"})

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def make_client(fake: FakeMineru, *, api_key: str | None = "test-token") -> MineruClient:
    return MineruClient(api_key, "https://mineru.test/api/v4", client=fake.client())


# ---- 客户端 ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_create_batch_upload_sends_documented_payload() -> None:
    fake = FakeMineru()
    async with make_client(fake) as client:
        upload = await client.create_batch_upload(
            "demo.pdf",
            "job-1",
            ParseOptions(model_version="vlm", language="en", enable_table=False),
        )
    assert upload.batch_id == "batch-1"
    assert upload.file_urls == ["https://upload.test/file-1"]
    body = fake.batch_requests[0]
    assert body["model_version"] == "vlm"
    assert body["language"] == "en"
    assert body["enable_table"] is False
    assert body["files"] == [{"name": "demo.pdf", "data_id": "job-1", "is_ocr": False}]
    assert fake.auth_headers[0] == "Bearer test-token"


@pytest.mark.asyncio
async def test_missing_token_raises_auth_error() -> None:
    fake = FakeMineru()
    async with make_client(fake, api_key=None) as client:
        with pytest.raises(MineruAuthError, match="MINERU_API_KEY"):
            await client.create_batch_upload("demo.pdf", "job-1")
    assert fake.batch_requests == []


@pytest.mark.asyncio
async def test_gateway_auth_failure_is_mapped() -> None:
    fake = FakeMineru(auth_failure=True)
    async with make_client(fake) as client:
        with pytest.raises(MineruAuthError, match="Token"):
            await client.create_batch_upload("demo.pdf", "job-1")


@pytest.mark.asyncio
async def test_business_error_code_is_mapped() -> None:
    fake = FakeMineru(batch_code=-60005)
    async with make_client(fake) as client:
        with pytest.raises(MineruError, match="200MB"):
            await client.create_batch_upload("demo.pdf", "job-1")


@pytest.mark.asyncio
async def test_wait_batch_polls_until_done() -> None:
    fake = FakeMineru(states=("pending", "running", "done"))
    progress: list[tuple[int, int]] = []
    async with make_client(fake) as client:
        item = await client.wait_batch(
            "batch-1",
            data_id="job",
            timeout=5,
            poll_interval=0,
            on_progress=lambda entry: progress.append((entry.extracted_pages, entry.total_pages)),
        )
    assert item.state == "done"
    assert item.full_zip_url == "https://cdn.test/result.zip"
    assert fake.polls == 3
    assert progress[0] == (1, 2)


@pytest.mark.asyncio
async def test_wait_batch_surfaces_failure() -> None:
    fake = FakeMineru(states=("failed",), err_msg="文件格式不支持")
    async with make_client(fake) as client:
        with pytest.raises(MineruTaskFailed, match="文件格式不支持"):
            await client.wait_batch("batch-1", data_id="job", timeout=5, poll_interval=0)


@pytest.mark.asyncio
async def test_wait_batch_timeout() -> None:
    fake = FakeMineru(states=("running",))
    async with make_client(fake) as client:
        with pytest.raises(MineruError, match="超时"):
            await client.wait_batch("batch-1", data_id="job", timeout=0.05, poll_interval=0)


# ---- 解包 -----------------------------------------------------------------
def test_extract_zip_collects_markdown_and_images(tmp_path: Path) -> None:
    summary = extract_result_zip(make_zip(), tmp_path / "job")
    assert summary.markdown_name == "full.md"
    assert summary.assets == 1
    assert summary.pages == 2
    assert summary.characters == len(MARKDOWN)
    assert (tmp_path / "job" / "full.md").is_file()
    assert (tmp_path / "job" / "images" / "img0.jpg").is_file()


def test_extract_zip_keeps_previous_result_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "job"
    extract_result_zip(make_zip("# 旧结果"), target)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("content_list.json", "[]")
    with pytest.raises(MineruError, match="Markdown"):
        extract_result_zip(buffer.getvalue(), target)

    assert (target / "full.md").read_text(encoding="utf-8") == "# 旧结果"


def test_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    payload = make_zip(extra={"../evil.md": b"x"})
    with pytest.raises(MineruError, match="非法路径"):
        extract_result_zip(payload, tmp_path / "job")
    assert not (tmp_path / "evil.md").exists()


def test_extract_zip_rejects_plain_bytes(tmp_path: Path) -> None:
    with pytest.raises(MineruError, match="zip"):
        extract_result_zip(b"not a zip", tmp_path / "job")


# ---- 调度器 ---------------------------------------------------------------
def service(
    tmp_path: Path,
    fake: FakeMineru,
    *,
    store: InMemoryJobStore | None = None,
    **overrides: Any,
) -> DocumentService:
    settings = Settings(
        _env_file=None,
        app_env="test",
        mineru_api_key="test-token",
        mineru_output_dir=tmp_path / "parsed",
        mineru_poll_interval=0,
        mineru_timeout=5,
        **overrides,
    )
    return DocumentService(
        settings,
        client_factory=lambda: MineruClient(
            "test-token", settings.mineru_base_url, client=fake.client()
        ),
        store=store if store is not None else InMemoryJobStore(),
    )


async def upload(service: DocumentService, payload: bytes, name: str = "demo.pdf") -> str:
    async def chunks():
        yield payload

    job = await service.create(name, chunks())
    await service.wait(job.id, timeout=10)
    return job.id


@pytest.mark.asyncio
async def test_service_parses_and_persists(tmp_path: Path) -> None:
    fake = FakeMineru()
    store = InMemoryJobStore()
    svc = service(tmp_path, fake, store=store)
    job_id = await upload(svc, b"%PDF-1.4 fake")

    job = await svc.get(job_id)
    assert job is not None
    assert job.state is JobState.done, job.err_msg
    assert job.markdown_file == "full.md"
    assert job.markdown_chars == len(MARKDOWN)
    assert job.asset_count == 1
    assert job.total_pages == 2
    assert job.assets_base == f"/parsed/{job_id}/"
    assert fake.uploaded == b"%PDF-1.4 fake"

    markdown = await svc.read_markdown(job_id)
    assert markdown.markdown.startswith("# 测试文档")
    assert markdown.assets_base == f"/parsed/{job_id}/"
    assert (tmp_path / "parsed" / job_id / "mineru.zip").is_file()

    stored = store.load_all()
    assert [item.id for item in stored] == [job_id]
    assert stored[0].state is JobState.done
    assert stored[0].markdown_file == "full.md"
    assert stored[0].markdown_chars == len(MARKDOWN)
    await svc.shutdown()


@pytest.mark.asyncio
async def test_service_records_failure(tmp_path: Path) -> None:
    fake = FakeMineru(states=("failed",), err_msg="文件页数超过 600 页上限")
    svc = service(tmp_path, fake)
    job_id = await upload(svc, b"%PDF-1.4 fake")
    job = await svc.get(job_id)
    assert job is not None
    assert job.state is JobState.failed
    assert "600 页" in job.err_msg
    await svc.shutdown()


@pytest.mark.asyncio
async def test_service_reloads_and_resumes_pending_job(tmp_path: Path) -> None:
    # 模拟「上传完成、云端解析中，进程重启」的场景：任务记录已在库里 + 源文件还在
    job = ParseJob(
        id="job-resume",
        file_name="demo.pdf",
        suffix="pdf",
        state=JobState.parsing,
        batch_id="batch-1",
        source_file="source.pdf",
    )
    job_dir = tmp_path / "parsed" / job.id
    job_dir.mkdir(parents=True)
    (job_dir / "source.pdf").write_bytes(b"%PDF-1.4 fake")
    store = InMemoryJobStore()
    store.save(job)

    fake = FakeMineru(states=("done",))
    reopened = service(tmp_path, fake, store=store)
    await reopened.ensure_loaded()
    restored = await reopened.wait("job-resume", timeout=10)

    assert restored.state is JobState.done, restored.err_msg
    assert fake.polls >= 1
    assert (job_dir / "full.md").is_file()
    await reopened.shutdown()


@pytest.mark.asyncio
async def test_service_rejects_bad_upload(tmp_path: Path) -> None:
    svc = service(tmp_path, FakeMineru(), mineru_max_file_mb=1)
    with pytest.raises(DocumentLoadError, match="不支持"):
        await svc.create("virus.exe", _one_chunk(b"x"))
    with pytest.raises(DocumentLoadError, match="为空"):
        await svc.create("empty.pdf", _one_chunk(b""))
    with pytest.raises(DocumentLoadError, match="MB"):
        await svc.create("big.pdf", _one_chunk(b"x" * (1024 * 1024 + 1)))
    await svc.shutdown()


@pytest.mark.asyncio
async def test_service_delete_removes_files(tmp_path: Path) -> None:
    svc = service(tmp_path, FakeMineru())
    job_id = await upload(svc, b"%PDF-1.4 fake")
    assert await svc.delete(job_id) is True
    assert await svc.get(job_id) is None
    assert not (tmp_path / "parsed" / job_id).exists()
    assert await svc.delete(job_id) is False
    await svc.shutdown()


@pytest.mark.asyncio
async def test_delete_cancels_in_flight_task_without_resurrecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """删除正在解析的任务：取消流程会先写一条 failed 记录，删除必须排在它之后。"""

    async def never_finishes(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(3600)

    monkeypatch.setattr(MineruClient, "wait_batch", never_finishes)
    store = InMemoryJobStore()
    svc = service(tmp_path, FakeMineru(), store=store)

    async def chunks():
        yield b"%PDF-1.4 fake"

    job = await svc.create("running.pdf", chunks())
    await asyncio.sleep(0.05)  # 让它跑到轮询阶段
    assert await svc.delete(job.id) is True
    await asyncio.sleep(0.05)  # 给「可能残留的写回」留出时间

    assert store.get(job.id) is None  # 没有被 _fail 的 upsert 写回来
    assert await svc.get(job.id) is None
    await svc.shutdown()


@pytest.mark.asyncio
async def test_get_reads_store_when_not_cached(tmp_path: Path) -> None:
    """别的进程创建的任务不在本进程缓存里，`get()` 直接读库（多实例可见）。"""
    store = InMemoryJobStore()
    store.save(
        ParseJob(id="job-other", file_name="别的进程上传.pdf", suffix="pdf", state=JobState.done)
    )
    svc = service(tmp_path, FakeMineru(), store=store)  # 不调用 ensure_loaded，缓存是空的

    job = await svc.get("job-other")
    assert job is not None
    assert job.file_name == "别的进程上传.pdf"
    assert job.state is JobState.done
    assert await svc.get("不存在") is None
    await svc.shutdown()


@pytest.mark.asyncio
async def test_get_keeps_in_flight_progress(tmp_path: Path) -> None:
    """本进程正在解析的任务：进度来自内存，不能被库里的旧值覆盖。"""
    job = ParseJob(
        id="job-progress",
        file_name="demo.pdf",
        suffix="pdf",
        state=JobState.parsing,
        total_pages=10,
        extracted_pages=1,
    )
    store = InMemoryJobStore()
    store.save(job)
    svc = service(tmp_path, FakeMineru(), store=store)
    svc._jobs[job.id] = job  # 模拟后台任务正持有这个对象
    job.extracted_pages = 7  # 只在内存里推进（还没 _touch 写库）

    assert await svc.get(job.id) is job

    # 任务结束后就以库为准：内存对象可以被库里的记录替换
    job.state = JobState.done
    store.save(job)
    loaded = await svc.get(job.id)
    assert loaded is not None
    assert loaded.extracted_pages == 7
    await svc.shutdown()


@pytest.mark.asyncio
async def test_list_jobs_reads_store_and_orders_by_created_at(tmp_path: Path) -> None:
    store = InMemoryJobStore()
    store.save(
        ParseJob(id="job-older", file_name="older.pdf", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    )
    store.save(
        ParseJob(id="job-newer", file_name="newer.pdf", created_at=datetime(2026, 2, 1, tzinfo=UTC))
    )
    svc = service(tmp_path, FakeMineru(), store=store)  # 缓存为空，列表必须读库

    assert [job.id for job in await svc.list_jobs()] == ["job-newer", "job-older"]
    await svc.shutdown()


async def _one_chunk(payload: bytes):
    yield payload


# ---- 杂项 -----------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\docs\\报告 2024.docx", "报告 2024.docx"),
        ("a<b>c.pdf", "a_b_c.pdf"),
        ("...pdf", "pdf"),
        ("", "document"),
    ],
)
def test_safe_file_name(raw: str, expected: str) -> None:
    assert safe_file_name(raw) == expected


def test_safe_file_name_truncates_long_names() -> None:
    name = safe_file_name("文" * 300 + ".pdf")
    assert len(name) <= 120
    assert name.endswith(".pdf")


# ---- 网络重试（代理/网络抖动导致的 TLS 错误）-------------------------------
@pytest.mark.asyncio
async def test_download_retries_transient_tls_error() -> None:
    payload = make_zip()
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ssl.SSLError("[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac")
        return httpx.Response(200, content=payload)

    client = MineruClient(
        "token",
        "https://mineru.test/api/v4",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_delay=0,
    )
    assert await client.download("https://cdn.test/result.zip") == payload
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_download_gives_up_after_retries() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise ssl.SSLError("bad record mac")

    client = MineruClient(
        "token",
        "https://mineru.test/api/v4",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_delay=0,
    )
    with pytest.raises(MineruError, match="下载解析结果失败"):
        await client.download("https://cdn.test/result.zip")
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_download_retries_server_error() -> None:
    payload = make_zip()
    responses = [httpx.Response(503), httpx.Response(200, content=payload)]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    client = MineruClient(
        "token",
        "https://mineru.test/api/v4",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_delay=0,
    )
    assert await client.download("https://cdn.test/result.zip") == payload


@pytest.mark.asyncio
async def test_polling_retries_transient_error() -> None:
    failures = {"left": 1}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/extract-results/batch/" in str(request.url):
            if failures["left"]:
                failures["left"] -= 1
                raise ssl.SSLError("bad record mac")
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "file_name": "a.pdf",
                                "data_id": "job",
                                "state": "done",
                                "full_zip_url": "https://cdn.test/result.zip",
                            }
                        ]
                    },
                },
            )
        return httpx.Response(404)

    client = MineruClient(
        "token",
        "https://mineru.test/api/v4",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_delay=0,
    )
    item = await client.wait_batch("batch-1", data_id="job", timeout=5, poll_interval=0)
    assert item.state == "done"
    assert failures["left"] == 0


@pytest.mark.asyncio
async def test_create_task_is_not_retried() -> None:
    """创建任务不是幂等的，网络错误时立即失败，避免重复建任务。"""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise ssl.SSLError("bad record mac")

    client = MineruClient(
        "token",
        "https://mineru.test/api/v4",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_delay=0,
    )
    with pytest.raises(MineruError, match="请求 MinerU"):
        await client.create_batch_upload("a.pdf", "job")
    assert calls["count"] == 1


@pytest.fixture(autouse=True)
def _no_real_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in ("MINERU_API_KEY", "MINERU_BASE_URL", "MINERU_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    yield
