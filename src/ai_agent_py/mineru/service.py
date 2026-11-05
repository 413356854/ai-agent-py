"""文档解析调度：上传 → 提交 MinerU → 轮询 → 下载解包 → 写入知识库。

`DocumentService` 是进程内单例（挂在 FastAPI `app.state` 上），负责：

* 把前端上传的文件落到 ``MINERU_OUTPUT_DIR/<job_id>/source.<ext>``；
* 用 `asyncio` 任务异步调用 MinerU，前端通过轮询任务状态获取进度；
* 结果解包后保存 Markdown 与图片，并把任务记录按行 upsert 到 MySQL 表
  ``tb_document_parse_job``（见 `ai_agent_py.db.document_jobs`），
  服务重启后可以继续轮询未完成的任务；
* 可选地把解析出的 Markdown 送进现有 RAG 索引（``/api/documents/{id}/ingest``）；
* 删除任务时同步删除 ``rag_chunk`` 里该文档的全部切片（删除对称）；Milvus 不可用时
  把待删的 ``doc_id`` 落盘到补删队列（``rag.deletions``），后台重试，不阻塞删除。

内存里的 `_jobs` 只用于两件事：让后台任务持有正在处理的任务对象；
以及服务启动时把未完成的任务恢复出来。**所有读路径（列表 / 详情 / 预览 / 下载 / 入库）
都以 MySQL 为准**，所以多进程或多实例部署时，任意实例都能看到别的实例创建的任务；
本进程正在解析的任务例外——它的页数进度在内存里比库里更新，直接返回内存对象。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shutil
import time
from collections.abc import AsyncIterable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.db.document_jobs import TABLE_NAME, JobStore, MySQLJobStore
from ai_agent_py.errors import AiAgentError, DocumentLoadError, MineruError, SplitCacheMiss
from ai_agent_py.mineru.archive import extract_result_zip
from ai_agent_py.mineru.client import MineruClient
from ai_agent_py.mineru.models import (
    BatchItem,
    JobState,
    MarkdownOut,
    ParseJob,
    ParseOptions,
    RagIngestInfo,
)
from ai_agent_py.rag.chunk_cache import ChunkCache
from ai_agent_py.rag.deletions import RagDeleteQueue
from ai_agent_py.rag.loaders import read_text
from ai_agent_py.rag.models import (
    ChunkIn,
    ChunkOut,
    RagIngestResultOut,
    SplitPreviewOut,
)

logger = logging.getLogger(__name__)

ClientFactory = Callable[[], MineruClient]

#: 精准解析 API 支持的文件类型（见官方文档「文件限制」）
ALLOWED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".png",
        ".jpg",
        ".jpeg",
        ".jp2",
        ".webp",
        ".gif",
        ".bmp",
        ".html",
        ".htm",
    }
)

#: 官方 `language` 取值里最常用的一批（UI 下拉框用）
SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "ch",
    "chinese_cht",
    "en",
    "japan",
    "korean",
    "latin",
    "arabic",
    "cyrillic",
    "east_slavic",
    "devanagari",
    "th",
    "ta",
    "te",
    "ka",
)

MODEL_VERSIONS: tuple[str, ...] = ("vlm", "pipeline", "MinerU-HTML")

#: 不需要 MinerU 解析、直接读取正文的文档（切分器按后缀区分）。
TEXT_DOCUMENT_SUFFIXES: frozenset[str] = frozenset({".md", ".markdown", ".json"})

#: 上传接口实际接受的全部后缀 = MinerU 支持的 + 直接切分的。
UPLOAD_SUFFIXES: frozenset[str] = ALLOWED_SUFFIXES | TEXT_DOCUMENT_SUFFIXES

_MAX_NAME_LENGTH = 120
_INVALID_NAME_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]+')

#: `.json` 直读任务的 ``full.md``：正文是一个 ```json 代码块。
_JSON_FENCE = re.compile(r"```json[ \t]*\r?\n(?P<payload>.*?)\r?\n```", re.DOTALL)

#: 前端回传切片时保留的 metadata（其余字段丢弃，避免任意 JSON 进向量库）。
_REVIEW_METADATA_KEYS = ("h1", "h2", "h3", "h4", "page", "start_index")

#: RAG 切片补删的后台重试间隔（秒）；队列清空后后台任务会自己退出。
DELETE_RETRY_SECONDS = 60.0


@dataclass(slots=True)
class RagDeleteFlush:
    """一次补删重试的结果：本轮删掉几条、还剩几条（失败的会在后台继续重试）。"""

    deleted: int
    remaining: int


def safe_file_name(raw: str | None, *, fallback: str = "document") -> str:
    """把用户上传的文件名收敛成安全的单段文件名。"""
    name = (raw or "").replace("\\", "/").split("/")[-1]
    name = _INVALID_NAME_CHARS.sub("_", name).strip().strip(".")
    if not name:
        return fallback
    if len(name) <= _MAX_NAME_LENGTH:
        return name
    suffix = Path(name).suffix
    stem = name[: max(1, _MAX_NAME_LENGTH - len(suffix))]
    return f"{stem}{suffix}"


def _now() -> datetime:
    return datetime.now(UTC)


class DocumentService:
    """解析任务的创建、调度、查询、删除与结果管理。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client_factory: ClientFactory | None = None,
        store: JobStore | None = None,
        chunk_cache: ChunkCache | None = None,
        delete_queue: RagDeleteQueue | None = None,
        delete_retry_seconds: float = DELETE_RETRY_SECONDS,
    ) -> None:
        self.settings = settings or get_settings()
        self._client_factory: ClientFactory = client_factory or (
            lambda: MineruClient.from_settings(self.settings)
        )
        self._store = store
        self._jobs: dict[str, ParseJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._load_lock = asyncio.Lock()
        self._loaded = False
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        #: 切分结果缓存：预览返回 split_id，入库时按 id 取回完整切片。
        self.chunk_cache = chunk_cache or ChunkCache()
        #: RAG 切片补删队列：删除文档时 Milvus 不可用，标记落盘后后台重试。
        self._delete_queue = delete_queue
        self._delete_retry_seconds = max(0.05, delete_retry_seconds)
        self._delete_retry_task: asyncio.Task[None] | None = None

    @property
    def store(self) -> JobStore:
        """任务记录存储；默认到第一次用时才创建 MySQL 连接池。"""
        if self._store is None:
            self._store = MySQLJobStore.from_settings(self.settings)
        return self._store

    @property
    def delete_queue(self) -> RagDeleteQueue:
        """RAG 补删队列；默认落在 ``RAG_DELETE_PENDING_DIR``。"""
        if self._delete_queue is None:
            self._delete_queue = RagDeleteQueue(self.settings.rag_delete_pending_dir)
        return self._delete_queue

    # ---- 路径 ----------------------------------------------------------
    @property
    def output_dir(self) -> Path:
        return self.settings.mineru_output_dir

    def job_dir(self, job_id: str) -> Path:
        return self.output_dir / job_id

    # ---- 配置 ----------------------------------------------------------
    def default_options(self) -> ParseOptions:
        cfg = self.settings
        return ParseOptions(
            model_version=cfg.mineru_model_version,
            language=cfg.mineru_language,
            enable_formula=cfg.mineru_enable_formula,
            enable_table=cfg.mineru_enable_table,
            is_ocr=cfg.mineru_is_ocr,
        )

    # ---- 生命周期 ------------------------------------------------------
    async def ensure_loaded(self) -> None:
        """首次使用时载入任务记录、恢复未完成任务与 RAG 补删队列。"""
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # 补删队列只在磁盘上、不依赖 MySQL：先恢复它，数据库挂了也能继续删向量。
            await self._resume_pending_deletes()
            jobs = await asyncio.to_thread(self.store.load_all)
            for job in jobs:
                self._jobs[job.id] = job
            if jobs:
                logger.info("已从 %s 载入 %d 个解析任务", TABLE_NAME, len(jobs))
            self._loaded = True
        for job in list(self._jobs.values()):
            if not job.is_terminal:
                logger.info("恢复未完成的解析任务 %s (%s)", job.id, job.file_name)
                self._spawn(job)

    async def shutdown(self) -> None:
        """取消所有正在轮询的任务与补删重试（测试与优雅退出用）。"""
        for task in list(self._tasks.values()):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        await self.aclose()

    async def aclose(self) -> None:
        """只停掉补删重试，不动正在解析的任务。

        应用关停时用这个：未解析完的任务保留在 ``parsing`` 状态，下次启动会从 MySQL
        恢复并继续轮询；补删标记本来就在磁盘上，重试任务停掉也不会丢。
        """
        retry = self._delete_retry_task
        self._delete_retry_task = None
        if retry is not None and not retry.done():
            retry.cancel()
            await asyncio.gather(retry, return_exceptions=True)

    # ---- 查询 ----------------------------------------------------------
    async def get(self, job_id: str) -> ParseJob | None:
        """按 id 读取任务。

        本进程正在解析的任务直接返回内存对象（页数进度比库里更新）；其余一律读库，
        这样多进程 / 多实例部署时也能看到别的实例创建或更新的任务。
        """
        cached = self._jobs.get(job_id)
        if cached is not None and not cached.is_terminal:
            return cached
        job = await asyncio.to_thread(self.store.get, job_id)
        if job is None:
            return None
        async with self._lock:
            current = self._jobs.get(job_id)
            if current is not None and not current.is_terminal:
                # 刚被本进程接手（ensure_loaded 恢复），内存对象才是最新的
                return current
            self._jobs[job_id] = job
        return job

    async def list_jobs(self) -> list[ParseJob]:
        """按创建时间倒序返回全部任务（以数据库为准）。"""
        jobs = await asyncio.to_thread(self.store.load_all)
        result: list[ParseJob] = []
        async with self._lock:
            for job in jobs:
                current = self._jobs.get(job.id)
                if current is not None and not current.is_terminal:
                    result.append(current)  # 本进程正在解析，内存里的进度更新
                else:
                    self._jobs[job.id] = job
                    result.append(job)
        return sorted(result, key=lambda job: job.created_at, reverse=True)

    async def wait(self, job_id: str, *, timeout: float = 60.0) -> ParseJob:
        """等待某个任务结束（CLI / 测试用）。"""
        task = self._tasks.get(job_id)
        if task is not None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    # ---- 上传 ----------------------------------------------------------
    async def create(
        self,
        file_name: str,
        chunks: AsyncIterable[bytes],
        options: ParseOptions | None = None,
    ) -> ParseJob:
        """接收上传的文件字节，创建任务并立刻开始后台解析。"""
        await self.ensure_loaded()

        clean_name = safe_file_name(file_name)
        suffix = Path(clean_name).suffix.lower()
        if suffix not in UPLOAD_SUFFIXES:
            raise DocumentLoadError(
                f"不支持的文件类型: {suffix or '(无扩展名)'}。"
                f"支持: {', '.join(sorted(UPLOAD_SUFFIXES))}"
            )

        max_bytes = self.settings.mineru_max_file_mb * 1024 * 1024
        job_id = uuid4().hex
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        source_name = f"source{suffix}"
        source_path = job_dir / source_name

        total = 0
        try:
            with source_path.open("wb") as sink:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise DocumentLoadError(
                            f"文件超过 {self.settings.mineru_max_file_mb} MB 上限"
                        )
                    sink.write(chunk)
        except BaseException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

        if total == 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise DocumentLoadError("上传的文件为空")

        job = ParseJob(
            id=job_id,
            file_name=clean_name,
            file_size=total,
            suffix=suffix.lstrip("."),
            options=options or self.default_options(),
            source_file=source_name,
        )
        async with self._lock:
            self._jobs[job.id] = job
        try:
            if suffix in TEXT_DOCUMENT_SUFFIXES:
                # Markdown / JSON 不经过 MinerU：读正文、校验、直接标记完成，
                # 这样「切分」按钮对它们立即可用。
                await asyncio.to_thread(self._finalize_text_document, job, source_path)
            await self._save(job)
        except BaseException:
            # 记录没写进库（MySQL 不可用等）：内存和磁盘都不留半个任务。
            async with self._lock:
                self._jobs.pop(job.id, None)
            await asyncio.to_thread(shutil.rmtree, job_dir, True)
            raise
        if suffix in TEXT_DOCUMENT_SUFFIXES:
            logger.info("创建文本任务 %s: %s（直接切分，不调用 MinerU）", job.id, job.file_name)
            return job
        self._spawn(job)
        logger.info("创建解析任务 %s: %s (%d bytes)", job.id, job.file_name, job.file_size)
        return job

    def _finalize_text_document(self, job: ParseJob, source_path: Path) -> None:
        """Markdown / JSON 直读任务：校验内容并伪造「解析完成」状态。"""
        text = read_text(source_path)
        if not text.strip():
            raise DocumentLoadError("文件内容为空")

        if source_path.suffix.lower() == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DocumentLoadError(
                    f"JSON 格式不正确（第 {exc.lineno} 行第 {exc.colno} 列）：{exc.msg}"
                ) from exc
            # 预览 / 下载仍给一份 Markdown，切分时读原始 source.json 保留结构。
            preview = json.dumps(payload, ensure_ascii=False, indent=2)
            job.markdown_file = "full.md"
            (source_path.parent / "full.md").write_text(
                f"# {job.file_name}\n\n```json\n{preview}\n```\n", encoding="utf-8"
            )
            job.markdown_chars = len(preview)
        else:
            job.markdown_file = job.source_file
            job.markdown_chars = len(text)

        job.options.model_version = "direct"
        job.state = JobState.done
        job.started_at = job.finished_at = _now()
        job.elapsed_seconds = 0.0
        job.err_msg = ""

    # ---- 删除 / 结果 ---------------------------------------------------
    async def delete(self, job_id: str) -> bool:
        """删除任务记录、结果目录与 RAG 切片。

        RAG 切片删除是「删除对称」的一环：入过库的文档必须同时清掉 ``rag_chunk``
        里该 ``doc_id`` 的全部切片，否则问答还会引用已删除的文档。Milvus 不可用时
        只把 ``doc_id`` 记进补删队列并转后台重试，**不阻塞任务删除**。
        """
        job = await self.get(job_id)
        if job is None:
            return False
        task = self._tasks.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()
            # 等取消流程走完（它会先写一条 failed 记录），再删库，否则会被重新写回。
            await asyncio.gather(task, return_exceptions=True)
        async with self._lock:
            self._jobs.pop(job_id, None)
        self.chunk_cache.drop_job(job_id)
        await self._purge_rag_chunks(job)
        await asyncio.to_thread(self.store.delete, job_id)
        await asyncio.to_thread(shutil.rmtree, self.job_dir(job_id), True)
        logger.info("删除解析任务 %s (%s)", job_id, job.file_name)
        return True

    async def _purge_rag_chunks(self, job: ParseJob) -> bool:
        """删除该文档在 ``rag_chunk`` 里的全部切片，返回是否一次成功。

        没有 ``rag_ingest`` 摘要的任务没写过向量，直接跳过：解析任务本身不需要
        连 Milvus，只有入过库的文档才需要「删除对称」。失败转入补删队列，异常
        不会抛给调用方。
        """
        if job.rag_ingest is None:
            return True
        try:
            deleted = await asyncio.to_thread(self._delete_rag_sync, job.id)
        except Exception as exc:
            error = str(exc).strip() or exc.__class__.__name__
            logger.warning("删除文档 %s 的 RAG 切片失败，转入补删队列：%s", job.id, error)
            await asyncio.to_thread(
                self._record_pending_delete,
                job.id,
                file_name=job.file_name,
                collection=job.rag_ingest.collection,
                error=error,
            )
            self._schedule_delete_retry()
            return False
        logger.info("已删除文档 %s 的 RAG 切片（%s 块）", job.id, deleted)
        # 上一次补删留下的标记（如果有）也该清掉：这次已经删成功了。
        await asyncio.to_thread(self.delete_queue.discard, job.id)
        return True

    def _delete_rag_sync(self, doc_id: str) -> int:
        from ai_agent_py.rag.milvus import MilvusRagWriter

        return MilvusRagWriter(self.settings).delete_document(doc_id)

    def _record_pending_delete(
        self,
        doc_id: str,
        *,
        file_name: str | None = None,
        collection: str | None = None,
        error: str = "",
    ) -> None:
        """把「还没删掉」的 doc_id 落盘；队列目录都写不了时只记日志，不丢任务删除。"""
        try:
            self.delete_queue.record(
                doc_id,
                file_name=file_name,
                collection=collection,
                error=error,
            )
        except Exception as exc:
            logger.error("记录 RAG 补删失败（doc_id=%s，需手工清理）：%s", doc_id, exc)

    async def flush_pending_deletes(self) -> RagDeleteFlush:
        """重试补删队列：成功一条删一条标记，失败累加 ``attempts`` 后保留。"""
        try:
            pending = await asyncio.to_thread(self.delete_queue.items)
        except Exception as exc:
            logger.warning("读取 RAG 补删队列失败：%s", exc)
            return RagDeleteFlush(deleted=0, remaining=0)

        deleted = 0
        remaining = 0
        for item in pending:
            try:
                await asyncio.to_thread(self._delete_rag_sync, item.doc_id)
            except Exception as exc:
                remaining += 1
                error = str(exc).strip() or exc.__class__.__name__
                logger.warning(
                    "补删 RAG 切片仍失败（doc_id=%s，第 %d 次）：%s",
                    item.doc_id,
                    item.attempts + 1,
                    error,
                )
                await asyncio.to_thread(
                    self._record_pending_delete,
                    item.doc_id,
                    file_name=item.file_name,
                    collection=item.collection,
                    error=error,
                )
                continue
            await asyncio.to_thread(self.delete_queue.discard, item.doc_id)
            deleted += 1
            logger.info("补删成功：doc_id=%s（第 %d 次尝试）", item.doc_id, item.attempts + 1)
        return RagDeleteFlush(deleted=deleted, remaining=remaining)

    async def _resume_pending_deletes(self) -> None:
        """启动时把上次没删成功的 RAG 切片重新排进后台重试。"""
        try:
            pending = await asyncio.to_thread(self.delete_queue.items)
        except Exception as exc:
            logger.warning("读取 RAG 补删队列失败：%s", exc)
            return
        if pending:
            logger.info("发现 %d 条待补删的 RAG 切片，转后台重试", len(pending))
            self._schedule_delete_retry()

    def _schedule_delete_retry(self) -> None:
        """确保补删后台任务在跑；队列清空后它会自己退出。"""
        if self._delete_retry_task is not None and not self._delete_retry_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # 没有事件循环（同步脚本 / 解释器退出中）：等下次启动再补
            return
        self._delete_retry_task = loop.create_task(self._delete_retry(), name="rag-delete-retry")

    async def _delete_retry(self) -> None:
        while True:
            await asyncio.sleep(self._delete_retry_seconds)
            report = await self.flush_pending_deletes()
            if report.remaining == 0:
                logger.info("RAG 补删队列已清空")
                return

    async def read_markdown(self, job_id: str) -> MarkdownOut:
        job = await self._require_job(job_id)
        text = await asyncio.to_thread(self._read_markdown_text, job)
        return MarkdownOut(
            job_id=job.id,
            file_name=job.file_name,
            markdown=text,
            assets_base=f"/parsed/{job.id}/",
            characters=len(text),
            assets=job.asset_count,
        )

    async def artifact(
        self, job_id: str, kind: Literal["markdown", "zip", "source"] = "markdown"
    ) -> tuple[Path, str]:
        """返回 (文件路径, 下载时的文件名)。"""
        job = await self._require_job(job_id)
        if kind == "source":
            if not job.source_file:
                raise FileNotFoundError("该任务没有保存源文件")
            return self.job_dir(job.id) / job.source_file, job.file_name
        if kind == "zip":
            if not job.zip_file:
                raise FileNotFoundError("解析结果还没有生成")
            return self.job_dir(job.id) / job.zip_file, f"{Path(job.file_name).stem}.zip"
        if not job.markdown_file:
            raise FileNotFoundError("解析结果还没有生成")
        return (
            self.job_dir(job.id) / job.markdown_file,
            f"{Path(job.file_name).stem}.md",
        )

    # ---- 切分预览 / 写入 Milvus RAG 表 ---------------------------------
    async def split_preview(
        self,
        job_id: str,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        limit: int = 5000,
    ) -> SplitPreviewOut:
        """读取 ``markdown_file`` 切分，返回给用户查阅的片段（不写库）。"""
        from ai_agent_py.rag.splitter import splitter_label

        job = await self._require_job(job_id)
        size, overlap = self._split_limits(chunk_size, chunk_overlap)
        chunks, strategy = await asyncio.to_thread(self._job_chunks, job, size, overlap)
        cached = self.chunk_cache.put(
            job_id=job.id,
            chunks=chunks,
            chunk_size=size,
            chunk_overlap=overlap,
            strategy=strategy,
        )
        return SplitPreviewOut(
            job_id=job.id,
            file_name=job.file_name,
            suffix=job.suffix,
            source_file=job.markdown_file or "",
            split_id=cached.split_id,
            strategy=strategy,
            strategy_label=splitter_label(strategy),
            chunk_size=size,
            chunk_overlap=overlap,
            total=len(chunks),
            characters=sum(len(chunk.page_content) for chunk in chunks),
            truncated=len(chunks) > limit,
            chunks=[
                ChunkOut(
                    index=index,
                    content=chunk.page_content,
                    characters=len(chunk.page_content),
                    metadata=self._preview_metadata(chunk.metadata),
                )
                for index, chunk in enumerate(chunks[:limit])
            ],
        )

    async def ingest_rag(
        self,
        job_id: str,
        *,
        split_id: str | None = None,
        chunks: Sequence[ChunkIn] | None = None,
        total: int | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> RagIngestResultOut:
        """把切片写入 Milvus ``rag_chunk``（允许重复入库）。

        重复入库会先删除该 ``doc_id`` 的旧切片，再写入本次的切片；成功后把
        `rag_ingest`（RAG 已入库摘要）写进任务记录，前端状态列显示「已入库」。

        三种来源，优先级从高到低：

        1. ``split_id``：从进程内缓存读取切分预览时缓存的**完整**切片（推荐，
           请求体里只有几十字节的 id）；
        2. ``chunks``：直接使用调用方回传的切片（无状态 / 多实例场景）；
        3. 都不传：读取 ``markdown_file`` 现场切分（CLI / 脚本）。

        无论哪种来源都不再受预览截断影响：写进 Milvus 的是切分结果全集。
        """
        if split_id is not None and chunks is not None:
            raise ValueError("split_id 与 chunks 只能二选一")

        job = await self._require_job(job_id)
        if job.state is not JobState.done:
            raise MineruError("文档还没有解析完成，暂时不能入库")

        mode: Literal["cached", "reviewed", "resplit"]
        if split_id is not None:
            cached = self.chunk_cache.get(split_id)
            if cached is None:
                raise SplitCacheMiss(
                    "切片缓存已失效（服务重启 / 多实例 / 超过 30 分钟），请重新切分后再入库"
                )
            if cached.job_id != job.id:
                raise SplitCacheMiss("该切片缓存不属于当前文档，请重新切分后再入库")
            documents = list(cached.chunks)
            mode = "cached"
        elif chunks is not None:
            documents = await asyncio.to_thread(self._chunks_from_payload, chunks)
            mode = "reviewed"
        else:
            size, overlap = self._split_limits(chunk_size, chunk_overlap)
            documents, _ = await asyncio.to_thread(self._job_chunks, job, size, overlap)
            mode = "resplit"

        if not documents:
            raise MineruError("切分结果为空，无法入库" if mode == "resplit" else "没有可入库的切片")
        self._check_total(total, len(documents), mode)

        report = await asyncio.to_thread(self._write_rag_sync, job.id, documents)
        # 重新入库 = 这篇文档还在：撤销可能残留的补删标记，别让后台把刚写的新切片删掉。
        await asyncio.to_thread(self.delete_queue.discard, job.id)
        # RAG 入库状态记在独立的 rag_ingest 上（不覆盖 Chroma 的 ingest），
        # 重复入库时只刷新摘要与时间；写入器会先删掉该文档的旧切片。
        job.rag_ingest = RagIngestInfo(
            chunks=report.chunks,
            characters=report.characters,
            collection=f"{report.database}.{report.collection}",
            ingested_at=_now(),
        )
        await self._touch(job)
        logger.info(
            "任务 %s 写入 Milvus %s.%s：%d chunks（%s）",
            job.id,
            report.database,
            report.collection,
            report.chunks,
            {
                "cached": "split_id 缓存",
                "reviewed": "回传切片",
                "resplit": "服务端重新切分",
            }[mode],
        )
        return RagIngestResultOut(
            job_id=job.id,
            chunks=report.chunks,
            characters=report.characters,
            database=report.database,
            collection=report.collection,
            store=report.store,
            elapsed_seconds=round(report.elapsed_seconds, 2),
            mode=mode,
        )

    def _check_total(
        self, total: int | None, count: int, mode: Literal["cached", "reviewed", "resplit"]
    ) -> None:
        """校验回传的切分总数，避免只写回一部分切片。"""
        if total is None or total == count:
            return
        source = "缓存" if mode == "cached" else "回传"
        raise ValueError(
            f"切片数量不一致：切分共 {total} 个，{source} {count} 个；请重新切分后再入库"
        )

    def _write_rag_sync(self, doc_id: str, chunks: list[Any]) -> Any:
        from ai_agent_py.rag.milvus import MilvusRagWriter

        return MilvusRagWriter(self.settings).index_document(doc_id, chunks)

    def _split_limits(self, chunk_size: int | None, chunk_overlap: int | None) -> tuple[int, int]:
        size = self.settings.chunk_size if chunk_size is None else chunk_size
        if size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap is None:
            overlap = min(self.settings.chunk_overlap, max(0, size - 1))
        else:
            overlap = chunk_overlap
        if overlap < 0 or overlap >= size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        return size, overlap

    def _chunks_from_payload(self, chunks: Sequence[ChunkIn]) -> list[Any]:
        """把前端查阅过的切片还原成 Document，并按数组顺序重新编号。"""
        from langchain_core.documents import Document

        documents: list[Document] = []
        for index, chunk in enumerate(chunks):
            content = chunk.content.strip()
            if not content:
                raise ValueError(f"第 {index + 1} 个切片内容为空，无法入库")
            metadata = {
                key: value
                for key, value in chunk.metadata.items()
                if key in _REVIEW_METADATA_KEYS and isinstance(value, str | int | float | bool)
            }
            metadata["chunk_index"] = index
            metadata["characters"] = len(content)
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def _job_chunks(self, job: ParseJob, size: int, overlap: int) -> tuple[list[Any], str]:
        """读取 ``markdown_file`` 切分，返回 ``(片段, 切分策略)``。

        预览与「不传 chunks 的入库」共用这一段，保证两条路径切出来的块一致。
        """
        from ai_agent_py.rag.splitter import split_text, splitter_for_suffix

        text, suffix = self._split_source(job)
        chunks = split_text(
            text,
            suffix=suffix,
            settings=self.settings,
            chunk_size=size,
            chunk_overlap=overlap,
        )
        return chunks, splitter_for_suffix(suffix)

    def _split_source(self, job: ParseJob) -> tuple[str, str]:
        """切分正文只取 ``markdown_file``，返回 ``(正文, 用于选切分器的后缀)``。

        用户在预览页看到的就是这份 Markdown，切分必须与它同源，所以不再回头读
        原始 HTML / JSON 源文件：

        * PDF / DOCX / 图片 / HTML / ``.md``：MinerU（或直读）产出的 Markdown，
          走 Markdown 标题切分；
        * ``.json`` 直读任务的 ``full.md`` 是 ```json 代码块，取出块内原文后
          仍按 JSON 结构切分（否则切出来的是 Markdown 片段而不是 JSON）。
        """
        from ai_agent_py.rag.splitter import splitter_for_suffix

        if not job.markdown_file:
            raise FileNotFoundError("解析结果还没有生成，不能切分")
        path = self.job_dir(job.id) / job.markdown_file
        if not path.is_file():
            raise FileNotFoundError("找不到可切分的 Markdown 正文，请确认文档已解析完成")

        text = read_text(path)
        if splitter_for_suffix(job.suffix) == "json":
            match = _JSON_FENCE.search(text)
            if match and match.group("payload").strip():
                return match.group("payload").strip(), ".json"
        return text, ".md"

    @staticmethod
    def _preview_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """只回传前端需要展示的 metadata（标题层级、页码、错误）。"""
        keep = ("h1", "h2", "h3", "h4", "page", "json_error", "source", "chunk_index")
        return {
            key: value
            for key, value in metadata.items()
            if key in keep and isinstance(value, str | int | float | bool)
        }

    # ---- 后台任务 ------------------------------------------------------
    def _spawn(self, job: ParseJob) -> None:
        task = asyncio.create_task(self._run(job), name=f"mineru-parse-{job.id}")
        self._tasks[job.id] = task

        def _done(_: asyncio.Task[None], job_id: str = job.id) -> None:
            if self._tasks.get(job_id) is task:
                self._tasks.pop(job_id, None)

        task.add_done_callback(_done)

    def _limiter(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(max(1, self.settings.mineru_max_concurrent))
            self._semaphore_loop = loop
        return self._semaphore

    async def _run(self, job: ParseJob) -> None:
        started = time.perf_counter()
        if job.started_at is None:
            job.started_at = _now()
        try:
            async with self._limiter(), self._client_factory() as client:
                if not job.batch_id:
                    await self._submit(client, job)
                item = await self._poll(client, job)
                await self._download(client, job, item)
        except asyncio.CancelledError:
            await self._fail(job, "任务已取消")
            raise
        except AiAgentError as exc:
            logger.warning("解析任务 %s 失败: %s", job.id, exc)
            await self._fail(job, str(exc))
        except Exception as exc:  # 网络、磁盘等意外错误
            logger.exception("解析任务 %s 异常", job.id)
            await self._fail(job, f"解析任务异常: {exc}")
        else:
            job.elapsed_seconds = round(time.perf_counter() - started, 2)
            job.finished_at = _now()
            if job.total_pages:
                job.extracted_pages = job.total_pages
            await self._touch(job, JobState.done)
            logger.info("解析完成 %s: %s, %d 字符", job.id, job.markdown_file, job.markdown_chars)

    async def _submit(self, client: MineruClient, job: ParseJob) -> None:
        await self._touch(job, JobState.uploading)
        if not job.source_file:
            raise MineruError("缺少源文件，请重新上传")
        source = self.job_dir(job.id) / job.source_file
        if not source.is_file():
            raise MineruError("源文件已丢失，请重新上传")

        content = await asyncio.to_thread(source.read_bytes)
        upload = await client.create_batch_upload(job.file_name, job.id, job.options)
        job.batch_id = upload.batch_id
        await self._touch(job)
        logger.info("申请到上传链接 %s，开始上传 %s", upload.batch_id, job.file_name)
        await client.upload(upload.file_urls[0], content)
        await self._touch(job, JobState.parsing)

    async def _poll(self, client: MineruClient, job: ParseJob) -> BatchItem:
        def on_progress(item: BatchItem) -> None:
            if item.extracted_pages:
                job.extracted_pages = item.extracted_pages
            if item.total_pages:
                job.total_pages = item.total_pages

        item = await client.wait_batch(
            job.batch_id or "",
            data_id=job.id,
            timeout=self.settings.mineru_timeout,
            poll_interval=self.settings.mineru_poll_interval,
            on_progress=on_progress,
        )
        if job.state is not JobState.parsing:
            job.state = JobState.parsing
        await self._touch(job)
        return item

    async def _download(self, client: MineruClient, job: ParseJob, item: BatchItem) -> None:
        if not item.full_zip_url:
            raise MineruError("解析完成但没有返回结果下载地址")
        await self._touch(job, JobState.downloading)
        payload = await client.download(item.full_zip_url)
        job_dir = self.job_dir(job.id)
        summary = await asyncio.to_thread(extract_result_zip, payload, job_dir)
        await asyncio.to_thread((job_dir / "mineru.zip").write_bytes, payload)

        job.zip_file = "mineru.zip"
        job.markdown_file = summary.markdown_name
        job.markdown_chars = summary.characters
        job.asset_count = summary.assets
        if item.total_pages:
            job.total_pages = item.total_pages
        elif summary.pages:
            job.total_pages = summary.pages
        job.extracted_pages = job.total_pages
        await self._touch(job)
        logger.info(
            "结果已保存 %s: %s（%d 字符, %d 张图片）",
            job.id,
            summary.markdown_name,
            summary.characters,
            summary.assets,
        )

    async def _fail(self, job: ParseJob, message: str) -> None:
        job.finished_at = _now()
        try:
            await self._touch(job, JobState.failed, message=(message or "未知错误").strip()[:500])
        except AiAgentError as exc:
            # 失败状态也写不进去时（MySQL 挂了）不能把取消/异常路径再炸一次。
            logger.error("任务 %s 的失败状态未能写入 %s: %s", job.id, TABLE_NAME, exc)

    # ---- 状态更新与持久化 ----------------------------------------------
    async def _touch(
        self,
        job: ParseJob,
        state: JobState | None = None,
        *,
        message: str | None = None,
    ) -> None:
        if state is not None:
            job.state = state
        if message is not None:
            job.err_msg = message
        job.updated_at = _now()
        await self._save(job)

    async def _save(self, job: ParseJob) -> None:
        """按主键 upsert 单条任务记录（串行化，避免同一任务并发写）。"""
        async with self._write_lock:
            await asyncio.to_thread(self.store.save, job)

    # ---- 杂项 ----------------------------------------------------------
    async def _require_job(self, job_id: str) -> ParseJob:
        """同 `get`，但任务不存在时抛 `KeyError`（API 层翻译成 404）。"""
        job = await self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def _read_markdown_text(self, job: ParseJob) -> str:
        if not job.markdown_file:
            raise FileNotFoundError("解析结果还没有生成")
        path = self.job_dir(job.id) / job.markdown_file
        if not path.is_file():
            raise FileNotFoundError(f"Markdown 文件不存在: {path.name}")
        return path.read_text(encoding="utf-8", errors="replace")
