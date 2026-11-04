"""文档解析 API：上传 → MinerU 精准解析 → 查询/预览/下载/入库。

路由前缀 ``/api/documents``，与前端「RAG 管理 → 文档解析」页面对应。
解析是异步的：``POST /parse`` 立即返回任务，前端轮询 ``GET /{job_id}`` 获取进度。
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from ai_agent_py.config import get_settings
from ai_agent_py.errors import (
    AiAgentError,
    ConfigurationError,
    DatabaseError,
    DocumentLoadError,
    MilvusError,
    MilvusSchemaError,
    MineruError,
    SplitCacheMiss,
)
from ai_agent_py.mineru.models import (
    DocumentConfigOut,
    IngestResultOut,
    MarkdownOut,
    ParseJob,
    ParseOptions,
)
from ai_agent_py.mineru.service import (
    MODEL_VERSIONS,
    SUPPORTED_LANGUAGES,
    UPLOAD_SUFFIXES,
    DocumentService,
)
from ai_agent_py.rag.models import RagIngestRequest, RagIngestResultOut, SplitPreviewOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])
asset_router = APIRouter(tags=["documents"])

_CHUNK = 1024 * 1024
# 支持 "1-10"、"2,4-6"、"2--2"（第 2 页到倒数第二页）、"-2"（倒数第二页）
_PAGE_RANGES = re.compile(r"^-?\d+(?:--?\d+)?(?:,-?\d+(?:--?\d+)?)*$")


# ---- 依赖 ---------------------------------------------------------------
def get_service(request: Request) -> DocumentService:
    """每个 app 一个服务实例（挂在 app.state，测试可直接替换）。"""
    service: DocumentService | None = getattr(request.app.state, "documents", None)
    if service is None:
        service = DocumentService(get_settings())
        request.app.state.documents = service
    return service


async def _stream(upload: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await upload.read(_CHUNK):
        yield chunk


def _not_found(job_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"解析任务不存在: {job_id}")


def _storage_error(exc: AiAgentError) -> HTTPException:
    """任务记录库（MySQL）不可用：区分于文件内容错误，返回 503。"""
    return HTTPException(
        status_code=503,
        detail=f"任务记录库不可用：{exc}（请检查 MYSQL_URL 并执行 docs/mysql 下的建表脚本）",
    )


# ---- 接口 ---------------------------------------------------------------
@router.get("/config", response_model=DocumentConfigOut, summary="解析配置与能力")
async def document_config(request: Request) -> DocumentConfigOut:
    service = get_service(request)
    cfg = service.settings
    return DocumentConfigOut(
        configured=bool(cfg.resolved_mineru_api_key),
        base_url=cfg.mineru_base_url,
        model_version=cfg.mineru_model_version,
        default_options=service.default_options(),
        model_versions=list(MODEL_VERSIONS),
        languages=list(SUPPORTED_LANGUAGES),
        max_file_mb=cfg.mineru_max_file_mb,
        accepted_suffixes=sorted(UPLOAD_SUFFIXES),
        output_dir=str(cfg.mineru_output_dir),
        rag_configured=bool(cfg.resolved_embedding_api_key),
        embedding_model=cfg.embedding_model,
        rag_database=cfg.milvus_database,
        rag_collection=cfg.milvus_collection,
        rag_dim=cfg.milvus_dim,
    )


@router.post("/parse", response_model=ParseJob, status_code=202, summary="上传文档并提交解析")
async def parse_document(
    request: Request,
    file: Annotated[UploadFile, File(description="待解析文档（pdf/docx/ppt/图片等）")],
    model_version: Annotated[str | None, Form(description="pipeline / vlm / MinerU-HTML")] = None,
    language: Annotated[str | None, Form(description="OCR 语言，默认 ch")] = None,
    enable_formula: Annotated[bool | None, Form(description="公式识别")] = None,
    enable_table: Annotated[bool | None, Form(description="表格识别")] = None,
    is_ocr: Annotated[bool | None, Form(description="扫描件 OCR")] = None,
    page_ranges: Annotated[str | None, Form(description='页码范围，如 "1-10,12"')] = None,
) -> ParseJob:
    service = get_service(request)
    defaults = service.default_options()
    raw_name = file.filename or "document"

    # HTML 文件只有 MinerU-HTML 模型能解析；用户没显式选模型时自动切换。
    default_model = defaults.model_version
    if Path(raw_name).suffix.lower() in {".html", ".htm"} and not model_version:
        default_model = "MinerU-HTML"
    chosen_model = (model_version or default_model).strip()
    if chosen_model not in MODEL_VERSIONS:
        raise HTTPException(
            status_code=422, detail=f"model_version 只能是 {', '.join(MODEL_VERSIONS)}"
        )
    ranges = (page_ranges or "").strip()
    if ranges and not _PAGE_RANGES.match(ranges):
        raise HTTPException(status_code=422, detail=f"page_ranges 格式不正确: {ranges}")

    options = ParseOptions(
        model_version=chosen_model,
        language=(language or defaults.language).strip() or defaults.language,
        enable_formula=defaults.enable_formula if enable_formula is None else enable_formula,
        enable_table=defaults.enable_table if enable_table is None else enable_table,
        is_ocr=defaults.is_ocr if is_ocr is None else is_ocr,
        page_ranges=ranges or None,
    )

    try:
        return await service.create(raw_name, _stream(file), options)
    except DocumentLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ConfigurationError, DatabaseError) as exc:
        raise _storage_error(exc) from exc
    finally:
        await file.close()


@router.get("", response_model=list[ParseJob], summary="解析任务列表")
async def list_documents(
    request: Request,
    state: Annotated[str | None, Query(description="按状态过滤")] = None,
    keyword: Annotated[str | None, Query(description="按文件名搜索")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[ParseJob]:
    service = get_service(request)
    try:
        await service.ensure_loaded()
    except (ConfigurationError, DatabaseError) as exc:
        raise _storage_error(exc) from exc
    jobs = await service.list_jobs()
    if state:
        jobs = [job for job in jobs if job.state.value == state]
    if keyword:
        needle = keyword.strip().lower()
        jobs = [job for job in jobs if needle in job.file_name.lower()]
    return jobs[:limit]


@router.get("/{job_id}", response_model=ParseJob, summary="查询任务状态")
async def get_document(request: Request, job_id: str) -> ParseJob:
    service = get_service(request)
    try:
        await service.ensure_loaded()
    except (ConfigurationError, DatabaseError) as exc:
        raise _storage_error(exc) from exc
    try:
        job = await service.get(job_id)
    except (ConfigurationError, DatabaseError) as exc:
        raise _storage_error(exc) from exc
    if job is None:
        raise _not_found(job_id)
    return job


@router.get("/{job_id}/markdown", response_model=MarkdownOut, summary="Markdown 预览")
async def get_markdown(request: Request, job_id: str) -> MarkdownOut:
    service = get_service(request)
    try:
        return await service.read_markdown(job_id)
    except KeyError as exc:
        raise _not_found(job_id) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ConfigurationError, DatabaseError) as exc:
        raise _storage_error(exc) from exc


@router.get("/{job_id}/download", summary="下载解析结果")
async def download(
    request: Request,
    job_id: str,
    kind: Annotated[Literal["markdown", "zip", "source"], Query()] = "markdown",
) -> FileResponse:
    service = get_service(request)
    try:
        path, filename = await service.artifact(job_id, kind)
    except KeyError as exc:
        raise _not_found(job_id) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ConfigurationError, DatabaseError) as exc:
        raise _storage_error(exc) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {path.name}")
    media_type = {
        "markdown": "text/markdown; charset=utf-8",
        "zip": "application/zip",
        "source": "application/octet-stream",
    }[kind]
    return FileResponse(path, filename=filename, media_type=media_type)


@router.delete("/{job_id}", status_code=204, summary="删除任务与结果")
async def delete_document(request: Request, job_id: str) -> None:
    service = get_service(request)
    try:
        deleted = await service.delete(job_id)
    except (ConfigurationError, DatabaseError) as exc:
        raise _storage_error(exc) from exc
    if not deleted:
        raise _not_found(job_id)
    return None


@router.post("/{job_id}/ingest", response_model=IngestResultOut, summary="解析结果写入知识库")
async def ingest_document(request: Request, job_id: str) -> IngestResultOut:
    service = get_service(request)
    try:
        return await service.ingest(job_id)
    except KeyError as exc:
        raise _not_found(job_id) from exc
    except DatabaseError as exc:
        raise _storage_error(exc) from exc
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=503, detail=f"向量库未就绪：{exc}（请先配置 EMBEDDING_API_KEY）"
        ) from exc
    except (MineruError, DocumentLoadError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AiAgentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # 向量模型/向量库的远端错误
        logger.exception("入库失败")
        raise HTTPException(status_code=502, detail=f"写入知识库失败: {exc}") from exc


@router.get("/{job_id}/chunks", response_model=SplitPreviewOut, summary="切分预览")
async def split_preview(
    request: Request,
    job_id: str,
    chunk_size: Annotated[
        int | None, Query(ge=20, le=4000, description="块大小（字符），默认 CHUNK_SIZE")
    ] = None,
    chunk_overlap: Annotated[
        int | None, Query(ge=0, le=1000, description="块重叠（字符），默认 CHUNK_OVERLAP")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=20000, description="最多返回的片段数")] = 5000,
) -> SplitPreviewOut:
    """读取 ``markdown_file`` 切分并缓存，返回 ``split_id`` 与给用户查阅的片段。"""
    service = get_service(request)
    try:
        return await service.split_preview(
            job_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            limit=limit,
        )
    except KeyError as exc:
        raise _not_found(job_id) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ConfigurationError, DatabaseError) as exc:
        raise _storage_error(exc) from exc
    except DocumentLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AiAgentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/{job_id}/rag",
    response_model=RagIngestResultOut,
    summary="把查阅过的切片写入 Milvus RAG 表",
)
async def ingest_rag(
    request: Request,
    job_id: str,
    payload: RagIngestRequest | None = None,
) -> RagIngestResultOut:
    """把用户查阅过的切片写入 Milvus ``rag_chunk``（按 doc_id 覆盖旧块）。

    * 传 ``split_id``（切分预览返回的缓存 id）：服务端从进程内缓存读取完整切片，
      不再重新切分，请求体只有几十字节；
    * 传 ``chunks``：无状态回传切片内容，适合多实例 / 脚本；
    * 都不传：退回服务端读取 ``markdown_file`` 现切，兼容 CLI 调用。
    """
    service = get_service(request)
    body = payload or RagIngestRequest()
    try:
        return await service.ingest_rag(
            job_id,
            split_id=body.split_id,
            chunks=body.chunks,
            total=body.total,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
        )
    except KeyError as exc:
        raise _not_found(job_id) from exc
    except SplitCacheMiss as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ConfigurationError, MilvusSchemaError) as exc:
        raise HTTPException(status_code=503, detail=f"RAG 入库未就绪：{exc}") from exc
    except DatabaseError as exc:
        raise _storage_error(exc) from exc
    except MilvusError as exc:
        raise HTTPException(status_code=502, detail=f"写入 Milvus 失败：{exc}") from exc
    except (MineruError, DocumentLoadError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AiAgentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG 入库失败")
        raise HTTPException(status_code=502, detail=f"写入 Milvus 失败: {exc}") from exc


# ---- 解析结果里的图片等静态资源 ------------------------------------------
@asset_router.get("/parsed/{job_id}/{file_path:path}", include_in_schema=False)
async def parsed_asset(request: Request, job_id: str, file_path: str) -> FileResponse:
    """`full.md` 里以相对路径引用的图片（``images/xxx.jpg``）。"""
    service = get_service(request)
    try:
        known = await service.get(job_id)
    except (ConfigurationError, DatabaseError) as exc:
        raise _storage_error(exc) from exc
    if known is None:
        raise _not_found(job_id)
    base = service.job_dir(job_id).resolve()
    target = (base / file_path).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target)
