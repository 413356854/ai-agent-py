"""私域 OSS 接口：上传图片、列举、换取 OSS 直链与删除。

路由前缀 ``/api/oss``，对应前端「私域 OSS」栏目。图片读取统一走 **OSS 直链**
（服务端预签名，带 ``OSSAccessKeyId`` + ``Expires`` + ``Signature``），本服务
不再代理图片字节：

* 上传 / 列举 / 换取直链 / 删除 → 必须带 access（页面用服务端会话 Cookie）；
* 上传与列表直接返回 ``url``（预览直链）和 ``download_url``（附件直链）；
* 所有对象都在私有 Bucket 的 ``OSS_PREFIX`` 目录下，接口拒绝访问其他前缀。
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile

from ai_agent_py.alioss.models import (
    OssImageOut,
    OssObjectListOut,
    OssOverviewOut,
    OssPresignOut,
)
from ai_agent_py.alioss.service import OssImageService
from ai_agent_py.config import get_settings
from ai_agent_py.errors import (
    OssAccessDenied,
    OssError,
    OssInvalidKey,
    OssNotConfigured,
    OssObjectNotFound,
    OssUploadRejected,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/oss", tags=["oss"])

ACCESS_HEADER = "X-OSS-Access"
#: 浏览器会话 Cookie：由 `GET /api/oss/overview` 签发，值由 OSS_ACCESS_TOKEN 派生，
#: 前端无需知道口令；脚本仍可用 `X-OSS-Access` / `?access=`。
SESSION_COOKIE = "oss_session"
_SESSION_MESSAGE = b"ai-agent-py:oss-session:v1"


# ---- 依赖 ---------------------------------------------------------------
def get_service(request: Request) -> OssImageService:
    """每个 app 一个服务实例（挂在 app.state，测试可直接替换）。"""
    service: OssImageService | None = getattr(request.app.state, "oss", None)
    if service is None:
        service = OssImageService(get_settings())
        request.app.state.oss = service
    return service


def _provided_access(request: Request, access: str | None) -> str | None:
    """access 口令来源：``X-OSS-Access`` → ``X-Access-Token`` → Bearer → ``?access=``。"""
    header = request.headers.get(ACCESS_HEADER) or request.headers.get("x-access-token")
    if header and header.strip():
        return header.strip()
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return access.strip() if access and access.strip() else None


def _master_access_ok(provided: str | None) -> bool:
    expected = get_settings().resolved_oss_access_token
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="未配置 OSS_ACCESS_TOKEN：请在 .env 里设置访问口令后再试",
        )
    if not provided:
        return False
    return secrets.compare_digest(provided, expected)


def _session_value() -> str | None:
    """由 OSS_ACCESS_TOKEN 派生的会话值；AccessToken 本身不会出现在 Cookie / JS 里。"""
    secret = get_settings().resolved_oss_access_token
    if not secret:
        return None
    return hmac.new(secret.encode(), _SESSION_MESSAGE, hashlib.sha256).hexdigest()


def _session_ok(request: Request) -> bool:
    expected = _session_value()
    provided = request.cookies.get(SESSION_COOKIE)
    if not expected or not provided:
        return False
    return secrets.compare_digest(provided, expected)


def _set_session_cookie(request: Request, response: Response) -> None:
    """页面打开时自动签发 HttpOnly 会话，前端不再要求输入访问口令。"""
    value = _session_value()
    if not value:
        return
    response.set_cookie(
        SESSION_COOKIE,
        value,
        path="/api/oss",
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
    )


async def require_access(
    request: Request,
    access: Annotated[
        str | None, Query(description="访问口令（也可用 X-OSS-Access 请求头）")
    ] = None,
) -> None:
    """上传 / 列举 / 换取直链 / 删除的 access 校验：显式口令或页面会话 Cookie。"""
    if not (_master_access_ok(_provided_access(request, access)) or _session_ok(request)):
        raise HTTPException(
            status_code=401,
            detail="access 口令缺失或错误：请确认服务端 OSS_ACCESS_TOKEN 配置，或刷新页面重新建立会话",
            headers={ACCESS_HEADER: "required"},
        )


def _error_response(exc: Exception) -> HTTPException:
    """把 OSS 异常翻译成 HTTP 状态码（中文原因直接展示给前端）。"""
    if isinstance(exc, OssNotConfigured):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, OssObjectNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (OssUploadRejected, OssInvalidKey)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, OssAccessDenied):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, OssError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


# ---- 接口 ---------------------------------------------------------------
@router.get("/overview", response_model=OssOverviewOut, summary="私域 OSS 配置与访问要求")
def overview(request: Request, response: Response) -> OssOverviewOut:
    """公开的配置概览；同时给浏览器签发 HttpOnly 会话 Cookie（页面无需输入口令）。"""
    _set_session_cookie(request, response)
    response.headers["Cache-Control"] = "no-store"
    return OssImageService(get_settings()).overview()


@router.post(
    "/images",
    response_model=OssImageOut,
    status_code=201,
    dependencies=[Depends(require_access)],
    summary="上传图片到私有 Bucket，返回 OSS 直链",
)
async def upload_image(
    request: Request,
    file: Annotated[UploadFile, File(description="图片文件（jpg/png/gif/webp/bmp）")],
) -> OssImageOut:
    service = get_service(request)
    cfg = get_settings()
    limit = cfg.oss_max_file_mb * 1024 * 1024
    try:
        data = await file.read(limit + 1)
        return await service.upload(file.filename or "image", data)
    except Exception as exc:
        logger.warning("OSS 上传失败：%s", exc)
        raise _error_response(exc) from exc
    finally:
        await file.close()


@router.get(
    "/objects",
    response_model=OssObjectListOut,
    dependencies=[Depends(require_access)],
    summary="列举 OSS_PREFIX 下的图片（最新在前）",
)
async def list_objects(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200, description="最多返回多少个对象")] = 50,
    marker: Annotated[str | None, Query(description="上一页返回的 next_marker")] = None,
) -> OssObjectListOut:
    service = get_service(request)
    try:
        return await service.list_objects(limit=limit, marker=marker)
    except Exception as exc:
        logger.warning("OSS 列举失败：%s", exc)
        raise _error_response(exc) from exc


@router.get(
    "/presign",
    response_model=OssPresignOut,
    dependencies=[Depends(require_access)],
    summary="按对象键换取新的 OSS 直链",
)
def presign_image(
    request: Request,
    key: Annotated[str, Query(min_length=1, description="对象键，如 image3/2026/11/05/x.png")],
) -> OssPresignOut:
    service = get_service(request)
    try:
        return service.presign(key)
    except Exception as exc:
        logger.warning("OSS 换取直链失败：%s", exc)
        raise _error_response(exc) from exc


@router.delete(
    "/images/{key:path}",
    status_code=204,
    dependencies=[Depends(require_access)],
    summary="删除一张图片",
)
async def delete_image(request: Request, key: str) -> None:
    service = get_service(request)
    try:
        await service.delete(key)
    except Exception as exc:
        logger.warning("OSS 删除失败：%s", exc)
        raise _error_response(exc) from exc
    return None
