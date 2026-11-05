"""私域 OSS 图片服务：上传校验、对象键规划、OSS 直链签名。

对象键形如 ``image3/2026/11/05/<uuid>.png``（目录来自 ``OSS_PREFIX``）。
Bucket 是私有的，图片统一通过 **OSS 直链**读取：

* 上传 / 列表 / 换取直链接口返回预签名 GET URL（``OSSAccessKeyId`` + ``Expires``
  + ``Signature``），有效期由 ``OSS_URL_EXPIRES`` 控制，可直接放 ``<img>``、
  浏览器打开或复制分享；
* ``download_url`` 在签名里额外带上 ``response-content-disposition=attachment``，
  浏览器点击即下载；
* 上传 / 列表 / 删除 / 换取直链这些管理操作仍要 access（页面用服务端会话 Cookie）。

直链由服务端 AccessKey 签名，AccessKey Secret 不会出现在 URL 里；但 URL 本身
是限时访问凭证，请勿转发。
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote
from uuid import uuid4

from ai_agent_py.alioss.client import OssClient
from ai_agent_py.alioss.models import (
    OssImageOut,
    OssObjectListOut,
    OssObjectOut,
    OssOverviewOut,
    OssPresignOut,
)
from ai_agent_py.config import Settings
from ai_agent_py.errors import (
    OssInvalidKey,
    OssNotConfigured,
    OssUploadRejected,
)

logger = logging.getLogger(__name__)

CONTENT_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def content_type_for(key: str) -> str:
    return CONTENT_TYPES.get(Path(key).suffix.lower(), "application/octet-stream")


def _looks_like_image(suffix: str, data: bytes) -> bool:
    """文件头校验，避免把任意文件改成 .png 后塞进私有桶。"""
    if suffix in {".jpg", ".jpeg"}:
        return data.startswith(b"\xff\xd8\xff")
    if suffix == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix == ".gif":
        return data[:6] in {b"GIF87a", b"GIF89a"}
    if suffix == ".bmp":
        return data.startswith(b"BM")
    if suffix == ".webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return True


def _parse_oss_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class OssImageService:
    """图片上传 / 列举 / 直链 / 删除；一个 app 一个实例（挂在 ``app.state``）。"""

    def __init__(self, settings: Settings, client: OssClient | None = None) -> None:
        self.settings = settings
        self._client = client

    @property
    def client(self) -> OssClient:
        if self._client is None:
            self._client = OssClient(self.settings)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "aclose", None)
            if close is not None:
                await close()

    # ---- 概览 ------------------------------------------------------------
    def overview(self) -> OssOverviewOut:
        cfg = self.settings
        warnings: list[str] = []
        if not cfg.oss_configured:
            warnings.append(
                "未配置 OSS：请补全 OSS_BUCKET / OSS_ENDPOINT / "
                "OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET"
            )
        if not cfg.oss_access_configured:
            warnings.append("未配置 OSS_ACCESS_TOKEN：上传与列表接口都会返回 503")
        return OssOverviewOut(
            configured=cfg.oss_configured,
            access_configured=cfg.oss_access_configured,
            bucket=cfg.oss_bucket,
            endpoint=cfg.oss_base_url or cfg.oss_endpoint,
            region=cfg.oss_region,
            prefix=cfg.oss_prefix,
            url_expires=cfg.oss_url_expires,
            max_file_mb=cfg.oss_max_file_mb,
            accepted_suffixes=list(cfg.oss_suffixes),
            warnings=warnings,
        )

    # ---- 上传 ------------------------------------------------------------
    async def upload(self, file_name: str, data: bytes) -> OssImageOut:
        self._ensure_ready()
        name = (file_name or "image").strip() or "image"
        suffix = Path(name).suffix.lower()
        if suffix not in self.settings.oss_suffixes:
            allowed = ", ".join(self.settings.oss_suffixes)
            raise OssUploadRejected(f"只支持上传图片（{allowed}），收到的是 {suffix or '无后缀'}")

        limit = self.settings.oss_max_file_mb * 1024 * 1024
        if not data:
            raise OssUploadRejected("上传内容为空")
        if len(data) > limit:
            raise OssUploadRejected(
                f"图片过大：{len(data) / 1024 / 1024:.2f} MB，上限 {self.settings.oss_max_file_mb} MB"
            )
        if not _looks_like_image(suffix, data):
            raise OssUploadRejected(f"文件内容不是有效的 {suffix} 图片")

        key = self._build_key(suffix)
        content_type = CONTENT_TYPES.get(suffix, "application/octet-stream")
        etag = await self.client.put_object(key, data, content_type)
        url, download_url, ttl = self._urls(key)
        return OssImageOut(
            key=key,
            name=name,
            size=len(data),
            content_type=content_type,
            etag=etag,
            url=url,
            download_url=download_url,
            expires_in=ttl,
            uploaded_at=datetime.now(UTC),
        )

    # ---- 列举 / 直链 / 删除 ----------------------------------------------
    async def list_objects(
        self,
        *,
        limit: int = 50,
        marker: str | None = None,
    ) -> OssObjectListOut:
        self._ensure_ready()
        prefix = f"{self.settings.oss_prefix}/"
        result = await self.client.list_objects(prefix=prefix, max_keys=limit, marker=marker)
        objects: list[OssObjectOut] = []
        for item in sorted(result.objects, key=lambda obj: obj.last_modified, reverse=True):
            url, download_url, ttl = self._urls(item.key)
            objects.append(
                OssObjectOut(
                    key=item.key,
                    name=item.key.rsplit("/", 1)[-1],
                    size=item.size,
                    content_type=content_type_for(item.key),
                    last_modified=_parse_oss_time(item.last_modified),
                    url=url,
                    download_url=download_url,
                    expires_in=ttl,
                )
            )
        return OssObjectListOut(
            prefix=prefix,
            count=len(objects),
            truncated=result.truncated,
            next_marker=result.next_marker,
            objects=objects,
        )

    def presign(self, key: str) -> OssPresignOut:
        """按对象键换取新的 OSS 直链（上传 / 列表之外的补充入口）。"""
        self._ensure_oss()
        normalized = self.normalize_key(key)
        url, download_url, ttl = self._urls(normalized)
        return OssPresignOut(
            key=normalized,
            url=url,
            download_url=download_url,
            expires_in=ttl,
        )

    async def delete(self, key: str) -> str:
        self._ensure_ready()
        normalized = self.normalize_key(key)
        await self.client.delete_object(normalized)
        return normalized

    # ---- 内部 ------------------------------------------------------------
    def _ensure_oss(self) -> None:
        if not self.settings.oss_configured:
            raise OssNotConfigured(
                "未配置 OSS：请补全 OSS_BUCKET / OSS_ENDPOINT / "
                "OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET"
            )

    def _ensure_ready(self) -> None:
        self._ensure_oss()
        if not self.settings.oss_access_configured:
            raise OssNotConfigured("未配置 OSS_ACCESS_TOKEN：上传 / 列表 / 删除接口都必须带 access")

    def _build_key(self, suffix: str) -> str:
        now = datetime.now()
        return f"{self.settings.oss_prefix}/{now:%Y/%m/%d}/{uuid4().hex}{suffix}"

    def _urls(self, key: str) -> tuple[str, str, int]:
        """返回（预览直链, 下载直链, 有效期）；下载直链带 attachment 响应头。"""
        url, ttl = self.client.presign_url(key)
        filename = Path(key).name or "image"
        # 响应头里不能出现引号 / 换行，避免自定义对象键破坏 Content-Disposition。
        filename = re.sub(r'["\\\r\n]', "_", filename)
        download_url, _ = self.client.presign_url(
            key,
            expires_in=ttl,
            response_content_disposition=f'attachment; filename="{filename}"',
        )
        return url, download_url, ttl

    def normalize_key(self, key: str) -> str:
        """只允许访问 ``OSS_PREFIX/`` 下的对象，并拒绝 ``.`` / ``..`` / 空片段。"""
        candidate = unquote((key or "").strip()).lstrip("/")
        if not candidate:
            raise OssInvalidKey("对象键不能为空")
        if "\\" in candidate:
            raise OssInvalidKey("对象键不能包含反斜杠")
        parts = candidate.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise OssInvalidKey(f"对象键包含非法路径片段: {candidate}")
        allowed_prefix = f"{self.settings.oss_prefix}/"
        if not candidate.startswith(allowed_prefix):
            raise OssInvalidKey(f"只允许访问 {allowed_prefix} 下的对象")
        return candidate
