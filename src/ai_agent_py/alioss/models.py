"""私域 OSS 接口模型（与前端 ``web/src/types.ts`` 对应）。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class OssOverviewOut(BaseModel):
    """``GET /api/oss/overview``：Bucket 配置与访问要求（不含任何密钥）。"""

    configured: bool = Field(description="Bucket / Endpoint / AccessKey 是否齐全")
    access_configured: bool = Field(description="是否配置了 OSS_ACCESS_TOKEN")
    access_required: bool = Field(
        default=True, description="上传 / 列举 / 删除 / 换取直链都需要 access（页面用会话 Cookie）"
    )
    bucket: str | None = None
    endpoint: str | None = None
    region: str = ""
    prefix: str = ""
    url_expires: int = Field(description="OSS 直链的有效期（秒）")
    max_file_mb: int = Field(description="单张图片大小上限（MB）")
    accepted_suffixes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class OssImageOut(BaseModel):
    """上传成功后的图片信息与 OSS 直链（可直接放 ``<img>`` / 浏览器打开）。"""

    key: str = Field(description="OSS 对象键，例如 image3/2026/11/05/<uuid>.png")
    name: str = Field(description="原始文件名")
    size: int = Field(description="字节数")
    content_type: str = "application/octet-stream"
    etag: str | None = None
    url: str = Field(description="OSS 直链（预签名 GET，可直接放 <img> 或浏览器打开）")
    download_url: str = Field(
        description="OSS 直链（预签名 GET + Content-Disposition: attachment）"
    )
    expires_in: int = Field(description="上述两个直链的有效期（秒）")
    uploaded_at: datetime


class OssObjectOut(BaseModel):
    """列表里的一个对象（含 OSS 直链）。"""

    key: str
    name: str
    size: int
    content_type: str = "application/octet-stream"
    last_modified: datetime | None = None
    url: str = Field(description="OSS 直链（预签名 GET）")
    download_url: str = Field(description="OSS 直链（附件下载）")
    expires_in: int


class OssPresignOut(BaseModel):
    """``GET /api/oss/presign``：按对象键换取新的 OSS 直链。"""

    key: str
    url: str
    download_url: str
    expires_in: int


class OssObjectListOut(BaseModel):
    """``GET /api/oss/objects``：``OSS_PREFIX`` 下的图片列表（最新在前）。"""

    prefix: str
    count: int
    truncated: bool = False
    next_marker: str | None = None
    objects: list[OssObjectOut] = Field(default_factory=list)
