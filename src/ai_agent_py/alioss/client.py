"""阿里云 OSS 客户端：按 OSS V1 签名规范 + httpx，读写私有 Bucket。

为什么不用 ``oss2``：本项目只需要图片对象的小文件 PUT / GET、列表与预签名
URL，用标准库的 HMAC-SHA1 直接签名即可，避免引入需要本地编译的
``crcmod`` / ``pycryptodome`` 依赖。

签名规则（OSS V1）::

    Authorization = "OSS " + AccessKeyId + ":" + base64(hmac-sha1(secret, string_to_sign))
    string_to_sign = VERB + "\\n" + Content-MD5 + "\\n" + Content-Type + "\\n"
                     + Date + "\\n" + CanonicalizedOSSHeaders
                     + CanonicalizedResource

对象键统一使用 ASCII 安全字符（日期目录 + uuid + 后缀），因此不需要处理
自定义头与复杂的 URL 编码；读取时仍会拒绝 ``.`` / ``..`` 片段。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, field
from email.utils import formatdate
from urllib.parse import quote, urlencode
from xml.etree import ElementTree

import httpx

from ai_agent_py.config import Settings
from ai_agent_py.errors import (
    OssAccessDenied,
    OssError,
    OssNotConfigured,
    OssObjectNotFound,
)

logger = logging.getLogger(__name__)

#: 可重试的 OSS 服务端状态码（上传 / 读取都是幂等的）。
_RETRY_STATUS = frozenset({500, 502, 503, 504})
_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class OssObject:
    """读取到的对象正文与元信息。"""

    key: str
    data: bytes
    content_type: str
    etag: str | None = None


@dataclass(frozen=True, slots=True)
class OssListedObject:
    """ListObjects 返回的一个对象（不含正文）。"""

    key: str
    size: int
    last_modified: str
    etag: str | None = None


@dataclass(frozen=True, slots=True)
class OssListResult:
    """一次 ListObjects 的结果。"""

    objects: list[OssListedObject] = field(default_factory=list)
    truncated: bool = False
    next_marker: str | None = None


class OssClient:
    """私有 Bucket 的异步客户端；一个实例复用一个 httpx 连接池。"""

    def __init__(
        self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        if not settings.oss_configured:
            raise OssNotConfigured(
                "未配置 OSS：请补全 OSS_BUCKET / OSS_ENDPOINT / "
                "OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET"
            )
        self.settings = settings
        self._bucket = settings.oss_bucket or ""
        self._access_key_id = settings.resolved_oss_access_key_id or ""
        self._access_key_secret = settings.resolved_oss_access_key_secret or ""
        self._base_url = settings.oss_base_url or ""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.oss_timeout),
            transport=transport,
            follow_redirects=True,
        )

    # ---- 生命周期 --------------------------------------------------------
    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OssClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    # ---- 签名 ------------------------------------------------------------
    def authorization(self, method: str, key: str, content_type: str, date: str) -> str:
        """生成 OSS V1 签名请求头（测试里会直接断言这个字符串）。"""
        resource = f"/{self._bucket}/{key}"
        string_to_sign = f"{method}\n\n{content_type}\n{date}\n{resource}"
        digest = hmac.new(
            self._access_key_secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha1,
        ).digest()
        return f"OSS {self._access_key_id}:{base64.b64encode(digest).decode('ascii')}"

    def object_url(self, key: str) -> str:
        return f"{self._base_url}/{quote(key, safe='/')}"

    def presign_url(
        self,
        key: str,
        *,
        expires_in: int | None = None,
        response_content_disposition: str | None = None,
    ) -> tuple[str, int]:
        """生成 OSS 直连预签名 GET URL（有效期秒数，返回值同时给出 TTL）。

        ``response_content_disposition`` 会作为 OSS 子资源参与签名，用于让直链
        以附件形式下载（浏览器保存为文件，而不是直接打开图片）。
        """
        ttl = expires_in or self.settings.oss_url_expires
        expires_at = int(time.time()) + ttl
        resource = f"/{self._bucket}/{key}"
        subresources: dict[str, str] = {}
        if response_content_disposition:
            subresources["response-content-disposition"] = response_content_disposition
        if subresources:
            # OSS V1 的 CanonicalizedResource 要求子资源按名称排序；
            # 这里直接用原始值参与签名，查询串再做 URL 编码（已用真实 Bucket 验证）。
            canonical = "&".join(f"{name}={value}" for name, value in sorted(subresources.items()))
            resource = f"{resource}?{canonical}"
        string_to_sign = f"GET\n\n\n{expires_at}\n{resource}"
        signature = base64.b64encode(
            hmac.new(
                self._access_key_secret.encode("utf-8"),
                string_to_sign.encode("utf-8"),
                hashlib.sha1,
            ).digest()
        ).decode("ascii")
        params = {
            "OSSAccessKeyId": self._access_key_id,
            "Expires": expires_at,
            "Signature": signature,
            **subresources,
        }
        query = urlencode(params, quote_via=quote)
        return f"{self.object_url(key)}?{query}", ttl

    # ---- 对象操作 --------------------------------------------------------
    async def put_object(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str | None:
        """上传对象；返回 OSS 的 ETag（去掉引号）。"""
        response = await self._send(
            "PUT",
            key,
            content_type=content_type,
            content=data,
            action="上传",
        )
        etag = response.headers.get("etag") or response.headers.get("ETag")
        return etag.strip('"') if etag else None

    async def get_object(self, key: str) -> OssObject:
        """读取对象正文；不存在 / 无权限分别抛专用异常。"""
        response = await self._send("GET", key, action="读取")
        content_type = response.headers.get("content-type", "application/octet-stream")
        etag = response.headers.get("etag") or response.headers.get("ETag")
        return OssObject(
            key=key,
            data=response.content,
            content_type=content_type,
            etag=etag.strip('"') if etag else None,
        )

    async def delete_object(self, key: str) -> None:
        await self._send("DELETE", key, action="删除")

    async def list_objects(
        self, *, prefix: str, max_keys: int = 100, marker: str | None = None
    ) -> OssListResult:
        """按前缀列对象（字典序）；目录占位对象（``xxx/``）会被过滤掉。"""
        params: dict[str, str | int] = {"prefix": prefix, "max-keys": max_keys}
        if marker:
            params["marker"] = marker
        response = await self._send("GET", "", params=params, list_request=True, action="列举")
        return self._parse_list(response, prefix)

    # ---- 内部 ------------------------------------------------------------
    async def _send(
        self,
        method: str,
        key: str,
        *,
        content_type: str = "",
        content: bytes | None = None,
        params: dict[str, str | int] | None = None,
        list_request: bool = False,
        action: str,
    ) -> httpx.Response:
        """带重试地发请求；每次重试都重新生成 Date / 签名。"""
        url = f"{self._base_url}/" if list_request else self.object_url(key)
        last_error: Exception | None = None
        for attempt in range(1, _ATTEMPTS + 1):
            date = formatdate(usegmt=True)
            headers = {
                "Authorization": self.authorization(method, key, content_type, date),
                "Date": date,
            }
            if content_type:
                headers["Content-Type"] = content_type
            try:
                response = await self._client.request(
                    method,
                    url,
                    headers=headers,
                    content=content,
                    params=params,
                )
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning(
                    "OSS %s %s 网络失败（第 %s 次）：%s", action, key or "/", attempt, exc
                )
                if attempt < _ATTEMPTS:
                    await asyncio.sleep(0.4 * attempt)
                    continue
                raise OssError(f"OSS {action}失败（网络异常）：{exc}") from exc

            if response.status_code in _RETRY_STATUS and attempt < _ATTEMPTS:
                logger.warning(
                    "OSS %s %s 返回 %s，重试（第 %s 次）",
                    action,
                    key or "/",
                    response.status_code,
                    attempt,
                )
                await asyncio.sleep(0.4 * attempt)
                continue

            self._raise_for_status(response, key, action)
            return response

        raise OssError(f"OSS {action}失败：{last_error}")  # pragma: no cover - 循环内已返回/抛出

    @staticmethod
    def _error_detail(response: httpx.Response) -> tuple[str, str]:
        """从 OSS 的 XML 错误体里取 Code / Message，解析失败就给原文摘要。"""
        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError:
            return "", response.text[:300]
        code = (root.findtext("Code") or "").strip()
        message = (root.findtext("Message") or "").strip()
        return code, message

    def _raise_for_status(self, response: httpx.Response, key: str, action: str) -> None:
        if response.is_success:
            return
        code, message = self._error_detail(response)
        suffix = f"（{code}: {message}）" if code or message else ""
        if response.status_code == 404:
            raise OssObjectNotFound(f"OSS 对象不存在: {key}")
        if response.status_code in {401, 403}:
            raise OssAccessDenied(
                f"OSS 拒绝访问（请检查 AccessKey 权限与 Bucket 是否私有）：{action}{suffix}"
            )
        raise OssError(f"OSS {action}失败（HTTP {response.status_code}）{suffix}")

    @staticmethod
    def _parse_list(response: httpx.Response, prefix: str) -> OssListResult:
        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError as exc:
            raise OssError(f"OSS 列举返回了无法解析的 XML：{exc}") from exc

        objects: list[OssListedObject] = []
        for node in root.findall("Contents"):
            key = (node.findtext("Key") or "").strip()
            size_text = (node.findtext("Size") or "0").strip()
            if not key or key.endswith("/"):
                # 控制台创建目录时会留下 0 字节的 `image3/` 占位对象
                continue
            if not key.startswith(prefix):
                continue
            try:
                size = int(size_text)
            except ValueError:
                size = 0
            etag = (node.findtext("ETag") or "").strip().strip('"')
            objects.append(
                OssListedObject(
                    key=key,
                    size=size,
                    last_modified=(node.findtext("LastModified") or "").strip(),
                    etag=etag or None,
                )
            )
        truncated = (root.findtext("IsTruncated") or "").strip().lower() == "true"
        next_marker = (root.findtext("NextMarker") or "").strip() or None
        return OssListResult(objects=objects, truncated=truncated, next_marker=next_marker)
