"""MinerU 精准解析 API 客户端（异步 httpx）。

接口文档：https://mineru.net/apiManage/docs

本地文件走「批量文件上传解析」这条链路（官方推荐，也是唯一支持上传的入口）：

1. ``POST /file-urls/batch``  申请上传链接，拿到 ``batch_id`` + ``file_urls``
2. ``PUT  <file_url>``         直接上传文件字节（**不要**设置 Content-Type），
                               上传完成后 MinerU 自动提交解析任务
3. ``GET  /extract-results/batch/{batch_id}``  轮询直到 ``state=done``
4. ``GET  <full_zip_url>``     下载结果 zip（full.md + images + content_list.json …）

单个 URL 解析则走 ``POST /extract/task`` + ``GET /extract/task/{task_id}``，
本客户端同样提供封装，方便以后接入「传链接」的场景。

响应信封有两套，客户端都做了兼容：

* 业务接口：``{"code": 0, "msg": "ok", "data": {...}}``
* 网关/鉴权：``{"success": false, "msgCode": "A0202", "msg": "user authenticate failed"}``
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import time
from collections.abc import Callable
from typing import Any

import httpx

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import MineruAuthError, MineruError, MineruTaskFailed
from ai_agent_py.mineru.models import BatchItem, BatchUpload, ParseOptions

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 60.0
UPLOAD_TIMEOUT = 600.0
DOWNLOAD_TIMEOUT = 600.0

#: 网络抖动（含代理导致的 TLS 错误）的重试次数与退避基准秒数
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0

#: 可以安全重试的异常：连接/读取失败、超时、代理或 TLS 层错误
#: （``ssl.SSLError`` 有时不会被 httpx 包装，例如 ``SSLV3_ALERT_BAD_RECORD_MAC``）
_RETRYABLE_EXCEPTIONS = (httpx.TransportError, ssl.SSLError)

# 官方文档里的错误码 → 中文提示
_ERROR_HINTS: dict[str, str] = {
    "A0202": "Token 无效，请检查 MINERU_API_KEY",
    "A0211": "Token 已过期，请重新申请",
    "-500": "请求参数错误",
    "-10001": "MinerU 服务内部错误，请稍后重试",
    "-10002": "请求参数不合法",
    "-60001": "未找到解析结果",
    "-60002": "不支持的文件格式（支持 pdf / 图片 / doc(x) / ppt(x) / html）",
    "-60003": "文件读取失败",
    "-60004": "文件内容为空",
    "-60005": "文件超过 200MB 上限",
    "-60006": "文件页数超过 600 页上限",
    "-60008": "文件读取超时（源地址不可访问）",
    "-60010": "文档解析失败",
    "-60012": "任务不存在或已过期",
    "-60015": "文件下载失败",
    "-60016": "文件格式转换失败",
    "-60018": "当日解析额度已用尽（2000 页/天）",
    "-60022": "网页内容读取失败（限频）",
}


class MineruClient:
    """MinerU 精准解析 API 的薄封装。

    ``client`` 参数用于注入 ``httpx.AsyncClient``（测试里传 ``MockTransport`` 即可完全离线）。
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://mineru.net/api/v4",
        *,
        client: httpx.AsyncClient | None = None,
        retries: int = RETRY_ATTEMPTS,
        retry_delay: float = RETRY_BASE_DELAY,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.retries = max(1, retries)
        self.retry_delay = max(0.0, retry_delay)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=15.0),
            follow_redirects=True,
        )

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, *, client: httpx.AsyncClient | None = None
    ) -> MineruClient:
        cfg = settings or get_settings()
        return cls(cfg.resolved_mineru_api_key, cfg.mineru_base_url, client=client)

    # ---- 生命周期 ------------------------------------------------------
    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> MineruClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @property
    def headers(self) -> dict[str, str]:
        if not self.api_key:
            raise MineruAuthError(
                "未配置 MinerU Token：请在 .env 中设置 MINERU_API_KEY"
                "（申请地址 https://mineru.net/apiManage/token）"
            )
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    # ---- 重试 ----------------------------------------------------------
    async def _sleep_before_retry(self, attempt: int, what: str, exc: Exception) -> None:
        """指数退避（1s、2s…），网络抖动时避免直接把任务判死。"""
        delay = self.retry_delay * attempt
        logger.warning(
            "%s失败（第 %d/%d 次尝试），%.1fs 后重试: %s",
            what,
            attempt,
            self.retries,
            delay,
            exc,
        )
        await asyncio.sleep(delay)

    # ---- 批量上传解析（本地文件）----------------------------------------
    async def create_batch_upload(
        self,
        file_name: str,
        data_id: str,
        options: ParseOptions | None = None,
    ) -> BatchUpload:
        """申请批量上传链接，返回 ``batch_id`` 与对应的 ``file_urls``。"""
        payload = (options or ParseOptions()).batch_payload(file_name, data_id)
        data = await self._request("POST", "/file-urls/batch", json=payload)
        batch_id = str(data.get("batch_id") or "").strip()
        urls = data.get("file_urls") or []
        if not batch_id or not urls:
            raise MineruError(f"申请上传链接返回异常: {data}")
        return BatchUpload(batch_id=batch_id, file_urls=[str(url) for url in urls])

    async def upload(self, file_url: str, content: bytes) -> None:
        """把文件字节 PUT 到预签名地址（不要带 Authorization / Content-Type）。"""
        what = "上传文件到 MinerU"
        for attempt in range(1, self.retries + 1):
            try:
                response = await self._client.put(
                    file_url,
                    content=content,
                    timeout=httpx.Timeout(UPLOAD_TIMEOUT, connect=15.0),
                )
            except _RETRYABLE_EXCEPTIONS as exc:
                if attempt >= self.retries:
                    raise MineruError(f"{what}失败: {exc}") from exc
                await self._sleep_before_retry(attempt, what, exc)
                continue
            if response.status_code >= 500 and attempt < self.retries:
                await self._sleep_before_retry(
                    attempt, what, MineruError(f"HTTP {response.status_code} {response.text[:120]}")
                )
                continue
            if response.status_code >= 400:
                raise MineruError(f"{what}失败: HTTP {response.status_code} {response.text[:200]}")
            return
        raise MineruError(f"{what}失败")  # pragma: no cover - 循环内已返回或抛出

    async def fetch_batch(self, batch_id: str) -> list[BatchItem]:
        """查询某批次下所有文件的解析状态（网络抖动时自动重试）。"""
        data = await self._request("GET", f"/extract-results/batch/{batch_id}", retry=True)
        raw_items = data.get("extract_result") or []
        return [BatchItem.from_api(item) for item in raw_items]

    async def wait_batch(
        self,
        batch_id: str,
        *,
        data_id: str | None = None,
        timeout: float = 600.0,
        poll_interval: float = 3.0,
        on_progress: Callable[[BatchItem], None] | None = None,
    ) -> BatchItem:
        """轮询批次直到任务完成；失败/超时抛出 ``MineruError``。"""
        deadline = time.monotonic() + timeout
        last_note = 0.0
        while True:
            items = await self.fetch_batch(batch_id)
            item = self._pick(items, data_id)
            if item is not None:
                if on_progress is not None:
                    on_progress(item)
                if item.state == "done":
                    if not item.full_zip_url:
                        raise MineruError("解析完成但未返回 full_zip_url")
                    return item
                if item.state == "failed":
                    raise MineruTaskFailed(f"MinerU 解析失败: {item.err_msg or '未知原因'}")
            if time.monotonic() >= deadline:
                raise MineruError(f"等待解析结果超时（{timeout:.0f}s），请稍后重试")
            # 排队时降低日志噪音：每 5 次轮询打印一次
            if time.monotonic() - last_note > poll_interval * 5:
                state = item.state if item else "waiting-file"
                logger.info("MinerU 任务 %s 状态: %s", batch_id, state)
                last_note = time.monotonic()
            await asyncio.sleep(max(poll_interval, 0.0))

    async def download(self, url: str) -> bytes:
        """下载解析结果 zip（网络抖动时自动重试）。"""
        what = "下载解析结果"
        for attempt in range(1, self.retries + 1):
            try:
                response = await self._client.get(
                    url, timeout=httpx.Timeout(DOWNLOAD_TIMEOUT, connect=15.0)
                )
            except _RETRYABLE_EXCEPTIONS as exc:
                if attempt >= self.retries:
                    raise MineruError(f"{what}失败: {exc}") from exc
                await self._sleep_before_retry(attempt, what, exc)
                continue
            if response.status_code >= 500 and attempt < self.retries:
                await self._sleep_before_retry(
                    attempt, what, MineruError(f"HTTP {response.status_code}")
                )
                continue
            if response.status_code >= 400:
                raise MineruError(f"{what}失败: HTTP {response.status_code}")
            return response.content
        raise MineruError(f"{what}失败")  # pragma: no cover - 循环内已返回或抛出

    # ---- 单 URL 解析（备用）---------------------------------------------
    async def create_url_task(self, url: str, options: ParseOptions | None = None) -> str:
        """提交一个 URL 解析任务，返回 ``task_id``。"""
        opts = options or ParseOptions()
        payload: dict[str, Any] = {
            "url": url,
            "model_version": opts.model_version,
            "language": opts.language,
            "enable_formula": opts.enable_formula,
            "enable_table": opts.enable_table,
            "is_ocr": opts.is_ocr,
        }
        if opts.page_ranges:
            payload["page_ranges"] = opts.page_ranges
        data = await self._request("POST", "/extract/task", json=payload)
        task_id = str(data.get("task_id") or "").strip()
        if not task_id:
            raise MineruError(f"创建解析任务返回异常: {data}")
        return task_id

    async def fetch_task(self, task_id: str) -> dict[str, Any]:
        """查询单个 URL 任务的状态（原样返回 ``data`` 字典）。"""
        return await self._request("GET", f"/extract/task/{task_id}", retry=True)

    # ---- 内部工具 ------------------------------------------------------
    def _pick(self, items: list[BatchItem], data_id: str | None) -> BatchItem | None:
        if not items:
            return None
        if data_id:
            for item in items:
                if item.data_id == data_id:
                    return item
        return items[0]

    async def _request(
        self, method: str, path: str, *, retry: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """发请求并把两套响应信封都归一成 ``data`` 字典。

        ``retry=True``（只用于幂等的 GET）时，网络抖动与 5xx 会自动重试；
        创建任务的 POST 不重试，避免重复建任务。
        """
        headers = dict(self.headers)
        headers.update(kwargs.pop("headers", {}) or {})
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        what = f"请求 MinerU（{method} {path}）"

        for attempt in range(1, self.retries + 1):
            try:
                response = await self._client.request(method, url, headers=headers, **kwargs)
            except _RETRYABLE_EXCEPTIONS as exc:
                if not retry or attempt >= self.retries:
                    raise MineruError(f"{what}失败: {exc}") from exc
                await self._sleep_before_retry(attempt, what, exc)
                continue
            if retry and response.status_code >= 500 and attempt < self.retries:
                await self._sleep_before_retry(
                    attempt, what, MineruError(f"HTTP {response.status_code}")
                )
                continue
            return self._unwrap(response, method=method, path=path)
        raise MineruError(f"{what}失败")  # pragma: no cover - 循环内已返回或抛出

    @staticmethod
    def _unwrap(response: httpx.Response, *, method: str, path: str) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError:
            snippet = response.text[:200]
            if response.status_code in (401, 403):
                raise MineruAuthError("MinerU 鉴权失败：请检查 MINERU_API_KEY 是否正确") from None
            raise MineruError(
                f"MinerU 返回非 JSON 响应（HTTP {response.status_code}）: {snippet}"
            ) from None

        if not isinstance(body, dict):
            raise MineruError(f"MinerU 返回了意外的响应: {body!r}")

        # 网关层（鉴权失败）用 success/msgCode
        if body.get("success") is False:
            code = str(body.get("msgCode") or "")
            message = str(body.get("msg") or "MinerU 鉴权失败")
            hint = _ERROR_HINTS.get(code)
            text = f"{message}（{code}）" if code else message
            if hint:
                text = f"{text}：{hint}"
            if code in {"A0202", "A0211"}:
                raise MineruAuthError(text)
            raise MineruError(f"{method} {path} 失败: {text}")

        if response.status_code in (401, 403):
            raise MineruAuthError(
                f"MinerU 鉴权失败（HTTP {response.status_code}）：请检查 MINERU_API_KEY 是否正确"
            )

        code_value: Any = body.get("code")
        if code_value not in (0, "0", None):
            key = str(code_value)
            hint = _ERROR_HINTS.get(key)
            message = str(body.get("msg") or "MinerU 请求失败")
            text = f"{message}（{key}）" if key else message
            if hint:
                text = f"{text}：{hint}"
            if key in {"A0202", "A0211"}:
                raise MineruAuthError(text)
            raise MineruError(f"{method} {path} 失败: {text}")

        data = body.get("data")
        return data if isinstance(data, dict) else {}
