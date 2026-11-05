"""Bailian (Model Studio) web extractor tool.

The web extractor is a server-side tool: you cannot POST a URL to a dedicated
"extract" endpoint, you ask a supporting model to read a page and it calls
``web_extractor`` internally. The OpenAI compatible Chat Completions path only
works with streaming, so this tool uses the Responses API, which supports
non-streaming calls and returns the ``web_extractor_call`` items plus the final
text in ``output``.

Docs: https://help.aliyun.com/zh/model-studio/web-extractor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import ToolError

WEB_EXTRACT_TOOL_NAME = "web_extract"
WEB_EXTRACT_TOOL_DESCRIPTION = (
    "抓取指定公开网页的正文（阿里云百炼 web_extractor）。web_search 返回来源后，"
    "用本工具阅读页面详情；返回页面主要内容，过长会自动截断。"
    "url 必须是完整的 http/https 地址；只抓取单页，不跟随站内链接。"
)


class WebExtractArgs(BaseModel):
    """``web_extract`` 的入参。"""

    url: str = Field(min_length=1, description="完整的 http/https 页面地址")


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_CHARS = 20_000
EXTRACTION_PROMPT = (
    "请使用网页抓取工具读取下面这个网页的正文，并尽量原样提取主要内容："
    "保留标题、段落、列表和代码块的结构，不要总结、不要评论、不要补充网页中没有的信息。"
    "如果页面无法访问或没有正文，请直接说明原因。\n\nURL：{url}"
)


@dataclass(frozen=True)
class WebExtractResult:
    """Extracted page content plus a little provenance for debugging."""

    url: str
    content: str
    extractor_calls: int = 0
    search_calls: int = 0
    truncated: bool = False
    request_id: str | None = None


class BailianWebExtractor:
    """Minimal Responses API client for Bailian's web extractor."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "qwen-plus",
        timeout: float = DEFAULT_TIMEOUT,
        max_chars: int = DEFAULT_MAX_CHARS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_chars = max_chars
        self._transport = transport

    @property
    def endpoint(self) -> str:
        """Responses URL; a base URL already ending with it is kept."""
        if self.base_url.endswith("/responses"):
            return self.base_url
        return f"{self.base_url}/responses"

    def extract(self, url: str) -> WebExtractResult:
        """Fetch one page through the model and return its main content."""
        target = _validate_url(url)
        if not self.api_key:
            raise ToolError(
                "未配置网页抓取 Key：请设置 WEB_EXTRACTOR_API_KEY 或 DASHSCOPE_API_KEY"
                "（百炼端点下也会复用 LLM_API_KEY）"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "input": EXTRACTION_PROMPT.format(url=target),
            # web_extractor must be requested together with web_search.
            "tools": [{"type": "web_search"}, {"type": "web_extractor"}],
            "enable_thinking": True,
        }
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                response = client.post(self.endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ToolError(f"请求百炼网页抓取失败：{exc}") from exc

        data = _json_payload(response)
        content, extractor_calls, search_calls = _parse_output(data)
        if not content:
            raise ToolError("抓取请求成功，但响应里没有正文")

        truncated = len(content) > self.max_chars
        if truncated:
            content = content[: self.max_chars].rstrip()

        return WebExtractResult(
            url=target,
            content=content,
            extractor_calls=extractor_calls,
            search_calls=search_calls,
            truncated=truncated,
            request_id=str(data.get("request_id") or data.get("id") or "") or None,
        )


def _validate_url(url: str) -> str:
    target = url.strip()
    if not target:
        raise ToolError("URL 不能为空")
    parts = urlsplit(target)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ToolError(f"URL 必须是完整的 http/https 地址：{target}")
    return target


def _json_payload(response: httpx.Response) -> dict[str, Any]:
    """Parse a Responses payload, turning API/HTTP errors into ToolError."""
    try:
        data = response.json()
    except ValueError as exc:
        raise ToolError(
            f"百炼返回了非 JSON 响应（HTTP {response.status_code}）：{response.text[:200]}"
        ) from exc
    if not isinstance(data, dict):
        raise ToolError(f"百炼返回了意外的响应结构（HTTP {response.status_code}）")

    error = data.get("error")
    if isinstance(error, dict):
        code = error.get("code") or error.get("type") or "error"
        message = error.get("message") or "未知错误"
        raise ToolError(
            f"百炼网页抓取失败（HTTP {response.status_code}，{code}）：{message}"
            + _auth_hint(response.status_code)
        )

    code = str(data.get("code") or "").strip()
    if response.status_code >= 400 or (code and code.lower() not in {"success", "ok"}):
        message = data.get("message") or "未知错误"
        raise ToolError(
            f"百炼网页抓取失败（HTTP {response.status_code}，{code or 'error'}）：{message}"
            + _auth_hint(response.status_code)
        )
    return data


def _auth_hint(status_code: int) -> str:
    if status_code in {401, 403}:
        return "；请检查 WEB_EXTRACTOR_API_KEY / WEB_EXTRACTOR_BASE_URL（默认跟随 LLM_*，需百炼兼容端点）"
    return ""


def _parse_output(data: dict[str, Any]) -> tuple[str, int, int]:
    """Return ``(content, extractor_calls, search_calls)`` from ``output``."""
    items = data.get("output")
    if not isinstance(items, list):
        return "", 0, 0

    message_parts: list[str] = []
    extractor_parts: list[str] = []
    extractor_calls = 0
    search_calls = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "web_extractor_call":
            extractor_calls += 1
            text = _text_content(item.get("output")).strip()
            if text:
                extractor_parts.append(text)
        elif kind == "web_search_call":
            search_calls += 1
        elif kind == "message":
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        message_parts.append(block["text"])

    # The model's message is the extraction result in the documented flow; the
    # raw web_extractor_call.output is only a fallback when no text came back.
    content = "\n".join(message_parts).strip() or "\n\n".join(extractor_parts).strip()
    return content, extractor_calls, search_calls


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "data"):
            nested = value.get(key)
            if isinstance(nested, str):
                return nested
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def format_result(result: WebExtractResult) -> str:
    """Render extracted content for the model."""
    lines = [f"URL：{result.url}", "", result.content]
    if result.extractor_calls == 0:
        lines.extend(
            [
                "",
                "（注意：本次响应没有 web_extractor 调用记录，内容可能不是页面正文）",
            ]
        )
    if result.truncated:
        lines.extend(["", "（内容过长，已截断）"])
    return "\n".join(lines)


def build_web_extractor_tool(
    *,
    settings: Settings | None = None,
    extractor: BailianWebExtractor | None = None,
) -> BaseTool:
    """Build the ``web_extract`` tool.

    A missing API key does not prevent building the tool: the call returns a
    configuration hint instead, so graphs can be assembled offline.
    """
    cfg = settings or get_settings()
    engine = extractor or BailianWebExtractor(
        api_key=cfg.resolved_web_extractor_api_key,
        base_url=cfg.resolved_web_extractor_base_url,
        model=cfg.resolved_web_extractor_model,
        timeout=cfg.web_extractor_timeout,
        max_chars=cfg.web_extractor_max_chars,
    )

    @tool(
        WEB_EXTRACT_TOOL_NAME,
        description=WEB_EXTRACT_TOOL_DESCRIPTION,
        args_schema=WebExtractArgs,
    )
    def web_extract(url: str) -> str:
        """用百炼 web_extractor 抓取指定网页的正文。"""
        try:
            return format_result(engine.extract(url))
        except ToolError as exc:
            return f"错误：{exc}"

    return web_extract
