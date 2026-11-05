"""Bailian (DashScope) web search tool.

Wraps the native DashScope text-generation endpoint with ``enable_search``:
the model searches the public web, answers from the results, and (because we
always set ``enable_source``) also returns the source list. The OpenAI
compatible endpoint can search too, but cannot return sources, so the tool
sticks to the native protocol.

Docs: https://help.aliyun.com/zh/model-studio/web-search
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.tools import BaseTool, tool

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import ToolError

WEB_SEARCH_TOOL_NAME = "web_search"
WEB_SEARCH_TOOL_DESCRIPTION = (
    "通过阿里云百炼联网搜索查询实时公开信息（新闻、政策、价格、人物/公司动态等）。"
    "返回基于网页结果的摘要和来源链接，摘要里的 [ref_n] 角标与来源列表一一对应。"
    "知识库没有、且需要最新信息时使用；query 用一句自然语言说明要查什么，一次只查一件事。"
)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_CITATION_FORMAT = "[ref_<number>]"
SYSTEM_PROMPT = (
    "你是一个联网搜索助手。根据网页搜索结果，用简体中文简洁、客观地回答用户问题；"
    "保留结果中的 [ref_n] 角标，只依据搜索结果作答，不要编造来源。"
)
MAX_SNIPPET_CHARS = 200


@dataclass(frozen=True)
class SearchSource:
    """One web page returned by Bailian's search."""

    title: str
    url: str
    index: int | None = None
    snippet: str = ""


@dataclass(frozen=True)
class WebSearchResult:
    """Search answer plus the pages it was built from."""

    query: str
    answer: str
    sources: tuple[SearchSource, ...] = ()
    request_id: str | None = None


class BailianWebSearch:
    """Minimal client for Bailian's native web search generation API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        strategy: str | None = None,
        forced_search: bool = True,
        enable_citation: bool = True,
        citation_format: str = DEFAULT_CITATION_FORMAT,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.strategy = strategy
        self.forced_search = forced_search
        self.enable_citation = enable_citation
        self.citation_format = citation_format
        self.timeout = timeout
        self._transport = transport

    @property
    def endpoint(self) -> str:
        """Full generation URL; a base URL already ending with it is kept."""
        if self.base_url.endswith("/text-generation/generation"):
            return self.base_url
        return f"{self.base_url}/services/aigc/text-generation/generation"

    def search(self, query: str) -> WebSearchResult:
        """Search the web for ``query`` and return the answer plus sources."""
        text = query.strip()
        if not text:
            raise ToolError("搜索内容不能为空")
        if not self.api_key:
            raise ToolError("未配置联网搜索 Key：请设置 WEB_SEARCH_API_KEY 或 DASHSCOPE_API_KEY")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                response = client.post(self.endpoint, headers=headers, json=self._payload(text))
        except httpx.HTTPError as exc:
            raise ToolError(f"请求百炼联网搜索失败：{exc}") from exc

        data = _json_payload(response)
        raw_output = data.get("output")
        output: dict[str, Any] = raw_output if isinstance(raw_output, dict) else {}
        raw_choices = output.get("choices")
        choices: list[Any] = raw_choices if isinstance(raw_choices, list) else []
        first: dict[str, Any] = choices[0] if choices and isinstance(choices[0], dict) else {}
        raw_message = first.get("message")
        message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
        answer = _text_content(message.get("content"))
        if not answer:
            answer = _text_content(output.get("text"))

        info = output.get("search_info")
        if not isinstance(info, dict):
            info = data.get("search_info")
        sources = _parse_sources(info if isinstance(info, dict) else {})

        return WebSearchResult(
            query=text,
            answer=answer.strip(),
            sources=sources,
            request_id=str(data.get("request_id") or "") or None,
        )

    def _payload(self, query: str) -> dict[str, Any]:
        options: dict[str, Any] = {
            "forced_search": self.forced_search,
            "enable_source": True,
        }
        if self.strategy:
            options["search_strategy"] = self.strategy
        if self.enable_citation:
            options["enable_citation"] = True
            options["citation_format"] = self.citation_format
        return {
            "model": self.model,
            "input": {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ]
            },
            "parameters": {
                "result_format": "message",
                "enable_search": True,
                "search_options": options,
            },
        }


def _json_payload(response: httpx.Response) -> dict[str, Any]:
    """Parse a DashScope response, turning API/HTTP errors into ToolError."""
    try:
        data = response.json()
    except ValueError as exc:
        raise ToolError(
            f"百炼返回了非 JSON 响应（HTTP {response.status_code}）：{response.text[:200]}"
        ) from exc
    if not isinstance(data, dict):
        raise ToolError(f"百炼返回了意外的响应结构（HTTP {response.status_code}）")

    code = str(data.get("code") or "").strip()
    failed = response.status_code >= 400 or (code and code.lower() not in {"success", "ok"})
    if failed:
        message = data.get("message") or data.get("error") or "未知错误"
        raise ToolError(
            f"百炼联网搜索失败（HTTP {response.status_code}，{code or 'error'}）：{message}"
        )
    return data


def _text_content(value: Any) -> str:
    """Flatten ``content`` (string or content-block list) to plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def _parse_sources(info: dict[str, Any]) -> tuple[SearchSource, ...]:
    raw = info.get("search_results")
    if not isinstance(raw, list):
        return ()
    sources: list[SearchSource] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if not url and not title:
            continue
        sources.append(
            SearchSource(
                title=title or url,
                url=url,
                index=_optional_int(item.get("index")),
                snippet=_first_text(item.get("snippet"), item.get("summary"), item.get("content"))[
                    :MAX_SNIPPET_CHARS
                ],
            )
        )
    sources.sort(key=lambda source: (source.index is None, source.index or 0))
    return tuple(sources)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def format_result(result: WebSearchResult) -> str:
    """Render a search result for the model (answer first, then sources)."""
    lines = [f"查询：{result.query}", "", result.answer or "（模型未返回摘要）"]
    if result.sources:
        lines.extend(["", "来源："])
        for position, source in enumerate(result.sources, start=1):
            number = source.index if source.index is not None else position
            lines.append(f"[ref_{number}] {source.title}")
            if source.url:
                lines.append(f"    {source.url}")
            if source.snippet:
                lines.append(f"    {source.snippet}")
    else:
        lines.extend(["", "（本次未返回搜索来源，回答可能未基于实时网页）"])
    return "\n".join(lines)


def build_web_search_tool(
    *,
    settings: Settings | None = None,
    searcher: BailianWebSearch | None = None,
) -> BaseTool:
    """Build the ``web_search`` tool.

    A missing API key does not prevent building the tool: the call returns a
    configuration hint instead, so graphs can be assembled offline.
    """
    cfg = settings or get_settings()
    engine = searcher or BailianWebSearch(
        api_key=cfg.resolved_web_search_api_key,
        base_url=cfg.web_search_base_url,
        model=cfg.web_search_model,
        strategy=cfg.web_search_strategy,
        forced_search=cfg.web_search_forced,
        enable_citation=cfg.web_search_enable_citation,
        timeout=cfg.web_search_timeout,
    )

    @tool(WEB_SEARCH_TOOL_NAME, description=WEB_SEARCH_TOOL_DESCRIPTION)
    def web_search(query: str) -> str:
        """通过百炼联网搜索查询实时网页信息。"""
        try:
            return format_result(engine.search(query))
        except ToolError as exc:
            return f"错误：{exc}"

    return web_search
