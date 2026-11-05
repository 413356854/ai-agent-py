"""Bailian web search: request shape, source parsing, errors, tool wrapper."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ai_agent_py.config import Settings
from ai_agent_py.errors import ToolError
from ai_agent_py.tools.web_search import (
    BailianWebSearch,
    build_web_search_tool,
    format_result,
)

SEARCH_PAYLOAD: dict[str, Any] = {
    "request_id": "req-1",
    "output": {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "根据公开资料，答案是 42 [ref_1]。",
                }
            }
        ],
        "search_info": {
            "search_results": [
                {"index": 2, "title": "来源 B", "url": "https://b.example", "snippet": "第二段"},
                {"index": 1, "title": "来源 A", "url": "https://a.example", "snippet": "第一段"},
            ]
        },
    },
}


def _transport(
    payload: dict[str, Any],
    *,
    status: int = 200,
    captured: dict[str, Any] | None = None,
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers.get("authorization")
            captured["body"] = json.loads(request.content)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handle)


def _searcher(transport: httpx.BaseTransport, **kwargs: Any) -> BailianWebSearch:
    options: dict[str, Any] = {
        "api_key": "sk-test",
        "model": "qwen-plus",
        "transport": transport,
    }
    options.update(kwargs)
    return BailianWebSearch(**options)


def test_request_enables_search_sources_and_parses_results() -> None:
    captured: dict[str, Any] = {}
    result = _searcher(_transport(SEARCH_PAYLOAD, captured=captured)).search("今天的热点")

    body = captured["body"]
    assert captured["url"].endswith("/services/aigc/text-generation/generation")
    assert captured["authorization"] == "Bearer sk-test"
    assert body["model"] == "qwen-plus"
    assert body["parameters"]["enable_search"] is True
    assert body["parameters"]["result_format"] == "message"
    options = body["parameters"]["search_options"]
    assert options["forced_search"] is True
    assert options["enable_source"] is True
    assert options["enable_citation"] is True
    assert "search_strategy" not in options  # unset -> server default

    assert result.answer == "根据公开资料，答案是 42 [ref_1]。"
    assert result.request_id == "req-1"
    assert [source.title for source in result.sources] == ["来源 A", "来源 B"]
    assert [source.index for source in result.sources] == [1, 2]


def test_strategy_is_forwarded_when_configured() -> None:
    captured: dict[str, Any] = {}
    _searcher(_transport(SEARCH_PAYLOAD, captured=captured), strategy="max").search("查询")

    assert captured["body"]["parameters"]["search_options"]["search_strategy"] == "max"


def test_api_error_becomes_tool_error() -> None:
    payload = {"code": "InvalidApiKey", "message": "Invalid API-key provided."}
    searcher = _searcher(_transport(payload, status=401))
    with pytest.raises(ToolError, match="InvalidApiKey"):
        searcher.search("查询")


def test_non_json_response_becomes_tool_error() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(502, text="<html>bad gateway</html>")
    )
    with pytest.raises(ToolError, match="非 JSON"):
        _searcher(transport).search("查询")


def test_missing_key_and_empty_query_are_reported() -> None:
    with pytest.raises(ToolError, match="未配置"):
        BailianWebSearch(api_key=None).search("查询")
    with pytest.raises(ToolError, match="不能为空"):
        _searcher(_transport(SEARCH_PAYLOAD)).search("   ")


def test_format_result_marks_sources_and_missing_sources() -> None:
    result = _searcher(_transport(SEARCH_PAYLOAD)).search("查询")
    rendered = format_result(result)
    assert "查询：查询" in rendered
    assert "[ref_1] 来源 A" in rendered
    assert "https://a.example" in rendered
    assert "第一段" in rendered

    empty = _searcher(
        _transport({"output": {"choices": [{"message": {"content": "无来源回答"}}]}})
    ).search("查询")
    assert "未返回搜索来源" in format_result(empty)


def test_tool_wrapper_returns_text_and_config_hint() -> None:
    transport = _transport(SEARCH_PAYLOAD)
    tool = build_web_search_tool(
        settings=Settings(_env_file=None),
        searcher=_searcher(transport),
    )
    assert tool.name == "web_search"
    output = str(tool.invoke({"query": "测试"}))
    assert "[ref_1] 来源 A" in output

    missing = build_web_search_tool(
        settings=Settings(_env_file=None),
        searcher=BailianWebSearch(api_key=None),
    )
    assert str(missing.invoke({"query": "测试"})).startswith("错误：")
