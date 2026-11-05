"""Bailian web extractor: Responses request shape, output parsing, errors."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ai_agent_py.config import Settings
from ai_agent_py.errors import ToolError
from ai_agent_py.tools.web_extractor import (
    BailianWebExtractor,
    build_web_extractor_tool,
    format_result,
)

EXTRACT_PAYLOAD: dict[str, Any] = {
    "id": "resp-1",
    "request_id": "req-1",
    "output": [
        {"type": "reasoning", "summary": []},
        {"type": "web_search_call", "status": "completed"},
        {
            "type": "web_extractor_call",
            "status": "completed",
            "goal": "读取 https://example.com/page",
            "output": "抓取到的原始片段",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "# 标题\n\n正文内容"}],
        },
    ],
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


def _extractor(transport: httpx.BaseTransport, **kwargs: Any) -> BailianWebExtractor:
    options: dict[str, Any] = {
        "api_key": "sk-test",
        "base_url": "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "transport": transport,
    }
    options.update(kwargs)
    return BailianWebExtractor(**options)


def test_request_uses_responses_tools_and_parses_message() -> None:
    captured: dict[str, Any] = {}
    result = _extractor(_transport(EXTRACT_PAYLOAD, captured=captured)).extract(
        "https://example.com/page"
    )

    body = captured["body"]
    assert captured["url"].endswith("/compatible-mode/v1/responses")
    assert captured["authorization"] == "Bearer sk-test"
    assert body["model"] == "qwen-plus"
    assert body["enable_thinking"] is True
    assert body["tools"] == [{"type": "web_search"}, {"type": "web_extractor"}]
    assert "https://example.com/page" in body["input"]

    assert result.content == "# 标题\n\n正文内容"
    assert result.extractor_calls == 1
    assert result.search_calls == 1
    assert result.request_id == "req-1"
    assert not result.truncated


def test_extractor_output_is_fallback_when_message_missing() -> None:
    payload = {
        "output": [
            {
                "type": "web_extractor_call",
                "goal": "读取",
                "output": "只有抓取结果的正文",
            }
        ]
    }
    result = _extractor(_transport(payload)).extract("https://example.com")
    assert result.content == "只有抓取结果的正文"


def test_missing_extractor_call_is_flagged() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "普通回答"}],
            }
        ]
    }
    result = _extractor(_transport(payload)).extract("https://example.com")
    rendered = format_result(result)
    assert "普通回答" in rendered
    assert "没有 web_extractor 调用记录" in rendered


def test_content_is_truncated_to_max_chars() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "x" * 100}],
            }
        ]
    }
    result = _extractor(_transport(payload), max_chars=10).extract("https://example.com")
    assert result.truncated
    assert result.content == "x" * 10
    assert "已截断" in format_result(result)


def test_api_error_becomes_tool_error_with_hint() -> None:
    payload = {
        "error": {
            "code": "InvalidParameter",
            "message": "The web_extractor tool must be executed with web_search tool.",
            "type": "invalid_request_error",
        }
    }
    with pytest.raises(ToolError, match="InvalidParameter"):
        _extractor(_transport(payload, status=400)).extract("https://example.com")

    with pytest.raises(ToolError, match="WEB_EXTRACTOR_API_KEY"):
        _extractor(_transport(payload, status=401)).extract("https://example.com")


def test_invalid_url_and_empty_input_are_rejected() -> None:
    transport = _transport(EXTRACT_PAYLOAD)
    with pytest.raises(ToolError, match="http/https"):
        _extractor(transport).extract("ftp://example.com/file")
    with pytest.raises(ToolError, match="http/https"):
        _extractor(transport).extract("example.com/page")
    with pytest.raises(ToolError, match="不能为空"):
        _extractor(transport).extract("   ")


def test_missing_key_is_reported() -> None:
    with pytest.raises(ToolError, match="未配置"):
        BailianWebExtractor(api_key=None).extract("https://example.com")


def test_tool_wrapper_returns_text_and_config_hint() -> None:
    tool = build_web_extractor_tool(
        settings=Settings(_env_file=None),
        extractor=_extractor(_transport(EXTRACT_PAYLOAD)),
    )
    assert tool.name == "web_extract"
    output = str(tool.invoke({"url": "https://example.com/page"}))
    assert "正文内容" in output
    assert "https://example.com/page" in output

    missing = build_web_extractor_tool(
        settings=Settings(_env_file=None),
        extractor=BailianWebExtractor(api_key=None),
    )
    assert str(missing.invoke({"url": "https://example.com"})).startswith("错误：")
