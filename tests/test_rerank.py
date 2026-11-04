"""精排（qwen3.7-text-rerank）：请求体、端点推导、结果映射与异常处理。"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from langchain_core.documents import Document

from ai_agent_py.config import Settings
from ai_agent_py.errors import RerankError
from ai_agent_py.rag.rerank import (
    DASHSCOPE_RERANK_ENDPOINT,
    DashScopeReranker,
    get_reranker,
    resolve_endpoint,
)

#: 百炼文档里的示例任务说明，仅 qwen3 系列 rerank 支持。
INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query."


def make_settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "_env_file": None,
        "rerank_api_key": "sk-test",
        "rerank_model": "qwen3.7-text-rerank",
        "rerank_candidates": 30,
        "retriever_k": 4,
    }
    payload.update(overrides)
    return Settings(**payload)


def make_documents(*texts: str) -> list[Document]:
    return [
        Document(
            page_content=text,
            metadata={"chunk_id": f"chunk-{index}", "score": 0.03, "retrieval": "hybrid"},
        )
        for index, text in enumerate(texts)
    ]


def reranker_with(
    handler: Callable[[httpx.Request], httpx.Response],
    **settings_kwargs: Any,
) -> tuple[DashScopeReranker, list[dict[str, Any]]]:
    """构造一个走 MockTransport 的精排器，并记录它真正发出的请求。"""
    captured: list[dict[str, Any]] = []

    def record(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "json": json.loads(request.content),
            }
        )
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(record))
    return DashScopeReranker(make_settings(**settings_kwargs), client=client), captured


def reply(results: list[dict[str, Any]], *, status: int = 200, **extra: Any):
    """成功的排序响应：按 relevance_score 降序的结果列表。"""

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = {
            "output": {"results": results},
            "usage": {"total_tokens": 42},
            "request_id": "req-1",
        }
        body.update(extra)
        return httpx.Response(status, json=body)

    return handler


def test_rerank_sends_expected_payload_and_keeps_order() -> None:
    reranker, captured = reranker_with(
        reply(
            [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.12},
            ]
        ),
        rerank_instruct=INSTRUCT,
    )

    documents = make_documents("无关内容：印花税税目", "业务招待费按发生额的 60% 扣除")
    ranked = reranker.rerank("业务招待费扣除标准", documents, top_n=2)

    assert [document.metadata["chunk_id"] for document in ranked] == ["chunk-1", "chunk-0"]
    sent = captured[0]
    assert sent["url"] == reranker.endpoint
    assert sent["headers"]["authorization"] == "Bearer sk-test"
    assert sent["json"]["model"] == "qwen3.7-text-rerank"
    assert sent["json"]["input"] == {
        "query": "业务招待费扣除标准",
        "documents": ["无关内容：印花税税目", "业务招待费按发生额的 60% 扣除"],
    }
    assert sent["json"]["parameters"] == {"top_n": 2, "instruct": INSTRUCT}


def test_rerank_backfills_scores_without_losing_rrf_score() -> None:
    reranker, _ = reranker_with(reply([{"index": 0, "relevance_score": 0.9334521}]))

    document = reranker.rerank("业务招待费", make_documents("正文"), top_n=1)[0]

    assert document.metadata["rerank_score"] == 0.933452
    assert document.metadata["score"] == 0.933452
    assert document.metadata["retrieval_score"] == 0.03
    assert document.metadata["reranked"] is True
    assert document.metadata["retrieval"] == "hybrid"


def test_rerank_omits_instruct_unless_configured() -> None:
    reranker, captured = reranker_with(reply([{"index": 0, "relevance_score": 0.5}]))

    reranker.rerank("问题", make_documents("正文"), top_n=1)

    assert captured[0]["json"]["parameters"] == {"top_n": 1}


def test_rerank_top_n_defaults_to_document_count() -> None:
    reranker, captured = reranker_with(reply([{"index": 0, "relevance_score": 0.5}]))

    reranker.rerank("问题", make_documents("a", "b", "c"))

    assert captured[0]["json"]["parameters"]["top_n"] == 3


def test_rerank_min_score_drops_weak_documents() -> None:
    reranker, _ = reranker_with(
        reply(
            [
                {"index": 0, "relevance_score": 0.82},
                {"index": 1, "relevance_score": 0.11},
            ]
        ),
        rerank_min_score=0.5,
    )

    ranked = reranker.rerank("业务招待费", make_documents("强相关", "弱相关"), top_n=2)

    assert [document.metadata["chunk_id"] for document in ranked] == ["chunk-0"]


@pytest.mark.parametrize(
    "base_url, expected",
    [
        (
            "https://ws-9lonm8753e95w4oc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            "https://ws-9lonm8753e95w4oc.cn-beijing.maas.aliyuncs.com"
            "/api/v1/services/rerank/text-rerank/text-rerank",
        ),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        ),
        # 非百炼域名不瞎推导：直接用公共网关，否则会拼出一个不存在的地址。
        ("https://api.openai.com/v1", DASHSCOPE_RERANK_ENDPOINT),
        ("", DASHSCOPE_RERANK_ENDPOINT),
    ],
)
def test_resolve_endpoint(base_url: str, expected: str) -> None:
    settings = make_settings(embedding_base_url=base_url or None)

    assert resolve_endpoint(settings) == expected


def test_explicit_endpoint_wins_over_derivation() -> None:
    settings = make_settings(
        embedding_base_url="https://api.openai.com/v1",
        rerank_endpoint="https://example.com/rerank",
    )

    assert resolve_endpoint(settings) == "https://example.com/rerank"


def test_http_error_raises_rerank_error() -> None:
    reranker, _ = reranker_with(reply([], status=401, code="InvalidApiKey", message="bad key"))

    with pytest.raises(RerankError, match="HTTP 401"):
        reranker.rerank("问题", make_documents("正文"), top_n=1)


def test_error_envelope_with_http_200_raises_rerank_error() -> None:
    reranker, _ = reranker_with(
        reply([], code="InvalidApiKey", message="Invalid API-key provided.")
    )

    with pytest.raises(RerankError, match="InvalidApiKey"):
        reranker.rerank("问题", make_documents("正文"), top_n=1)


def test_empty_results_raise_rerank_error() -> None:
    reranker, _ = reranker_with(reply([]))

    with pytest.raises(RerankError, match="结果为空"):
        reranker.rerank("问题", make_documents("正文"), top_n=1)


def test_bad_indexes_are_skipped() -> None:
    reranker, _ = reranker_with(
        reply(
            [
                {"index": 99, "relevance_score": 0.99},  # 越界
                {"index": 0, "relevance_score": 0.4},
                {"index": 0, "relevance_score": 0.8},  # 重复下标，保留第一次出现的
                {"index": 1, "relevance_score": None},  # 分数非法
            ]
        )
    )

    ranked = reranker.rerank("问题", make_documents("a", "b"), top_n=2)

    assert [document.metadata["chunk_id"] for document in ranked] == ["chunk-0"]


def test_empty_query_or_documents_skips_request() -> None:
    reranker, captured = reranker_with(reply([{"index": 0, "relevance_score": 0.5}]))

    assert reranker.rerank("   ", make_documents("正文")) == []
    assert reranker.rerank("问题", []) == []
    assert captured == []


def test_get_reranker_returns_none_without_key() -> None:
    assert get_reranker(make_settings(rerank_api_key=None, dashscope_api_key=None)) is None


def test_get_reranker_returns_none_when_disabled() -> None:
    assert get_reranker(make_settings(rerank_enabled=False)) is None


def test_get_reranker_falls_back_to_dashscope_key() -> None:
    reranker = get_reranker(make_settings(rerank_api_key=None, dashscope_api_key="sk-dash"))

    assert isinstance(reranker, DashScopeReranker)
    assert reranker.api_key == "sk-dash"


def test_get_reranker_reuses_embedding_key_on_dashscope_domain() -> None:
    """同一个百炼业务空间里 embedding 与 rerank 共用一把 Key，不必填两遍。"""
    reranker = get_reranker(
        make_settings(
            rerank_api_key=None,
            dashscope_api_key=None,
            embedding_api_key="sk-workspace",
            embedding_base_url="https://ws-9lon.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )
    )

    assert isinstance(reranker, DashScopeReranker)
    assert reranker.api_key == "sk-workspace"


def test_get_reranker_ignores_embedding_key_of_other_providers() -> None:
    """OpenAI 的 Key 发到阿里云只会 401，所以非百炼域名不复用。"""
    settings = make_settings(
        rerank_api_key=None,
        dashscope_api_key=None,
        embedding_api_key="sk-openai",
        embedding_base_url="https://api.openai.com/v1",
    )

    assert get_reranker(settings) is None
