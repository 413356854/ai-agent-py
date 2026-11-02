"""FastAPI layer with injected offline models (no network)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import ai_agent_py.api.app as api_app
from ai_agent_py.rag.ingest import ingest
from ai_agent_py.rag.retriever import get_retriever
from fakes import ScriptedChatModel


def _client(monkeypatch, resources) -> TestClient:
    monkeypatch.setattr(api_app, "_resources", lambda: resources)
    return TestClient(api_app.app)


def test_health_reports_configuration(monkeypatch, settings) -> None:
    monkeypatch.setattr(api_app, "get_settings", lambda: settings)
    client = TestClient(api_app.app)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["vector_store"] == settings.vector_store


def test_query_returns_answer_and_sources(
    monkeypatch, settings, store, embeddings, corpus_dir
) -> None:
    ingest([corpus_dir], store=store, embeddings=embeddings, settings=settings, recreate=True)
    retriever = get_retriever(store=store, settings=settings)
    llm = ScriptedChatModel(responses=[AIMessage(content="答案来自知识库")])
    client = _client(monkeypatch, (settings, llm, store, retriever))

    response = client.post("/query", json={"question": "用什么管理依赖？", "k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "答案来自知识库"
    assert body["sources"]
    assert {"index", "source", "page", "snippet"} <= set(body["sources"][0])


def test_query_validates_empty_question(monkeypatch, settings, store) -> None:
    client = _client(
        monkeypatch, (settings, ScriptedChatModel(responses=[AIMessage(content="x")]), store, None)
    )
    assert client.post("/query", json={"question": ""}).status_code == 422
