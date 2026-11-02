"""End-to-end RAG behaviour with offline models: ingest -> retrieve -> answer."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from ai_agent_py.rag.chain import (
    NO_CONTEXT_ANSWER,
    answer_question,
    build_rag_chain,
    format_documents,
)
from ai_agent_py.rag.ingest import ingest
from ai_agent_py.rag.retriever import get_retriever
from ai_agent_py.rag.store import store_size
from fakes import ScriptedChatModel


def test_ingest_then_answer(settings, store, embeddings, corpus_dir) -> None:
    report = ingest(
        [corpus_dir], store=store, embeddings=embeddings, settings=settings, recreate=True
    )

    assert report.files == 3
    assert report.chunks >= 3
    assert report.characters > 0
    assert store_size(store) == report.chunks

    retriever = get_retriever(store=store, settings=settings, k=3)
    llm = ScriptedChatModel(responses=[AIMessage(content="本项目用 uv 管理依赖。")])

    result = answer_question("这个项目用什么管理依赖？", llm=llm, retriever=retriever)

    assert result.answer == "本项目用 uv 管理依赖。"
    assert result.documents
    assert [source.index for source in result.sources] == list(range(1, len(result.documents) + 1))
    # The retrieved context really made it into the prompt.
    prompt = "\n".join(str(message.content) for message in llm.received[0])
    assert "【参考资料】" in prompt
    assert "uv" in prompt


def test_lcel_chain_returns_answer_and_documents(settings, store, embeddings, corpus_dir) -> None:
    ingest([corpus_dir], store=store, embeddings=embeddings, settings=settings, recreate=True)
    retriever = get_retriever(store=store, settings=settings, k=2)
    llm = ScriptedChatModel(responses=[AIMessage(content="来自 LCEL 链的回答")])

    output = build_rag_chain(llm, retriever).invoke({"question": "LangGraph 用来做什么？"})

    assert output["answer"] == "来自 LCEL 链的回答"
    assert output["documents"]
    assert "来源" in output["context"]


def test_answer_without_documents_skips_the_llm(settings, store) -> None:
    retriever = get_retriever(store=store, settings=settings)
    llm = ScriptedChatModel(responses=[AIMessage(content="不应被调用")])

    result = answer_question("知识库里没有的问题", llm=llm, retriever=retriever)

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert llm.calls == 0


def test_reingest_is_idempotent(settings, store, embeddings, corpus_dir) -> None:
    first = ingest(
        [corpus_dir], store=store, embeddings=embeddings, settings=settings, recreate=True
    )
    second = ingest([corpus_dir], store=store, embeddings=embeddings, settings=settings)

    assert second.chunks == first.chunks
    assert store_size(store) == first.chunks


def test_format_documents_numbers_sources_and_pages() -> None:
    documents = [
        Document(page_content="第一段", metadata={"source": "a.md"}),
        Document(page_content="第二段", metadata={"source": "b.pdf", "page": 3}),
    ]

    rendered = format_documents(documents)

    assert rendered.startswith("[1] 来源：a.md")
    assert "[2] 来源：b.pdf（第 3 页）" in rendered
    assert format_documents([]) == "（未检索到相关资料）"
