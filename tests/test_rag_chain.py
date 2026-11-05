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
from ai_agent_py.rag.filters import SearchFilter
from ai_agent_py.rag.ingest import ingest
from fakes import ScriptedChatModel


def test_ingest_then_answer(settings, searcher, corpus_dir) -> None:
    report = ingest([corpus_dir], writer=searcher, settings=settings)

    assert report.files == 3
    assert report.chunks >= 3
    assert report.characters > 0
    assert report.store == "OfflineChunkSearcher"
    assert len(searcher) == report.chunks

    llm = ScriptedChatModel(responses=[AIMessage(content="本项目用 uv 管理依赖。")])

    result = answer_question("这个项目用什么管理依赖？", llm=llm, searcher=searcher, k=3)

    assert result.answer == "本项目用 uv 管理依赖。"
    assert result.documents
    assert [source.index for source in result.sources] == list(range(1, len(result.documents) + 1))
    # The retrieved context really made it into the prompt.
    prompt = "\n".join(str(message.content) for message in llm.received[0])
    assert "【参考资料】" in prompt
    assert "uv" in prompt


def test_lcel_chain_returns_answer_and_documents(settings, searcher, corpus_dir) -> None:
    ingest([corpus_dir], writer=searcher, settings=settings)
    llm = ScriptedChatModel(responses=[AIMessage(content="来自 LCEL 链的回答")])

    output = build_rag_chain(llm, searcher).invoke({"question": "LangGraph 用来做什么？"})

    assert output["answer"] == "来自 LCEL 链的回答"
    assert output["documents"]
    assert "来源" in output["context"]


def test_answer_without_documents_skips_the_llm(settings, searcher) -> None:
    llm = ScriptedChatModel(responses=[AIMessage(content="不应被调用")])

    result = answer_question("知识库里没有的问题", llm=llm, searcher=searcher)

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert llm.calls == 0


def test_reingest_is_idempotent(settings, searcher, corpus_dir) -> None:
    first = ingest([corpus_dir], writer=searcher, settings=settings)
    second = ingest([corpus_dir], writer=searcher, settings=settings)

    assert second.chunks == first.chunks
    assert len(searcher) == first.chunks


def test_offline_searcher_filters_by_doc_ids(searcher) -> None:
    searcher.index_document("doc-a", [Document(page_content="住宿费标准：每晚 500 元")])
    searcher.index_document("doc-b", [Document(page_content="住宿费标准：每晚 800 元")])

    unfiltered = searcher.search("住宿费标准", k=5)
    assert {document.metadata["doc_id"] for document in unfiltered} == {"doc-a", "doc-b"}

    filtered = searcher.search("住宿费标准", k=5, filter=SearchFilter(doc_ids=("doc-b",)))
    assert [document.metadata["doc_id"] for document in filtered] == ["doc-b"]


def test_format_documents_numbers_sources_and_pages() -> None:
    documents = [
        Document(page_content="第一段", metadata={"source": "a.md"}),
        Document(page_content="第二段", metadata={"source": "b.pdf", "page": 3}),
    ]

    rendered = format_documents(documents)

    assert rendered.startswith("[1] 来源：a.md")
    assert "[2] 来源：b.pdf（第 3 页）" in rendered
    assert format_documents([]) == "（未检索到相关资料）"


# ---- 答案引用校验 ---------------------------------------------------------
def test_answer_question_verifies_citations(searcher) -> None:
    searcher.index_document(
        "doc-a",
        [Document(page_content="业务招待费按发生额的60%扣除，最高不超过当年销售收入的5‰。")],
    )
    llm = ScriptedChatModel(responses=[AIMessage(content="业务招待费按60%扣除，上限为5‰[1]。")])

    result = answer_question("业务招待费怎么扣？", llm=llm, searcher=searcher)

    assert result.answer == "业务招待费按60%扣除，上限为5‰[1]。"
    assert [citation.status for citation in result.citations] == ["ok"]
    assert result.citations[0].start is not None
    assert result.citation_stats.removed == 0


def test_answer_question_removes_out_of_range_citation(searcher) -> None:
    searcher.index_document("doc-a", [Document(page_content="业务招待费按发生额的60%扣除。")])
    llm = ScriptedChatModel(responses=[AIMessage(content="业务招待费按60%扣除[9]。")])

    result = answer_question("业务招待费怎么扣？", llm=llm, searcher=searcher)

    assert result.answer == "业务招待费按60%扣除。"
    assert result.citation_stats.invalid == 1
    assert result.citations[0].removed is True
