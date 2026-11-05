"""End-to-end RAG behaviour with offline models: ingest -> retrieve -> answer."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from ai_agent_py.rag.chain import (
    NO_CONTEXT_ANSWER,
    answer_question,
    build_rag_chain,
    format_documents,
    normalize_citations,
)
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


def test_format_documents_numbers_sources_and_pages() -> None:
    documents = [
        Document(page_content="第一段", metadata={"source": "a.md"}),
        Document(page_content="第二段", metadata={"source": "b.pdf", "page": 3}),
    ]

    rendered = format_documents(documents)

    assert rendered.startswith("[1] 来源：a.md")
    assert "[2] 来源：b.pdf（第 3 页）" in rendered
    assert format_documents([]) == "（未检索到相关资料）"


# ---- 越界引用校正（法条号被当成来源编号）----------------------------------
def test_normalize_citations_maps_article_number_to_its_source() -> None:
    documents = [
        Document(
            page_content="第四十二条 职工教育经费支出，不超过工资薪金总额2.5%的部分，准予扣除。"
        ),
        Document(
            page_content=(
                "第四十三条 企业发生的与生产经营活动有关的业务招待费支出，"
                "按照发生额的60%扣除，但最高不得超过当年销售（营业）收入的5‰。"
            )
        ),
    ]

    text, fixed = normalize_citations("最高不得超过当年销售（营业）收入的 5‰[43]。", documents)

    assert text == "最高不得超过当年销售（营业）收入的 5‰[2]。"
    assert fixed == [43]


def test_normalize_citations_keeps_valid_indexes() -> None:
    documents = [Document(page_content="甲"), Document(page_content="乙")]

    text, fixed = normalize_citations("依据[1]和[2]。", documents)

    assert text == "依据[1]和[2]。"
    assert fixed == []


def test_normalize_citations_falls_back_when_article_is_ambiguous() -> None:
    documents = [
        Document(page_content="第四十三条 业务招待费……"),
        Document(page_content="另见第四十三条的衔接规定。"),
    ]

    # 两条资料都提到这条法条，无法判定来源：补成汉字法条引用
    text, _ = normalize_citations("最高不超过 5‰[43]。", documents)
    assert text == "最高不超过 5‰（第四十三条）。"

    # 句子本身已经写了法条号，删掉方括号即可，避免重复
    text, _ = normalize_citations("第四十三条规定最高不超过 5‰[43]。", documents)
    assert text == "第四十三条规定最高不超过 5‰。"


def test_normalize_citations_ignores_year_like_numbers() -> None:
    documents = [Document(page_content="甲")]

    text, fixed = normalize_citations("2024 年[2024]发布。", documents)

    assert text == "2024 年[2024]发布。"
    assert fixed == []


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (5, "第五条"),
        (10, "第十条"),
        (21, "第二十一条"),
        (105, "第一百零五条"),
        (110, "第一百一十条"),
    ],
)
def test_normalize_citations_uses_chinese_article_labels(number: int, expected: str) -> None:
    text, _ = normalize_citations(f"见[{number}]。", [Document(page_content="无关正文")])

    assert text == f"见（{expected}）。"
