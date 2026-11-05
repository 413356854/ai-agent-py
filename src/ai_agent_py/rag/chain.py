"""The plain (non-agentic) RAG chain: retrieve -> prompt -> answer.

This is the "one shot" path used by `ai-agent-py ask`, and the piece that the
LangGraph agent wraps as a tool. ``answer_question`` also verifies the answer's
``[n]`` citations via :mod:`ai_agent_py.rag.citations`; ``generate_answer`` /
``stream_answer`` stay as raw LLM calls for callers that need the token stream.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from ai_agent_py.rag.citations import Citation, CitationStats, verify_citations

if TYPE_CHECKING:
    from ai_agent_py.rag.milvus_search import DocumentSearcher

NO_CONTEXT_ANSWER = (
    "知识库中没有检索到与这个问题相关的内容。\n"
    "请换一种说法，或先运行 `ai-agent-py ingest <目录>` 建立索引。"
)

SYSTEM_PROMPT = """你是「企业知识库问答助手」，必须遵守以下规则：
1. 只使用【参考资料】中的信息作答，不要依赖先验知识，不要编造事实、文件名、文号、日期。
2. 引用编号只能取资料开头的 [n]；法条序号要写成「第四十三条」或「实施条例第四十三条」，
   不能写成 [43]，文号、年份同理。
3. 每句事实在句末标注来源编号，例如 [1]、[2]；编号必须与资料序号一一对应。
4. 资料不足以回答时，直接说明「资料中未找到相关内容」，并指出缺少什么信息。
5. 涉及税率、金额、期限、比例等数字时原样引用，不要改写、换算或估算。
6. 使用简体中文回答：先给结论，再给要点，必要时使用列表；不要另列参考资料清单。
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}"),
    ]
)


@dataclass(slots=True)
class Source:
    """One retrieved chunk, as shown to the user."""

    index: int
    source: str
    page: int | None
    snippet: str

    @property
    def label(self) -> str:
        return f"{self.source} (p.{self.page})" if self.page is not None else self.source

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source": self.source,
            "page": self.page,
            "snippet": self.snippet,
        }


@dataclass(slots=True)
class RagAnswer:
    """Answer text, the chunks it was grounded on, and citation verification."""

    answer: str
    documents: list[Document]
    citations: list[Citation] = field(default_factory=list)
    citation_stats: CitationStats = field(default_factory=CitationStats)

    @property
    def sources(self) -> list[Source]:
        sources: list[Source] = []
        for index, document in enumerate(self.documents, start=1):
            page = document.metadata.get("page")
            sources.append(
                Source(
                    index=index,
                    source=str(document.metadata.get("source", "unknown")),
                    page=int(page) if isinstance(page, int) else None,
                    snippet=_snippet(document.page_content),
                )
            )
        return sources


def _snippet(text: str, limit: int = 160) -> str:
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else f"{flattened[:limit]}…"


def format_documents(documents: Sequence[Document]) -> str:
    """Render retrieved chunks as numbered context blocks for the prompt."""
    if not documents:
        return "（未检索到相关资料）"

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        header = f"[{index}] 来源：{document.metadata.get('source', 'unknown')}"
        page = document.metadata.get("page")
        if page is not None:
            header += f"（第 {page} 页）"
        blocks.append(f"{header}\n{document.page_content.strip()}")
    return "\n\n".join(blocks)


def generate_answer(
    llm: BaseChatModel,
    question: str,
    documents: Sequence[Document],
) -> str:
    """Single blocking call: context + question -> answer text."""
    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": format_documents(documents), "question": question})


def stream_answer(
    llm: BaseChatModel,
    question: str,
    documents: Sequence[Document],
) -> Iterator[str]:
    """Same as `generate_answer`, but yields answer tokens."""
    chain = RAG_PROMPT | llm | StrOutputParser()
    yield from chain.stream({"context": format_documents(documents), "question": question})


def answer_question(
    question: str,
    *,
    llm: BaseChatModel,
    searcher: DocumentSearcher,
    k: int | None = None,
) -> RagAnswer:
    """Retrieve, then answer. Skips the LLM entirely when nothing was found."""
    documents = [
        document for document in searcher.search(question, k=k) if document.page_content.strip()
    ]
    if not documents:
        return RagAnswer(answer=NO_CONTEXT_ANSWER, documents=[])
    report = verify_citations(generate_answer(llm, question, documents), documents)
    return RagAnswer(
        answer=report.answer,
        documents=documents,
        citations=list(report.citations),
        citation_stats=report.stats,
    )


def build_rag_chain(
    llm: BaseChatModel,
    searcher: DocumentSearcher,
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Composable LCEL equivalent: ``{"question": str} -> {answer, documents, ...}``.

    Useful when embedding this RAG into another Runnable graph or an API layer.
    """

    def _retrieve(payload: dict[str, Any]) -> list[Document]:
        return list(searcher.search(payload["question"]))

    def _format(payload: dict[str, Any]) -> str:
        return format_documents(payload["documents"])

    return (
        RunnablePassthrough.assign(documents=_retrieve)
        | RunnablePassthrough.assign(context=_format)
        | RunnablePassthrough.assign(answer=RAG_PROMPT | llm | StrOutputParser())
    )
