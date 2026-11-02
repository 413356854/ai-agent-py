"""The plain (non-agentic) RAG chain: retrieve -> prompt -> answer.

This is the "one shot" path used by `ai-agent-py ask`, and the piece that the
LangGraph agent wraps as a tool.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

NO_CONTEXT_ANSWER = (
    "知识库中没有检索到与这个问题相关的内容。\n"
    "请换一种说法，或先运行 `ai-agent-py ingest <目录>` 建立索引。"
)

SYSTEM_PROMPT = """你是「企业知识库问答助手」，必须遵守以下规则：
1. 只使用【参考资料】中的信息作答，不要依赖先验知识，不要编造事实。
2. 如果资料不足以回答，直接说明“资料中未找到相关内容”，并指出缺少什么信息。
3. 引用资料时标注来源编号，例如 [1]、[2]，编号必须与资料的序号对应。
4. 使用简体中文回答：先给结论，再给要点，必要时使用列表。
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
    """Answer text plus the chunks it was grounded on."""

    answer: str
    documents: list[Document]

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
    retriever: BaseRetriever,
) -> RagAnswer:
    """Retrieve, then answer. Skips the LLM entirely when nothing was found."""
    documents = list(retriever.invoke(question))
    if not documents:
        return RagAnswer(answer=NO_CONTEXT_ANSWER, documents=[])
    return RagAnswer(answer=generate_answer(llm, question, documents), documents=documents)


def build_rag_chain(
    llm: BaseChatModel,
    retriever: BaseRetriever,
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Composable LCEL equivalent: ``{"question": str} -> {answer, documents, ...}``.

    Useful when embedding this RAG into another Runnable graph or an API layer.
    """

    def _retrieve(payload: dict[str, Any]) -> list[Document]:
        return list(retriever.invoke(payload["question"]))

    def _format(payload: dict[str, Any]) -> str:
        return format_documents(payload["documents"])

    return (
        RunnablePassthrough.assign(documents=_retrieve)
        | RunnablePassthrough.assign(context=_format)
        | RunnablePassthrough.assign(answer=RAG_PROMPT | llm | StrOutputParser())
    )
