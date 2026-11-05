"""The plain (non-agentic) RAG chain: retrieve -> prompt -> answer.

This is the "one shot" path used by `ai-agent-py ask`, and the piece that the
LangGraph agent wraps as a tool.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

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


#: 回答里的方括号数字：可能是合法来源编号，也可能是模型误写的法条序号。
_CITATION_RE = re.compile(r"\[(\d{1,4})\]")

#: 判断「这句话有没有提到同一条法条」用的句读。
_SENTENCE_MARKS = "。！？；\n"

_DIGITS = "零一二三四五六七八九"


def normalize_citations(
    text: str,
    documents: Sequence[Document],
) -> tuple[str, list[int]]:
    """把越界的 ``[n]`` 修正成真实来源编号或汉字法条引用，返回 ``(文本, 被改写的编号)``。

    模型偶尔把法条序号当成来源编号（第四十三条 → ``[43]``）。这里用**本轮召回的
    片段**做确定性校正，而不是猜：

    * 只有一条资料正文写了「第 N 条」→ 这句话就来自该资料，改写成它的序号 ``[n]``；
    * 没有资料包含这条法条 → 句子本身已提到就删掉方括号，否则写成「（第 N 条）」；
    * ``n`` 本来就 ≤ 资料条数，或不在 1~999 之间（年份、金额）→ 原样保留。
    """
    total = len(documents)
    if not text or total <= 0:
        return text, []

    fixed: list[int] = []

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if 1 <= number <= total or not 1 <= number <= 999:
            return match.group(0)
        label = f"第{_chinese_number(number)}条"
        start = match.start()
        sentence_start = max(text.rfind(mark, 0, start) for mark in _SENTENCE_MARKS) + 1
        sentence = text[sentence_start:start]
        fixed.append(number)
        owner = _article_owner(label, documents)
        if owner is not None:
            return f"[{owner}]"
        return "" if label in sentence else f"（{label}）"

    return _CITATION_RE.sub(replace, text), fixed


def _article_owner(label: str, documents: Sequence[Document]) -> int | None:
    """哪条资料正文提到了这条法条；唯一命中时才返回它的序号（从 1 开始）。"""
    hits = [
        index for index, document in enumerate(documents, start=1) if label in document.page_content
    ]
    return hits[0] if len(hits) == 1 else None


def _chinese_number(value: int) -> str:
    """1~999 → 汉字序号（43 → 四十三），只用于法条引用。"""
    if value < 10:
        return _DIGITS[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        prefix = "十" if tens == 1 else f"{_DIGITS[tens]}十"
        return prefix + (_DIGITS[ones] if ones else "")
    hundreds, rest = divmod(value, 100)
    text = f"{_DIGITS[hundreds]}百"
    if rest == 0:
        return text
    if rest < 10:
        return f"{text}零{_DIGITS[rest]}"
    tens, ones = divmod(rest, 10)
    return text + f"{_DIGITS[tens]}十" + (_DIGITS[ones] if ones else "")


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
    documents = list(searcher.search(question, k=k))
    if not documents:
        return RagAnswer(answer=NO_CONTEXT_ANSWER, documents=[])
    answer, _ = normalize_citations(generate_answer(llm, question, documents), documents)
    return RagAnswer(answer=answer, documents=documents)


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
