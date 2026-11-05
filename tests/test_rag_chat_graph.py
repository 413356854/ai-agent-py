"""LangGraph RAG 问答图：首轮直答、多轮改写、无召回兜底、流式 token。"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_agent_py.config import Settings
from ai_agent_py.graph.agent import message_text
from ai_agent_py.graph.rag_chat import (
    NO_CONTEXT_ANSWER,
    ask_rag_chat,
    build_rag_chat_graph,
    rag_chat_config,
)
from ai_agent_py.rag.filters import SearchFilter
from fakes import FakeDocumentNameResolver, ScriptedChatModel, StreamingScriptedChatModel

_FILE_NAME = "中华人民共和国企业所得税法实施条例.docx"


class FakeSearcher:
    def __init__(self, count: int = 2) -> None:
        self.queries: list[str] = []
        self.filters: list[SearchFilter | None] = []
        self.count = count

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: SearchFilter | None = None,
    ) -> list[Document]:
        self.queries.append(query)
        self.filters.append(filter)
        return [
            Document(
                page_content=f"第 {index} 条资料：与「{query}」有关的内容",
                metadata={
                    "chunk_id": f"chunk-{index}",
                    "doc_id": "doc-a",
                    "chunk_index": index,
                    "page": index + 1,
                    "source": "doc-a",
                },
            )
            for index in range(self.count)
        ]


def make_graph(
    model: Any,
    *,
    count: int = 2,
    checkpointer: Any = None,
    resolver: Any | None = None,
) -> tuple[Any, FakeSearcher]:
    searcher = FakeSearcher(count=count)
    name_resolver = (
        resolver if resolver is not None else FakeDocumentNameResolver({"doc-a": _FILE_NAME})
    )
    graph = build_rag_chat_graph(
        llm=model,
        searcher=searcher,
        resolver=name_resolver,
        settings=Settings(_env_file=None),
        checkpointer=checkpointer,
    )
    return graph, searcher


class LegalSearcher:
    """固定正文的检索器：用来构造「引用被支持 / 不被支持」的答案。"""

    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.queries: list[str] = []
        self.filters: list[SearchFilter | None] = []

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: SearchFilter | None = None,
    ) -> list[Document]:
        self.queries.append(query)
        self.filters.append(filter)
        return [
            Document(
                page_content=content,
                metadata={
                    "chunk_id": f"chunk-{index}",
                    "doc_id": "doc-a",
                    "chunk_index": index,
                    "page": 1,
                    "source": "doc-a",
                },
            )
            for index, content in enumerate(self.contents)
        ]


def make_legal_graph(
    model: Any,
    searcher: LegalSearcher,
    *,
    checkpointer: Any = None,
) -> Any:
    return build_rag_chat_graph(
        llm=model,
        searcher=searcher,
        resolver=FakeDocumentNameResolver({"doc-a": _FILE_NAME}),
        settings=Settings(_env_file=None),
        checkpointer=checkpointer,
    )


def test_first_turn_answers_without_rewriting() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="答案一")])
    graph, searcher = make_graph(model)

    turn = ask_rag_chat(graph, "企业所得税的扣除项目有哪些？")

    assert turn.answer == "答案一"
    assert turn.query == "企业所得税的扣除项目有哪些？"
    assert [document.metadata["chunk_id"] for document in turn.documents] == ["chunk-0", "chunk-1"]
    # 首轮没有历史，不额外调用模型做改写
    assert model.calls == 1
    assert searcher.queries == ["企业所得税的扣除项目有哪些？"]


def test_answer_removes_out_of_range_citation() -> None:
    """模型把法条号写成 [43] 时，answer 节点删除编号，不再改写成汉字法条。"""
    model = ScriptedChatModel(responses=[AIMessage(content="扣除标准是 60%[43]。")])
    graph, _ = make_graph(model, count=2)

    turn = ask_rag_chat(graph, "业务招待费的扣除标准是什么？")

    assert turn.answer == "扣除标准是 60%。"
    assert turn.citation_stats.invalid == 1
    assert turn.citation_stats.removed == 1
    assert turn.citations[0].removed is True


def test_prompt_bounds_citation_numbers_and_bans_article_numbers() -> None:
    """「[43]」这类法条号被当成引用编号的问题，根因在提示词，这里锁住约束。"""
    model = ScriptedChatModel(responses=[AIMessage(content="答案一")])
    graph, _ = make_graph(model, count=3)

    ask_rag_chat(graph, "业务招待费的扣除标准是什么？")

    system = model.received[0][0].content
    human = model.received[0][-1].content
    assert isinstance(system, str)
    assert isinstance(human, str)
    # 点名反例：法条序号必须写汉字，不能进方括号
    assert "第四十三条" in system
    assert "[43]" in system
    # 参考资料标出总数，模型才知道合法编号的上界
    assert "共 3 条" in human
    assert "编号 1~3" in human


def test_retrieved_chunks_get_mysql_file_names() -> None:
    """Milvus 只给 doc_id；resolve 节点回查 MySQL 后，模型看到的是文件名。"""
    model = ScriptedChatModel(responses=[AIMessage(content="答案一 [1]")])
    resolver = FakeDocumentNameResolver({"doc-a": "企业所得税法实施条例.docx"})
    graph, _ = make_graph(model, resolver=resolver)

    turn = ask_rag_chat(graph, "业务招待费的扣除标准是什么？")

    assert [document.metadata["source"] for document in turn.documents] == [
        "企业所得税法实施条例.docx",
        "企业所得税法实施条例.docx",
    ]
    assert all(
        document.metadata["doc_name"] == "企业所得税法实施条例.docx" for document in turn.documents
    )
    # 同属一篇文档的多个片段只查一次
    assert resolver.calls == [["doc-a"]]
    prompt = model.received[0][-1].content
    assert isinstance(prompt, str)
    assert "来源：企业所得税法实施条例.docx（第 1 页）" in prompt
    assert "doc-a" not in prompt


def test_unknown_doc_id_falls_back_to_doc_id() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="答案一")])
    graph, _ = make_graph(model, resolver=FakeDocumentNameResolver())

    turn = ask_rag_chat(graph, "业务招待费的扣除标准是什么？")

    assert turn.documents[0].metadata["source"] == "doc-a"
    prompt = model.received[0][-1].content
    assert isinstance(prompt, str)
    assert "来源：doc-a（第 1 页）" in prompt


def test_name_lookup_failure_does_not_break_answer() -> None:
    class BrokenResolver:
        def resolve(self, doc_ids: Any) -> Any:
            raise RuntimeError("MySQL 挂了")

    model = ScriptedChatModel(responses=[AIMessage(content="答案一")])
    graph, _ = make_graph(model, resolver=BrokenResolver())

    turn = ask_rag_chat(graph, "业务招待费的扣除标准是什么？")

    assert turn.answer == "答案一"
    assert turn.documents[0].metadata["source"] == "doc-a"


def test_graph_without_mysql_keeps_doc_id_source() -> None:
    """没配 MySQL 时默认解析器静默跳过，问答仍可用，来源退回 doc_id。"""
    model = ScriptedChatModel(responses=[AIMessage(content="答案一")])
    graph = build_rag_chat_graph(
        llm=model,
        searcher=FakeSearcher(),
        settings=Settings(_env_file=None),
    )

    turn = ask_rag_chat(graph, "业务招待费的扣除标准是什么？")

    assert turn.answer == "答案一"
    assert turn.documents[0].metadata["source"] == "doc-a"


def test_follow_up_rewrites_query_with_history() -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="答案一"),
            AIMessage(content="业务招待费的扣除标准是什么？"),
            AIMessage(content="答案二"),
        ]
    )
    graph, searcher = make_graph(model, checkpointer=InMemorySaver())

    first = graph.invoke(
        {"messages": [HumanMessage(content="企业所得税的扣除项目有哪些？")]},
        rag_chat_config(thread_id="t1"),
    )
    second = graph.invoke(
        {"messages": [HumanMessage(content="那业务招待费呢？")]},
        rag_chat_config(thread_id="t1"),
    )

    assert first["answer"] == "答案一"
    assert second["answer"] == "答案二"
    assert second["query"] == "业务招待费的扣除标准是什么？"
    assert searcher.queries == [
        "企业所得税的扣除项目有哪些？",
        "业务招待费的扣除标准是什么？",
    ]
    assert [message.type for message in second["messages"]] == ["human", "ai", "human", "ai"]

    # 改写提示词里带着上一轮问答，才能消解「那…呢」的指代
    condense_prompt = model.received[1][-1].content
    assert isinstance(condense_prompt, str)
    assert "企业所得税的扣除项目有哪些" in condense_prompt
    assert "那业务招待费呢" in condense_prompt

    # 生成答案的提示词：system + 上一轮问答 + 本轮（参考资料 + 问题）
    answer_messages = model.received[2]
    assert [message.type for message in answer_messages] == ["system", "human", "ai", "human"]
    answer_prompt = answer_messages[-1].content
    assert isinstance(answer_prompt, str)
    assert "【参考资料】" in answer_prompt
    assert f"[1] 来源：{_FILE_NAME}（第 1 页）" in answer_prompt
    assert "那业务招待费呢" in answer_prompt


def test_no_context_skips_llm() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="不应被调用")])
    resolver = FakeDocumentNameResolver({"doc-a": _FILE_NAME})
    graph, searcher = make_graph(model, count=0, resolver=resolver)

    turn = ask_rag_chat(graph, "知识库里没有的问题")

    assert turn.answer == NO_CONTEXT_ANSWER
    assert turn.documents == []
    assert model.calls == 0
    assert searcher.queries == ["知识库里没有的问题"]
    # 没有召回片段就不必回查文件名
    assert resolver.calls == []


def test_stream_mode_emits_answer_tokens_only() -> None:
    model = StreamingScriptedChatModel(
        responses=[AIMessage(content="企业所得税的扣除项目")],
        chunk_size=3,
    )
    graph, _ = make_graph(model)

    events = list(
        graph.stream(
            {"messages": [HumanMessage(content="企业所得税的扣除项目有哪些？")]},
            {"recursion_limit": 12},
            stream_mode=["messages", "updates"],
        )
    )

    tokens: list[str] = []
    for mode, payload in events:
        if mode != "messages":
            continue
        message, metadata = payload
        if metadata.get("langgraph_node") == "answer" and isinstance(message, AIMessageChunk):
            tokens.append(message_text(message))

    assert "".join(tokens) == "企业所得税的扣除项目"
    updates = [payload for mode, payload in events if mode == "updates"]
    assert any("retrieve" in update for update in updates)
    assert any("resolve" in update for update in updates)
    assert any("answer" in update for update in updates)


# ---- 文档级过滤（R-01）----------------------------------------------------
def test_doc_ids_are_forwarded_as_search_filter() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="答案一")])
    graph, searcher = make_graph(model)

    ask_rag_chat(graph, "业务招待费的扣除标准是什么？", doc_ids=["doc-a", "doc-b"])

    assert searcher.filters == [SearchFilter(doc_ids=("doc-a", "doc-b"))]


def test_doc_ids_scope_is_reset_on_next_turn() -> None:
    """上一轮选了文档，下一轮不传 doc_ids 时必须回到全库，不能沿用旧范围。"""
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="答案一"),
            AIMessage(content="业务招待费的扣除标准是什么？"),
            AIMessage(content="答案二"),
        ]
    )
    graph, searcher = make_graph(model, checkpointer=InMemorySaver())

    ask_rag_chat(graph, "企业所得税的扣除项目有哪些？", thread_id="t-filter", doc_ids=["doc-a"])
    ask_rag_chat(graph, "那业务招待费呢？", thread_id="t-filter")

    assert searcher.filters == [SearchFilter(doc_ids=("doc-a",)), None]


# ---- 引用校验与历史隔离（R-03 / D-01）-------------------------------------
def test_history_strips_previous_citation_numbers() -> None:
    """上一轮答案里的 [n] 进下一轮 prompt 前必须剥掉，否则会和本轮编号冲突。"""
    searcher = LegalSearcher(["业务招待费按发生额的60%扣除。"])
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="业务招待费按60%扣除[1]。"),
            AIMessage(content="业务招待费的扣除标准是什么？"),
            AIMessage(content="答案二"),
        ]
    )
    graph = make_legal_graph(model, searcher, checkpointer=InMemorySaver())

    first = ask_rag_chat(graph, "业务招待费怎么扣？", thread_id="t-history")
    assert first.citation_stats.ok == 1

    ask_rag_chat(graph, "那它的标准呢？", thread_id="t-history")

    condense_prompt = model.received[1][-1].content
    assert isinstance(condense_prompt, str)
    assert "业务招待费按60%扣除" in condense_prompt
    assert "[1]" not in condense_prompt

    answer_messages = model.received[2]
    history = [message_text(message) for message in answer_messages if message.type == "ai"]
    assert history == ["业务招待费按60%扣除。"]


def test_no_context_turn_clears_previous_citations() -> None:
    """无召回时不能把上一轮 checkpointer 里的 citations 带出来。"""
    model = ScriptedChatModel(responses=[AIMessage(content="答案[1]"), AIMessage(content="第二问")])
    graph, searcher = make_graph(model, checkpointer=InMemorySaver())

    first = ask_rag_chat(graph, "第一问", thread_id="t-clear")
    assert first.citation_stats.total == 1

    searcher.count = 0
    second = ask_rag_chat(graph, "第二问", thread_id="t-clear")

    assert second.answer == NO_CONTEXT_ANSWER
    assert second.citations == []
    assert second.citation_stats.total == 0


def test_empty_documents_are_filtered_before_numbering() -> None:
    """空片段被过滤后，答案里的 [1] 必须对应过滤后的第一条，而不是原始下标。"""
    searcher = LegalSearcher(["", "业务招待费按60%扣除。"])
    model = ScriptedChatModel(responses=[AIMessage(content="业务招待费按60%扣除[1]。")])
    graph = make_legal_graph(model, searcher)

    turn = ask_rag_chat(graph, "业务招待费怎么扣？")

    assert [document.metadata["chunk_id"] for document in turn.documents] == ["chunk-1"]
    assert turn.citations[0].chunk_id == "chunk-1"
    assert turn.citation_stats.ok == 1
