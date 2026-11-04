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
from fakes import ScriptedChatModel, StreamingScriptedChatModel


class FakeSearcher:
    def __init__(self, count: int = 2) -> None:
        self.queries: list[str] = []
        self.count = count

    def search(self, query: str, *, k: int | None = None) -> list[Document]:
        self.queries.append(query)
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


def make_graph(model: Any, *, count: int = 2, checkpointer: Any = None) -> tuple[Any, FakeSearcher]:
    searcher = FakeSearcher(count=count)
    graph = build_rag_chat_graph(
        llm=model,
        searcher=searcher,
        settings=Settings(_env_file=None),
        checkpointer=checkpointer,
    )
    return graph, searcher


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


def test_answer_normalizes_out_of_range_citation() -> None:
    """模型把法条号写成 [43] 时，answer 节点会校正成汉字法条引用。"""
    model = ScriptedChatModel(responses=[AIMessage(content="扣除标准是 60%[43]。")])
    graph, _ = make_graph(model, count=2)

    turn = ask_rag_chat(graph, "业务招待费的扣除标准是什么？")

    assert turn.answer == "扣除标准是 60%（第四十三条）。"


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
    assert "[1] 来源：doc-a（第 1 页）" in answer_prompt
    assert "那业务招待费呢" in answer_prompt


def test_no_context_skips_llm() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="不应被调用")])
    graph, searcher = make_graph(model, count=0)

    turn = ask_rag_chat(graph, "知识库里没有的问题")

    assert turn.answer == NO_CONTEXT_ANSWER
    assert turn.documents == []
    assert model.calls == 0
    assert searcher.queries == ["知识库里没有的问题"]


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
    assert any("answer" in update for update in updates)
