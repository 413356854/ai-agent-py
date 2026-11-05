"""The LangGraph agent: routing, tool execution, end-to-end turn."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_agent_py.graph.agent import ask_agent, build_agent_graph, message_text
from ai_agent_py.graph.nodes import route_after_agent
from ai_agent_py.rag.ingest import ingest
from fakes import ScriptedChatModel, tool_call


def test_router_only_goes_to_tools_when_calls_requested() -> None:
    assert route_after_agent({"messages": [AIMessage(content="直接回答")]}) == "end"
    assert route_after_agent({"messages": [tool_call("search_knowledge_base", "uv")]}) == "tools"


def test_agent_retrieves_then_answers(settings, searcher, corpus_dir) -> None:
    ingest([corpus_dir], writer=searcher, settings=settings)
    model = ScriptedChatModel(
        responses=[
            tool_call("search_knowledge_base", "uv 依赖管理"),
            AIMessage(content="项目使用 uv 管理依赖。"),
        ]
    )
    graph = build_agent_graph(llm=model, searcher=searcher, settings=settings)

    result = graph.invoke(
        {"messages": [HumanMessage(content="这个项目用什么管理依赖？")]},
        {"recursion_limit": 10},
    )
    messages = result["messages"]

    assert [message.type for message in messages] == ["human", "ai", "tool", "ai"]
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "call-1"
    assert "uv" in str(messages[2].content)
    assert messages[-1].content == "项目使用 uv 管理依赖。"
    assert model.calls == 2
    # Second call must include the system prompt, history and the tool result.
    second_prompt = model.received[1]
    assert second_prompt[0].type == "system"
    assert any(isinstance(message, ToolMessage) for message in second_prompt)


def test_ask_agent_returns_final_text(settings, searcher, corpus_dir) -> None:
    ingest([corpus_dir], writer=searcher, settings=settings)
    model = ScriptedChatModel(
        responses=[
            tool_call("search_knowledge_base", "LangGraph"),
            AIMessage(content="LangGraph 负责流程编排。"),
        ]
    )
    graph = build_agent_graph(llm=model, searcher=searcher, settings=settings)

    assert ask_agent(graph, "LangGraph 做什么？") == "LangGraph 负责流程编排。"


def test_message_text_flattens_content_blocks() -> None:
    message = AIMessage(
        content=[{"type": "text", "text": "你好"}, {"type": "text", "text": "世界"}],
    )
    assert message_text(message) == "你好世界"
    assert message_text(AIMessage(content="纯文本")) == "纯文本"
