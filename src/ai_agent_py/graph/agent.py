"""The LangGraph agent: a ReAct style loop over the retrieval tool.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

`agent_max_turns` caps the loop through the graph's `recursion_limit`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.graph.nodes import build_agent_node, build_tool_node, route_after_agent
from ai_agent_py.graph.state import AgentState
from ai_agent_py.llm import get_chat_model
from ai_agent_py.rag.retriever import get_retriever
from ai_agent_py.tools import build_retriever_tool

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """你是「ai-agent-py」的知识库助手，可以调用工具来查找资料。

可用工具：
- `search_knowledge_base(query)`：在内部知识库中做向量检索，返回带编号的文档片段。

工作方式：
1. 需要事实依据的问题，先调用 `search_knowledge_base` 检索，再基于检索结果作答。
2. 只依据检索到的内容回答，不要编造；检索不到就直说，并建议用户换个问法或补充文档。
3. 引用资料时标注片段编号，例如 [1]、[2]。
4. 寒暄或与知识库无关的闲聊可以直接回答，不必调用工具。
5. 使用简体中文回答。

【重要】开始回答前，先判断是否已检索过；未检索就先调用工具，不要凭记忆直接回答。
"""


def _add_node(builder: StateGraph, name: str, node: Callable[[AgentState], Any]) -> None:
    """Register a node function on the graph.

    mypy cannot match a `Callable[[TypedDict], Any]` *variable* against
    LangGraph's node-protocol overloads (plain `def`s work), so the ignored
    call is kept in one place instead of at every call site.
    """
    builder.add_node(name, node)  # type: ignore[call-overload]


def build_agent_graph(
    *,
    llm: BaseChatModel | None = None,
    retriever: BaseRetriever | None = None,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: Any | None = None,
    settings: Settings | None = None,
    system_prompt: str = AGENT_SYSTEM_PROMPT,
) -> CompiledStateGraph:
    """Wire and compile the agent graph.

    `llm` / `retriever` / `tools` are injectable so tests can run offline with
    fake models, and so extra tools can be added without touching this file.
    """
    cfg = settings or get_settings()
    model = llm or get_chat_model(cfg)
    tool_list: list[BaseTool] = (
        list(tools)
        if tools is not None
        else [build_retriever_tool(retriever or get_retriever(settings=cfg))]
    )

    builder = StateGraph(AgentState)
    _add_node(builder, "agent", build_agent_node(model, tool_list, system_prompt=system_prompt))
    _add_node(builder, "tools", build_tool_node(tool_list))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


def agent_config(*, thread_id: str | None = None, max_turns: int | None = None) -> RunnableConfig:
    """Runnable config for one agent invocation (thread + recursion limit)."""
    cfg = get_settings()
    turns = max_turns or cfg.agent_max_turns
    config: dict[str, Any] = {"recursion_limit": max(2, turns * 2 + 1)}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    return cast("RunnableConfig", config)


def message_text(message: AnyMessage) -> str:
    """Flatten message content (string or content-block list) to plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
    return "".join(parts)


def ask_agent(
    graph: CompiledStateGraph,
    question: str,
    *,
    thread_id: str | None = None,
    max_turns: int | None = None,
) -> str:
    """One turn against the agent graph; returns the final answer text."""
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        agent_config(thread_id=thread_id, max_turns=max_turns),
    )
    messages: list[AnyMessage] = result["messages"]
    for message in reversed(messages):
        if message.type == "ai" and not getattr(message, "tool_calls", None):
            return message_text(message)
    return message_text(messages[-1])
