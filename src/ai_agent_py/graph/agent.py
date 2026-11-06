"""The LangGraph agent: a ReAct style loop over the retrieval tool.

    START -> agent --(tool calls and budget left?)--> tools -> agent -> ...
              |                    |
              |                    +--(budget exhausted)--> finalize -> END
              +--(no tool calls)---------------------------> END

The loop is bounded twice (A-02):

* the **budget** (:mod:`ai_agent_py.graph.budget`) is the normal stop — once the
  step / tool-call / token / time budget is used up, the graph runs ``finalize``,
  which answers from the material already collected;
* the graph's ``recursion_limit`` is only a safety net; :func:`agent_config`
  derives it from the budget so the budget always wins.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.graph.budget import LoopBudget, turn_messages
from ai_agent_py.graph.messages import message_text
from ai_agent_py.graph.nodes import (
    NO_FINAL_ANSWER,
    build_agent_node,
    build_finalize_node,
    build_route_after_agent,
    build_tool_node,
)
from ai_agent_py.graph.state import AgentState
from ai_agent_py.llm import get_chat_model
from ai_agent_py.rag.milvus_search import DocumentSearcher, get_milvus_searcher
from ai_agent_py.tools import build_knowledge_tool

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """你是「ai-agent-py」的知识库助手，可以调用工具来查找资料。

可用工具：
- `search_knowledge_base(query)`：在内部知识库中做向量检索，返回带编号的文档片段。

工作方式：
1. 需要事实依据的问题，先调用 `search_knowledge_base` 检索，再基于检索结果作答。
2. 只依据检索到的内容回答，不要编造；检索不到就直说，并建议用户换个问法或补充文档。
3. 引用资料时标注片段编号，例如 [1]、[2]，编号只能取工具返回内容里的 [n]；
   法条序号要写成「第四十三条」，不要写成 [43]，文号、年份同理。
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
    searcher: DocumentSearcher | None = None,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: Any | None = None,
    settings: Settings | None = None,
    system_prompt: str = AGENT_SYSTEM_PROMPT,
) -> CompiledStateGraph:
    """Wire and compile the agent graph.

    `llm` / `searcher` / `tools` are injectable so tests can run offline with
    fake models, and so extra tools can be added without touching this file.
    """
    cfg = settings or get_settings()
    model = llm or get_chat_model(cfg)
    budget = LoopBudget.from_settings(cfg)
    tool_list: list[BaseTool] = (
        list(tools)
        if tools is not None
        else [build_knowledge_tool(searcher or get_milvus_searcher(settings=cfg))]
    )

    builder = StateGraph(AgentState)
    _add_node(builder, "agent", build_agent_node(model, tool_list, system_prompt=system_prompt))
    _add_node(
        builder,
        "tools",
        build_tool_node(tool_list, settings=cfg, budget=budget),
    )
    _add_node(
        builder,
        "finalize",
        build_finalize_node(model, system_prompt=system_prompt, budget=budget),
    )
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        build_route_after_agent(budget),
        {"tools": "tools", "finalize": "finalize", "end": END},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


def agent_config(
    *,
    thread_id: str | None = None,
    max_turns: int | None = None,
    settings: Settings | None = None,
) -> RunnableConfig:
    """Runnable config for one agent invocation (thread + recursion limit).

    ``recursion_limit`` 只是兜底：正常路径由循环预算在 ``finalize`` 收尾，这里保证
    预算耗尽前不会先撞上 LangGraph 的上限（两者取较大值）。
    """
    cfg = settings or get_settings()
    budget = LoopBudget.from_settings(cfg)
    turns = max_turns or cfg.agent_max_turns
    config: dict[str, Any] = {
        "recursion_limit": max(2, turns * 2 + 2, budget.recursion_limit()),
    }
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    return cast("RunnableConfig", config)


def ask_agent(
    graph: CompiledStateGraph,
    question: str,
    *,
    thread_id: str | None = None,
    max_turns: int | None = None,
    settings: Settings | None = None,
) -> str:
    """One turn against the agent graph; returns the final answer text.

    只看**本轮**（最后一条 `HumanMessage` 之后）的 AI 消息：上一轮的回答不能被当成本轮
    答案；本轮没有最终回答时返回明确的 finalize 文案，而不是把 `ToolMessage` 当答案。
    """
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            agent_config(thread_id=thread_id, max_turns=max_turns, settings=settings),
        )
    except GraphRecursionError:
        logger.warning("智能体达到 recursion_limit，返回明确的收尾文案")
        return NO_FINAL_ANSWER

    messages: list[AnyMessage] = result["messages"]
    for message in reversed(turn_messages(messages)):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            text = message_text(message).strip()
            if text:
                return text
    return NO_FINAL_ANSWER
