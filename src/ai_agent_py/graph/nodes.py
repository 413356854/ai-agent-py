"""Graph nodes: the LLM step and the tool-execution step, plus the router.

Everything is a plain function over `AgentState` so nodes stay easy to test
and easy to swap.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from ai_agent_py.graph.state import AgentState

logger = logging.getLogger(__name__)


def build_agent_node(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str,
) -> Callable[[AgentState], Any]:
    """Node that calls the LLM (with tools bound) and appends its reply."""
    model = llm.bind_tools(list(tools)) if tools else llm

    def agent_node(state: AgentState) -> dict[str, list[AnyMessage]]:
        messages: list[AnyMessage] = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system_prompt))
        response = model.invoke(messages)
        return {"messages": [response]}

    return agent_node


def build_tool_node(
    tools: Sequence[BaseTool],
) -> Callable[[AgentState], Any]:
    """Node that runs every tool call requested by the last AI message."""
    by_name = {tool.name: tool for tool in tools}

    def tool_node(state: AgentState) -> dict[str, list[AnyMessage]]:
        messages = state["messages"]
        last = messages[-1] if messages else None
        calls = list(getattr(last, "tool_calls", None) or [])
        outputs: list[AnyMessage] = []

        for call in calls:
            payload = dict(call)
            name = str(payload.get("name") or "")
            call_id = str(payload.get("id") or "")
            tool = by_name.get(name)
            if tool is None:
                outputs.append(ToolMessage(content=f"未知工具：{name}", tool_call_id=call_id))
                continue
            try:
                result = tool.invoke(payload.get("args") or {})
            except Exception as exc:  # report back to the model
                logger.exception("Tool %s failed", name)
                outputs.append(ToolMessage(content=f"工具执行失败：{exc}", tool_call_id=call_id))
            else:
                outputs.append(ToolMessage(content=str(result), tool_call_id=call_id))

        return {"messages": outputs}

    return tool_node


def route_after_agent(state: AgentState) -> Literal["tools", "end"]:
    """Conditional edge: run tools if the model asked for them, else finish."""
    messages = state["messages"]
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"
