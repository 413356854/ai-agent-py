"""State definition shared by the graph nodes."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Graph state.

    Only the message list for now; `add_messages` appends/merges instead of
    overwriting. Add more keys here (documents, plan, iterations, ...) as the
    graph grows - any key written by parallel nodes needs a reducer such as
    `operator.add`.
    """

    messages: Annotated[list[AnyMessage], add_messages]
