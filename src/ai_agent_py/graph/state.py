"""State definition shared by the graph nodes."""

from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Graph state.

    `messages` 由 `add_messages` 追加；另外两个键是循环预算（A-02）需要的辅助状态：

    * ``started_at`` — 本轮开始时间（`time.monotonic()` 秒），由 `agent` 节点在每轮
      第一次调用模型时写入，时间预算据此计算；
    * ``budget_reason`` — 预算耗尽的原因，由 `finalize` 节点写入，接口据此告知前端。

    用量本身（步数 / 工具次数 / token）**不落状态**，而是每轮从消息里现算（见
    :mod:`ai_agent_py.graph.budget`），这样带 checkpointer 的多轮会话不会把上一轮的
    用量累计进来。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    started_at: NotRequired[float]
    budget_reason: NotRequired[str]
