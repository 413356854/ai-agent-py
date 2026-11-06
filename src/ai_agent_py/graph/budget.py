"""A-02 循环预算：步数 / 工具次数 / token / 墙钟，以及「预算耗尽」的判定。

设计要点
--------

* **用量每轮现算**：只统计消息列表里最后一条 `HumanMessage` 之后的部分
  （:func:`turn_messages`），所以带 checkpointer 的多轮会话不会把上一轮的用量累计
  进来，也不需要额外的状态计数器；
* **时间预算需要状态**：本轮开始时间 `started_at`（monotonic 秒）由 agent 节点在
  每轮第一次调用模型时写入，`LoopBudget.usage()` 用它算已用时间；
* **耗尽不是异常**：路由（:func:`ai_agent_py.graph.nodes.build_route_after_agent`）
  在预算用完时改走 `finalize` 节点，用已获得的资料给出「信息可能不完整」的答案，
  而不是让 LangGraph 抛 `GraphRecursionError` 变成 502；
* **预算是护栏**：`0` 表示该项不限制；默认只关掉 token（provider 不一定返回 usage），
  步数 / 工具次数 / 墙钟默认开启。
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from ai_agent_py.config import Settings, get_settings

__all__ = [
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_TOOL_CALLS",
    "DEFAULT_MAX_TURNS",
    "REASON_LABELS",
    "BudgetReason",
    "LoopBudget",
    "TurnUsage",
    "measure_turn",
    "tool_status",
    "turn_messages",
    "usage_tokens",
]

#: 预算耗尽原因；`recursion_limit` 是「没走预算、被 LangGraph 兜底截断」的情况。
BudgetReason = Literal[
    "max_steps",
    "max_tool_calls",
    "max_tokens",
    "max_seconds",
    "recursion_limit",
]

#: 提示词与接口展示用的中文标签。
REASON_LABELS: dict[str, str] = {
    "max_steps": "模型调用步数",
    "max_tool_calls": "工具调用次数",
    "max_tokens": "token 预算",
    "max_seconds": "单轮时间预算",
    "recursion_limit": "图循环上限",
}

#: 与 `Settings` 的默认值保持一致（直接使用 `route_after_agent` 时按它判定）。
DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_TOOL_CALLS = 24
DEFAULT_MAX_SECONDS = 600.0


@dataclass(frozen=True, slots=True)
class TurnUsage:
    """一轮对话已经用掉的预算。"""

    steps: int = 0
    tool_calls: int = 0
    skipped_calls: int = 0
    tokens: int = 0
    elapsed_seconds: float = 0.0


def turn_messages(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """本轮消息：最后一条 `HumanMessage` 之后的部分（没有 human 时返回全部）。"""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return list(messages[index:])
    return list(messages)


def tool_status(message: ToolMessage) -> str:
    """工具状态：优先取工具节点写入的 artifact，老消息没有 artifact 时按成功处理。"""
    artifact = getattr(message, "artifact", None)
    if isinstance(artifact, dict):
        status = artifact.get("status")
        if isinstance(status, str) and status:
            return status
    return "ok"


def measure_turn(
    messages: Sequence[AnyMessage],
    *,
    elapsed_seconds: float = 0.0,
) -> TurnUsage:
    """统计本轮用量：模型调用次数、真正执行的工具次数、token、已用时间。"""
    steps = 0
    tool_calls = 0
    skipped = 0
    tokens = 0
    for message in turn_messages(messages):
        if isinstance(message, AIMessage):
            steps += 1
            tokens += usage_tokens(message)
        elif isinstance(message, ToolMessage):
            if tool_status(message) == "skipped":
                skipped += 1
            else:
                tool_calls += 1
    return TurnUsage(
        steps=steps,
        tool_calls=tool_calls,
        skipped_calls=skipped,
        tokens=tokens,
        elapsed_seconds=max(0.0, elapsed_seconds),
    )


def usage_tokens(message: AIMessage) -> int:
    """从 `usage_metadata`（优先）或 OpenAI 风格的 `response_metadata` 取总 token。"""
    usage = getattr(message, "usage_metadata", None)
    total = _total_tokens(usage)
    if total is not None:
        return total
    metadata = getattr(message, "response_metadata", None)
    if isinstance(metadata, dict):
        for key in ("token_usage", "usage"):
            total = _total_tokens(metadata.get(key))
            if total is not None:
                return total
    return 0


def _total_tokens(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    if isinstance(total, (int, float)) and total > 0:
        return int(total)
    return None


@dataclass(frozen=True, slots=True)
class LoopBudget:
    """一轮对话的循环预算；`0` 表示该项不限制。"""

    max_steps: int = DEFAULT_MAX_TURNS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_tokens: int = 0
    max_seconds: float = DEFAULT_MAX_SECONDS

    def __post_init__(self) -> None:
        if self.max_steps < 0 or self.max_tool_calls < 0 or self.max_tokens < 0:
            raise ValueError("循环预算（步数 / 工具次数 / token）不能为负数")
        if self.max_seconds < 0:
            raise ValueError("单轮时间预算不能为负数")

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LoopBudget:
        """按 `Settings` 构造：`AGENT_MAX_TURNS` 是步数预算，其余见 config.py。"""
        cfg = settings or get_settings()
        return cls(
            max_steps=cfg.agent_max_turns,
            max_tool_calls=cfg.agent_max_tool_calls,
            max_tokens=cfg.agent_max_tokens,
            max_seconds=cfg.agent_max_seconds,
        )

    def usage(self, state: Mapping[str, Any]) -> TurnUsage:
        """按图状态算本轮的用量（含 `started_at` 起算的时间）。"""
        messages = state.get("messages") or []
        started_at = state.get("started_at")
        elapsed = 0.0
        if isinstance(started_at, (int, float)):
            elapsed = max(0.0, time.monotonic() - float(started_at))
        return measure_turn(messages, elapsed_seconds=elapsed)

    def check(
        self,
        state: Mapping[str, Any],
        *,
        usage: TurnUsage | None = None,
    ) -> BudgetReason | None:
        """返回第一个耗尽的预算项；都还有余量时返回 `None`。"""
        current = usage if usage is not None else self.usage(state)
        if self.max_steps > 0 and current.steps >= self.max_steps:
            return "max_steps"
        if self.max_tool_calls > 0 and current.tool_calls >= self.max_tool_calls:
            return "max_tool_calls"
        if self.max_tokens > 0 and current.tokens >= self.max_tokens:
            return "max_tokens"
        if self.max_seconds > 0 and current.elapsed_seconds >= self.max_seconds:
            return "max_seconds"
        return None

    def recursion_limit(self) -> int:
        """LangGraph 的兜底上限：预算耗尽后还要再走一次 `finalize`。

        每次模型调用之间夹一个 `tools` 节点，`2 × max_steps` 步足够走到 `finalize`；
        再加 2 步余量，保证预算一定先于 `GraphRecursionError` 生效。
        """
        steps = max(0, self.max_steps)
        return max(2, steps * 2 + 2)

    def describe(self) -> dict[str, float | int]:
        """JSON 友好的配置快照（`GET /api/general-agent/overview`）。"""
        return {
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_tokens": self.max_tokens,
            "max_seconds": self.max_seconds,
        }


#: 默认预算：供直接使用 `route_after_agent` 的调用方（图里会注入配置后的实例）。
DEFAULT_BUDGET = LoopBudget()
