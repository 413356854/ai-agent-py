"""Graph nodes: the LLM step and the tool-execution step, plus the router.

The tool node is the single place where a model-requested tool call is
governed (see :mod:`ai_agent_py.tools.policy`):

* **timeout** — a call runs on a daemon thread with the per-tool deadline, so a
  stuck tool returns ``工具执行超时`` instead of hanging the graph;
* **retry** — transient failures (network / 429 / 5xx) are retried with a small
  backoff; invalid arguments and ordinary tool errors are not;
* **concurrency** — calls of the same turn run in parallel (``asyncio`` is not
  needed because LangGraph nodes are sync); side-effecting tools stay serial;
* **structured args** — every built-in tool declares a Pydantic ``args_schema``,
  so a bad argument comes back as a correctable ``参数不合法`` message instead of
  a generic crash;
* **observability** — every ``ToolMessage.artifact`` carries ``status`` /
  ``error_code`` / ``elapsed_ms`` / ``attempts`` for the API and MySQL rows.

Everything is a plain function over `AgentState` so nodes stay easy to test and
easy to swap.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from ai_agent_py.config import Settings
from ai_agent_py.graph.state import AgentState
from ai_agent_py.tools.policy import (
    ToolPolicies,
    ToolPolicy,
    is_retryable_exception,
    is_transient_result,
    retry_delay,
)

logger = logging.getLogger(__name__)

__all__ = ["ToolOutcome", "build_agent_node", "build_tool_node", "route_after_agent"]

ToolStatus = Literal["ok", "error", "timeout", "invalid_args"]

#: 兜底等待：正常情况下 ``_run_call`` 自己的超时先生效，这里只防线程卡死。
#: 有超时的工具按 ``timeout * (重试次数 + 1) + 余量`` 计算等待窗口。
_WAIT_GRACE_SECONDS = 30.0


@dataclass(slots=True)
class ToolOutcome:
    """One tool call's result plus the bookkeeping the API needs."""

    content: str
    status: ToolStatus = "ok"
    error_code: str | None = None
    elapsed_ms: int = 0
    attempts: int = 1
    raw_truncated: bool = False


def build_agent_node(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str,
) -> Callable[[AgentState], Any]:
    """Node that calls the LLM (with tools bound) and appends its reply."""
    model = llm.bind_tools(list(tools)) if tools else llm

    def agent_node(state: AgentState) -> dict[str, list[AnyMessage]]:
        """Node: call the model once with the system prompt and append its reply."""
        messages: list[AnyMessage] = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system_prompt))
        response = model.invoke(messages)
        return {"messages": [response]}

    return agent_node


def build_tool_node(
    tools: Sequence[BaseTool],
    *,
    result_formatter: Callable[[str], str] | None = None,
    policies: ToolPolicies | None = None,
    settings: Settings | None = None,
    parallel: bool | None = None,
) -> Callable[[AgentState], Any]:
    """Node that runs every tool call requested by the last AI message.

    ``result_formatter`` 只改模型可见的 ``ToolMessage.content``（用于通用 Agent 的
    工具输出预算）；完整结果始终放在 ``ToolMessage.artifact['raw']``，供落库/界面展示。
    ``policies`` / ``settings`` 决定超时、重试与并发；``parallel=False`` 可整体关掉并行。
    """
    by_name = {tool.name: tool for tool in tools}
    registry = policies or ToolPolicies.from_settings(settings)
    allow_parallel = registry.parallel if parallel is None else parallel
    formatter = result_formatter or str

    def _tool_message(outcome: ToolOutcome, *, call_id: str, name: str) -> ToolMessage:
        visible = formatter(outcome.content)
        return ToolMessage(
            content=visible,
            tool_call_id=call_id,
            name=name,
            artifact={
                "raw": outcome.content,
                "raw_chars": len(outcome.content),
                "truncated": visible != outcome.content,
                "status": outcome.status,
                "error_code": outcome.error_code,
                "elapsed_ms": outcome.elapsed_ms,
                "attempts": outcome.attempts,
                "raw_truncated": outcome.raw_truncated,
            },
        )

    def tool_node(state: AgentState) -> dict[str, list[AnyMessage]]:
        """Node: run every tool call from the last AI message and append ToolMessages."""
        messages = state["messages"]
        last = messages[-1] if messages else None
        calls = list(getattr(last, "tool_calls", None) or [])
        if not calls:
            return {"messages": []}

        results: list[ToolMessage | None] = [None] * len(calls)
        # 并行中的调用：(结果下标, 槽位, 名称, 等待截止时刻)
        pending: list[tuple[int, _Slot, str, float | None]] = []

        def drain_pending() -> None:
            for index, slot, name, deadline in pending:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if not slot.done.wait(remaining):
                    logger.error("工具 %s 线程未在预期时间内结束，按超时处理", name)
                    results[index] = _tool_message(
                        ToolOutcome(
                            content=(
                                f"工具执行超时：{name} 未在预期时间内返回，已中止等待。"
                                "请缩小任务范围或拆成多次调用后重试。"
                            ),
                            status="timeout",
                            error_code="timeout",
                        ),
                        call_id=slot.call_id,
                        name=name,
                    )
                else:
                    results[index] = _tool_message(
                        slot.outcome or _internal_failure(name),
                        call_id=slot.call_id,
                        name=name,
                    )
            pending.clear()

        for index, raw_call in enumerate(calls):
            call = dict(raw_call)
            name = str(call.get("name") or "")
            call_id = str(call.get("id") or "")
            tool = by_name.get(name)
            if tool is None:
                results[index] = _tool_message(
                    ToolOutcome(
                        content=f"未知工具：{name}", status="error", error_code="unknown_tool"
                    ),
                    call_id=call_id,
                    name=name,
                )
                continue

            args = call.get("args")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                results[index] = _tool_message(
                    ToolOutcome(
                        content=(
                            f"工具参数不合法（{name}）：参数必须是 JSON 对象，"
                            f"收到 {type(args).__name__}。请按工具说明重新组装参数后重试。"
                        ),
                        status="invalid_args",
                        error_code="invalid_args",
                    ),
                    call_id=call_id,
                    name=name,
                )
                continue

            policy = registry.get(name)
            if allow_parallel and policy.concurrent:
                slot = _Slot(call_id=call_id)
                slot.thread = threading.Thread(
                    target=slot.run,
                    args=(tool, args, policy, name),
                    name=f"tool-{name}",
                    daemon=True,
                )
                slot.thread.start()
                budget = _wait_budget(policy)
                pending.append(
                    (index, slot, name, None if budget is None else time.monotonic() + budget)
                )
            else:
                # 有副作用 / 关闭并行：先等已启动的并行调用结束，保持调用顺序可预期。
                drain_pending()
                results[index] = _tool_message(
                    _run_call(tool, args, policy),
                    call_id=call_id,
                    name=name,
                )

        drain_pending()
        return {"messages": [message for message in results if message is not None]}

    return tool_node


class _Slot:
    """Result holder for one tool call running on its own daemon thread."""

    __slots__ = ("call_id", "done", "outcome", "thread")

    def __init__(self, *, call_id: str) -> None:
        self.call_id = call_id
        self.done = threading.Event()
        self.outcome: ToolOutcome | None = None
        self.thread: threading.Thread | None = None

    def run(self, tool: BaseTool, args: dict[str, Any], policy: ToolPolicy, name: str) -> None:
        try:
            self.outcome = _run_call(tool, args, policy)
        except BaseException:  # 线程里的意外不能让 slot 永远等下去
            logger.exception("工具 %s 线程异常终止", name)
            self.outcome = _internal_failure(name)
        finally:
            self.done.set()


def _wait_budget(policy: ToolPolicy) -> float | None:
    """Outer wait window: a safety net around ``_run_call``'s own deadline."""
    if policy.timeout is None:
        return None
    return policy.timeout * (policy.max_retries + 1) + _WAIT_GRACE_SECONDS


def _run_call(tool: BaseTool, args: dict[str, Any], policy: ToolPolicy) -> ToolOutcome:
    """Run one tool call with retry, deadline and output bounding."""
    started = time.perf_counter()
    attempts = 0
    while True:
        attempts += 1
        try:
            finished, value = _invoke_once(tool, args, policy.timeout)
        except ValidationError as exc:
            return _invalid_args(tool, exc, started, attempts)
        except Exception as exc:  # 工具内部异常（网络 / 上层库错误）
            if policy.retryable and attempts <= policy.max_retries and is_retryable_exception(exc):
                logger.warning("工具 %s 第 %d 次执行失败（%s），准备重试", tool.name, attempts, exc)
                time.sleep(retry_delay(attempts))
                continue
            logger.exception("工具 %s 执行失败", tool.name)
            suffix = f"（已重试 {attempts - 1} 次）" if attempts > 1 else ""
            return ToolOutcome(
                content=f"工具执行失败（{tool.name}）{suffix}：{exc}",
                status="error",
                error_code=exc.__class__.__name__,
                elapsed_ms=_elapsed(started),
                attempts=attempts,
            )

        if not finished:
            # 超时不重试：再来一次很可能还是同样的耗时，只会把整轮拖得更久。
            return ToolOutcome(
                content=(
                    f"工具执行超时：{tool.name} 超过 {policy.timeout:g} 秒未返回，已中止等待。"
                    "请缩小任务范围、拆成多次调用，或调整参数后重试。"
                ),
                status="timeout",
                error_code="timeout",
                elapsed_ms=_elapsed(started),
                attempts=attempts,
            )

        content = str(value)
        if policy.retryable and attempts <= policy.max_retries and is_transient_result(content):
            logger.warning(
                "工具 %s 第 %d 次返回临时错误，准备重试：%.80s", tool.name, attempts, content
            )
            time.sleep(retry_delay(attempts))
            continue

        clipped, truncated = _clip_raw(content, policy.max_output_chars)
        return ToolOutcome(
            content=clipped,
            elapsed_ms=_elapsed(started),
            attempts=attempts,
            raw_truncated=truncated,
        )


def _invoke_once(
    tool: BaseTool,
    args: dict[str, Any],
    timeout: float | None,
) -> tuple[bool, Any]:
    """Run ``tool.invoke`` on a daemon thread; ``(False, None)`` means timeout."""
    slot = _InvokeSlot()

    def target() -> None:
        try:
            slot.value = tool.invoke(args)
        except BaseException as exc:  # 在调用线程重新抛出
            slot.error = exc
        finally:
            slot.done.set()

    thread = threading.Thread(target=target, name=f"tool-call-{tool.name}", daemon=True)
    thread.start()
    if not slot.done.wait(timeout):
        return False, None
    if slot.error is not None:
        if isinstance(slot.error, Exception):
            raise slot.error
        raise RuntimeError(f"工具线程异常终止：{slot.error!r}")
    return True, slot.value


class _InvokeSlot:
    """Shared state between the calling thread and one execution thread."""

    __slots__ = ("done", "error", "value")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.value: Any = None
        self.error: BaseException | None = None


def _invalid_args(
    tool: BaseTool, exc: ValidationError, started: float, attempts: int
) -> ToolOutcome:
    details: list[str] = []
    for error in exc.errors()[:5]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "参数"
        message = str(error.get("msg") or "参数不合法")
        details.append(f"{location}: {message}")
    detail = "；".join(details) or "参数不合法"
    return ToolOutcome(
        content=(
            f"工具参数不合法（{tool.name}）：{detail}。"
            "请对照工具说明修正参数后重试，不要重复提交同样的参数。"
        ),
        status="invalid_args",
        error_code="invalid_args",
        elapsed_ms=_elapsed(started),
        attempts=attempts,
    )


def _internal_failure(name: str) -> ToolOutcome:
    return ToolOutcome(
        content=f"工具执行失败（{name}）：内部错误，请稍后重试。",
        status="error",
        error_code="internal_error",
    )


def _clip_raw(text: str, limit: int | None) -> tuple[str, bool]:
    if limit is None or len(text) <= limit:
        return text, False
    omitted = len(text) - limit
    return f"{text[:limit]}\n…（工具输出过长，已省略 {omitted} 个字符）", True


def _elapsed(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def route_after_agent(state: AgentState) -> Literal["tools", "end"]:
    """Conditional edge: run tools if the model asked for them, else finish."""
    messages = state["messages"]
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"
