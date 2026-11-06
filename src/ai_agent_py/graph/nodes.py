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

A-02 adds the **loop budget** to the same module:

* :func:`build_route_after_agent` sends the graph to ``finalize`` instead of
  ``tools`` once a budget (steps / tool calls / tokens / wall clock) is used up;
* :func:`build_finalize_node` answers from the material already collected and
  tells the user that the information may be incomplete — a budget stop is a
  degraded but *usable* answer, never a 502;
* the tool node skips repeated identical retrieval calls (``dedupe``) and
  refuses calls past ``AGENT_MAX_TOOL_CALLS``.

Everything is a plain function over `AgentState` so nodes stay easy to test and
easy to swap.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from ai_agent_py.config import Settings
from ai_agent_py.graph.budget import (
    DEFAULT_BUDGET,
    REASON_LABELS,
    LoopBudget,
    TurnUsage,
    measure_turn,
    tool_status,
    turn_messages,
)
from ai_agent_py.graph.messages import message_text
from ai_agent_py.graph.state import AgentState
from ai_agent_py.tools.policy import (
    ToolPolicies,
    ToolPolicy,
    is_failure_result,
    is_retryable_exception,
    is_transient_result,
    retry_delay,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NO_FINAL_ANSWER",
    "ToolOutcome",
    "build_agent_node",
    "build_finalize_node",
    "build_route_after_agent",
    "build_tool_node",
    "route_after_agent",
]

ToolStatus = Literal["ok", "error", "timeout", "invalid_args", "skipped"]

#: 预算耗尽后 `finalize` 节点的系统提示后缀。故意不绑定工具：已经没有预算再调了。
FINALIZE_SYSTEM_SUFFIX = """【执行预算已用尽】本轮已经用掉{reason_label}（{usage}），从现在起不能再调用任何工具。
请立刻基于上面已经获得的资料和工具结果给出最终回答：
1. 先给出目前能确定的最佳答案；引用资料时保留已有的 [n] 编号，不要编造工具没有返回过的内容；
2. 明确说明「信息可能不完整」，并指出还缺哪一步、哪些资料；
3. 如果现有结果不足以回答，直说不足在哪里，并给出下一步建议（缩小范围、换关键词、分步执行）。"""

#: 没有最终 AI 回答时的统一文案（`ask_agent` 与通用 Agent 的 `_collect_turn` 共用）。
NO_FINAL_ANSWER = (
    "本轮没有生成最终答案：可能已经到达执行预算上限，或者模型只返回了工具调用。"
    "请缩小任务范围、分步提问后重试。"
)

#: `finalize` 节点连模型都调用失败时的静态兜底文案。
BUDGET_FALLBACK_ANSWER = (
    "本轮执行已到达{reason_label}上限（{usage}），模型没有给出最终答案。"
    "信息可能不完整，请缩小任务范围后重试，或调大 AGENT_MAX_TURNS / AGENT_MAX_TOOL_CALLS。"
)

#: 兜底文案里附带的上一次工具结果最多保留多少字符。
_FALLBACK_SNIPPET_CHARS = 800

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

    def agent_node(state: AgentState) -> dict[str, Any]:
        """Node: call the model once with the system prompt and append its reply."""
        messages: list[AnyMessage] = list(state["messages"])
        update: dict[str, Any] = {}
        if messages and isinstance(messages[-1], HumanMessage):
            # 每轮第一次调用模型时记下开始时间，供时间预算使用（每轮覆盖）。
            update["started_at"] = time.monotonic()
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system_prompt))
        response = model.invoke(messages)
        update["messages"] = [response]
        return update

    return agent_node


def build_tool_node(
    tools: Sequence[BaseTool],
    *,
    result_formatter: Callable[[str], str] | None = None,
    policies: ToolPolicies | None = None,
    settings: Settings | None = None,
    parallel: bool | None = None,
    budget: LoopBudget | None = None,
) -> Callable[[AgentState], Any]:
    """Node that runs every tool call requested by the last AI message.

    ``result_formatter`` 只改模型可见的 ``ToolMessage.content``（用于通用 Agent 的
    工具输出预算）；完整结果始终放在 ``ToolMessage.artifact['raw']``，供落库/界面展示。
    ``policies`` / ``settings`` 决定超时、重试与并发；``parallel=False`` 可整体关掉并行。
    ``budget`` 给定时，超过 ``AGENT_MAX_TOOL_CALLS`` 的调用直接回一条「预算用尽」，
    不再真正执行；同一轮里重复的只读检索（``ToolPolicy.dedupe``）也会被跳过。
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

        # 本轮此前「成功执行过」的调用签名（失败/超时的不算，模型可以再试一次）。
        completed = _completed_signatures(messages[:-1])
        # 已经真正执行过的工具调用次数（含失败/超时，不含被跳过的）。
        used = measure_turn(messages[:-1]).tool_calls

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
            signature = call_signature(name, args)
            if policy.dedupe and signature in completed:
                results[index] = _tool_message(
                    ToolOutcome(
                        content=_duplicate_message(name, args),
                        status="skipped",
                        error_code="duplicate_call",
                    ),
                    call_id=call_id,
                    name=name,
                )
                continue
            if (
                budget is not None
                and budget.max_tool_calls > 0
                and used >= budget.max_tool_calls
            ):
                results[index] = _tool_message(
                    ToolOutcome(
                        content=_tool_budget_message(name, budget.max_tool_calls),
                        status="skipped",
                        error_code="tool_budget",
                    ),
                    call_id=call_id,
                    name=name,
                )
                continue

            # 真执行了才占用预算 / 进重复调用表：未知工具、参数错误不算。
            used += 1
            if policy.dedupe:
                completed.add(signature)

            if allow_parallel and policy.concurrent:
                slot = _Slot(call_id=call_id)
                slot.thread = threading.Thread(
                    target=slot.run,
                    args=(tool, args, policy, name),
                    name=f"tool-{name}",
                    daemon=True,
                )
                slot.thread.start()
                wait_budget = _wait_budget(policy)
                pending.append(
                    (
                        index,
                        slot,
                        name,
                        None if wait_budget is None else time.monotonic() + wait_budget,
                    )
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


def build_finalize_node(
    llm: BaseChatModel,
    *,
    system_prompt: str,
    budget: LoopBudget,
    compressor: Callable[[list[AnyMessage]], list[AnyMessage]] | None = None,
    system_extra: Callable[[Mapping[str, Any]], str] | None = None,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Node that closes the turn once a loop budget is exhausted.

    故意**不** `bind_tools`：预算已经用完，不允许再请求工具。模型如果仍然返回
    `tool_calls` 或空内容，就退化成静态的「预算用尽」文案，保证接口不会 502、
    也不会把上一条 `ToolMessage` 当答案返回。

    ``system_extra`` 用来追加调用方自己的系统信息（通用 Agent 用它拼「更早对话摘要」），
    保证 finalize 的答案仍然带着压缩过的历史背景。

    注意 ``state`` 注解故意用 ``Mapping[str, Any]`` 而不是 ``AgentState``：LangGraph 会按
    节点参数的类型注解**裁剪**传给它的状态，写成 ``AgentState`` 时通用 Agent 的
    ``summary`` 会被裁掉（finalize 就看不到压缩历史了）。
    """

    def finalize_node(state: Mapping[str, Any]) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        usage = budget.usage(state)
        reason = budget.check(state, usage=usage) or "max_steps"
        extra = system_extra(state).strip() if system_extra is not None else ""
        system = f"{system_prompt}\n\n{extra}" if extra else system_prompt
        system = f"{system}\n\n{_finalize_instruction(reason, usage)}"
        prompt = _with_system_message(messages, system)
        prompt = _with_skipped_tool_results(prompt, _skipped_tool_note(reason))
        if compressor is not None:
            prompt = compressor(prompt)
        try:
            response = llm.invoke(prompt)
        except Exception:  # 模型 / 网络失败也要给用户一个明确的收尾
            logger.exception("finalize 节点调用模型失败，改用静态兜底文案")
            return {
                "messages": [AIMessage(content=_fallback_answer(reason, usage, messages))],
                "budget_reason": reason,
            }
        text = message_text(response).strip()
        if not text:
            text = _fallback_answer(reason, usage, messages)
        return {"messages": [AIMessage(content=text)], "budget_reason": reason}

    return finalize_node


def build_route_after_agent(
    budget: LoopBudget,
) -> Callable[[Mapping[str, Any]], Literal["tools", "finalize", "end"]]:
    """条件边：还有工具调用且预算未耗尽 → `tools`；预算耗尽 → `finalize`；否则结束。"""

    def route(state: Mapping[str, Any]) -> Literal["tools", "finalize", "end"]:
        return _route(state, budget)

    return route


def route_after_agent(state: AgentState) -> Literal["tools", "finalize", "end"]:
    """默认预算下的条件边（与 Settings 默认值一致）。

    图里用 :func:`build_route_after_agent` 注入配置后的预算；这里是给直接引用
    `route_after_agent` 的调用方保留的便捷入口。
    """
    return _route(state, DEFAULT_BUDGET)


def _route(state: Mapping[str, Any], budget: LoopBudget) -> Literal["tools", "finalize", "end"]:
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return "end"
    if budget.check(state) is not None:
        return "finalize"
    return "tools"


def _finalize_instruction(reason: str, usage: TurnUsage) -> str:
    return FINALIZE_SYSTEM_SUFFIX.format(
        reason_label=REASON_LABELS.get(reason, reason),
        usage=_usage_text(usage),
    )


def _usage_text(usage: TurnUsage) -> str:
    parts = [f"{usage.steps} 次模型调用", f"{usage.tool_calls} 次工具调用"]
    if usage.tokens:
        parts.append(f"{usage.tokens} tokens")
    if usage.elapsed_seconds >= 1:
        parts.append(f"{usage.elapsed_seconds:.0f} 秒")
    return "、".join(parts)


def _fallback_answer(reason: str, usage: TurnUsage, messages: Sequence[AnyMessage]) -> str:
    text = BUDGET_FALLBACK_ANSWER.format(
        reason_label=REASON_LABELS.get(reason, reason),
        usage=_usage_text(usage),
    )
    snippet = _last_tool_snippet(messages)
    if snippet:
        text += f"\n\n最后一次工具结果（节选）：\n{snippet}"
    return text


def _last_tool_snippet(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(turn_messages(messages)):
        if not isinstance(message, ToolMessage):
            continue
        artifact = getattr(message, "artifact", None)
        raw = artifact.get("raw") if isinstance(artifact, dict) else None
        text = raw if isinstance(raw, str) and raw.strip() else message_text(message)
        text = text.strip()
        if not text:
            continue
        if len(text) > _FALLBACK_SNIPPET_CHARS:
            omitted = len(text) - _FALLBACK_SNIPPET_CHARS
            return f"{text[:_FALLBACK_SNIPPET_CHARS]}\n…（已省略 {omitted} 个字符）"
        return text
    return ""


def _with_system_message(messages: list[AnyMessage], system: str) -> list[AnyMessage]:
    """把系统提示放到最前面（替换已有的系统消息，避免出现两条）。"""
    rest = [message for message in messages if not isinstance(message, SystemMessage)]
    return [SystemMessage(content=system), *rest]


def _skipped_tool_note(reason: str) -> str:
    label = REASON_LABELS.get(reason, reason)
    return f"工具未执行：本轮{label}已用尽，请基于已经获得的结果作答，不要再请求工具。"


def _with_skipped_tool_results(
    messages: list[AnyMessage],
    note: str,
) -> list[AnyMessage]:
    """给「预算耗尽后没有执行」的工具调用补一条占位 ToolMessage。

    OpenAI 兼容接口要求每个 `tool_call` 后面必须跟一条 `tool` 消息；预算耗尽时最后
    一条 `AIMessage` 里的调用还没执行，直接拿去做 finalize 的 prompt 会 400。
    这些占位消息只进 prompt，不写回图状态（`finalize` 只返回一条 AI 回答）。
    """
    answered = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    result: list[AnyMessage] = []
    for message in messages:
        result.append(message)
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or []:
            call_id = str(call.get("id") or "")
            if not call_id or call_id in answered:
                continue
            result.append(
                ToolMessage(
                    content=note,
                    tool_call_id=call_id,
                    name=str(call.get("name") or ""),
                    artifact={"raw": note, "status": "skipped", "error_code": "tool_budget"},
                )
            )
            answered.add(call_id)
    return result


def call_signature(name: str, args: Any) -> str:
    """`(工具名, 参数)` 的稳定签名，用于同一轮里的重复调用检测。"""
    try:
        payload = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - 参数来自 JSON，正常不会失败
        payload = repr(args)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"{name}:{digest}"


def _completed_signatures(messages: Sequence[AnyMessage]) -> set[str]:
    """本轮已经**成功**执行过的调用签名（失败 / 超时 / 参数错误的允许再试）。"""
    signatures: dict[str, str] = {}
    completed: set[str] = set()
    for message in turn_messages(messages):
        if isinstance(message, AIMessage):
            for raw in message.tool_calls or []:
                call_id = str(raw.get("id") or "")
                if not call_id:
                    continue
                args = raw.get("args")
                signatures[call_id] = call_signature(
                    str(raw.get("name") or ""),
                    args if isinstance(args, dict) else {},
                )
        elif isinstance(message, ToolMessage):
            signature = signatures.get(message.tool_call_id)
            if signature is not None and _succeeded(message):
                completed.add(signature)
    return completed


def _succeeded(message: ToolMessage) -> bool:
    if tool_status(message) != "ok":
        return False
    return not is_failure_result(message_text(message))


def _duplicate_message(name: str, args: dict[str, Any]) -> str:
    rendered = json.dumps(args, ensure_ascii=False, default=str)
    if len(rendered) > 300:
        rendered = f"{rendered[:300]}…"
    return (
        f"重复调用已跳过：本轮已经成功执行过 {name}，参数完全相同（{rendered}）。"
        "请直接使用上一次的结果；需要不同范围的资料时请修改参数后再调用。"
    )


def _tool_budget_message(name: str, limit: int) -> str:
    return (
        f"工具调用预算已用尽：本轮最多执行 {limit} 次工具调用，{name} 没有执行。"
        "请基于已经获得的结果作答，或缩小任务范围后重新提问。"
    )


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
