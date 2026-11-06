"""Per-tool execution policy: timeout, retry, output bound and risk level.

Every tool call the model asks for goes through :func:`ai_agent_py.graph.nodes.build_tool_node`,
which looks the tool up in a :class:`ToolPolicies` registry.  Keeping the policy
out of the tool functions has two benefits:

* the same tool is governed identically whether it is reached through the
  knowledge agent, the general agent or a custom graph;
* one place controls the guarantees: a stuck tool cannot hang the graph, a
  flaky network call is retried, and a huge result is bounded before it reaches
  the model or MySQL.

Design notes
------------
* ``timeout=None`` means "the tool enforces its own deadline".  Only ``shell``
  uses it: the command timeout is a tool argument (``AGENT_SHELL_TIMEOUT``,
  capped at 3600s), so an outer cap would silently override what the model asked
  for.
* Timeouts are enforced in a daemon thread, so a tool that ignores its own
  deadline still cannot block the graph (the abandoned thread is daemon and
  dies with the process).
* Retry only covers *transient* failures: network/5xx/429/timeout-shaped
  exceptions or ``错误：...`` strings.  Invalid arguments and ordinary
  "file not found" errors are never retried.
* ``concurrent=False`` serialises a tool (file writes and shell commands may
  touch shared state); the other calls of the same model turn still run in
  parallel.
* ``dedupe=True`` lets the tool node skip a **repeated identical** call of a
  read-only retrieval tool within one turn (same name + same arguments), so the
  model cannot keep re-searching the same query.  Side-effecting tools and
  filesystem reads are never deduped: re-reading a file after editing it is a
  legitimate pattern.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import httpx

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import DatabaseError, MilvusError, RerankError, ToolError
from ai_agent_py.tools.filesystem import (
    EDIT_TOOL_NAME,
    GLOB_TOOL_NAME,
    GREP_TOOL_NAME,
    READ_TOOL_NAME,
    WRITE_TOOL_NAME,
)
from ai_agent_py.tools.shell import SHELL_TOOL_NAME
from ai_agent_py.tools.web_extractor import WEB_EXTRACT_TOOL_NAME
from ai_agent_py.tools.web_search import WEB_SEARCH_TOOL_NAME

__all__ = [
    "DEFAULT_MAX_OUTPUT_CHARS",
    "KNOWLEDGE_TOOL_NAME",
    "RiskLevel",
    "ToolPolicies",
    "ToolPolicy",
    "ToolPolicyView",
    "is_failure_result",
    "is_retryable_exception",
    "is_transient_result",
    "retry_delay",
]

#: 与 :data:`ai_agent_py.tools.KNOWLEDGE_TOOL_NAME` 一致；这里写字面量是为了避免
#: 包初始化期的循环导入（``tools/__init__`` 会导入本模块用到的工具模块）。
KNOWLEDGE_TOOL_NAME = "search_knowledge_base"

RiskLevel = Literal["low", "medium", "high"]


class ToolPolicyView(TypedDict):
    """JSON friendly view of one policy (``GET /api/general-agent/overview``)."""

    timeout_seconds: float | None
    retryable: bool
    max_retries: int
    risk_level: str
    concurrent: bool
    dedupe: bool


#: 工具结果落 ``ToolMessage.artifact['raw']`` 的兜底上限（字符）。
#: 工具自己已经截断的（shell / web / grep）不会触发；read 一次最多 5000 行，
#: 极端情况下仍可能有几 MB 文本，这里防止它整段进内存 / MySQL。
DEFAULT_MAX_OUTPUT_CHARS = 200_000

#: 给「工具自身超时」留的余量：策略超时必须晚于工具内部超时，
#: 否则模型看到的是策略超时，而不是工具更具体的超时说明。
TIMEOUT_GRACE_SECONDS = 15.0

_RETRY_DELAY_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Execution policy for one tool.

    ``max_output_chars`` bounds the raw result kept in ``artifact['raw']``;
    the model-visible content is clipped by the caller's ``result_formatter``
    (the general agent uses a head+tail budget) or by this limit.
    """

    timeout: float | None = 60.0
    max_output_chars: int | None = DEFAULT_MAX_OUTPUT_CHARS
    retryable: bool = False
    max_retries: int = 1
    risk_level: RiskLevel = "low"
    concurrent: bool = True
    #: 同一轮里「同名 + 同参数」的重复调用是否直接跳过（只读检索类工具打开）。
    dedupe: bool = False

    def __post_init__(self) -> None:
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("ToolPolicy.timeout 必须是正数或 None")
        if self.max_output_chars is not None and self.max_output_chars <= 0:
            raise ValueError("ToolPolicy.max_output_chars 必须是正整数或 None")
        if self.max_retries < 0:
            raise ValueError("ToolPolicy.max_retries 不能为负数")
        if self.risk_level not in ("low", "medium", "high"):
            raise ValueError(f"未知的工具风险等级：{self.risk_level!r}")


@dataclass(frozen=True, slots=True)
class ToolPolicies:
    """Name → policy registry with a default for unknown / custom tools."""

    default: ToolPolicy
    overrides: Mapping[str, ToolPolicy] = field(default_factory=dict)
    parallel: bool = True

    def get(self, name: str) -> ToolPolicy:
        """Policy for ``name``; unknown tools fall back to the default."""
        return self.overrides.get(name, self.default)

    def describe(self, name: str) -> ToolPolicyView:
        """JSON friendly view of one policy (used by ``GET /overview``)."""
        policy = self.get(name)
        return ToolPolicyView(
            timeout_seconds=policy.timeout,
            retryable=policy.retryable,
            max_retries=policy.max_retries,
            risk_level=policy.risk_level,
            concurrent=policy.concurrent and self.parallel,
            dedupe=policy.dedupe,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        overrides: Mapping[str, ToolPolicy] | None = None,
    ) -> ToolPolicies:
        """Build the registry for the built-in tool set.

        ``settings`` controls the global timeout/retry defaults and the network
        tool deadlines; per-tool overrides can still be injected for tests or
        custom tool sets.
        """
        cfg = settings or get_settings()
        timeout = cfg.agent_tool_timeout
        retries = cfg.agent_tool_retries

        default = ToolPolicy(
            timeout=timeout,
            max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
            max_retries=retries,
        )
        policies: dict[str, ToolPolicy] = {
            # 检索是只读 + 网络调用：可重试，超时给短一点；同一轮重复同样的查询直接跳过。
            KNOWLEDGE_TOOL_NAME: ToolPolicy(
                timeout=timeout,
                max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
                retryable=True,
                max_retries=retries,
                dedupe=True,
            ),
            READ_TOOL_NAME: ToolPolicy(
                timeout=timeout,
                max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
            ),
            # 写文件不可重试：失败重放可能把半截内容写两遍；并发也串行化，
            # 避免同一轮两次 write 同时改同一个文件。
            WRITE_TOOL_NAME: ToolPolicy(
                timeout=timeout,
                max_output_chars=20_000,
                risk_level="medium",
                concurrent=False,
            ),
            EDIT_TOOL_NAME: ToolPolicy(
                timeout=timeout,
                max_output_chars=20_000,
                risk_level="medium",
                concurrent=False,
            ),
            GLOB_TOOL_NAME: ToolPolicy(
                timeout=timeout,
                max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
            ),
            GREP_TOOL_NAME: ToolPolicy(
                timeout=timeout,
                max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
            ),
            # shell 的截止时间由工具参数控制（subprocess timeout），这里不再叠加；
            # 命令可能互相影响（同一目录的构建 / git），同一轮串行执行。
            SHELL_TOOL_NAME: ToolPolicy(
                timeout=None,
                max_output_chars=100_000,
                risk_level="high",
                concurrent=False,
            ),
            WEB_SEARCH_TOOL_NAME: ToolPolicy(
                timeout=cfg.web_search_timeout + TIMEOUT_GRACE_SECONDS,
                max_output_chars=100_000,
                retryable=True,
                max_retries=retries,
                dedupe=True,
            ),
            WEB_EXTRACT_TOOL_NAME: ToolPolicy(
                timeout=cfg.web_extractor_timeout + TIMEOUT_GRACE_SECONDS,
                max_output_chars=100_000,
                retryable=True,
                max_retries=retries,
                dedupe=True,
            ),
        }
        if overrides:
            policies.update(overrides)
        return cls(default=default, overrides=policies, parallel=cfg.agent_tool_parallel)


# ---- 重试判定 -------------------------------------------------------------
_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
    httpx.HTTPError,
    MilvusError,
    RerankError,
    DatabaseError,
)

#: ``错误：...`` 文本里出现这些片段时视为「临时性失败」，值得重试一次。
#: 用 ``http 5xx`` / ``http 429`` 这种带前缀的形式，避免把金额、行号当成状态码。
_TRANSIENT_HINTS: tuple[str, ...] = (
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "status 429",
    "too many requests",
    "rate limit",
    "timed out",
    "timeout",
    "超时",
    "connection reset",
    "connection refused",
    "connection aborted",
    "连接失败",
    "网络错误",
    "限流",
    "暂时不可用",
)

_ERROR_PREFIXES = ("错误：", "工具执行失败", "工具执行超时", "工具参数不合法")


def is_retryable_exception(exc: BaseException) -> bool:
    """True when ``exc`` looks transient and the call is safe to replay."""
    if isinstance(exc, ToolError):
        # ToolError 是「预期内的失败」：参数错、文件不存在、Key 没配。
        return False
    return isinstance(exc, _RETRYABLE_EXCEPTIONS)


def is_failure_result(text: str) -> bool:
    """True when a tool result string is a known failure (not a normal answer)."""
    return text.lstrip().startswith(_ERROR_PREFIXES)


def is_transient_result(text: str) -> bool:
    """True when a tool's ``错误：...`` result looks like a transient failure.

    Tools catch their own ``ToolError`` and return a string (so a bad path never
    breaks the graph); this is how the node recognises the subset that a retry
    can plausibly fix.
    """
    if not is_failure_result(text):
        return False
    lowered = text.lstrip().lower()
    return any(hint in lowered for hint in _TRANSIENT_HINTS)


def retry_delay(attempt: int) -> float:
    """Small linear backoff between attempts (capped at 2s)."""
    return min(_RETRY_DELAY_SECONDS * attempt, 2.0)
