"""工具层标准化（A-01）：策略注册表、并行、超时、重试、参数校验与 artifact。

全部使用假工具 / 临时目录，不联网、不连 Milvus。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

from ai_agent_py.config import Settings
from ai_agent_py.errors import ToolError
from ai_agent_py.general.agent import build_general_tools
from ai_agent_py.graph.nodes import build_tool_node
from ai_agent_py.tools.filesystem import MAX_READ_LINES, build_filesystem_tools
from ai_agent_py.tools.policy import (
    ToolPolicies,
    ToolPolicy,
    is_retryable_exception,
    is_transient_result,
)
from ai_agent_py.tools.shell import MAX_TIMEOUT


class FakeSearcher:
    """知识库检索器的空实现（只用于装配工具清单）。"""

    def search(self, query: str, *, k: int | None = None) -> list[Any]:
        return []


@tool("sleepy", description="睡一会儿再返回，用来测超时与并行")
def sleepy(seconds: float = 0.05) -> str:
    """Sleep for the given number of seconds."""
    time.sleep(seconds)
    return f"slept {seconds}"


def tool_calls_message(calls: list[tuple[str, dict[str, Any], str]]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
            for name, args, call_id in calls
        ],
    )


def run_tool_node(
    tools: list[BaseTool],
    calls: list[tuple[str, dict[str, Any], str]],
    *,
    policies: ToolPolicies,
    parallel: bool | None = None,
) -> list[Any]:
    node = build_tool_node(tools, policies=policies, parallel=parallel)
    state = {"messages": [HumanMessage(content="go"), tool_calls_message(calls)]}
    return list(node(state)["messages"])


def single_policy(**kwargs: Any) -> ToolPolicies:
    return ToolPolicies(default=ToolPolicy(**kwargs))


# ---- 策略注册表 -----------------------------------------------------------
def test_registry_covers_every_builtin_tool() -> None:
    settings = Settings(_env_file=None)
    policies = ToolPolicies.from_settings(settings)
    names = {item.name for item in build_general_tools(settings, searcher=FakeSearcher())}

    assert names <= set(policies.overrides)
    # 未知 / 自定义工具退回默认策略，不会因为没有登记就失去超时保护。
    assert policies.get("my-custom-tool") == policies.default


def test_registry_follows_settings() -> None:
    settings = Settings(
        _env_file=None,
        agent_tool_timeout=7.0,
        agent_tool_retries=3,
        agent_tool_parallel=False,
    )
    policies = ToolPolicies.from_settings(settings)

    assert policies.get("read").timeout == 7.0
    assert policies.get("search_knowledge_base").retryable is True
    assert policies.get("search_knowledge_base").max_retries == 3
    assert policies.get("web_search").timeout == settings.web_search_timeout + 15.0
    # 写文件不可重试且串行；shell 的截止时间交给命令参数。
    assert policies.get("write").concurrent is False
    assert policies.get("edit").risk_level == "medium"
    assert policies.get("shell").timeout is None
    assert policies.get("shell").risk_level == "high"
    # 全局关闭并行后，概览里的 concurrent 也应为 False。
    assert policies.describe("read")["concurrent"] is False
    assert policies.describe("shell")["timeout_seconds"] is None


# ---- 并行与串行 -----------------------------------------------------------
def test_parallel_calls_take_max_time_not_sum() -> None:
    calls = [("sleepy", {"seconds": 0.3}, f"c{index}") for index in range(3)]
    started = time.perf_counter()
    messages = run_tool_node([sleepy], calls, policies=single_policy(timeout=5.0))
    elapsed = time.perf_counter() - started

    assert [message.tool_call_id for message in messages] == ["c0", "c1", "c2"]
    assert all(message.content == "slept 0.3" for message in messages)
    assert elapsed < 0.75, f"三个 0.3s 工具并行应接近 0.3s，实际 {elapsed:.2f}s"


def test_parallel_can_be_switched_off() -> None:
    started = time.perf_counter()
    run_tool_node(
        [sleepy],
        [("sleepy", {"seconds": 0.2}, "c0"), ("sleepy", {"seconds": 0.2}, "c1")],
        policies=single_policy(timeout=5.0, concurrent=False),
    )
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.35, f"关闭并行后应按顺序执行（>=0.4s），实际 {elapsed:.2f}s"


# ---- 超时 -----------------------------------------------------------------
def test_slow_tool_times_out_without_hanging_the_node() -> None:
    started = time.perf_counter()
    messages = run_tool_node(
        [sleepy],
        [("sleepy", {"seconds": 3}, "c1")],
        policies=single_policy(timeout=0.2),
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, "工具卡住时必须按策略超时返回，不能挂住整张图"
    assert "工具执行超时" in messages[0].content
    assert "0.2" in messages[0].content
    assert messages[0].artifact["status"] == "timeout"
    assert messages[0].artifact["error_code"] == "timeout"


# ---- 重试 -----------------------------------------------------------------
def test_retryable_tool_retries_transient_exception() -> None:
    attempts: list[int] = []

    @tool("flaky", description="第一次抛网络异常，第二次成功")
    def flaky() -> str:
        """Fail once, then succeed."""
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("connection reset by peer")
        return "recovered"

    messages = run_tool_node(
        [flaky],
        [("flaky", {}, "c1")],
        policies=single_policy(timeout=5.0, retryable=True, max_retries=1),
    )

    assert messages[0].content == "recovered"
    assert messages[0].artifact["attempts"] == 2
    assert len(attempts) == 2


def test_non_retryable_tool_fails_fast() -> None:
    attempts: list[int] = []

    @tool("flaky", description="第一次抛异常，第二次成功")
    def flaky() -> str:
        """Fail once, then succeed."""
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("connection reset by peer")
        return "recovered"

    messages = run_tool_node(
        [flaky],
        [("flaky", {}, "c1")],
        policies=single_policy(timeout=5.0, retryable=False),
    )

    assert messages[0].content.startswith("工具执行失败")
    assert messages[0].artifact["status"] == "error"
    assert messages[0].artifact["attempts"] == 1
    assert len(attempts) == 1


def test_transient_error_text_is_retried() -> None:
    results = ["错误：请求失败（HTTP 503）：服务暂时不可用", "正常结果"]

    @tool("flaky_text", description="先返回 503 文本，再返回正常结果")
    def flaky_text() -> str:
        """Return one canned result per call."""
        return results.pop(0)

    messages = run_tool_node(
        [flaky_text],
        [("flaky_text", {}, "c1")],
        policies=single_policy(timeout=5.0, retryable=True, max_retries=1),
    )

    assert messages[0].content == "正常结果"
    assert messages[0].artifact["attempts"] == 2


def test_retry_helpers_separate_transient_from_expected_errors() -> None:
    assert is_retryable_exception(ConnectionError("reset"))
    assert is_retryable_exception(TimeoutError("slow"))
    assert not is_retryable_exception(ToolError("文件不存在"))
    assert not is_retryable_exception(ValueError("bad"))

    assert is_transient_result("错误：请求失败（HTTP 503）：服务不可用")
    assert is_transient_result("错误：连接失败，请稍后重试")
    assert not is_transient_result("错误：文件不存在")
    assert not is_transient_result("住宿标准 503 元")


# ---- 参数校验 -------------------------------------------------------------
def test_invalid_arguments_become_a_correctable_message(tmp_path: Path) -> None:
    tools = build_filesystem_tools(workspace=tmp_path)
    messages = run_tool_node(
        tools,
        [("read", {"path": "a.txt", "limit": MAX_READ_LINES + 1}, "c1")],
        policies=single_policy(timeout=5.0),
    )
    content = messages[0].content

    assert content.startswith("工具参数不合法（read）")
    assert "limit" in content
    assert "修正参数" in content
    assert messages[0].artifact["status"] == "invalid_args"
    assert messages[0].artifact["error_code"] == "invalid_args"


def test_tool_schemas_expose_enum_bounds(tmp_path: Path) -> None:
    filesystem = {item.name: item for item in build_filesystem_tools(workspace=tmp_path)}
    read_schema = filesystem["read"].args
    assert read_schema["limit"]["minimum"] == 1
    assert read_schema["limit"]["maximum"] == MAX_READ_LINES

    general = {item.name: item for item in build_general_tools(Settings(_env_file=None))}
    # 可空数值在 JSON Schema 里是 anyOf[number, null]，约束挂在 number 分支上。
    timeout_schema = general["shell"].args["timeout"]["anyOf"][0]
    assert timeout_schema["maximum"] == MAX_TIMEOUT
    assert timeout_schema["exclusiveMinimum"] == 0
    assert general["search_knowledge_base"].args["query"]["minLength"] == 1


# ---- artifact / 输出上限 ---------------------------------------------------
def test_artifact_records_status_and_elapsed() -> None:
    messages = run_tool_node(
        [sleepy],
        [("sleepy", {"seconds": 0.05}, "c1")],
        policies=single_policy(timeout=5.0),
    )
    artifact = messages[0].artifact

    assert artifact["status"] == "ok"
    assert artifact["error_code"] is None
    assert artifact["attempts"] == 1
    assert artifact["elapsed_ms"] >= 40
    assert artifact["raw"].startswith("slept")
    assert artifact["raw_chars"] == len(artifact["raw"])


def test_unknown_tool_is_reported_without_execution() -> None:
    messages = run_tool_node(
        [sleepy],
        [("nope", {}, "c1")],
        policies=single_policy(timeout=5.0),
    )

    assert "未知工具" in messages[0].content
    assert messages[0].artifact["status"] == "error"
    assert messages[0].artifact["error_code"] == "unknown_tool"


def test_policy_bounds_huge_results() -> None:
    payload = "x" * 500

    @tool("big", description="返回超长文本")
    def big() -> str:
        """Return a large string."""
        return payload

    messages = run_tool_node(
        [big],
        [("big", {}, "c1")],
        policies=single_policy(timeout=5.0, max_output_chars=100),
    )

    assert len(messages[0].content) < 200
    assert "省略" in messages[0].content
    assert messages[0].artifact["raw_truncated"] is True
