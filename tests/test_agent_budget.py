"""A-02 循环预算：用量统计、预算路由、finalize 与重复调用跳过。

全部离线：脚本化模型 + 假检索器 + 临时工作目录，不连 MySQL / Milvus / 真实模型。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from ai_agent_py.config import Settings
from ai_agent_py.general.agent import build_general_agent_graph, build_general_tools
from ai_agent_py.graph.agent import agent_config, ask_agent, build_agent_graph
from ai_agent_py.graph.budget import LoopBudget, measure_turn, turn_messages
from ai_agent_py.graph.nodes import (
    NO_FINAL_ANSWER,
    build_route_after_agent,
    route_after_agent,
)
from ai_agent_py.tools.policy import ToolPolicies
from fakes import ScriptedChatModel


class FakeSearcher:
    """假检索器：记录查询，返回一条固定片段。"""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, k: int | None = None) -> list[Document]:
        self.queries.append(query)
        return [
            Document(
                page_content=f"检索结果：{query} 的答案在这里。",
                metadata={"doc_id": "doc-a", "chunk_id": "chunk-1"},
            )
        ]


def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def tool_message(call_id: str, name: str = "search_knowledge_base", status: str = "ok") -> ToolMessage:
    return ToolMessage(
        content="工具结果",
        tool_call_id=call_id,
        name=name,
        artifact={"raw": "工具结果", "status": status},
    )


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "app_env": "test",
        "agent_workspace": tmp_path,
        "agent_max_turns": 4,
    }
    values.update(overrides)
    return Settings(**values)


def build_graph(
    tmp_path: Path,
    model: ScriptedChatModel,
    *,
    searcher: FakeSearcher | None = None,
    **overrides: Any,
) -> Any:
    settings = make_settings(tmp_path, **overrides)
    tools = build_general_tools(settings, searcher=searcher or FakeSearcher())
    return build_general_agent_graph(llm=model, tools=tools, settings=settings)


# ---- 用量统计 -------------------------------------------------------------
def test_measure_turn_only_counts_the_current_turn() -> None:
    messages = [
        HumanMessage(content="上一轮"),
        AIMessage(content="上一轮回答"),
        ToolMessage(content="旧工具结果", tool_call_id="old", name="read"),
        HumanMessage(content="这一轮"),
        tool_call("search_knowledge_base", {"query": "q"}, "c1"),
        tool_message("c1"),
        AIMessage(
            content="这一轮回答",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        ),
    ]

    usage = measure_turn(messages, elapsed_seconds=2.5)

    assert (usage.steps, usage.tool_calls, usage.tokens) == (2, 1, 15)
    assert usage.elapsed_seconds == 2.5


def test_measure_turn_counts_skipped_and_usage_from_response_metadata() -> None:
    messages = [
        HumanMessage(content="这一轮"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read", "args": {"path": "a.txt"}, "id": "c1", "type": "tool_call"}],
            response_metadata={"token_usage": {"total_tokens": 42}},
        ),
        tool_message("c1", "read", status="skipped"),
        tool_message("c2", "read", status="ok"),
    ]

    usage = measure_turn(messages)

    assert usage.tool_calls == 1
    assert usage.skipped_calls == 1
    assert usage.tokens == 42


def test_turn_messages_falls_back_to_all_messages() -> None:
    messages = [AIMessage(content="没有用户消息")]
    assert turn_messages(messages) == messages


def test_budget_describe_matches_settings(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, agent_max_tool_calls=7, agent_max_tokens=1000)
    budget = LoopBudget.from_settings(settings)

    assert budget.describe() == {
        "max_steps": 4,
        "max_tool_calls": 7,
        "max_tokens": 1000,
        "max_seconds": 600.0,
    }


# ---- 路由 -----------------------------------------------------------------
def test_default_router_keeps_old_behaviour() -> None:
    assert route_after_agent({"messages": [HumanMessage(content="hi")]}) == "end"
    assert route_after_agent({"messages": [AIMessage(content="直接回答")]}) == "end"
    assert (
        route_after_agent({"messages": [tool_call("search_knowledge_base", {"query": "uv"})]})
        == "tools"
    )


def test_router_finalizes_on_step_budget() -> None:
    route = build_route_after_agent(LoopBudget(max_steps=2, max_tool_calls=10, max_seconds=0))
    state = {
        "messages": [
            HumanMessage(content="问题"),
            tool_call("search_knowledge_base", {"query": "a"}, "c1"),
            tool_message("c1"),
            tool_call("search_knowledge_base", {"query": "b"}, "c2"),
        ]
    }

    assert route(state) == "finalize"


def test_router_finalizes_on_tool_call_budget() -> None:
    route = build_route_after_agent(LoopBudget(max_steps=10, max_tool_calls=1, max_seconds=0))
    state = {
        "messages": [
            HumanMessage(content="问题"),
            tool_call("search_knowledge_base", {"query": "a"}, "c1"),
            tool_message("c1"),
            tool_call("search_knowledge_base", {"query": "b"}, "c2"),
        ]
    }

    assert route(state) == "finalize"


def test_router_finalizes_when_time_budget_is_gone() -> None:
    route = build_route_after_agent(LoopBudget(max_steps=10, max_tool_calls=10, max_seconds=5))
    state = {
        "messages": [HumanMessage(content="问题"), tool_call("search_knowledge_base", {"query": "a"})],
        "started_at": time.monotonic() - 30,
    }

    assert route(state) == "finalize"


def test_router_finalizes_on_token_budget() -> None:
    route = build_route_after_agent(
        LoopBudget(max_steps=10, max_tool_calls=10, max_seconds=0, max_tokens=10)
    )
    state = {
        "messages": [
            HumanMessage(content="问题"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_knowledge_base", "args": {"query": "a"}, "id": "c1"}
                ],
                usage_metadata={"input_tokens": 8, "output_tokens": 8, "total_tokens": 16},
            ),
        ]
    }

    assert route(state) == "finalize"


def test_skipped_calls_do_not_consume_the_tool_budget() -> None:
    budget = LoopBudget(max_steps=10, max_tool_calls=2, max_seconds=0)
    messages = [
        HumanMessage(content="问题"),
        tool_call("read", {"path": "a.txt"}, "c1"),
        tool_message("c1", "read"),
        tool_message("c2", "read", status="skipped"),
        tool_message("c3", "read", status="skipped"),
        tool_call("read", {"path": "b.txt"}, "c4"),
    ]

    usage = measure_turn(messages)

    assert (usage.tool_calls, usage.skipped_calls) == (1, 2)
    assert budget.check({"messages": messages}) is None


# ---- 图：finalize ---------------------------------------------------------
def test_general_graph_records_the_turn_start(tmp_path: Path) -> None:
    """时间预算依赖 agent 节点每轮写入 started_at（通用 Agent 的自定义节点也要写）。"""
    model = ScriptedChatModel(responses=[AIMessage(content="直接回答")])
    graph = build_graph(tmp_path, model)

    result = graph.invoke({"messages": [HumanMessage(content="你好")]})

    assert isinstance(result.get("started_at"), float)


def test_general_graph_finalizes_when_the_time_budget_ran_out(tmp_path: Path) -> None:
    model = ScriptedChatModel(
        responses=[
            tool_call("search_knowledge_base", {"query": "q"}, "c1"),
            AIMessage(content="收尾答案"),
        ]
    )
    graph = build_graph(tmp_path, model, agent_max_turns=4, agent_max_seconds=0.000001)

    result = graph.invoke({"messages": [HumanMessage(content="问题")]})

    assert result["budget_reason"] == "max_seconds"
    assert result["messages"][-1].content == "收尾答案"


def test_general_graph_finalizes_when_step_budget_runs_out(tmp_path: Path) -> None:
    searcher = FakeSearcher()
    model = ScriptedChatModel(
        responses=[
            tool_call("search_knowledge_base", {"query": "住宿费"}, "c1"),
            tool_call("search_knowledge_base", {"query": "住宿费 标准"}, "c2"),
            AIMessage(content="基于已检索到的资料：一线城市住宿上限 500 元（信息可能不完整）。"),
        ]
    )
    graph = build_graph(tmp_path, model, searcher=searcher, agent_max_turns=2)

    result = graph.invoke({"messages": [HumanMessage(content="住宿费标准是多少？")]})

    assert result["budget_reason"] == "max_steps"
    assert [message.type for message in result["messages"]] == [
        "human",
        "ai",
        "tool",
        "ai",
        "ai",
    ]
    assert result["messages"][-1].content.startswith("基于已检索到的资料")
    # 预算用完后第二个工具调用没有执行（工具结果只有一条）。
    assert searcher.queries == ["住宿费"]
    assert model.calls == 3
    # finalize 的系统提示词明确要求「基于已有资料 + 说明信息可能不完整 + 不再调工具」。
    finalize_prompt = model.received[-1]
    finalize_system = finalize_prompt[0]
    assert isinstance(finalize_system, SystemMessage)
    assert "执行预算已用尽" in finalize_system.content
    assert "信息可能不完整" in finalize_system.content
    # 被预算拦下的 c2 在 prompt 里有占位 ToolMessage：tool_call 配对合法，不会 400。
    tool_messages = [item for item in finalize_prompt if isinstance(item, ToolMessage)]
    assert [item.tool_call_id for item in tool_messages] == ["c1", "c2"]
    assert "工具未执行" in tool_messages[-1].content


def test_general_graph_finalizes_on_tool_call_budget(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read", "args": {"path": "a.txt"}, "id": "c1"},
                    {"name": "read", "args": {"path": "b.txt"}, "id": "c2"},
                ],
            ),
            AIMessage(content="预算内的答案"),
        ]
    )
    graph = build_graph(tmp_path, model, agent_max_turns=4, agent_max_tool_calls=1)

    result = graph.invoke({"messages": [HumanMessage(content="读两个文件")]})

    tool_messages = [item for item in result["messages"] if isinstance(item, ToolMessage)]
    assert [item.artifact["status"] for item in tool_messages] == ["ok", "skipped"]
    assert "1→A" in tool_messages[0].content
    assert "工具调用预算已用尽" in tool_messages[1].content
    assert result["messages"][-1].content == "预算内的答案"
    assert result.get("budget_reason") is None


def test_finalize_falls_back_when_model_returns_nothing(tmp_path: Path) -> None:
    searcher = FakeSearcher()
    model = ScriptedChatModel(
        responses=[
            tool_call("search_knowledge_base", {"query": "住宿费"}, "c1"),
            tool_call("search_knowledge_base", {"query": "住宿费 标准"}, "c2"),
            AIMessage(content=""),
        ]
    )
    graph = build_graph(tmp_path, model, searcher=searcher, agent_max_turns=2)

    result = graph.invoke({"messages": [HumanMessage(content="住宿费标准是多少？")]})

    text = result["messages"][-1].content
    assert result["budget_reason"] == "max_steps"
    assert "上限" in text and "模型没有给出最终答案" in text
    # 兜底文案带上最后一次工具结果，至少不是空回答。
    assert "检索结果：住宿费" in text
    assert searcher.queries == ["住宿费"]


def test_finalize_keeps_the_compressed_summary_in_the_system_prompt(tmp_path: Path) -> None:
    """finalize 也要带上「更早对话摘要」，不能丢掉压缩过的历史背景。"""
    model = ScriptedChatModel(
        responses=[
            tool_call("search_knowledge_base", {"query": "住宿费"}, "c1"),
            AIMessage(content="收尾答案"),
        ]
    )
    graph = build_graph(tmp_path, model, agent_max_turns=1)

    graph.invoke(
        {
            "messages": [HumanMessage(content="继续")],
            "summary": "更早的结论：一线城市住宿上限 500 元。",
        }
    )

    system = model.received[-1][0]
    assert isinstance(system, SystemMessage)
    assert "更早的结论" in system.content
    assert "执行预算已用尽" in system.content


def test_finalize_uses_static_answer_when_model_fails(tmp_path: Path) -> None:
    model = ScriptedChatModel(
        responses=[tool_call("search_knowledge_base", {"query": "住宿费"}, "c1")],
        fail_at=2,
    )
    graph = build_graph(tmp_path, model, agent_max_turns=1)

    result = graph.invoke({"messages": [HumanMessage(content="住宿费标准是多少？")]})

    assert result["budget_reason"] == "max_steps"
    assert "上限" in result["messages"][-1].content
    # 失败被记录成可恢复的收尾答案，而不是让整张图抛异常。
    assert result["messages"][-1].type == "ai"


def test_knowledge_agent_graph_finalizes_too(tmp_path: Path) -> None:
    """知识库 Agent 复用同一套预算 / finalize（不是只有通用 Agent 有）。"""
    searcher = FakeSearcher()
    model = ScriptedChatModel(
        responses=[
            tool_call("search_knowledge_base", {"query": "uv 依赖"}, "c1"),
            tool_call("search_knowledge_base", {"query": "uv 依赖管理"}, "c2"),
            AIMessage(content="收尾答案：使用 uv 管理依赖（信息可能不完整）。"),
        ]
    )
    settings = make_settings(tmp_path, agent_max_turns=2)
    graph = build_agent_graph(llm=model, searcher=searcher, settings=settings)

    answer = ask_agent(graph, "这个项目用什么管理依赖？", settings=settings)

    assert answer.startswith("收尾答案")
    assert searcher.queries == ["uv 依赖"]


def test_budget_resets_every_turn_with_a_checkpointer(tmp_path: Path) -> None:
    """带 checkpointer 时上一轮的步数不能累计到下一轮（预算按「本轮」算）。"""
    from langgraph.checkpoint.memory import InMemorySaver

    searcher = FakeSearcher()
    model = ScriptedChatModel(
        responses=[
            tool_call("search_knowledge_base", {"query": "第一问"}, "c1"),
            AIMessage(content="第一轮回答"),
            tool_call("search_knowledge_base", {"query": "第二问"}, "c2"),
            AIMessage(content="第二轮回答"),
        ]
    )
    settings = make_settings(tmp_path, agent_max_turns=2)
    graph = build_agent_graph(
        llm=model,
        searcher=searcher,
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    config = agent_config(thread_id="turns", settings=settings)

    first = graph.invoke({"messages": [HumanMessage(content="第一问")]}, config)
    second = graph.invoke({"messages": [HumanMessage(content="第二问")]}, config)

    assert first["messages"][-1].content == "第一轮回答"
    assert second["messages"][-1].content == "第二轮回答"
    assert second.get("budget_reason") is None
    assert searcher.queries == ["第一问", "第二问"]


# ---- 图：重复调用 ---------------------------------------------------------
def test_duplicate_retrieval_is_skipped(tmp_path: Path) -> None:
    searcher = FakeSearcher()
    model = ScriptedChatModel(
        responses=[
            tool_call("search_knowledge_base", {"query": "住宿费"}, "c1"),
            tool_call("search_knowledge_base", {"query": "住宿费"}, "c2"),
            AIMessage(content="答案：一线城市住宿上限 500 元。"),
        ]
    )
    graph = build_graph(tmp_path, model, searcher=searcher, agent_max_turns=4)

    result = graph.invoke({"messages": [HumanMessage(content="住宿费标准？")]})

    tool_messages = [item for item in result["messages"] if isinstance(item, ToolMessage)]
    assert [item.artifact["status"] for item in tool_messages] == ["ok", "skipped"]
    assert "重复调用已跳过" in tool_messages[1].content
    assert searcher.queries == ["住宿费"]
    assert result["messages"][-1].content.startswith("答案")
    # 跳过之后工具配对仍完整：模型看到的仍是「一次调用 + 一条结果」。
    assert [item.type for item in model.received[-1]].count("tool") == 2


def test_failed_call_is_not_treated_as_a_duplicate(tmp_path: Path) -> None:
    attempts: list[int] = []

    @tool("flaky", description="第一次失败、第二次成功的假工具")
    def flaky() -> str:
        """Return a failure once, then succeed."""
        attempts.append(1)
        return "错误：连接失败" if len(attempts) == 1 else "成功结果"

    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "flaky", "args": {}, "id": "c1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "flaky", "args": {}, "id": "c2"}],
            ),
            AIMessage(content="第二次成功了。"),
        ]
    )
    settings = make_settings(tmp_path, agent_max_turns=4)
    graph = build_general_agent_graph(llm=model, tools=[flaky], settings=settings)

    result = graph.invoke({"messages": [HumanMessage(content="试一下 flaky")]})

    tool_messages = [item for item in result["messages"] if isinstance(item, ToolMessage)]
    assert len(attempts) == 2
    assert tool_messages[0].content == "错误：连接失败"
    assert tool_messages[1].content == "成功结果"


# ---- 回退：ask_agent / 没有最终回答 ---------------------------------------
class _FixedGraph:
    """返回固定消息的假图（只实现 invoke）。"""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = messages

    def invoke(self, state: Any, config: Any = None) -> dict[str, Any]:
        return {"messages": self._messages}


class _RecursionGraph:
    def invoke(self, state: Any, config: Any = None) -> dict[str, Any]:
        raise GraphRecursionError("Recursion limit of 3 reached")


def test_ask_agent_does_not_return_a_tool_message() -> None:
    graph = _FixedGraph(
        [
            HumanMessage(content="问题"),
            tool_call("search_knowledge_base", {"query": "q"}),
            ToolMessage(content="工具输出的原文", tool_call_id="c1"),
        ]
    )

    assert ask_agent(graph, "问题") == NO_FINAL_ANSWER


def test_ask_agent_ignores_the_previous_turn_answer() -> None:
    graph = _FixedGraph(
        [
            HumanMessage(content="旧问题"),
            AIMessage(content="旧答案"),
            HumanMessage(content="新问题"),
            tool_call("search_knowledge_base", {"query": "q"}),
            ToolMessage(content="工具输出的原文", tool_call_id="c1"),
        ]
    )

    assert ask_agent(graph, "新问题") == NO_FINAL_ANSWER


def test_ask_agent_handles_recursion_error() -> None:
    assert ask_agent(_RecursionGraph(), "问题") == NO_FINAL_ANSWER


def test_agent_config_reserves_room_for_finalize() -> None:
    settings = Settings(_env_file=None, agent_max_turns=3)
    config = agent_config(settings=settings)

    assert config["recursion_limit"] >= 3 * 2 + 2


# ---- 策略登记 -------------------------------------------------------------
def test_dedupe_is_enabled_only_for_read_only_retrieval(tmp_path: Path) -> None:
    policies = ToolPolicies.from_settings(make_settings(tmp_path))

    assert policies.get("search_knowledge_base").dedupe is True
    assert policies.get("web_search").dedupe is True
    assert policies.get("web_extract").dedupe is True
    # 读文件 / 写文件 / shell 不能去重：改完文件再读一次是正常操作。
    for name in ("read", "write", "edit", "glob", "grep", "shell"):
        assert policies.get(name).dedupe is False
    assert policies.describe("search_knowledge_base")["dedupe"] is True
