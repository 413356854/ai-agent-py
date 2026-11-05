"""通用 Agent：工具清单与 LangGraph 图（脚本化模型，不联网）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ai_agent_py.config import Settings
from ai_agent_py.general.agent import (
    GENERAL_AGENT_SYSTEM_PROMPT,
    build_general_agent_graph,
    build_general_tools,
    describe_tools,
)
from ai_agent_py.general.context import TOOL_RESULT_PLACEHOLDER
from fakes import ScriptedChatModel

EXPECTED_TOOLS = [
    "search_knowledge_base",
    "read",
    "write",
    "edit",
    "glob",
    "grep",
    "shell",
    "web_search",
    "web_extract",
]


class FakeSearcher:
    """假的知识库检索器：返回一条带来源的片段。"""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, k: int | None = None) -> list[Document]:
        self.queries.append(query)
        return [
            Document(
                page_content="住宿费按城市分级：一线城市每晚不超过 500 元。",
                metadata={"doc_id": "doc-a", "chunk_id": "chunk-1", "page": 2},
            )
        ]


def tool_call_message(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    """一条请求调用工具的 AI 消息。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
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


def test_general_tools_cover_every_group(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    tools = build_general_tools(settings, searcher=FakeSearcher())

    assert [tool.name for tool in tools] == EXPECTED_TOOLS
    metas = describe_tools(tools)
    assert [meta.group for meta in metas] == [
        "knowledge",
        "filesystem",
        "filesystem",
        "filesystem",
        "filesystem",
        "filesystem",
        "shell",
        "web",
        "web",
    ]
    assert {meta.group_label for meta in metas} == {"知识库", "文件", "命令", "联网"}
    assert all(meta.description for meta in metas)


def test_general_tools_can_drop_shell_only(tmp_path: Path) -> None:
    tools = build_general_tools(
        make_settings(tmp_path), searcher=FakeSearcher(), include_shell=False
    )

    names = [tool.name for tool in tools]
    assert "shell" not in names
    assert names[-2:] == ["web_search", "web_extract"]
    assert len(names) == len(EXPECTED_TOOLS) - 1


def test_general_graph_runs_tool_calls_and_answers(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# 笔记\n\n差旅住宿按城市分级。", encoding="utf-8")
    searcher = FakeSearcher()
    model = ScriptedChatModel(
        responses=[
            tool_call_message("search_knowledge_base", {"query": "住宿费标准"}),
            AIMessage(content="依据知识库 [1]，一线城市住宿上限 500 元。"),
        ]
    )
    settings = make_settings(tmp_path)
    tools = build_general_tools(settings, searcher=searcher)
    graph = build_general_agent_graph(llm=model, tools=tools, settings=settings)

    result = graph.invoke({"messages": [HumanMessage(content="一线城市住宿标准是多少？")]})

    messages = result["messages"]
    assert [message.type for message in messages] == ["human", "ai", "tool", "ai"]
    assert messages[1].tool_calls[0]["name"] == "search_knowledge_base"
    assert "500 元" in messages[2].content
    assert messages[-1].content.startswith("依据知识库")
    assert searcher.queries == ["住宿费标准"]
    # 系统提示词里列出了全部工具，模型知道有哪些工具可用
    system = model.received[0][0]
    assert isinstance(system, SystemMessage)
    assert system.content == GENERAL_AGENT_SYSTEM_PROMPT
    for tool in tools:
        assert tool.name in system.content


def test_general_graph_can_read_workspace_files(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# 笔记\n\n差旅住宿按城市分级。", encoding="utf-8")
    model = ScriptedChatModel(
        responses=[
            tool_call_message("read", {"path": "notes.md"}),
            AIMessage(content="文件里写着：差旅住宿按城市分级。"),
        ]
    )
    settings = make_settings(tmp_path)
    graph = build_general_agent_graph(
        llm=model,
        tools=build_general_tools(settings, searcher=FakeSearcher()),
        settings=settings,
    )

    result = graph.invoke({"messages": [HumanMessage(content="读一下 notes.md")]})

    tool_message = result["messages"][2]
    assert "差旅住宿按城市分级" in tool_message.content
    assert "1→# 笔记" in tool_message.content
    assert result["messages"][-1].content == "文件里写着：差旅住宿按城市分级。"


def test_tool_result_is_clipped_for_model_but_kept_in_artifact(tmp_path: Path) -> None:
    large = "开始\n" + "x" * 20_000 + "\n结束"
    (tmp_path / "large.txt").write_text(large, encoding="utf-8")
    model = ScriptedChatModel(
        responses=[
            tool_call_message("read", {"path": "large.txt"}),
            AIMessage(content="读完了。"),
        ]
    )
    settings = make_settings(
        tmp_path,
        agent_tool_result_max_chars=400,
        agent_tool_result_head_chars=200,
        agent_tool_result_tail_chars=80,
    )
    graph = build_general_agent_graph(
        llm=model,
        tools=build_general_tools(settings, searcher=FakeSearcher()),
        settings=settings,
    )

    result = graph.invoke({"messages": [HumanMessage(content="读 large.txt")]})

    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert len(tool_message.content) <= 400
    assert "已省略" in tool_message.content
    raw = tool_message.artifact["raw"]
    assert raw.startswith("large.txt")
    assert len(raw) > 400
    assert "开始" in raw and "结束" in raw
    assert tool_message.artifact["truncated"] is True


def test_turn_tool_budget_replaces_older_results_before_model_call(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A" * 1000, encoding="utf-8")
    (tmp_path / "b.txt").write_text("B" * 1000, encoding="utf-8")
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read",
                        "args": {"path": "a.txt"},
                        "id": "c1",
                        "type": "tool_call",
                    },
                    {
                        "name": "read",
                        "args": {"path": "b.txt"},
                        "id": "c2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="两个文件都读完了。"),
        ]
    )
    settings = make_settings(
        tmp_path,
        agent_tool_result_max_chars=400,
        agent_tool_result_head_chars=200,
        agent_tool_result_tail_chars=80,
        agent_tool_turn_max_chars=450,
    )
    graph = build_general_agent_graph(
        llm=model,
        tools=build_general_tools(settings, searcher=FakeSearcher()),
        settings=settings,
    )

    result = graph.invoke({"messages": [HumanMessage(content="读两个文件")]})

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 2
    assert tool_messages[0].artifact["raw"]
    # 第二次模型调用的 prompt 里，较早的工具结果已被占位符替换。
    prompt = model.received[-1]
    prompt_tools = [message for message in prompt if isinstance(message, ToolMessage)]
    assert len(prompt_tools) == 2
    assert prompt_tools[0].content == TOOL_RESULT_PLACEHOLDER
    assert prompt_tools[1].content.startswith("b.txt")
    # 原始消息对象未被修改，落库/界面仍然能拿到完整结果。
    assert len(tool_messages[0].content) <= 400
