"""通用 Agent 上下文预算：token 估算、历史选择、工具输出修剪（纯函数，不连库）。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_agent_py.config import Settings
from ai_agent_py.db.agent_chat import ChatMessage
from ai_agent_py.general.agent import GENERAL_AGENT_SYSTEM_PROMPT, build_general_tools
from ai_agent_py.general.context import (
    TOOL_RESULT_PLACEHOLDER,
    build_history_budget,
    build_summarization_transcript,
    clip_text,
    clip_tool_result,
    compress_tool_messages,
    estimate_records_tokens,
    estimate_text_tokens,
    estimate_tools_tokens,
    select_history,
)


class EmptySearcher:
    def search(self, query: str, *, k: int | None = None):
        return []


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "app_env": "test",
        "agent_workspace": tmp_path,
        "agent_token_chars": 2.0,
        "agent_model_context_tokens": 8192,
        "agent_context_max_tokens": 0,
        "agent_output_reserve_tokens": 512,
        "agent_context_safety_tokens": 128,
        "agent_history_keep_messages": 4,
        "agent_message_max_chars": 1000,
    }
    values.update(overrides)
    return Settings(**values)


def records(count: int, *, chars: int = 100) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="user" if index % 2 == 0 else "assistant",
            content=f"第{index}条：" + "内容" * (chars // 2),
            seq=index + 1,
        )
        for index in range(count)
    ]


def test_token_estimates_follow_chinese_density(tmp_path: Path) -> None:
    short = estimate_text_tokens("你好", chars_per_token=2.0)
    long = estimate_text_tokens("你好" * 100, chars_per_token=2.0)
    assert 0 < short < long
    assert long >= 100

    tools = build_general_tools(make_settings(tmp_path), searcher=EmptySearcher())
    assert estimate_tools_tokens(tools, chars_per_token=2.0) > 0


def test_history_budget_subtracts_system_tools_and_output(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, agent_model_context_tokens=4096)
    tools = build_general_tools(settings, searcher=EmptySearcher())

    budget = build_history_budget(
        settings,
        tools,
        system_prompt=GENERAL_AGENT_SYSTEM_PROMPT,
    )

    assert budget >= 512
    assert budget < 4096
    capped = build_history_budget(
        make_settings(tmp_path, agent_context_max_tokens=700),
        tools,
        system_prompt=GENERAL_AGENT_SYSTEM_PROMPT,
    )
    assert capped == 700


def test_clip_text_respects_budget_and_keeps_head_and_tail() -> None:
    text = "HEAD" + "x" * 5000 + "TAIL"
    clipped = clip_text(text, max_chars=200, head_chars=80, tail_chars=40)
    assert len(clipped) <= 200
    assert clipped.startswith("HEAD")
    assert clipped.endswith("TAIL")
    assert "省略" in clipped
    assert clip_text("short", max_chars=200, head_chars=80, tail_chars=40) == "short"


def test_clip_tool_result_uses_settings(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    clipped = clip_tool_result("a" * 20_000, settings)
    assert len(clipped) <= settings.agent_tool_result_max_chars
    assert "工具结果" in clipped


def test_select_history_keeps_recent_tail_within_budget(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, agent_history_keep_messages=2)
    rows = records(20)

    selection = select_history(rows, settings=settings, budget_tokens=260)

    assert selection.used >= 2
    assert selection.used < len(rows)
    assert selection.dropped == len(rows) - selection.used
    assert selection.estimated_tokens <= 260
    assert isinstance(selection.messages[0], HumanMessage)
    assert isinstance(selection.messages[-1], AIMessage)


def test_select_history_always_keeps_latest_user_and_starts_with_user(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, agent_history_keep_messages=1)
    rows = records(5)
    rows.append(ChatMessage(role="user", content="最新问题" * 100, seq=6))

    selection = select_history(rows, settings=settings, budget_tokens=1000)

    assert selection.messages
    assert isinstance(selection.messages[0], HumanMessage)
    assert selection.messages[-1].content.startswith("最新问题")
    assert selection.estimated_tokens <= 1000


def test_compress_tool_messages_replaces_old_results_but_keeps_pairing() -> None:
    messages = [
        ToolMessage(content="A" * 100, tool_call_id="a", name="read"),
        ToolMessage(content="B" * 100, tool_call_id="b", name="read"),
        ToolMessage(content="C" * 100, tool_call_id="c", name="read"),
        ToolMessage(content="D" * 100, tool_call_id="d", name="read"),
    ]

    result = compress_tool_messages(messages, max_total_chars=150, keep_recent_results=1)

    assert result.trimmed >= 2
    assert result.after_chars < result.before_chars
    assert all(isinstance(message, ToolMessage) for message in result.messages)
    assert [message.tool_call_id for message in result.messages] == ["a", "b", "c", "d"]
    assert result.messages[-1].content == "D" * 100
    assert result.messages[0].content == TOOL_RESULT_PLACEHOLDER


def test_estimate_records_skips_tool_rows_and_transcript_has_roles(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    rows = [
        ChatMessage(role="user", content="问题", seq=1),
        ChatMessage(role="tool", content="工具结果", seq=2),
        ChatMessage(role="assistant", content="回答", seq=3),
    ]

    tokens = estimate_records_tokens(rows, settings=settings)
    transcript = build_summarization_transcript(rows, max_chars=1000)

    assert tokens > 0
    assert "用户：问题" in transcript
    assert "助手：回答" in transcript
    assert "工具结果" not in transcript
