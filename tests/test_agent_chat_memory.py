"""通用 Agent 的 MySQL 会话记忆：内存实现 + 历史还原（不连数据库）。"""

from __future__ import annotations

from ai_agent_py.db.agent_chat import (
    DEFAULT_TITLE,
    ChatMessage,
    InMemoryChatMemoryStore,
    build_history,
    session_title,
)


def test_session_title_flattens_and_clips() -> None:
    assert session_title("  帮我   看看\n项目结构  ") == "帮我 看看 项目结构"
    assert session_title("   ") == DEFAULT_TITLE
    assert session_title("很长的问题" * 20).endswith("…")
    assert len(session_title("很长的问题" * 20)) == 41


def test_append_assigns_seq_and_only_first_title_wins() -> None:
    store = InMemoryChatMemoryStore()

    session = store.ensure_session("t-1")
    assert session.title == DEFAULT_TITLE
    assert session.message_count == 0

    first = store.append_messages(
        "t-1",
        [ChatMessage(role="user", content="帮我列一下项目里的 Python 文件")],
        title="帮我列一下项目里的 Python 文件",
    )
    assert first.message_count == 1
    assert first.title == "帮我列一下项目里的 Python 文件"

    second = store.append_messages(
        "t-1",
        [
            ChatMessage(
                role="tool",
                content="src/main.py",
                tool_name="glob",
                tool_call_id="c1",
                tool_args='{"pattern": "*.py"}',
                tool_status="ok",
                elapsed_ms=8,
            ),
            ChatMessage(role="assistant", content="找到 1 个 Python 文件：src/main.py。"),
        ],
        title="这句不该覆盖标题",
    )
    assert second.title == "帮我列一下项目里的 Python 文件"
    assert second.message_count == 3

    messages = store.load_messages("t-1")
    assert [message.seq for message in messages] == [1, 2, 3]
    assert [message.role for message in messages] == ["user", "tool", "assistant"]
    assert messages[1].tool_name == "glob"
    assert messages[1].tool_args == '{"pattern": "*.py"}'
    assert messages[1].elapsed_ms == 8
    assert all(message.created_at is not None for message in messages)


def test_build_history_keeps_only_user_and_assistant() -> None:
    messages = [
        ChatMessage(role="user", content="第一个问题", seq=1),
        ChatMessage(role="assistant", content="第一个回答", seq=2),
        ChatMessage(role="tool", content="工具输出", tool_name="read", seq=3),
        ChatMessage(role="assistant", content="  ", seq=4),
        ChatMessage(role="user", content="第二个问题", seq=5),
    ]

    history = build_history(messages)

    assert [type(item).__name__ for item in history] == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
    ]
    assert [item.content for item in history] == ["第一个问题", "第一个回答", "第二个问题"]


def test_load_messages_limit_returns_latest_window() -> None:
    store = InMemoryChatMemoryStore()
    for index in range(5):
        store.append_messages("t-2", [ChatMessage(role="user", content=f"问题 {index}")])

    messages = store.load_messages("t-2", limit=2)

    assert [message.seq for message in messages] == [4, 5]
    assert [message.content for message in messages] == ["问题 3", "问题 4"]


def test_sessions_are_listed_by_recent_update_and_deleted() -> None:
    store = InMemoryChatMemoryStore()
    store.append_messages("older", [ChatMessage(role="user", content="旧会话")], title="旧会话")
    store.append_messages("newer", [ChatMessage(role="user", content="新会话")], title="新会话")

    sessions = store.list_sessions()

    assert [session.thread_id for session in sessions] == ["newer", "older"]
    assert store.count_sessions() == 2
    assert store.delete_session("newer") is True
    assert store.get_session("newer") is None
    assert store.load_messages("newer") == []
    assert store.count_sessions() == 1
    assert store.delete_session("missing") is False


def test_sessions_can_be_filtered_by_agent() -> None:
    store = InMemoryChatMemoryStore()
    store.append_messages(
        "general-1", [ChatMessage(role="user", content="通用")], agent_id="general"
    )
    store.append_messages(
        "reimburse-1", [ChatMessage(role="user", content="报销")], agent_id="reimburse"
    )

    assert store.count_sessions(agent_id="general") == 1
    assert store.count_sessions(agent_id=None) == 2
    assert [item.thread_id for item in store.list_sessions(agent_id="reimburse")] == ["reimburse-1"]
