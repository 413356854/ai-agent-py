"""通用 Agent HTTP 接口：概览 / MySQL 会话 / 同步对话 / SSE 流式。

全部使用内存版会话记忆 + 脚本化模型 + 假检索器，不连 MySQL / Milvus / 真实模型。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

import ai_agent_py.api.general_agent as api_general
from ai_agent_py.api.app import app as fastapi_app
from ai_agent_py.config import Settings
from ai_agent_py.db.agent_chat import ChatMessage, InMemoryChatMemoryStore
from ai_agent_py.errors import ConfigurationError, DatabaseError
from ai_agent_py.general.agent import build_general_agent_graph, build_general_tools
from ai_agent_py.graph.nodes import NO_FINAL_ANSWER
from fakes import ScriptedChatModel, StreamingScriptedChatModel

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
    def search(self, query: str, *, k: int | None = None) -> list[Document]:
        return [
            Document(
                page_content="住宿费按城市分级：一线城市每晚不超过 500 元。",
                metadata={"doc_id": "doc-a", "chunk_id": "chunk-1", "page": 2},
            )
        ]


def tool_call_message(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
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
        "agent_memory_candidates": 6,
        "mysql_url": "mysql+pymysql://admin:secret@localhost:3306/db_py_agent",
        "llm_api_key": "sk-test",
        "web_search_api_key": "sk-test",
    }
    values.update(overrides)
    return Settings(**values)


def make_client(
    monkeypatch: Any,
    tmp_path: Path,
    *,
    model: Any | None = None,
    store: Any | None = None,
    settings: Settings | None = None,
    agent_graph: Any | None = None,
) -> tuple[TestClient, Any, Any]:
    cfg = settings or make_settings(tmp_path)
    memory = store if store is not None else InMemoryChatMemoryStore()
    chat_model = model or ScriptedChatModel(responses=[AIMessage(content="好的")])
    tools = build_general_tools(cfg, searcher=FakeSearcher())
    graph = agent_graph or build_general_agent_graph(llm=chat_model, tools=tools, settings=cfg)
    monkeypatch.setattr(api_general, "get_settings", lambda: cfg)
    monkeypatch.setattr(api_general, "_store", lambda: memory)
    monkeypatch.setattr(api_general, "_tools", lambda: tools)
    monkeypatch.setattr(api_general, "_graph", lambda: graph)
    monkeypatch.setattr(api_general, "_chat_model", lambda: chat_model)
    return TestClient(fastapi_app), memory, chat_model


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = "message"
        data = "{}"
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        events.append((event, json.loads(data)))
    return events


class FailingStore(InMemoryChatMemoryStore):
    """模拟 MySQL 不可用（未配置 / 表不存在 / 连接失败）。"""

    def ensure_session(self, *args: Any, **kwargs: Any) -> Any:
        raise DatabaseError("MySQL 会话记忆表不存在")


class BrokenStore:
    """概览只用到 count_sessions，所以单独做一个最小假实现。"""

    def count_sessions(self, **kwargs: Any) -> int:
        raise DatabaseError("MySQL 会话记忆表不存在")


def seed_history(store: Any, thread_id: str, count: int = 6) -> None:
    """直接往内存存储里灌入 user/assistant 交替历史。"""
    for index in range(count):
        store.append_messages(
            thread_id,
            [
                ChatMessage(
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"第{index}条：" + "内容" * 60,
                )
            ],
            title="历史测试" if index == 0 else None,
        )


def summary_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "agent_model_context_tokens": 4096,
        "agent_output_reserve_tokens": 256,
        "agent_context_safety_tokens": 64,
        "agent_summary_trigger_tokens": 10,
        "agent_summary_keep_messages": 2,
        "agent_summary_max_chars": 400,
        "agent_summary_max_tokens": 200,
    }
    values.update(overrides)
    return make_settings(tmp_path, **values)


# ---- /overview ------------------------------------------------------------
def test_overview_lists_all_tools_and_memory(monkeypatch, tmp_path: Path) -> None:
    client, store, _ = make_client(monkeypatch, tmp_path)
    store.append_messages("t-1", [ChatMessage(role="user", content="你好")], title="你好")

    response = client.get("/api/general-agent/overview")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["llm_configured"] is True
    assert body["llm_model"] == "gpt-4o-mini"
    assert body["workspace"] == str(tmp_path)
    assert [tool["name"] for tool in body["tools"]] == EXPECTED_TOOLS
    assert body["tools"][0]["group_label"] == "知识库"
    # A-01：工具策略随概览暴露（超时 / 重试 / 风险 / 并行）。
    assert body["tools"][0]["timeout_seconds"] == 60.0
    assert body["tools"][0]["retryable"] is True
    assert body["tools"][0]["risk_level"] == "low"
    shell = next(tool for tool in body["tools"] if tool["name"] == "shell")
    assert shell["timeout_seconds"] is None
    assert shell["risk_level"] == "high"
    assert shell["concurrent"] is False
    assert body["shell_enabled"] is True
    assert body["memory"] == {
        "configured": True,
        "ok": True,
        "tables": [
            "tb_agent_chat_session",
            "tb_agent_chat_message",
            "tb_agent_chat_summary",
        ],
        "agent_id": "general",
        "candidate_messages": 6,
        "session_count": 1,
        "summary_table_ready": True,
        "error": None,
    }
    assert body["context"]["model_context_tokens"] == 32768
    assert body["context"]["history_budget_tokens"] > 0
    assert body["context"]["summary_enabled"] is True
    assert body["warnings"] == []


def test_overview_warns_without_llm_key(monkeypatch, tmp_path: Path) -> None:
    settings = make_settings(tmp_path, llm_api_key=None, dashscope_api_key=None)
    client, _, _ = make_client(monkeypatch, tmp_path, settings=settings)

    body = client.get("/api/general-agent/overview").json()

    assert body["ready"] is False
    assert body["llm_configured"] is False
    assert any("对话模型 Key" in warning for warning in body["warnings"])


def test_overview_reports_memory_failure(monkeypatch, tmp_path: Path) -> None:
    client, _, _ = make_client(monkeypatch, tmp_path, store=BrokenStore())

    response = client.get("/api/general-agent/overview")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert body["memory"]["ok"] is False
    assert "MySQL" in body["memory"]["error"]
    assert any("会话记忆不可用" in warning for warning in body["warnings"])


def test_overview_shell_flag_follows_tool_list(monkeypatch, tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    client, _, _ = make_client(monkeypatch, tmp_path, settings=settings)
    tools = build_general_tools(settings, searcher=FakeSearcher(), include_shell=False)
    monkeypatch.setattr(api_general, "_tools", lambda: tools)

    body = client.get("/api/general-agent/overview").json()

    assert body["shell_enabled"] is False
    assert "shell" not in [tool["name"] for tool in body["tools"]]


# ---- /sessions ------------------------------------------------------------
def test_sessions_endpoints_read_and_delete_mysql_memory(monkeypatch, tmp_path: Path) -> None:
    client, store, _ = make_client(monkeypatch, tmp_path)
    store.append_messages(
        "t-a", [ChatMessage(role="user", content="会话 A 的第一个问题")], title="会话 A"
    )
    store.append_messages(
        "t-a",
        [
            ChatMessage(
                role="tool",
                content="README.md",
                tool_name="glob",
                tool_call_id="c1",
                tool_args='{"pattern": "*.md"}',
                tool_status="ok",
                elapsed_ms=3,
            ),
            ChatMessage(role="assistant", content="找到了 README.md。"),
        ],
    )

    sessions = client.get("/api/general-agent/sessions").json()
    assert [session["thread_id"] for session in sessions] == ["t-a"]
    assert sessions[0]["title"] == "会话 A"
    assert sessions[0]["message_count"] == 3
    assert sessions[0]["created_at"]

    messages = client.get("/api/general-agent/sessions/t-a/messages").json()
    assert messages["thread_id"] == "t-a"
    assert messages["total"] == 3
    assert [item["role"] for item in messages["items"]] == ["user", "tool", "assistant"]
    assert messages["items"][1]["tool_name"] == "glob"
    assert messages["items"][1]["tool_status"] == "ok"

    assert client.delete("/api/general-agent/sessions/t-a").status_code == 204
    assert store.load_messages("t-a") == []
    assert client.get("/api/general-agent/sessions/t-a/messages").status_code == 404
    assert client.delete("/api/general-agent/sessions/t-a").status_code == 404


def test_sessions_reject_invalid_limit(monkeypatch, tmp_path: Path) -> None:
    client, _, _ = make_client(monkeypatch, tmp_path)

    assert client.get("/api/general-agent/sessions", params={"limit": 0}).status_code == 422


# ---- /chat ----------------------------------------------------------------
def test_chat_persists_messages_and_reuses_history(monkeypatch, tmp_path: Path) -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="第一轮回答")])
    client, store, _ = make_client(monkeypatch, tmp_path, model=model)

    first = client.post(
        "/api/general-agent/chat",
        json={"message": "第一个问题", "thread_id": "t-1"},
    )

    assert first.status_code == 200
    body = first.json()
    assert body["answer"] == "第一轮回答"
    assert body["thread_id"] == "t-1"
    assert body["tool_calls"] == 0
    assert body["session"]["message_count"] == 2
    assert [item["role"] for item in body["messages"]] == ["assistant"]
    assert body["messages"][0]["seq"] == 2

    model.responses.append(AIMessage(content="第二轮回答"))
    second = client.post(
        "/api/general-agent/chat",
        json={"message": "第二个问题", "thread_id": "t-1"},
    )

    assert second.json()["answer"] == "第二轮回答"
    # 第二轮提示词里带上了第一轮的问答（MySQL 记忆）
    contents = [message.content for message in model.received[-1]]
    assert "第一个问题" in contents
    assert "第一轮回答" in contents
    assert "第二个问题" in contents
    assert [message.seq for message in store.load_messages("t-1")] == [1, 2, 3, 4]


def test_chat_attaches_uploaded_image_urls(monkeypatch, tmp_path: Path) -> None:
    oss_url = (
        "https://agent3.oss-cn-shenzhen.aliyuncs.com/image3/2026/11/05/abc.png"
        "?OSSAccessKeyId=AK&Expires=1793944492&Signature=abc"
    )
    settings = make_settings(
        tmp_path,
        oss_bucket="agent3",
        oss_endpoint="https://oss-cn-shenzhen.aliyuncs.com",
        oss_access_key_id="AK",
        oss_access_key_secret="SK",
        oss_prefix="image3",
        oss_access_token="token",
    )
    model = ScriptedChatModel(responses=[AIMessage(content="收到图片")])
    client, store, _ = make_client(monkeypatch, tmp_path, model=model, settings=settings)

    response = client.post(
        "/api/general-agent/chat",
        json={"message": "看看这张图", "thread_id": "t-img", "images": [oss_url]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "收到图片"
    user = store.load_messages("t-img")[0]
    assert user.role == "user"
    assert user.content.startswith("看看这张图")
    assert f"![用户上传的图片]({oss_url})" in user.content
    human = next(
        message for message in model.received[-1] if isinstance(message, HumanMessage)
    )
    assert isinstance(human.content, list)
    assert {"type": "image", "url": oss_url} in human.content
    assert any(
        block.get("type") == "text" and "看看这张图" in block.get("text", "")
        for block in human.content
    )

    # 第二轮没有新图片，但历史里的用户消息仍要还原成 image 内容块。
    model.responses.append(AIMessage(content="第二轮回答"))
    second = client.post(
        "/api/general-agent/chat",
        json={"message": "继续", "thread_id": "t-img"},
    )
    assert second.status_code == 200
    history_human = [
        message for message in model.received[-1] if isinstance(message, HumanMessage)
    ]
    assert any(
        isinstance(message.content, list)
        and {"type": "image", "url": oss_url} in message.content
        for message in history_human
    )


def test_chat_rejects_image_urls_outside_bucket(monkeypatch, tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        oss_bucket="agent3",
        oss_endpoint="https://oss-cn-shenzhen.aliyuncs.com",
        oss_access_key_id="AK",
        oss_access_key_secret="SK",
        oss_prefix="image3",
        oss_access_token="token",
    )
    client, _, _ = make_client(monkeypatch, tmp_path, settings=settings)

    wrong_host = client.post(
        "/api/general-agent/chat",
        json={
            "message": "看图",
            "images": ["https://evil.example.com/image3/2026/11/05/abc.png"],
        },
    )
    assert wrong_host.status_code == 400

    wrong_prefix = client.post(
        "/api/general-agent/chat",
        json={
            "message": "看图",
            "images": ["https://agent3.oss-cn-shenzhen.aliyuncs.com/other/abc.png"],
        },
    )
    assert wrong_prefix.status_code == 400


def test_chat_records_tool_calls(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# 笔记\n\n差旅住宿按城市分级。", encoding="utf-8")
    model = ScriptedChatModel(
        responses=[
            tool_call_message("read", {"path": "notes.md"}),
            AIMessage(content="笔记写的是差旅住宿按城市分级。"),
        ]
    )
    client, _, _ = make_client(monkeypatch, tmp_path, model=model)

    response = client.post(
        "/api/general-agent/chat",
        json={"message": "读一下 notes.md", "thread_id": "t-tool"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tool_calls"] == 1
    assert body["answer"] == "笔记写的是差旅住宿按城市分级。"
    tool_row, answer_row = body["messages"]
    assert tool_row["role"] == "tool"
    assert tool_row["tool_name"] == "read"
    assert tool_row["tool_status"] == "ok"
    # 同步路径从工具节点的 artifact 读取真实耗时（A-01）。
    assert tool_row["elapsed_ms"] is not None
    assert json.loads(tool_row["tool_args"]) == {"path": "notes.md"}
    assert "差旅住宿按城市分级" in tool_row["content"]
    assert answer_row["role"] == "assistant"


def test_chat_returns_503_when_model_not_configured(monkeypatch, tmp_path: Path) -> None:
    settings = make_settings(tmp_path, llm_api_key=None, dashscope_api_key=None)
    client, _, _ = make_client(monkeypatch, tmp_path, settings=settings)

    def no_graph() -> Any:
        raise ConfigurationError("Missing chat model API key")

    monkeypatch.setattr(api_general, "_graph", no_graph)

    response = client.post("/api/general-agent/chat", json={"message": "你好"})

    assert response.status_code == 503
    assert "通用 Agent 未就绪" in response.json()["detail"]


def test_chat_returns_503_when_memory_unavailable(monkeypatch, tmp_path: Path) -> None:
    client, _, _ = make_client(monkeypatch, tmp_path, store=FailingStore())

    response = client.post("/api/general-agent/chat", json={"message": "你好"})

    assert response.status_code == 503
    assert "MySQL 会话记忆不可用" in response.json()["detail"]


def test_chat_validates_empty_message(monkeypatch, tmp_path: Path) -> None:
    client, _, _ = make_client(monkeypatch, tmp_path)

    assert client.post("/api/general-agent/chat", json={"message": ""}).status_code == 422


def test_chat_generates_thread_id_when_omitted(monkeypatch, tmp_path: Path) -> None:
    client, store, _ = make_client(monkeypatch, tmp_path)

    body = client.post("/api/general-agent/chat", json={"message": "你好"}).json()

    assert len(body["thread_id"]) == 32
    assert [session.thread_id for session in store.list_sessions()] == [body["thread_id"]]


# ---- /chat/stream ---------------------------------------------------------
def test_chat_stream_emits_tokens_and_persists(monkeypatch, tmp_path: Path) -> None:
    model = StreamingScriptedChatModel(
        responses=[AIMessage(content="流式回答内容")],
        chunk_size=2,
    )
    client, store, _ = make_client(monkeypatch, tmp_path, model=model)

    response = client.post(
        "/api/general-agent/chat/stream",
        json={"message": "你好", "thread_id": "t-stream"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "session"
    assert names[-1] == "done"
    assert "tool_call" not in names
    assert events[0][1]["thread_id"] == "t-stream"
    assert events[0][1]["message_count"] == 1

    tokens = "".join(payload["text"] for name, payload in events if name == "token")
    assert tokens == "流式回答内容"
    done = events[-1][1]
    assert done["answer"] == "流式回答内容"
    assert done["tool_calls"] == 0
    assert [item["role"] for item in done["messages"]] == ["assistant"]
    assert [message.role for message in store.load_messages("t-stream")] == ["user", "assistant"]


def test_chat_stream_reports_tool_calls(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# 笔记\n\n差旅住宿按城市分级。", encoding="utf-8")
    model = ScriptedChatModel(
        responses=[
            tool_call_message("glob", {"pattern": "*.md"}),
            AIMessage(content="工作目录里有 notes.md。"),
        ]
    )
    client, store, _ = make_client(monkeypatch, tmp_path, model=model)

    response = client.post(
        "/api/general-agent/chat/stream",
        json={"message": "找一下 markdown 文件", "thread_id": "t-tools"},
    )

    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "session"
    assert "tool_call" in names
    assert "tool_result" in names
    assert names[-1] == "done"
    call = next(payload for name, payload in events if name == "tool_call")
    assert call["name"] == "glob"
    assert json.loads(call["args"]) == {"pattern": "*.md"}
    result = next(payload for name, payload in events if name == "tool_result")
    assert result["name"] == "glob"
    assert result["status"] == "ok"
    assert "notes.md" in result["content"]
    assert events[-1][1]["answer"] == "工作目录里有 notes.md。"
    assert [message.role for message in store.load_messages("t-tools")] == [
        "user",
        "tool",
        "assistant",
    ]
    assert store.get_session("t-tools").title == "找一下 markdown 文件"


def test_chat_stream_reports_memory_failure_as_event(monkeypatch, tmp_path: Path) -> None:
    client, _, _ = make_client(monkeypatch, tmp_path, store=FailingStore())

    response = client.post(
        "/api/general-agent/chat/stream",
        json={"message": "你好", "thread_id": "t-broken"},
    )

    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["error"]
    assert "MySQL 会话记忆不可用" in events[0][1]["message"]


# ---- 上下文压缩 L2/L3 -----------------------------------------------------
def test_chat_auto_compact_summarizes_old_history(monkeypatch, tmp_path: Path) -> None:
    settings = summary_settings(tmp_path)
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="## 主要请求与意图\n- 用户在做历史压缩测试"),
            AIMessage(content="压缩后的回答"),
        ]
    )
    client, store, _ = make_client(monkeypatch, tmp_path, model=model, settings=settings)
    seed_history(store, "t-sum", 6)

    response = client.post(
        "/api/general-agent/chat",
        json={"message": "继续", "thread_id": "t-sum"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "压缩后的回答"
    context = body["context"]
    assert context["summary_used"] is True
    assert context["summarized_messages"] == 4
    assert context["compressed_messages"] == 4
    assert context["summary_chars"] > 0

    summary_row = store.get_summary("t-sum")
    assert summary_row is not None
    assert summary_row.until_seq == 4
    assert "主要请求与意图" in summary_row.summary

    # 模型第二轮（agent 调用）的系统提示词里带上了摘要
    system = model.received[-1][0]
    assert isinstance(system, SystemMessage)
    assert "更早对话摘要（已压缩）" in system.content

    messages = client.get("/api/general-agent/sessions/t-sum/messages").json()
    assert messages["summary"]["until_seq"] == 4
    compressed = [item for item in messages["items"] if item["compressed"]]
    assert len(compressed) == 4
    assert all(item["seq"] <= 4 for item in compressed)


def test_manual_compact_endpoint_updates_summary_and_flags(monkeypatch, tmp_path: Path) -> None:
    settings = summary_settings(tmp_path, agent_summary_trigger_tokens=10_000)
    model = ScriptedChatModel(responses=[AIMessage(content="## 主要请求与意图\n- 手动压缩")])
    client, store, _ = make_client(monkeypatch, tmp_path, model=model, settings=settings)
    seed_history(store, "t-manual", 6)

    response = client.post(
        "/api/general-agent/sessions/t-manual/compact",
        json={"keep": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["compacted"] is True
    assert body["summarized"] == 4
    assert body["until_seq"] == 4
    assert body["summary"].startswith("## 主要请求与意图")

    summary_row = store.get_summary("t-manual")
    assert summary_row is not None and summary_row.until_seq == 4
    messages = client.get("/api/general-agent/sessions/t-manual/messages").json()
    assert messages["summary"] is not None
    assert sum(1 for item in messages["items"] if item["compressed"]) == 4


def test_manual_compact_with_no_history_is_noop(monkeypatch, tmp_path: Path) -> None:
    client, store, _ = make_client(monkeypatch, tmp_path)
    store.append_messages("t-empty", [ChatMessage(role="user", content="只有一句")])

    response = client.post("/api/general-agent/sessions/t-empty/compact", json={})

    assert response.status_code == 200
    assert response.json()["compacted"] is False


def test_chat_stream_reports_compact_stage(monkeypatch, tmp_path: Path) -> None:
    settings = summary_settings(tmp_path)
    model = StreamingScriptedChatModel(
        responses=[
            AIMessage(content="## 主要请求与意图\n- 流式压缩"),
            AIMessage(content="流式压缩后的回答"),
        ],
        chunk_size=4,
    )
    client, store, _ = make_client(monkeypatch, tmp_path, model=model, settings=settings)
    seed_history(store, "t-stream-sum", 6)

    response = client.post(
        "/api/general-agent/chat/stream",
        json={"message": "继续", "thread_id": "t-stream-sum"},
    )

    assert response.status_code == 200
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "stage"
    assert events[0][1]["stage"] == "compact"
    assert "session" in names
    assert names[-1] == "done"
    done = events[-1][1]
    assert done["context"]["summary_used"] is True
    assert done["context"]["summarized_messages"] == 4
    assert store.get_summary("t-stream-sum") is not None


# ---- 上下文溢出重试 L4 -----------------------------------------------------
class OverflowOnceGraph:
    """第一次调用模拟 provider 上下文超限，第二次成功。"""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, state: dict[str, Any], config: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError(
                "This model's maximum context length is 8192 tokens, "
                "please reduce the length of the messages"
            )
        return {"messages": [*state["messages"], AIMessage(content="重试成功")]}


def test_chat_retries_once_after_context_overflow(monkeypatch, tmp_path: Path) -> None:
    settings = summary_settings(
        tmp_path,
        agent_summary_enabled=False,
        agent_context_overflow_retries=1,
    )
    graph = OverflowOnceGraph()
    client, store, _ = make_client(
        monkeypatch,
        tmp_path,
        settings=settings,
        agent_graph=graph,
    )
    seed_history(store, "t-overflow", 2)

    response = client.post(
        "/api/general-agent/chat",
        json={"message": "继续", "thread_id": "t-overflow"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "重试成功"
    assert graph.calls == 2
    assert [message.role for message in store.load_messages("t-overflow")][-2:] == [
        "user",
        "assistant",
    ]


# ---- A-02 循环预算 / finalize --------------------------------------------
class ToolOnlyGraph:
    """只返回工具消息、没有最终 AI 回答的假图（D-03 / A-02 的回退场景）。"""

    def invoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        return {
            "messages": [
                HumanMessage(content="读一下"),
                ToolMessage(content="文件原文：差旅住宿按城市分级。", tool_call_id="c1"),
            ]
        }


def test_chat_returns_finalize_text_instead_of_the_tool_message(
    monkeypatch, tmp_path: Path
) -> None:
    client, store, _ = make_client(monkeypatch, tmp_path, agent_graph=ToolOnlyGraph())

    response = client.post(
        "/api/general-agent/chat",
        json={"message": "读一下", "thread_id": "t-noanswer"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == NO_FINAL_ANSWER
    # 落库的是明确的 finalize 文案，不是 ToolMessage 的原文。
    messages = store.load_messages("t-noanswer")
    assert messages[-1].role == "assistant"
    assert messages[-1].content == NO_FINAL_ANSWER


def test_chat_reports_finalize_answer_and_exhausted_budget(monkeypatch, tmp_path: Path) -> None:
    model = ScriptedChatModel(
        responses=[
            tool_call_message("search_knowledge_base", {"query": "住宿费"}),
            AIMessage(content="基于已有资料：一线城市住宿上限 500 元（信息可能不完整）。"),
        ]
    )
    settings = make_settings(tmp_path, agent_max_turns=1)
    client, store, _ = make_client(monkeypatch, tmp_path, model=model, settings=settings)

    response = client.post(
        "/api/general-agent/chat",
        json={"message": "住宿费标准？", "thread_id": "t-budget"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("基于已有资料")
    assert body["budget"]["exhausted"] is True
    assert body["budget"]["reason"] == "max_steps"
    assert body["budget"]["max_steps"] == 1
    # 预算=1：第一次模型调用请求工具后就直接 finalize，工具没有执行。
    assert body["tool_calls"] == 0
    assert store.load_messages("t-budget")[-1].role == "assistant"


def test_chat_survives_graph_recursion_error(monkeypatch, tmp_path: Path) -> None:
    class RecursionGraph:
        def invoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
            raise GraphRecursionError("Recursion limit of 3 reached")

    client, store, _ = make_client(monkeypatch, tmp_path, agent_graph=RecursionGraph())

    response = client.post(
        "/api/general-agent/chat",
        json={"message": "长任务", "thread_id": "t-recursion"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == NO_FINAL_ANSWER
    assert body["budget"]["exhausted"] is True
    assert body["budget"]["reason"] == "recursion_limit"
    assert store.load_messages("t-recursion")[-1].content == NO_FINAL_ANSWER


def test_chat_stream_emits_finalize_tokens_and_budget(monkeypatch, tmp_path: Path) -> None:
    model = StreamingScriptedChatModel(
        responses=[
            tool_call_message("search_knowledge_base", {"query": "住宿费"}),
            AIMessage(content="收尾答案：信息可能不完整。"),
        ],
        chunk_size=4,
    )
    settings = make_settings(tmp_path, agent_max_turns=1)
    client, store, _ = make_client(monkeypatch, tmp_path, model=model, settings=settings)

    response = client.post(
        "/api/general-agent/chat/stream",
        json={"message": "住宿费标准？", "thread_id": "t-sbudget"},
    )

    assert response.status_code == 200
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[-1] == "done"
    stage = next(payload for name, payload in events if name == "stage")
    assert stage["stage"] == "finalize"
    tokens = "".join(payload["text"] for name, payload in events if name == "token")
    assert tokens == "收尾答案：信息可能不完整。"
    done = events[-1][1]
    assert done["answer"] == "收尾答案：信息可能不完整。"
    assert done["budget"]["exhausted"] is True
    assert done["budget"]["reason"] == "max_steps"
    assert done["budget"]["steps"] == 2
    assert [message.role for message in store.load_messages("t-sbudget")] == [
        "user",
        "assistant",
    ]


def test_chat_stream_survives_graph_recursion_error(monkeypatch, tmp_path: Path) -> None:
    class RecursionStreamGraph:
        def stream(self, state: dict[str, Any], config: Any, stream_mode: Any = None) -> Any:
            raise GraphRecursionError("Recursion limit of 3 reached")

    client, store, _ = make_client(monkeypatch, tmp_path, agent_graph=RecursionStreamGraph())

    response = client.post(
        "/api/general-agent/chat/stream",
        json={"message": "长任务", "thread_id": "t-srecursion"},
    )

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [name for name, _ in events][-1] == "done"
    done = events[-1][1]
    assert done["answer"] == NO_FINAL_ANSWER
    assert done["budget"]["exhausted"] is True
    assert done["budget"]["reason"] == "recursion_limit"
    assert store.load_messages("t-srecursion")[-1].content == NO_FINAL_ANSWER


def test_overview_exposes_loop_budget(monkeypatch, tmp_path: Path) -> None:
    client, _, _ = make_client(monkeypatch, tmp_path)

    body = client.get("/api/general-agent/overview").json()

    assert body["budget"] == {
        "max_steps": 4,
        "max_tool_calls": 24,
        "max_tokens": 0,
        "max_seconds": 600.0,
    }
    knowledge = body["tools"][0]
    assert knowledge["dedupe"] is True
