"""人工审批（HITL）运行状态机：暂停 / 恢复 / 取消 / 超时，以及检查点持久化。

不连 MySQL：运行记录用内存实现，检查点用临时目录里的 SQLite 文件——测的正是
「图停在 G8 之后还能不能接着跑」这条链路。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import create_engine

from ai_agent_py.config import Settings
from ai_agent_py.errors import RunConflict, RunNotFound
from ai_agent_py.graph.checkpoint import SqliteCheckpointer
from ai_agent_py.reimburse import SAMPLES_BY_ID, build_context, build_graph, build_result, run_claim
from ai_agent_py.runs.models import RunRecord, event, with_event
from ai_agent_py.runs.service import RunService
from ai_agent_py.runs.store import InMemoryRunStore, MySQLRunStore, agent_run


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "app_env": "test",
        "agent_run_checkpoint_path": tmp_path / "agent_runs.sqlite3",
        "agent_approval_timeout": 600,
        "agent_approval_escalate_seconds": 60,
    }
    values.update(overrides)
    return Settings(**values)


def _service(
    tmp_path: Path,
    *,
    store: InMemoryRunStore | None = None,
    settings: Settings | None = None,
    graph: Any | None = None,
    checkpointer: SqliteCheckpointer | None = None,
    agent_id: str = "reimburse",
    result_builder: Any | None = build_result,
) -> RunService:
    cfg = settings or _settings(tmp_path)
    pointer = checkpointer or SqliteCheckpointer(cfg.agent_run_checkpoint_path)
    compiled = graph if graph is not None else build_graph(checkpointer=pointer)
    return RunService(
        agent_id=agent_id,
        store=store or InMemoryRunStore(),
        graph=compiled,
        checkpointer=pointer,
        result_builder=result_builder,
        settings=cfg,
    )


def _inputs(sample_id: str = "C003") -> dict[str, Any]:
    claim = SAMPLES_BY_ID[sample_id]["claim"]
    return {"claim": claim, "context": build_context(claim), "findings": [], "trace": []}


def _statuses(record: RunRecord) -> list[str]:
    return [str(item["status"]) for item in record.events]


# ---- 启动与暂停 -------------------------------------------------------------
def test_auto_decision_completes_without_pausing(tmp_path: Path) -> None:
    """硬规则能拍板的单子不该停下来：没有审批点，直接 completed。"""
    service = _service(tmp_path)
    record = service.start(_inputs("C002"), title="BX-2026-10-002")

    assert record.status == "completed"
    assert record.request is None
    assert record.result is not None
    assert record.result["status"] == "退回"
    assert _statuses(record) == ["running", "completed"]


def test_claim_pauses_at_g8_with_approval_request(tmp_path: Path) -> None:
    """语义疑点走到 G8 就真的停住：状态 pending，检查点的下一个节点是 human_review。"""
    service = _service(tmp_path)
    record = service.start(_inputs("C003"), title="BX-2026-10-003")

    assert record.status == "pending"
    assert record.terminal is False
    assert record.expires_at is not None
    request = record.request
    assert request is not None
    assert request.type == "approval"
    assert request.node == "human_review"
    assert request.claim_id == "BX-2026-10-003"
    assert request.risk == "high"  # R4 的语义疑点属于高风险
    assert request.details["findings"]
    assert _statuses(record) == ["running", "pending"]

    snapshot = service.graph.get_state(service.config(record.thread_id))
    assert snapshot.next == ("human_review",)
    assert snapshot.values["decision"]["status"] == "转人工"


def test_approval_continues_to_archive(tmp_path: Path) -> None:
    """审批通过后从 G8 继续：结论改为「通过」，并执行到归档节点。"""
    service = _service(tmp_path)
    record = service.start(_inputs("C003"))
    assert record.status == "pending"

    approved = service.resume(
        record.run_id,
        decision="approve",
        approver="李四",
        comment="招待确为业务需要",
    )

    assert approved.status == "approved"
    assert approved.approver == "李四"
    assert approved.comment == "招待确为业务需要"
    assert approved.finished_at is not None
    assert approved.result is not None
    assert approved.result["status"] == "通过"
    assert approved.result["approval"]["decision"] == "approve"
    assert approved.result["approval"]["approver"] == "李四"
    # G8 之后确实走到了归档，而不是只改了个状态；恢复只重跑 human_review，不重跑前序节点
    nodes = [event["node"] for event in approved.result["trace"]]
    assert nodes[-2:] == ["human_review", "archive"]
    assert nodes.count("decide") == 1
    assert nodes.count("hard_rules") == 1
    assert _statuses(approved) == ["running", "pending", "running", "approved"]
    assert service.graph.get_state(service.config(record.thread_id)).next == ()


def test_rejection_returns_the_claim(tmp_path: Path) -> None:
    """审批拒绝 = 单据退回，不归档为通过。"""
    service = _service(tmp_path)
    record = service.start(_inputs("C004"))
    assert record.status == "pending"

    rejected = service.resume(
        record.run_id, decision="reject", approver="王五", comment="发票与事由不符"
    )

    assert rejected.status == "rejected"
    assert rejected.result is not None
    assert rejected.result["status"] == "退回"
    assert "发票与事由不符" in rejected.result["decision"]["reason"]
    assert rejected.result["approval"]["approved"] is False


def test_second_resume_is_rejected(tmp_path: Path) -> None:
    """重复审批（两个标签页同时点）只能有一个成功。"""
    service = _service(tmp_path)
    record = service.start(_inputs("C003"))
    service.resume(record.run_id, decision="approve", approver="李四")

    with pytest.raises(RunConflict):
        service.resume(record.run_id, decision="reject", approver="王五")


def test_unknown_run_raises_not_found(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(RunNotFound):
        service.get("missing-run")
    with pytest.raises(RunNotFound):
        service.resume("missing-run", decision="approve")


# ---- 取消 -------------------------------------------------------------------
def test_cancel_drops_the_checkpoint(tmp_path: Path) -> None:
    """取消后不再能恢复：状态 cancelled，检查点线程被删掉。"""
    service = _service(tmp_path)
    record = service.start(_inputs("C003"))

    cancelled = service.cancel(record.run_id, approver="李四", comment="单据重复提交")

    assert cancelled.status == "cancelled"
    assert cancelled.finished_at is not None
    assert _statuses(cancelled)[-1] == "cancelled"
    assert cancelled.result is None
    assert service.checkpointer.get_tuple(
        {"configurable": {"thread_id": record.thread_id}}
    ) is None
    with pytest.raises(RunConflict):
        service.resume(record.run_id, decision="approve")


# ---- 超时 -------------------------------------------------------------------
def test_timeout_escalates_then_auto_rejects(tmp_path: Path) -> None:
    """超时未审批：先升级并延长宽限期，宽限期满自动拒绝（终态 timeout）。"""
    service = _service(tmp_path)
    record = service.start(_inputs("C003"), timeout_seconds=60)
    assert record.expires_at is not None

    first = service.sweep(now=record.expires_at + timedelta(seconds=1))
    assert [item.status for item in first] == ["escalated"]
    escalated = service.get(record.run_id)
    assert escalated.status == "escalated"
    assert escalated.expires_at is not None and escalated.expires_at > record.expires_at
    assert _statuses(escalated)[-1] == "escalated"

    second = service.sweep(now=escalated.expires_at + timedelta(seconds=1))
    assert [item.status for item in second] == ["timeout"]
    timed_out = service.get(record.run_id)
    assert timed_out.status == "timeout"
    assert timed_out.approver == "系统（超时）"
    assert timed_out.result is not None
    assert timed_out.result["status"] == "退回"
    assert timed_out.result["approval"]["timeout"] is True
    assert _statuses(timed_out) == ["running", "pending", "escalated", "running", "timeout"]


def test_timeout_without_grace_rejects_directly(tmp_path: Path) -> None:
    """宽限期为 0 时到点直接自动拒绝，不经过 escalated。"""
    settings = _settings(tmp_path, agent_approval_escalate_seconds=0)
    service = _service(tmp_path, settings=settings)
    record = service.start(_inputs("C003"), timeout_seconds=60)
    assert record.expires_at is not None

    resolved = service.sweep(now=record.expires_at + timedelta(seconds=1))

    assert [item.status for item in resolved] == ["timeout"]
    assert _statuses(service.get(record.run_id)) == ["running", "pending", "running", "timeout"]


def test_unlimited_timeout_never_sweeps(tmp_path: Path) -> None:
    """超时设成 0 = 一直等人：永远不会被巡检处理掉。"""
    service = _service(tmp_path)
    record = service.start(_inputs("C003"), timeout_seconds=0)

    assert record.expires_at is None
    assert service.sweep(now=record.created_at + timedelta(days=365)) == []
    assert service.get(record.run_id).status == "pending"


# ---- 持久化 -----------------------------------------------------------------
def test_resume_after_restart_uses_persisted_checkpoint(tmp_path: Path) -> None:
    """进程重启（新图 / 新 checkpointer）后仍能审批恢复：检查点在 SQLite 里。"""
    settings = _settings(tmp_path)
    store = InMemoryRunStore()  # 相当于 MySQL 里的 tb_agent_run 仍在
    first_service = _service(tmp_path, store=store, settings=settings)
    record = first_service.start(_inputs("C003"))
    assert record.status == "pending"

    restarted = _service(tmp_path, store=store, settings=settings)
    resumed = restarted.resume(record.run_id, decision="approve", approver="李四")

    assert resumed.status == "approved"
    assert resumed.result is not None
    assert resumed.result["status"] == "通过"


# ---- 通用性：工具审批共用同一套状态机 ---------------------------------------
class ToolState(TypedDict, total=False):
    path: str
    written: bool
    approval: dict[str, Any]


def _tool_graph(checkpointer: SqliteCheckpointer) -> Any:
    """一个最小的「写文件前审批」图：与报销图共用 interrupt / resume 契约。"""

    def approve_write(state: ToolState, config: RunnableConfig) -> dict[str, Any]:
        decision = interrupt(
            {
                "type": "tool_call",
                "agent_id": "filesystem",
                "node": "approve_write",
                "title": "写文件需要审批",
                "summary": f"即将写入 {state.get('path', '')}",
                "tool": "write",
                "args": {"path": state.get("path", ""), "content": "hello"},
                "risk": "high",
            }
        )
        return {
            "approval": decision if isinstance(decision, dict) else {},
            "written": isinstance(decision, dict) and decision.get("decision") == "approve",
        }

    builder = StateGraph(ToolState)
    builder.add_node("approve_write", approve_write)
    builder.add_edge(START, "approve_write")
    builder.add_edge("approve_write", END)
    return builder.compile(checkpointer=checkpointer)


def test_tool_approval_shares_the_run_state_machine(tmp_path: Path) -> None:
    """破坏性工具审批（type=tool_call）与报销审批走同一套运行记录与恢复接口。"""
    settings = _settings(tmp_path)
    pointer = SqliteCheckpointer(settings.agent_run_checkpoint_path)
    service = _service(
        tmp_path,
        settings=settings,
        graph=_tool_graph(pointer),
        checkpointer=pointer,
        agent_id="filesystem",
        result_builder=None,
    )

    record = service.start({"path": "notes.md"}, title="write notes.md")
    assert record.status == "pending"
    request = record.request
    assert request is not None
    assert request.type == "tool_call"
    assert request.tool == "write"
    assert request.args == {"path": "notes.md", "content": "hello"}

    approved = service.resume(record.run_id, decision="approve", approver="安全负责人")

    assert approved.status == "approved"
    assert approved.result is not None
    assert approved.result["values"]["written"] is True


def test_preview_does_not_pause(tmp_path: Path) -> None:
    """单次试跑（无 checkpointer / 无 hitl）保持原来的「转人工」预览语义。"""
    result = run_claim(SAMPLES_BY_ID["C003"]["claim"])
    assert result["status"] == "转人工"
    assert result["approval"]["decision"] == "pending"


# ---- 存储层：条件更新（原子状态流转） ---------------------------------------
def test_store_transition_is_conditional(tmp_path: Path) -> None:
    """状态流转用 ``WHERE status IN (...)`` 抢占：同一状态只能被改一次。

    这里用 SQLite 引擎跑同一套 SQLAlchemy 语句，验证状态机依赖的原子语义
    （不连 MySQL；建表脚本本身由 ``docs/mysql/tb_agent_run.sql`` 负责）。
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'runs.db'}")
    agent_run.create(engine)
    store = MySQLRunStore(engine)
    now = datetime.now(UTC)
    payload = with_event({"request": None, "events": []}, event("running"))

    store.create(
        RunRecord(
            run_id="run-1",
            agent_id="reimburse",
            thread_id="run-1",
            status="running",
            title="BX-1",
            payload=payload,
            created_at=now,
            updated_at=now,
        )
    )

    pending = store.transition(
        "run-1",
        expected=["running"],
        status="pending",
        expires_at=now + timedelta(seconds=60),
        payload=payload,
    )
    assert pending is not None
    assert pending.status == "pending"
    assert pending.expires_at is not None

    # 并发请求拿同一个 expected 抢不到状态；先到的那次才能改
    assert store.transition("run-1", expected=["running"], status="approved") is None
    assert store.transition("run-1", expected=["pending", "escalated"], status="approved")
    assert store.transition("run-1", expected=["pending"], status="rejected") is None

    assert [item.status for item in store.list_runs(agent_id="reimburse")] == ["approved"]
    assert store.list_runs(statuses=["pending"]) == []
    assert store.due(now + timedelta(seconds=120)) == []
