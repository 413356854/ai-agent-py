"""人工审批（HITL）HTTP 接口：提交运行、列表、详情、审批、取消。

用 TestClient + 临时 SQLite 检查点 + 内存运行记录库，不连 MySQL。
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from ai_agent_py.api.app import create_app
from ai_agent_py.config import Settings
from ai_agent_py.graph.checkpoint import SqliteCheckpointer
from ai_agent_py.reimburse import SAMPLES_BY_ID, build_context, build_graph, build_result
from ai_agent_py.runs.models import RunRecord
from ai_agent_py.runs.service import RunService
from ai_agent_py.runs.store import InMemoryRunStore


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


def _service(tmp_path: Path, **overrides: Any) -> RunService:
    """一个完整的运行服务：SQLite 检查点 + 内存运行库（与线上装配一致）。"""
    settings = _settings(tmp_path, **overrides)
    pointer = SqliteCheckpointer(settings.agent_run_checkpoint_path)
    return RunService(
        agent_id="reimburse",
        store=InMemoryRunStore(),
        graph=build_graph(checkpointer=pointer),
        checkpointer=pointer,
        result_builder=build_result,
        settings=settings,
    )


def _client(tmp_path: Path, **overrides: Any) -> tuple[TestClient, RunService]:
    """每个测试一个 app + 一个运行服务（挂在 app.state，和线上装配一致）。"""
    service = _service(tmp_path, **overrides)
    app = create_app()
    app.state.agent_runs = service
    return TestClient(app), service


def test_start_list_and_get_run(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)

    created = client.post("/api/agents/reimburse/runs", json={"sample_id": "C003"})
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "pending"
    assert body["title"] == "BX-2026-10-003"
    assert body["approval"]["type"] == "approval"
    assert body["approval"]["claim_id"] == "BX-2026-10-003"
    assert body["approval"]["risk"] == "high"
    assert body["expires_at"]
    assert body["result"] is None
    assert [item["status"] for item in body["events"]] == ["running", "pending"]

    listed = client.get("/api/agents/reimburse/runs").json()
    assert [item["run_id"] for item in listed] == [body["run_id"]]

    pending = client.get("/api/agents/reimburse/runs?status=pending").json()
    assert len(pending) == 1
    assert client.get("/api/agents/reimburse/runs?status=completed").json() == []

    detail = client.get(f"/api/agents/reimburse/runs/{body['run_id']}")
    assert detail.status_code == 200
    assert detail.json()["run_id"] == body["run_id"]

    assert client.get("/api/agents/reimburse/runs/missing").status_code == 404
    assert client.get("/api/agents/nope/runs").status_code == 404


def test_auto_decision_is_completed(tmp_path: Path) -> None:
    """不需要人拍板的单子一次跑完，接口直接给 completed + 结果。"""
    client, _ = _client(tmp_path)
    body = client.post("/api/agents/reimburse/runs", json={"sample_id": "C002"}).json()

    assert body["status"] == "completed"
    assert body["approval"] is None
    assert body["result"]["status"] == "退回"
    assert {"R2.1"} <= {finding["rule_id"] for finding in body["result"]["findings"]}


def test_resume_approve_then_reject_requests(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)

    approved_body = client.post("/api/agents/reimburse/runs", json={"sample_id": "C003"}).json()
    approved = client.post(
        f"/api/agents/reimburse/runs/{approved_body['run_id']}/resume",
        json={"decision": "approve", "approver": "李四", "comment": "业务确认"},
    )
    assert approved.status_code == 200
    payload = approved.json()
    assert payload["status"] == "approved"
    assert payload["approver"] == "李四"
    assert payload["result"]["status"] == "通过"
    assert payload["result"]["approval"]["approver"] == "李四"
    # 审批内容在审批后仍能看到（审批留痕）
    assert payload["approval"]["claim_id"] == approved_body["approval"]["claim_id"]

    # 重复审批 → 409
    again = client.post(
        f"/api/agents/reimburse/runs/{approved_body['run_id']}/resume",
        json={"decision": "reject", "approver": "王五"},
    )
    assert again.status_code == 409
    assert "已结束" in again.json()["detail"]

    rejected_body = client.post("/api/agents/reimburse/runs", json={"sample_id": "C004"}).json()
    rejected = client.post(
        f"/api/agents/reimburse/runs/{rejected_body['run_id']}/resume",
        json={"decision": "reject", "approver": "王五", "comment": "事由不充分"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert rejected.json()["result"]["status"] == "退回"


def test_cancel_run_api(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    body = client.post("/api/agents/reimburse/runs", json={"sample_id": "C003"}).json()

    cancelled = client.post(
        f"/api/agents/reimburse/runs/{body['run_id']}/cancel",
        json={"approver": "李四", "comment": "重复提交"},
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["comment"] == "重复提交"

    # 取消后既不能再取消，也不能再审批
    assert client.post(f"/api/agents/reimburse/runs/{body['run_id']}/cancel").status_code == 409
    resume = client.post(
        f"/api/agents/reimburse/runs/{body['run_id']}/resume",
        json={"decision": "approve"},
    )
    assert resume.status_code == 409


def test_invalid_start_payload(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    assert client.post("/api/agents/reimburse/runs", json={}).status_code == 400
    assert (
        client.post("/api/agents/reimburse/runs", json={"sample_id": "NOPE"}).status_code == 404
    )
    bad = client.post(
        "/api/agents/reimburse/runs",
        json={"sample_id": "C003", "timeout_seconds": -1},
    )
    assert bad.status_code == 422


def test_timeout_visible_through_api(tmp_path: Path) -> None:
    """超时巡检在接口读路径上生效：到点后列表 / 详情看到 escalated，再超时看到 timeout。"""
    client, _ = _client(tmp_path, agent_approval_escalate_seconds=0)
    body = client.post(
        "/api/agents/reimburse/runs",
        json={"sample_id": "C003", "timeout_seconds": 0.001},
    ).json()
    assert body["status"] == "pending"

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        detail = client.get(f"/api/agents/reimburse/runs/{body['run_id']}").json()
        if detail["status"] == "timeout":
            break
        time.sleep(0.01)

    assert detail["status"] == "timeout"
    assert detail["approver"] == "系统（超时）"
    assert detail["result"]["approval"]["timeout"] is True
    assert client.get("/api/agents/reimburse/runs?status=pending").json() == []


def test_run_from_inline_claim(tmp_path: Path) -> None:
    """直接传单据（不走样例）：标题仍取 claim_id。"""
    client, _ = _client(tmp_path)
    claim = {
        "claim_id": "BX-INLINE-1",
        "employee": {"id": "E009", "name": "赵六", "dept": "销售部"},
        "period": "2026-10",
        "category": "差旅",
        "purpose": "客户拜访",
        "total": "100.00",
        "invoices": [],
    }
    body = client.post("/api/agents/reimburse/runs", json={"claim": claim}).json()
    # 缺票据是硬规则阻断，不需要人工审批
    assert body["status"] == "completed"
    assert body["title"] == "BX-INLINE-1"
    assert body["result"]["status"] == "退回"


def test_checkpoint_file_is_reused(tmp_path: Path) -> None:
    """检查点落在配置的文件上，而不是进程内存里。"""
    client, _ = _client(tmp_path)
    body = client.post("/api/agents/reimburse/runs", json={"sample_id": "C003"}).json()
    assert body["status"] == "pending"

    path = tmp_path / "agent_runs.sqlite3"
    assert path.is_file()
    assert path.stat().st_size > 0


def test_run_of_another_agent_is_not_visible(tmp_path: Path) -> None:
    """运行记录按应用隔离：别的应用的 run_id 在这个前缀下就是 404。"""
    client, service = _client(tmp_path)
    now = datetime.now(UTC)
    service.store.create(
        RunRecord(
            run_id="other-1",
            agent_id="filesystem",
            thread_id="other-1",
            status="pending",
            payload={"request": None, "events": []},
            created_at=now,
            updated_at=now,
        )
    )
    assert client.get("/api/agents/reimburse/runs/other-1").status_code == 404
    assert (
        client.post(
            "/api/agents/reimburse/runs/other-1/resume", json={"decision": "approve"}
        ).status_code
        == 404
    )
    assert client.post("/api/agents/reimburse/runs/other-1/cancel").status_code == 404


def test_background_sweeper_starts_and_stops(tmp_path: Path, monkeypatch: Any) -> None:
    """应用启动后应有审批巡检任务（没人打开页面也会处理超时），关停时干净取消。"""
    import asyncio
    from types import SimpleNamespace

    from ai_agent_py.api import agents as api_agents

    monkeypatch.setattr(api_agents, "APPROVAL_SWEEP_INTERVAL", 0.02)
    service = _service(tmp_path)

    async def scenario() -> None:
        app = SimpleNamespace(state=SimpleNamespace())
        api_agents.start_approval_sweeper(app)
        task = app.state.approval_sweeper
        assert task is not None
        # 巡检不会自己建服务：先确认没有服务时也能正常空转
        await asyncio.sleep(0.05)
        assert task.done() is False

        # 首次用到审批接口才装配服务；巡检下一轮就会处理到期的审批
        app.state.agent_runs = service
        claim = SAMPLES_BY_ID["C003"]["claim"]
        record = await asyncio.to_thread(
            service.start,
            {"claim": claim, "context": build_context(claim), "findings": [], "trace": []},
            timeout_seconds=0.001,
        )
        assert record.status == "pending"
        await asyncio.sleep(0.2)
        # 直接读 store，绕开「接口读路径也会巡检」，确保状态是后台任务改的
        assert service.store.get(record.run_id).status in {"escalated", "timeout"}

        await api_agents.stop_approval_sweeper(app)
        assert task.done()
        assert app.state.approval_sweeper is None

    asyncio.run(scenario())
