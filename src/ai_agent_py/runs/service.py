"""人工审批（HITL）运行服务：启动、暂停、审批、取消、超时。

一次「运行」的生命周期都收敛在 :class:`RunService` 里，业务图只负责在需要人拍板的地方
调用 ``interrupt(...)``：

    service.start(inputs)                  # 跑到审批点自动暂停，返回 status=pending
    service.resume(run_id, decision="approve")   # Command(resume=...) 接着跑完
    service.cancel(run_id)                 # 撤单，清掉检查点
    service.sweep()                        # 超时巡检：先升级，再自动拒绝

审批点由 ``interrupt()`` 的 payload 描述（业务审批 / 工具审批同构）：状态机只认
``type / risk / tool / args`` 这些字段，所以「报销单复核」和「写文件前审批」共用同一套
``tb_agent_run`` 记录与同一组接口，不需要第二套实现。

**持久化**：图停在审批节点时，完整状态在 :class:`SqliteCheckpointer` 里（默认
``data/agent_runs.sqlite3``），所以进程重启后 ``resume`` 依然有效；``pending`` 记录本身
落在 ``tb_agent_run``（MySQL），没有配置数据库时退化为内存实现（重启即丢，只适合本地演示）。
审批结束的运行保留检查点（可用 ``get_state_history`` 回看，便于排障）；取消的运行不保留，
``cancel`` 会删掉对应线程，避免文件无限增长。
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import RunConflict, RunNotFound
from ai_agent_py.runs.models import (
    TERMINAL_STATUSES,
    WAITING_STATUSES,
    ApprovalRequest,
    RunDecision,
    RunRecord,
    RunStatus,
    event,
    resume_value,
    with_event,
)
from ai_agent_py.runs.store import RunStore

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

#: 由图的最终状态构造「对外结果」；第二个参数是本段执行耗时（毫秒）。
ResultBuilder = Callable[[dict[str, Any], float], dict[str, Any]]

#: 审批记录里 comment / error 的最大长度（与库里的 VARCHAR(500) 对齐）。
_MAX_NOTE_CHARS = 500


class RunService:
    """一次运行的状态机；同一个实例可服务多个 run（按 ``run_id`` 区分检查点线程）。"""

    def __init__(
        self,
        *,
        agent_id: str,
        store: RunStore,
        graph: CompiledStateGraph,
        checkpointer: Any | None = None,
        result_builder: ResultBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.store = store
        self.graph = graph
        self.checkpointer = (
            checkpointer if checkpointer is not None else getattr(graph, "checkpointer", None)
        )
        self._result_builder = result_builder
        self.settings = settings or get_settings()

    # ---- 配置 -----------------------------------------------------------
    def timeout_seconds(self, override: float | None = None) -> float:
        """本单的审批时限（秒）；``0`` 表示不限时（一直等人）。"""
        if override is not None:
            return max(0.0, float(override))
        return max(0.0, float(self.settings.agent_approval_timeout))

    @property
    def escalate_seconds(self) -> float:
        """首次超时后的升级宽限期（秒）；``0`` 表示到点直接自动拒绝。"""
        return max(0.0, float(self.settings.agent_approval_escalate_seconds))

    def config(self, thread_id: str) -> RunnableConfig:
        """图执行配置：检查点线程 + ``hitl`` 开关（业务图据此决定是否真的 ``interrupt``）。"""
        return {
            "configurable": {"thread_id": thread_id, "hitl": True},
            "recursion_limit": 50,
        }

    # ---- 启动 -----------------------------------------------------------
    def start(
        self,
        inputs: dict[str, Any],
        *,
        title: str = "",
        timeout_seconds: float | None = None,
        created_by: str | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
    ) -> RunRecord:
        """执行一次图：撞上审批点就停在 ``pending``，跑完了就是 ``completed``。

        ``inputs`` 由调用方按图自己的状态结构组装（报销是 claim + context），服务不猜测
        业务字段；``title`` 只用于列表展示（例如报销单号）。
        """
        now = datetime.now(UTC)
        run_id = run_id or uuid.uuid4().hex
        thread_id = thread_id or run_id
        timeout = self.timeout_seconds(timeout_seconds)
        payload = with_event(
            {"request": None, "events": []},
            event("running", by=created_by, note="开始执行"),
        )
        record = self.store.create(
            RunRecord(
                run_id=run_id,
                agent_id=self.agent_id,
                thread_id=thread_id,
                status="running",
                title=title[:200],
                payload=payload,
                created_at=now,
                updated_at=now,
                expires_at=now + timedelta(seconds=timeout) if timeout > 0 else None,
            )
        )
        started = time.perf_counter()
        try:
            values = self._invoke(inputs, record.thread_id)
        except Exception as exc:
            logger.exception("run %s failed", run_id)
            self._mark_failed(record, exc)
            raise
        return self._settle(
            record,
            values,
            started,
            timeout=timeout,
            final_status="completed",
            note=f"{created_by} 提交的执行完成" if created_by else "执行完成",
        )

    # ---- 审批 -----------------------------------------------------------
    def resume(
        self,
        run_id: str,
        *,
        decision: RunDecision,
        approver: str | None = None,
        comment: str | None = None,
    ) -> RunRecord:
        """按人工结论继续执行：``approve`` → ``approved``，``reject`` → ``rejected``。"""
        self.sweep()
        record = self._require(run_id)
        self._ensure_waiting(record, "审批")
        note = "人工通过" if decision == "approve" else "人工拒绝"
        claimed = self.store.transition(
            run_id,
            expected=sorted(WAITING_STATUSES),
            status="running",
            approver=approver,
            comment=comment,
            updated_at=datetime.now(UTC),
            payload=with_event(
                record.payload,
                event("running", by=approver, note=f"{note}，继续执行"),
            ),
        )
        if claimed is None:
            raise RunConflict("运行状态已被其他操作改变，请刷新后重试")
        return self._continue(
            claimed,
            decision=decision,
            approver=approver,
            comment=comment,
            timeout=self.timeout_seconds(),
            final_status="approved" if decision == "approve" else "rejected",
            note=note,
        )

    def cancel(
        self,
        run_id: str,
        *,
        approver: str | None = None,
        comment: str | None = None,
    ) -> RunRecord:
        """取消一次等待中的运行（不执行后续节点，检查点一并删除）。"""
        self.sweep()
        record = self._require(run_id)
        self._ensure_waiting(record, "取消")
        now = datetime.now(UTC)
        cancelled = self.store.transition(
            run_id,
            expected=sorted(WAITING_STATUSES),
            status="cancelled",
            approver=approver,
            comment=comment,
            updated_at=now,
            finished_at=now,
            payload=with_event(record.payload, event("cancelled", by=approver, note=comment or "运行已取消")),
        )
        if cancelled is None:
            raise RunConflict("运行状态已被其他操作改变，请刷新后重试")
        self._discard_checkpoint(record.thread_id)
        return cancelled

    # ---- 查询 -----------------------------------------------------------
    def get(self, run_id: str) -> RunRecord:
        """读取一次运行（先做一轮超时巡检，避免看到过期的 ``pending``）。"""
        self.sweep()
        record = self.store.get(run_id)
        if record is None:
            raise RunNotFound(f"运行不存在：{run_id}")
        return record

    def find(self, run_id: str) -> RunRecord | None:
        """按 id 查找；不存在返回 `None`（接口层决定 404 文案）。"""
        self.sweep()
        return self.store.get(run_id)

    def list_runs(
        self,
        *,
        statuses: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        """列出本应用的运行记录（最新在前）。"""
        self.sweep()
        return self.store.list_runs(agent_id=self.agent_id, statuses=statuses, limit=limit)

    # ---- 超时巡检 -------------------------------------------------------
    def sweep(self, *, now: datetime | None = None) -> list[RunRecord]:
        """处理到期的审批：``pending`` → ``escalated``，宽限期再到期 → ``timeout``。

        巡检查询只看 ``pending`` / ``escalated`` 且 ``expires_at <= now`` 的记录，所以
        反复调用是安全的；单条处理失败只记日志，不会挡住其它运行。
        """
        moment = now or datetime.now(UTC)
        resolved: list[RunRecord] = []
        for record in self.store.due(moment):
            try:
                updated = self._resolve_due(record, moment)
            except Exception:
                logger.exception("运行 %s 超时处理失败", record.run_id)
                continue
            if updated is not None:
                resolved.append(updated)
        return resolved

    def _resolve_due(self, record: RunRecord, now: datetime) -> RunRecord | None:
        """一条到期的运行：先升级（宽限一次），宽限期满再自动拒绝。"""
        if record.status == "pending" and self.escalate_seconds > 0:
            grace = self.escalate_seconds
            upgraded = self.store.transition(
                record.run_id,
                expected=("pending",),
                status="escalated",
                expires_at=now + timedelta(seconds=grace),
                updated_at=now,
                payload=with_event(
                    record.payload,
                    event(
                        "escalated",
                        by="系统",
                        note=f"超过审批时限未处理，已升级并延长 {grace:g} 秒",
                    ),
                ),
            )
            if upgraded is not None:
                logger.info("审批超时升级：%s", record.run_id)
            return upgraded
        return self._auto_reject(record, now)

    def _auto_reject(self, record: RunRecord, now: datetime) -> RunRecord | None:
        """超时自动拒绝：用 ``reject`` 恢复图，运行终态记为 ``timeout``。"""
        note = "审批超时，自动拒绝"
        claimed = self.store.transition(
            record.run_id,
            expected=sorted(WAITING_STATUSES),
            status="running",
            updated_at=now,
            payload=with_event(record.payload, event("running", by="系统", note=note)),
        )
        if claimed is None:
            return None
        logger.warning("审批超时自动拒绝：%s", record.run_id)
        return self._continue(
            claimed,
            decision="reject",
            approver="系统（超时）",
            comment=note,
            timeout=self.timeout_seconds(),
            final_status="timeout",
            note=note,
            timeout_resume=True,
        )

    # ---- 内部 -----------------------------------------------------------
    def _invoke(self, payload: Any, thread_id: str) -> dict[str, Any]:
        """跑图（首跑或 ``Command(resume=...)``），返回最终状态的值。"""
        result = self.graph.invoke(payload, self.config(thread_id))
        return dict(result or {})

    def _continue(
        self,
        record: RunRecord,
        *,
        decision: RunDecision,
        approver: str | None,
        comment: str | None,
        timeout: float,
        final_status: RunStatus,
        note: str,
        timeout_resume: bool = False,
    ) -> RunRecord:
        """恢复图执行并结算：可能再次暂停（多审批点），也可能跑完。"""
        started = time.perf_counter()
        value = resume_value(
            decision,
            approver=approver,
            comment=comment,
            timeout=timeout_resume,
        )
        try:
            values = self._invoke(Command(resume=value), record.thread_id)
        except Exception as exc:
            logger.exception("run %s resume failed", record.run_id)
            self._mark_failed(record, exc)
            raise
        return self._settle(
            record,
            values,
            started,
            timeout=timeout,
            final_status=final_status,
            approver=approver,
            comment=comment,
            note=note,
        )

    def _settle(
        self,
        record: RunRecord,
        values: dict[str, Any],
        started: float,
        *,
        timeout: float,
        final_status: RunStatus,
        approver: str | None = None,
        comment: str | None = None,
        note: str = "",
    ) -> RunRecord:
        """把一次图执行的结果落成运行状态：还有审批点 → pending，否则 → 终态。"""
        now = datetime.now(UTC)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        interrupts = self._interrupts(record.thread_id)
        if interrupts:
            request = ApprovalRequest.from_dict(interrupts[-1].value)
            payload = with_event(
                {**record.payload, "request": request.as_dict()},
                event("pending", note=f"等待审批：{request.title or request.summary or request.type}"),
            )
            waiting = self.store.transition(
                record.run_id,
                expected=("running",),
                status="pending",
                updated_at=now,
                expires_at=now + timedelta(seconds=timeout) if timeout > 0 else None,
                payload=payload,
            )
            if waiting is None:  # pragma: no cover - 状态被外部改动才会发生
                raise RunConflict(f"运行 {record.run_id} 已不在执行中，无法进入待审批")
            return waiting

        result = self._build_result(values, elapsed_ms)
        payload = with_event(
            {**record.payload, "request": record.request.as_dict() if record.request else None},
            event(final_status, by=approver, note=(note or final_status)[:_MAX_NOTE_CHARS]),
        )
        settled = self.store.transition(
            record.run_id,
            expected=("running",),
            status=final_status,
            approver=approver,
            comment=(comment or "")[:_MAX_NOTE_CHARS] or None,
            result=result,
            payload=payload,
            updated_at=now,
            finished_at=now,
        )
        if settled is None:  # pragma: no cover - 并发取消等极端情况
            raise RunConflict(f"运行 {record.run_id} 状态已变化，结果未写入")
        return settled

    def _interrupts(self, thread_id: str) -> tuple[Any, ...]:
        """读取检查点里待处理的审批请求（没有就是空元组）。"""
        state = self.graph.get_state(self.config(thread_id))
        return tuple(getattr(state, "interrupts", ()) or ())

    def _build_result(self, values: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
        if self._result_builder is None:
            return {"values": values, "elapsed_ms": elapsed_ms}
        return self._result_builder(values, elapsed_ms)

    def _mark_failed(self, record: RunRecord, exc: Exception) -> None:
        """把执行异常写进运行记录；调用方决定是否再把异常抛给接口。"""
        now = datetime.now(UTC)
        message = f"{exc.__class__.__name__}: {exc}"[:_MAX_NOTE_CHARS]
        with contextlib.suppress(Exception):
            self.store.transition(
                record.run_id,
                expected=("running",),
                status="failed",
                error=message,
                updated_at=now,
                finished_at=now,
                payload=with_event(record.payload, event("failed", by="系统", note=message)),
            )

    def _discard_checkpoint(self, thread_id: str) -> None:
        """删掉检查点线程（取消后不再需要恢复，也避免 SQLite 文件无限增长）。"""
        if self.checkpointer is None:
            return
        delete = getattr(self.checkpointer, "delete_thread", None)
        if delete is None:
            return
        try:
            delete(thread_id)
        except Exception:
            logger.warning("删除检查点线程 %s 失败", thread_id, exc_info=True)

    def _require(self, run_id: str) -> RunRecord:
        record = self.store.get(run_id)
        if record is None:
            raise RunNotFound(f"运行不存在：{run_id}")
        return record

    def _ensure_waiting(self, record: RunRecord, action: str) -> None:
        if record.status in TERMINAL_STATUSES:
            raise RunConflict(f"运行已结束（{record.status}），不能再{action}")
        if record.status not in WAITING_STATUSES:
            raise RunConflict(f"运行当前是 {record.status}，不在等待审批，不能{action}")
