"""人工审批（HITL）的运行状态模型。

一张运行（run）= 一次「可能停下来等人」的图执行。状态机：

    running ──(撞上 interrupt)──> pending ──(同意)──> approved
       │                            │  └──(拒绝)──> rejected
       │                            ├──(超时未批)──> escalated ──(宽限期内还没批)──> timeout
       │                            └──(取消)──────> cancelled
       └──(没有审批点，直接跑完)──> completed
       任意执行异常 ──────────────> failed

* ``pending`` / ``escalated`` 是**暂停**态：图停在审批节点，检查点在 SQLite 里，随时可恢复；
* ``approved`` / ``rejected`` / ``timeout`` / ``cancelled`` / ``completed`` / ``failed`` 是**终态**；
* 报销审批与工具审批共用这套状态机，区别只在 ``ApprovalRequest`` 的 ``type`` / ``tool`` / ``args``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

#: 运行状态（与 ``docs/mysql/tb_agent_run.sql`` 的 CHECK 约束一致）。
RunStatus = Literal[
    "running",
    "pending",
    "escalated",
    "approved",
    "rejected",
    "timeout",
    "cancelled",
    "completed",
    "failed",
]

#: 等人审批的状态：可以 ``resume`` / ``cancel``，也会被超时巡检处理。
WAITING_STATUSES: frozenset[str] = frozenset({"pending", "escalated"})

#: 终态：不再接受 resume / cancel。
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"approved", "rejected", "timeout", "cancelled", "completed", "failed"}
)

#: 审批结论（``POST /runs/{id}/resume`` 的 decision 字段）。
RunDecision = Literal["approve", "reject"]

#: 审批请求的类型：业务审批（报销单）、工具审批（写文件 / 删除等破坏性工具）。
APPROVAL_TYPES: tuple[str, ...] = ("approval", "tool_call")


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """一次审批请求，就是 ``interrupt()`` 的 payload。

    ``type="approval"`` 表示业务审批（报销单等），``type="tool_call"`` 表示工具审批；
    两者字段一致，前端与状态机不需要区分处理。
    """

    type: str = "approval"
    agent_id: str = ""
    node: str = ""
    title: str = ""
    summary: str = ""
    risk: str = "medium"
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    claim_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的 dict（``interrupt()`` 的入参与接口响应共用）。"""
        return {
            "type": self.type,
            "agent_id": self.agent_id,
            "node": self.node,
            "title": self.title,
            "summary": self.summary,
            "risk": self.risk,
            "tool": self.tool,
            "args": dict(self.args),
            "claim_id": self.claim_id,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> ApprovalRequest:
        """从 ``interrupt()`` 的值还原；字段缺失时按默认值补全（不抛异常）。"""
        data = raw if isinstance(raw, dict) else {}
        risk = str(data.get("risk") or "medium")
        return cls(
            type=str(data.get("type") or "approval"),
            agent_id=str(data.get("agent_id") or ""),
            node=str(data.get("node") or ""),
            title=str(data.get("title") or ""),
            summary=str(data.get("summary") or ""),
            risk=risk if risk in {"low", "medium", "high"} else "medium",
            tool=str(data["tool"]) if data.get("tool") else None,
            args=dict(data.get("args") or {}),
            claim_id=str(data["claim_id"]) if data.get("claim_id") else None,
            details=dict(data.get("details") or {}),
        )


@dataclass(frozen=True, slots=True)
class RunRecord:
    """一行 ``tb_agent_run``：一次运行的全部可见状态。"""

    run_id: str
    agent_id: str
    thread_id: str
    status: RunStatus
    title: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    approver: str | None = None
    comment: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def waiting(self) -> bool:
        """是否停在审批节点等人。"""
        return self.status in WAITING_STATUSES

    @property
    def terminal(self) -> bool:
        """是否已经是终态（不再接受审批 / 取消）。"""
        return self.status in TERMINAL_STATUSES

    @property
    def request(self) -> ApprovalRequest | None:
        """待审批内容；不是暂停态或 payload 损坏时返回 `None`。"""
        raw = (self.payload or {}).get("request")
        return ApprovalRequest.from_dict(raw) if raw else None

    @property
    def events(self) -> list[dict[str, Any]]:
        """状态流转流水（谁、什么时候、把状态改成了什么）。"""
        raw = (self.payload or {}).get("events")
        return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def resume_value(
    decision: RunDecision,
    *,
    approver: str | None = None,
    comment: str | None = None,
    timeout: bool = False,
) -> dict[str, Any]:
    """组装 ``Command(resume=...)`` 的值：审批节点读它，运行记录也读它。

    ``timeout=True`` 表示这不是人批的，而是超时巡检自动拒绝；节点与前端据此显示
    「审批超时自动拒绝」，而不是伪造一个审批人。
    """
    return {
        "decision": decision,
        "approver": approver or ("系统（超时）" if timeout else "审批人"),
        "comment": (comment or "").strip(),
        "timeout": timeout,
        "at": datetime.now(UTC).isoformat(),
    }


def event(
    status: str,
    *,
    by: str | None = None,
    note: str = "",
    at: datetime | None = None,
) -> dict[str, Any]:
    """一条状态流转记录（追加到 ``payload["events"]``）。"""
    moment = at or datetime.now(UTC)
    return {
        "at": moment.isoformat(),
        "status": status,
        "by": by,
        "note": note,
    }


def with_event(payload: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    """返回追加了一条事件的 payload（不改原对象）。"""
    events = list(payload.get("events") or [])
    events.append(item)
    return {**payload, "events": events}
