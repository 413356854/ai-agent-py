"""Agent 应用接口：列表、LangGraph 图导出、样例、试跑与人工审批（HITL）。

图不是手画的：`describe_graph()` 调 `compiled.get_graph().draw_mermaid()`，
节点与边直接来自 LangGraph，前端只做渲染。

人工审批分两条路径：

* ``POST /{agent_id}/run``：单次试跑，不停顿（G8 只产出「转人工」待办），用于快速预览；
* ``POST /{agent_id}/runs``：正式运行，图停在 G8 并落检查点，``runs/{id}/resume`` 用
  ``Command(resume=...)`` 继续；``runs/{id}/cancel`` 撤单；超时由运行服务巡检自动升级 / 拒绝。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from functools import lru_cache
from typing import Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ai_agent_py.config import PROJECT_ROOT, Settings, get_settings
from ai_agent_py.errors import AiAgentError, DatabaseError, RunConflict, RunNotFound
from ai_agent_py.graph.checkpoint import SqliteCheckpointer
from ai_agent_py.reimburse import (
    SAMPLES,
    SAMPLES_BY_ID,
    build_context,
    build_graph,
    build_result,
    describe_graph,
    run_claim,
)
from ai_agent_py.reimburse.io import build_template, dump_claim_json, parse_upload
from ai_agent_py.reimburse.params import ParamError, get_store
from ai_agent_py.reimburse.rules import DATA_DIR, load_params, load_rules
from ai_agent_py.runs.models import RunRecord
from ai_agent_py.runs.service import RunService
from ai_agent_py.runs.store import InMemoryRunStore, MySQLRunStore, RunStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])

AGENTS: dict[str, dict[str, Any]] = {
    "reimburse": {
        "id": "reimburse",
        "name": "报销审核 Agent",
        "summary": "读报销单与发票，按规则库判定通过 / 退回 / 转人工，并给出依据与建议。",
        "status": "MVP",
        "version": "0.1.0",
        "tags": ["LangGraph", "规则引擎", "财务共享"],
    }
}


class AgentSummary(BaseModel):
    id: str
    name: str
    summary: str
    status: str
    version: str
    tags: list[str]
    node_count: int
    rule_count: int


class GraphNode(BaseModel):
    id: str
    label: str
    gate: str
    kind: str
    description: str


class GraphEdge(BaseModel):
    source: str
    target: str
    conditional: bool


class AgentGraph(BaseModel):
    mermaid: str
    mermaid_source: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    node_count: int
    edge_count: int
    rule_count: int
    param_status: str


class AgentDetail(AgentSummary):
    graph: AgentGraph


class SampleBrief(BaseModel):
    sample_id: str
    title: str
    expected: str
    note: str
    claim: dict[str, Any] = Field(description="样例单据原文，供试跑页对照结果")


class RunRequest(BaseModel):
    sample_id: str | None = Field(default=None, description="用内置样例试跑")
    claim: dict[str, Any] | None = Field(default=None, description="直接传一张报销单")


class ApprovalOut(BaseModel):
    """一次审批请求（``interrupt()`` 的 payload）：业务审批与工具审批同构。"""

    type: str = "approval"
    agent_id: str = ""
    node: str = ""
    title: str = ""
    summary: str = ""
    risk: str = "medium"
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    claim_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RunEventOut(BaseModel):
    """运行状态流转的一条记录。"""

    at: str
    status: str
    by: str | None = None
    note: str = ""


class AgentRunOut(BaseModel):
    """一次运行（``tb_agent_run``）：待审批内容、结果与流转记录。"""

    run_id: str
    agent_id: str
    thread_id: str
    status: str = Field(
        description="running / pending / escalated / approved / rejected / timeout / cancelled / completed / failed"
    )
    title: str = ""
    created_at: str
    updated_at: str
    expires_at: str | None = Field(default=None, description="审批截止时间（UTC）")
    finished_at: str | None = None
    approver: str | None = None
    comment: str | None = None
    error: str | None = None
    approval: ApprovalOut | None = Field(default=None, description="需要人拍板的内容")
    result: dict[str, Any] | None = Field(default=None, description="跑完后的结论 / Finding / 轨迹")
    events: list[RunEventOut] = Field(default_factory=list)


class RunStartRequest(BaseModel):
    """提交一次正式运行（可能停在审批点）。"""

    sample_id: str | None = Field(default=None, description="用内置样例运行")
    claim: dict[str, Any] | None = Field(default=None, description="直接传一张报销单")
    timeout_seconds: float | None = Field(
        default=None,
        ge=0,
        description="本次审批时限（秒）；默认取 AGENT_APPROVAL_TIMEOUT，0 表示一直等人",
    )
    created_by: str | None = Field(default=None, max_length=64, description="提交人")


class RunResumeRequest(BaseModel):
    """审批结论：同意则继续归档，拒绝则退回。"""

    decision: Literal["approve", "reject"]
    approver: str | None = Field(default=None, max_length=64, description="审批人")
    comment: str | None = Field(default=None, max_length=500, description="审批意见")


class RunCancelRequest(BaseModel):
    """取消一次等待中的运行。"""

    approver: str | None = Field(default=None, max_length=64)
    comment: str | None = Field(default=None, max_length=500)


class RuleOut(BaseModel):
    id: str
    name: str
    category: str
    severity: str
    stage: str
    citation: str
    params: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    rules: list[RuleOut]
    params: dict[str, Any]
    param_status: str
    param_note: str
    param_file: str
    param_version: str
    param_updated_at: str | None = None


class ParamsUpdate(BaseModel):
    params: dict[str, Any] = Field(description="完整的参数对象，整体覆盖写入")


class UploadResult(BaseModel):
    filename: str
    claim: dict[str, Any]
    warnings: list[str]


@lru_cache(maxsize=1)
def _graph_payload() -> dict[str, Any]:
    return describe_graph()


# ---- 人工审批（HITL）运行服务 ----------------------------------------------
_HOST = "agent_runs"


def _run_store(settings: Settings) -> RunStore:
    """运行记录库：配了 MySQL 就用 ``tb_agent_run``，否则退化为内存（重启即丢）。"""
    if not settings.mysql_url:
        logger.warning(
            "未配置 MYSQL_URL：审批运行记录只存在内存里（重启即丢；图检查点仍落 SQLite）"
        )
        return InMemoryRunStore()
    return MySQLRunStore.from_settings(settings)


def build_run_service(settings: Settings | None = None, agent_id: str = "reimburse") -> RunService:
    """装配一个应用的运行服务：图 + 检查点 + 运行记录库。

    检查点默认落 ``AGENT_RUN_CHECKPOINT_PATH``（SQLite），图停在 G8 后进程重启仍能恢复；
    运行记录默认落 ``tb_agent_run``（MySQL），未配置数据库时用内存实现。
    """
    cfg = settings or get_settings()
    if agent_id != "reimburse":
        raise ValueError(f"应用 {agent_id} 还没有审批图")
    checkpointer = SqliteCheckpointer(cfg.agent_run_checkpoint_path)
    return RunService(
        agent_id=agent_id,
        store=_run_store(cfg),
        graph=build_graph(checkpointer=checkpointer),
        checkpointer=checkpointer,
        result_builder=build_result,
        settings=cfg,
    )


def get_run_service(request: Request) -> RunService:
    """每个 app 一个运行服务实例（挂在 ``app.state``，测试可直接替换）。"""
    service: RunService | None = getattr(request.app.state, _HOST, None)
    if service is None:
        try:
            service = build_run_service(get_settings())
        except AiAgentError as exc:
            raise HTTPException(status_code=503, detail=f"审批运行服务不可用：{exc}") from exc
        request.app.state.agent_runs = service
    return service


def _run_out(record: RunRecord) -> AgentRunOut:
    """运行记录 → 接口响应（时间统一 ISO 8601，事件按发生顺序）。"""
    request = record.request
    return AgentRunOut(
        run_id=record.run_id,
        agent_id=record.agent_id,
        thread_id=record.thread_id,
        status=record.status,
        title=record.title,
        created_at=record.created_at.isoformat(),
        updated_at=record.updated_at.isoformat(),
        expires_at=record.expires_at.isoformat() if record.expires_at else None,
        finished_at=record.finished_at.isoformat() if record.finished_at else None,
        approver=record.approver,
        comment=record.comment,
        error=record.error,
        approval=ApprovalOut(**request.as_dict()) if request else None,
        result=record.result,
        events=[
            RunEventOut(
                at=str(item.get("at") or ""),
                status=str(item.get("status") or ""),
                by=str(item["by"]) if item.get("by") else None,
                note=str(item.get("note") or ""),
            )
            for item in record.events
        ],
    )


def _claim_from(payload: RunRequest | RunStartRequest) -> dict[str, Any]:
    """样例 id / 直接传单据 → 一张报销单（两个入口共用）。"""
    if payload.claim is not None:
        return payload.claim
    if payload.sample_id:
        sample = SAMPLES_BY_ID.get(payload.sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail=f"未找到样例：{payload.sample_id}")
        return sample["claim"]
    raise HTTPException(status_code=400, detail="需要 sample_id 或 claim 之一")


def _run_error(exc: Exception, action: str) -> HTTPException:
    """运行异常 → HTTP：找不到 404、状态冲突 409、库不可用 503、其余 500。

    404 / 409 是客户端可预期的结果，只回中文原因；真正的故障（数据库、图执行）
    才打堆栈，避免重复审批把日志刷满。
    """
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, RunNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, RunConflict):
        return HTTPException(status_code=409, detail=str(exc))
    logger.exception("%s失败", action)
    if isinstance(exc, DatabaseError):
        return HTTPException(status_code=503, detail=f"{action}失败：{exc}")
    return HTTPException(status_code=500, detail=f"{action}失败：{exc}")


def _owned(record: RunRecord, agent_id: str) -> RunRecord:
    """运行记录必须属于路径里的应用，否则当成不存在。"""
    if record.agent_id != agent_id:
        raise HTTPException(status_code=404, detail=f"运行不存在：{record.run_id}")
    return record


def _require_run(service: RunService, run_id: str, agent_id: str) -> RunRecord:
    """读一条运行并校验归属（顺带触发一次超时巡检）。"""
    record = service.find(run_id)
    if record is None:
        raise RunNotFound(f"运行不存在：{run_id}")
    return _owned(record, agent_id)


#: 后台审批巡检间隔（秒）：没人打开页面也会按时升级 / 自动拒绝。
APPROVAL_SWEEP_INTERVAL = 30.0


def start_approval_sweeper(app: Any) -> None:
    """启动后台审批巡检（应用 startup 调用）。

    巡检不自己建运行服务：首次用到审批接口时 `get_run_service` 才装配（SQLite 检查点 /
    MySQL 连接池），巡检只负责定时 `sweep` 已经存在的那个实例。这样应用启动不会因为
    审批相关的依赖缺失而失败，测试里也不会凭空建出 ``data/agent_runs.sqlite3``。
    """
    if getattr(app.state, "approval_sweeper", None) is not None:
        return

    async def _loop() -> None:
        while True:
            await asyncio.sleep(APPROVAL_SWEEP_INTERVAL)
            service = getattr(app.state, _HOST, None)
            if service is None:
                continue
            try:
                await asyncio.to_thread(service.sweep)
            except Exception:  # 单轮失败不影响下一轮
                logger.exception("审批超时巡检失败")

    app.state.approval_sweeper = asyncio.create_task(_loop(), name="approval-sweeper")
    logger.info("审批超时巡检已启动（每 %.0f 秒）", APPROVAL_SWEEP_INTERVAL)


async def stop_approval_sweeper(app: Any) -> None:
    """停止后台审批巡检（应用 shutdown 调用）。"""
    task = getattr(app.state, "approval_sweeper", None)
    app.state.approval_sweeper = None
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def _require(agent_id: str) -> dict[str, Any]:
    agent = AGENTS.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"未登记的应用：{agent_id}")
    return agent


def _summary(agent: dict[str, Any]) -> AgentSummary:
    graph = _graph_payload()
    return AgentSummary(
        **agent,
        node_count=graph["node_count"],
        rule_count=graph["rule_count"],
    )


@router.get("", response_model=list[AgentSummary], summary="已登记的 Agent 应用")
def list_agents() -> list[AgentSummary]:
    return [_summary(agent) for agent in AGENTS.values()]


@router.get("/{agent_id}", response_model=AgentDetail, summary="应用详情（含流程图）")
def get_agent(agent_id: str) -> AgentDetail:
    agent = _require(agent_id)
    graph = _graph_payload()
    return AgentDetail(**_summary(agent).model_dump(), graph=AgentGraph(**graph))


@router.get("/{agent_id}/graph", response_model=AgentGraph, summary="LangGraph 导出的流程图")
def get_graph(agent_id: str) -> AgentGraph:
    _require(agent_id)
    return AgentGraph(**_graph_payload())


@router.get("/{agent_id}/samples", response_model=list[SampleBrief], summary="内置样例")
def list_samples(agent_id: str) -> list[SampleBrief]:
    _require(agent_id)
    return [
        SampleBrief(
            sample_id=item["sample_id"],
            title=item["title"],
            expected=item["expected"],
            note=item["note"],
            claim=item["claim"],
        )
        for item in SAMPLES
    ]


@router.get("/{agent_id}/config", response_model=AgentConfig, summary="规则库与参数")
def get_config(agent_id: str) -> AgentConfig:
    _require(agent_id)
    params = load_params()
    try:
        param_file = (DATA_DIR / "policy_params.yaml").relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        param_file = (DATA_DIR / "policy_params.yaml").as_posix()
    return AgentConfig(
        rules=[
            RuleOut(
                id=rule.id,
                name=rule.name,
                category=rule.category,
                severity=rule.severity,
                stage=rule.stage,
                citation=rule.citation,
                params=list(rule.params),
            )
            for rule in load_rules()
        ],
        params=params.get("params", {}),
        param_status=str(params.get("status", "")),
        param_note=str(params.get("note", "")),
        param_file=param_file,
        param_version=str(params.get("version", "")),
        param_updated_at=params.get("updated_at"),
    )


@router.put("/{agent_id}/config/params", response_model=AgentConfig, summary="修改参数（落盘并刷新缓存）")
def update_params(agent_id: str, payload: ParamsUpdate) -> AgentConfig:
    _require(agent_id)
    try:
        get_store().save(payload.params)
    except ParamError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("saving params failed")
        raise HTTPException(status_code=500, detail=f"参数落盘失败：{exc}") from exc
    return get_config(agent_id)


@router.post("/{agent_id}/config/reload", response_model=AgentConfig, summary="重新读取参数文件")
def reload_config(agent_id: str) -> AgentConfig:
    _require(agent_id)
    get_store().invalidate()
    return get_config(agent_id)


@router.post("/{agent_id}/run", summary="试跑一单（不在审批点暂停）")
def run(agent_id: str, payload: RunRequest) -> dict[str, Any]:
    """单次试跑：一口气跑完，G8 只产出「转人工」待办。

    需要真正暂停 / 审批 / 恢复请用 ``POST /api/agents/{agent_id}/runs``。
    """
    _require(agent_id)
    claim = _claim_from(payload)
    try:
        return run_claim(claim)
    except Exception as exc:
        logger.exception("reimburse run failed")
        raise HTTPException(status_code=500, detail=f"试跑失败：{exc}") from exc


# ---- 人工审批（HITL）-------------------------------------------------------
@router.post(
    "/{agent_id}/runs",
    response_model=AgentRunOut,
    status_code=201,
    summary="提交运行（遇到审批点自动暂停）",
)
def start_run(agent_id: str, payload: RunStartRequest, request: Request) -> AgentRunOut:
    """提交一次正式运行：需要人拍板的单子会停在 G8，返回 ``status=pending`` 与审批内容。"""
    _require(agent_id)
    claim = _claim_from(payload)
    service = get_run_service(request)
    try:
        record = service.start(
            {"claim": claim, "context": build_context(claim), "findings": [], "trace": []},
            title=str(claim.get("claim_id") or ""),
            timeout_seconds=payload.timeout_seconds,
            created_by=payload.created_by,
        )
    except Exception as exc:
        raise _run_error(exc, "提交运行") from exc
    return _run_out(record)


@router.get("/{agent_id}/runs", response_model=list[AgentRunOut], summary="运行 / 待审批列表")
def list_runs(
    agent_id: str,
    request: Request,
    status: str | None = Query(
        default=None,
        description="按状态过滤，多个用逗号分隔（pending,escalated 只看待审批）",
    ),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[AgentRunOut]:
    _require(agent_id)
    service = get_run_service(request)
    statuses = [item.strip() for item in status.split(",") if item.strip()] if status else None
    try:
        records = service.list_runs(statuses=statuses, limit=limit)
    except Exception as exc:
        raise _run_error(exc, "读取运行列表") from exc
    return [_run_out(record) for record in records]


@router.get("/{agent_id}/runs/{run_id}", response_model=AgentRunOut, summary="运行详情")
def get_run(agent_id: str, run_id: str, request: Request) -> AgentRunOut:
    _require(agent_id)
    service = get_run_service(request)
    try:
        record = _require_run(service, run_id, agent_id)
    except Exception as exc:
        raise _run_error(exc, "读取运行") from exc
    return _run_out(record)


@router.post(
    "/{agent_id}/runs/{run_id}/resume",
    response_model=AgentRunOut,
    summary="审批并继续执行",
)
def resume_run(
    agent_id: str,
    run_id: str,
    payload: RunResumeRequest,
    request: Request,
) -> AgentRunOut:
    """人工审批：``approve`` 继续归档，``reject`` 退回；重复提交返回 409。"""
    _require(agent_id)
    service = get_run_service(request)
    try:
        _require_run(service, run_id, agent_id)
        record = service.resume(
            run_id,
            decision=payload.decision,
            approver=payload.approver,
            comment=payload.comment,
        )
    except Exception as exc:
        raise _run_error(exc, "审批") from exc
    return _run_out(record)


@router.post(
    "/{agent_id}/runs/{run_id}/cancel",
    response_model=AgentRunOut,
    summary="取消等待中的运行",
)
def cancel_run(
    agent_id: str,
    run_id: str,
    request: Request,
    payload: RunCancelRequest | None = None,
) -> AgentRunOut:
    """撤单：不再执行后续节点，检查点一并删除；已审批 / 已取消的运行返回 409。"""
    _require(agent_id)
    service = get_run_service(request)
    body = payload or RunCancelRequest()
    try:
        _require_run(service, run_id, agent_id)
        record = service.cancel(run_id, approver=body.approver, comment=body.comment)
    except Exception as exc:
        raise _run_error(exc, "取消运行") from exc
    return _run_out(record)


XLSX_MEDIA = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _attachment(content: bytes, media_type: str, filename: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )


@router.get("/{agent_id}/template", summary="空白报销单模板（xlsx）")
def download_template(agent_id: str) -> Response:
    _require(agent_id)
    return _attachment(build_template(), XLSX_MEDIA, "报销单模板.xlsx")


@router.get("/{agent_id}/samples/{sample_id}/download", summary="样例单据下载（xlsx / json）")
def download_sample(agent_id: str, sample_id: str, fmt: str = "xlsx") -> Response:
    _require(agent_id)
    sample = SAMPLES_BY_ID.get(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"未找到样例：{sample_id}")
    if fmt == "json":
        return _attachment(dump_claim_json(sample["claim"]), "application/json", f"报销单样板_{sample_id}.json")
    return _attachment(build_template(sample["claim"]), XLSX_MEDIA, f"报销单样板_{sample_id}.xlsx")


@router.post("/{agent_id}/upload", response_model=UploadResult, summary="上传单据并解析")
async def upload_claim(agent_id: str, file: UploadFile = File(...)) -> UploadResult:
    _require(agent_id)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件超过 5 MB")
    filename = file.filename or "upload.json"
    try:
        claim, warnings = parse_upload(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UploadResult(filename=filename, claim=claim, warnings=warnings)
