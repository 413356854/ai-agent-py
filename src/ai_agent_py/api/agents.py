"""Agent 应用接口：列表、LangGraph 图导出、样例与试跑。

图不是手画的：`describe_graph()` 调 `compiled.get_graph().draw_mermaid()`，
节点与边直接来自 LangGraph，前端只做渲染。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ai_agent_py import __version__
from ai_agent_py.config import PROJECT_ROOT
from ai_agent_py.reimburse import SAMPLES, SAMPLES_BY_ID, describe_graph, run_claim
from ai_agent_py.reimburse.io import build_template, dump_claim_json, parse_upload
from ai_agent_py.reimburse.params import ParamError, get_store
from ai_agent_py.reimburse.rules import DATA_DIR, load_params, load_rules

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


@router.post("/{agent_id}/run", summary="试跑一单")
def run(agent_id: str, payload: RunRequest) -> dict[str, Any]:
    _require(agent_id)
    if payload.claim is not None:
        claim = payload.claim
    elif payload.sample_id:
        sample = SAMPLES_BY_ID.get(payload.sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail=f"未找到样例：{payload.sample_id}")
        claim = sample["claim"]
    else:
        raise HTTPException(status_code=400, detail="需要 sample_id 或 claim 之一")
    try:
        return run_claim(claim)
    except Exception as exc:  # noqa: BLE001 - 单次试跑失败不该影响服务
        logger.exception("reimburse run failed")
        raise HTTPException(status_code=500, detail=f"试跑失败：{exc}") from exc


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
