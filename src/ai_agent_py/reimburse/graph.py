"""报销审核 Agent 的 LangGraph 装配。

    START → 文档接入 → 要素抽取 ─┬→ 硬规则   ─┐
                                 ├→ 参数比对 ─┼→ 判定融合 ─┬→ 结论输出 → 归档 → END
                                 └→ 语义审核 ─┘           └→ 人工复核 → 归档 → END

三类检查互不依赖，并行执行；`判定融合` 是汇合点（barrier），
只有它能决定"退回/通过/转人工"——语义疑点一律转人工，绝不自动退回。
"""

from __future__ import annotations

import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .rules import (
    Finding,
    load_params,
    load_rules,
    period_bounds,
    run_stage,
)
from .samples import COMPANY, HISTORY

NODE_META: dict[str, dict[str, str]] = {
    "ingest": {"gate": "G1", "label": "文档接入", "kind": "node", "desc": "归一化单据、必填字段校验"},
    "extract": {"gate": "G2", "label": "要素抽取", "kind": "node", "desc": "结构化发票字段并给出置信度"},
    "hard_rules": {"gate": "G3", "label": "硬规则", "kind": "checker", "desc": "票据/勾稽/审批（R1/R2/R6，纯程序）"},
    "policy_check": {"gate": "G4", "label": "参数比对", "kind": "checker", "desc": "标准与税务合规（R3/R5，查参数库）"},
    "semantic_review": {"gate": "G5", "label": "语义审核", "kind": "agent", "desc": "事由与票据相关性（R4，启发式可换 LLM）"},
    "decide": {"gate": "G6", "label": "判定融合", "kind": "node", "desc": "汇总 Finding，唯一的裁决点"},
    "conclude": {"gate": "G7", "label": "结论输出", "kind": "node", "desc": "通过/退回，附依据与建议"},
    "human_review": {"gate": "G8", "label": "人工复核", "kind": "human", "desc": "低置信或语义疑点，生成待办"},
    "archive": {"gate": "G9", "label": "归档", "kind": "node", "desc": "写入审计包与规则命中统计"},
}


class ClaimState(TypedDict, total=False):
    """图状态：``findings`` / ``trace`` 用 reducer 追加归并，其余键由节点覆盖。"""

    claim: dict[str, Any]
    context: dict[str, Any]
    normalized: dict[str, Any]
    confidence: float
    findings: Annotated[list[dict[str, Any]], operator.add]
    decision: dict[str, Any]
    trace: Annotated[list[dict[str, Any]], operator.add]
    status: str


def _event(node: str, started: float, summary: str) -> dict[str, Any]:
    """生成一条节点轨迹：门禁号、中文标签、耗时与一句话摘要。

    ``trace`` 用 ``operator.add`` 归并，每个节点只返回自己这一条；
    前端时间轴与归档留痕都读它。
    """
    meta = NODE_META.get(node, {})
    return {
        "node": node,
        "label": meta.get("label", node),
        "gate": meta.get("gate", ""),
        "ms": round((time.perf_counter() - started) * 1000, 2),
        "summary": summary,
    }


def _ctx(state: ClaimState) -> dict[str, Any]:
    """取运行上下文（企业主体、历史发票、报销期间与起止日），缺省按空对象处理。"""
    return state.get("context") or {}


# --------------------------------------------------------------------------- 节点


def node_ingest(state: ClaimState) -> dict[str, Any]:
    """G1 文档接入：归一化报销单字段，并把必填项校验结果记成 R0.1 阻断。

    业务上这是"收单"环节——单据号、报销人、期间、金额、发票缺一不可，缺了就不该
    往下走；``normalized`` 会补齐可选字段，后续节点只读它、不再碰原始 claim。
    """
    started = time.perf_counter()
    claim = state.get("claim") or {}
    findings: list[dict[str, Any]] = []
    missing = [key for key in ("claim_id", "employee", "period", "total", "invoices") if not claim.get(key)]
    if missing:
        findings.append(
            Finding(
                rule_id="R0.1",
                name="必填字段缺失",
                category="数据完整性",
                severity="阻断",
                stage="hard",
                message=f"缺少必填字段：{'、'.join(missing)}",
                citation="报销单填写规范",
                evidence={"missing": missing},
            ).as_dict()
        )
    normalized = {
        "claim_id": claim.get("claim_id", ""),
        "employee": claim.get("employee") or {},
        "period": claim.get("period", ""),
        "category": claim.get("category", ""),
        "purpose": claim.get("purpose", ""),
        "total": claim.get("total", "0"),
        "invoices": claim.get("invoices") or [],
        "approvals": claim.get("approvals") or [],
        "meta": claim.get("meta") or {},
        "attachments": claim.get("attachments") or [],
    }
    summary = f"{normalized['claim_id'] or '未编号'} · {len(normalized['invoices'])} 张票据"
    return {"normalized": normalized, "findings": findings, "trace": [_event("ingest", started, summary)]}


def node_extract(state: ClaimState) -> dict[str, Any]:
    """G2 要素抽取：按票据字段完整度折算 OCR 置信度，供判定融合决定能否自动裁决。

    业务含义：发票号 / 开票日期 / 价税合计任一缺失，就视作这一张没抽全，整体置信度
    打七折；低于 80% 时 ``node_decide`` 不再自动判定，直接转人工复核。
    """
    started = time.perf_counter()
    claim = state.get("claim") or {}
    base = float(claim.get("ocr_confidence", 0.9))
    invoices = claim.get("invoices") or []
    incomplete = sum(
        1
        for inv in invoices
        if not inv.get("number") or not inv.get("total") or not inv.get("date")
    )
    confidence = base if not incomplete else round(base * 0.7, 3)
    summary = f"票据 {len(invoices)} 张，字段缺失 {incomplete} 张，置信度 {confidence:.0%}"
    return {"confidence": confidence, "trace": [_event("extract", started, summary)]}


def _stage_node(name: str, stage: str):
    """工厂：把一个规则阶段（hard / policy / semantic）包装成 LangGraph 节点。

    三类检查共用同一执行逻辑——读归一化后的单据，跑该阶段全部规则，把 Finding 转成
    可 JSON 的 dict 并记一条轨迹；差异只在 ``stage`` 与节点名。
    """

    def _run(state: ClaimState) -> dict[str, Any]:
        """节点执行体：跑本阶段规则库，返回命中的 Finding 与轨迹。"""
        started = time.perf_counter()
        claim = state.get("normalized") or {}
        found = run_stage(stage, claim, load_params(), _ctx(state))
        summary = f"命中 {len(found)} 条" + ("：" + "、".join(f.rule_id for f in found) if found else "")
        return {"findings": [item.as_dict() for item in found], "trace": [_event(name, started, summary)]}

    return _run


node_hard_rules = _stage_node("hard_rules", "hard")
node_policy_check = _stage_node("policy_check", "policy")
node_semantic_review = _stage_node("semantic_review", "semantic")


def node_decide(state: ClaimState) -> dict[str, Any]:
    """G6 判定融合：整张图唯一的裁决点，按"置信度 → 阻断 → 风险 → 警告"定结论。

    业务规则：
    - 抽取置信度 < 80%：不做自动判定，转人工（不拿可能抽错的数据下结论）；
    - 有"阻断"级 Finding：退回，并列出规则号；
    - 有"高 / 中"风险（语义疑点等）：转人工，绝不自动退回；
    - 只有"警告"或没有命中：通过，警告作为风险提示保留。
    """
    started = time.perf_counter()
    findings = state.get("findings") or []
    confidence = float(state.get("confidence", 1.0))

    if confidence < 0.80:
        decision = {
            "status": "转人工",
            "reason": f"关键字段抽取置信度 {confidence:.0%} 低于 80%，不做自动判定",
            "confidence": confidence,
        }
    else:
        blocked = [item for item in findings if item.get("severity") == "阻断"]
        risky = [item for item in findings if item.get("severity") in ("高", "中")]
        warned = [item for item in findings if item.get("severity") == "警告"]
        if blocked:
            decision = {
                "status": "退回",
                "reason": f"存在 {len(blocked)} 项硬性不合规：" + "、".join(item["rule_id"] for item in blocked),
                "confidence": confidence,
            }
        elif risky:
            decision = {
                "status": "转人工",
                "reason": f"{len(risky)} 项疑点需人工确认：" + "、".join(item["rule_id"] for item in risky),
                "confidence": confidence,
            }
        elif warned:
            decision = {
                "status": "通过",
                "reason": f"附 {len(warned)} 条风险提示（不阻断）",
                "confidence": confidence,
            }
        else:
            decision = {"status": "通过", "reason": "未发现不合规项", "confidence": confidence}

    summary = f"{decision['status']} · {decision['reason']}"
    return {"decision": decision, "trace": [_event("decide", started, summary)]}


def node_conclude(state: ClaimState) -> dict[str, Any]:
    """G7 结论输出：把判定结果落成单据状态（通过 / 退回），供接口与前端展示。"""
    started = time.perf_counter()
    decision = state.get("decision") or {}
    status = decision.get("status", "通过")
    return {"status": status, "trace": [_event("conclude", started, f"输出结论：{status}")]}


def node_human_review(state: ClaimState) -> dict[str, Any]:
    """G8 人工复核：低置信或语义疑点生成人工待办，不做自动退回。

    业务约定：语义类疑点只能由人确认业务实质，机器只提示、不裁决。
    """
    started = time.perf_counter()
    decision = state.get("decision") or {}
    return {
        "status": "转人工",
        "trace": [_event("human_review", started, f"生成待办：{decision.get('reason', '')}")],
    }


def node_archive(state: ClaimState) -> dict[str, Any]:
    """G9 归档：把 Finding 与节点轨迹写入审计留痕（当前为轨迹占位，可接审计库）。"""
    started = time.perf_counter()
    findings = state.get("findings") or []
    return {
        "trace": [_event("archive", started, f"归档 {len(findings)} 条 Finding，留痕可追溯")],
    }


def route_after_decide(state: ClaimState) -> str:
    """条件边：判定为"转人工"走 human_review，其余（通过 / 退回）走结论输出。"""
    decision = state.get("decision") or {}
    return "human" if decision.get("status") == "转人工" else "conclude"


# --------------------------------------------------------------------------- 装配


def build_graph():
    """构建并编译报销审核图（无 checkpointer，便于导出与单次试跑）。"""
    builder = StateGraph(ClaimState)
    builder.add_node("ingest", node_ingest)
    builder.add_node("extract", node_extract)
    builder.add_node("hard_rules", node_hard_rules)
    builder.add_node("policy_check", node_policy_check)
    builder.add_node("semantic_review", node_semantic_review)
    builder.add_node("decide", node_decide)
    builder.add_node("conclude", node_conclude)
    builder.add_node("human_review", node_human_review)
    builder.add_node("archive", node_archive)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "extract")

    # 三类检查并行：互不依赖，且只有一条汇合到 decide
    for checker in ("hard_rules", "policy_check", "semantic_review"):
        builder.add_edge("extract", checker)
        builder.add_edge(checker, "decide")

    builder.add_conditional_edges(
        "decide", route_after_decide, {"conclude": "conclude", "human": "human_review"}
    )
    builder.add_edge("conclude", "archive")
    builder.add_edge("human_review", "archive")
    builder.add_edge("archive", END)
    return builder.compile()


def run_claim(claim: dict[str, Any], *, history: list[str] | None = None) -> dict[str, Any]:
    """跑一单：装配上下文、执行整张图，返回结论 / Finding / 节点轨迹。

    ``history`` 是已报销过的发票号（默认用样例里的演示数据），用于重复报销判定；
    Finding 按"阻断 → 高 → 中 → 警告 → 提示"排序，前端直接按序展示。
    """
    started = time.perf_counter()
    context = {
        "company": COMPANY,
        "history": sorted(history if history is not None else HISTORY),
        "period": claim.get("period", ""),
        "period_range": period_bounds(claim.get("period", "")),
    }
    graph = build_graph()
    result = graph.invoke({"claim": claim, "context": context, "findings": [], "trace": []})

    findings = sorted(
        result.get("findings") or [],
        key=lambda item: ({"阻断": 0, "高": 1, "中": 2, "警告": 3, "提示": 4}.get(item.get("severity"), 9), item.get("rule_id", "")),
    )
    return {
        "claim_id": (result.get("normalized") or {}).get("claim_id", ""),
        "status": result.get("status", ""),
        "decision": result.get("decision") or {},
        "confidence": result.get("confidence", 0.0),
        "findings": findings,
        "trace": result.get("trace") or [],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


# --------------------------------------------------------------------------- 图导出


def _localize_mermaid(text: str) -> str:
    """把 mermaid 里的节点标签换成中文（结构仍来自 LangGraph 导出）。"""
    for node_id, meta in NODE_META.items():
        label = meta["label"]
        text = re.sub(
            rf"(?<![\w]){re.escape(node_id)}\(([^)]*)\)",
            lambda _m, node_id=node_id, label=label: f"{node_id}({label})",
            text,
        )
    text = text.replace("__start__([<p>__start__</p>])", "__start__([<p>开始</p>])")
    text = text.replace("__end__([<p>__end__</p>])", "__end__([<p>结束</p>])")

    # LangGraph 的条件边会输出 `-. &nbsp;human&nbsp; .->`，这里改成标准带标签的边，
    # 只影响展示，不改图的拓扑（节点与边仍来自 get_graph()）。
    text = text.replace("decide -.-> conclude;", "decide -->|通过 / 退回| conclude;")
    text = re.sub(
        r"decide -\.\s*&nbsp;(\w+)&nbsp;\s*\.-> human_review;",
        lambda _m: "decide -.->|转人工| human_review;",
        text,
    )
    return text


def describe_graph() -> dict[str, Any]:
    """导出 LangGraph 的图：mermaid 源码 + 节点/边（供前端渲染与节点清单）。"""
    compiled = build_graph()
    raw = compiled.get_graph()
    mermaid_raw = raw.draw_mermaid()
    mermaid = _localize_mermaid(mermaid_raw)

    nodes = []
    for node in raw.nodes.values():
        if node.id.startswith("__"):
            continue
        meta = NODE_META.get(node.id, {})
        nodes.append(
            {
                "id": node.id,
                "label": meta.get("label", node.name),
                "gate": meta.get("gate", ""),
                "kind": meta.get("kind", "node"),
                "description": meta.get("desc", ""),
            }
        )

    edges = [
        {"source": edge.source, "target": edge.target, "conditional": bool(edge.conditional)}
        for edge in raw.edges
        if not edge.source.startswith("__") and not edge.target.startswith("__")
    ]

    return {
        "mermaid": mermaid,
        "mermaid_source": mermaid_raw,
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "rule_count": len(load_rules()),
        "param_status": load_params().get("status", ""),
    }
