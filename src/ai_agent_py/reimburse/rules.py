"""报销审核规则库：YAML 元数据 + 判定器注册表。

设计约定：
- YAML 只放**元数据与参数引用**，判定逻辑在 Python 注册表里按 `check` 名查找；
- 每条规则标注 `stage`（hard / policy / semantic），决定它跑在图的哪个节点；
- 判定器返回"证据字典列表"，文案由规则模板格式化，保证同一规则输出稳定。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

from .params import get_store

DATA_DIR = Path(__file__).with_name("data")
ZERO = Decimal("0")

SEVERITY_ORDER = {"阻断": 0, "高": 1, "中": 2, "警告": 3, "提示": 4}


@dataclass(frozen=True)
class RuleSpec:
    """一条规则的元数据（来自 rules.yaml）：编号、级别、阶段与判定器名。"""

    id: str
    name: str
    category: str
    severity: str
    stage: str
    check: str
    message: str
    citation: str
    # 该规则依赖的参数键（对应 policy_params.yaml 的 params 段）
    params: tuple[str, ...] = ()


@dataclass(frozen=True)
class Finding:
    """一条命中记录：规则号 + 定级 + 文案 + 证据，是图中流转的最小业务单元。"""

    rule_id: str
    name: str
    category: str
    severity: str
    stage: str
    message: str
    citation: str
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典，供图状态、接口与归档使用。"""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "category": self.category,
            "severity": self.severity,
            "stage": self.stage,
            "message": self.message,
            "citation": self.citation,
            "evidence": self.evidence,
        }


def dec(value: Any, default: Decimal = ZERO) -> Decimal:
    """把金额字段稳妥地转成 Decimal：空值取默认值，千分位/非法输入也不抛异常。"""
    if value is None or value == "":
        return default
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return default


def money(value: Any) -> str:
    """金额展示格式：千分位 + 两位小数，Finding 文案统一用它保证输出稳定。"""
    return f"{dec(value):,.2f}"


def safe_format(template: str, evidence: dict[str, Any]) -> str:
    """用证据字典渲染规则文案；缺字段时保留 ``{占位符}``，不让文案格式化把规则跑挂。"""

    class _Missing(dict):
        def __missing__(self, key: str) -> str:  # 缺字段时保留占位符，不抛异常
            """缺字段时返回占位符本身，保证文案渲染不中断。"""
            return "{" + key + "}"

    try:
        return template.format_map(_Missing(**evidence))
    except (ValueError, IndexError):
        return template


@lru_cache(maxsize=1)
def load_rules() -> tuple[RuleSpec, ...]:
    """读取 rules.yaml 的规则元数据（进程内缓存，改 YAML 后需重启或清缓存）。"""
    raw = yaml.safe_load((DATA_DIR / "rules.yaml").read_text(encoding="utf-8"))
    return tuple(
        RuleSpec(**{**item, "params": tuple(item.get("params") or ())}) for item in raw["rules"]
    )


def load_params() -> dict[str, Any]:
    """参数走 PolicyParamsStore 的自定义缓存（可落盘、可失效），不用 lru_cache。"""
    return get_store().load()


# --------------------------------------------------------------------------- 判定器
# 每个判定器签名： (claim, params, ctx) -> list[evidence dict]


def _invoices(claim: dict[str, Any]) -> list[dict[str, Any]]:
    """取单据下的发票列表，缺字段按空列表处理。"""
    return list(claim.get("invoices") or [])


def _items(invoice: dict[str, Any]) -> list[dict[str, Any]]:
    """取单张发票的明细行（品名 + 金额），住宿与语义类判定要看它。"""
    return list(invoice.get("items") or [])


def check_duplicate_invoice(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R1.1 重复报销：发票号命中历史已报销清单即阻断（一票多报）。"""
    history = ctx.get("history") or set()
    return [{"number": str(inv.get("number"))} for inv in _invoices(claim) if str(inv.get("number")) in history]


def check_buyer_mismatch(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R1.3 抬头核对：购买方名称或税号不是本企业主体，发票不得税前扣除。"""
    company = ctx.get("company") or {}
    out = []
    for inv in _invoices(claim):
        if inv.get("buyer") != company.get("name") or inv.get("buyer_tax_id") != company.get("tax_id"):
            out.append({"number": inv.get("number"), "buyer": inv.get("buyer", "")})
    return out


def check_date_out_of_period(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R1.4 跨期票据：开票日期不在报销期间内（日期解析失败按跨期处理）。"""
    start, end = ctx["period_range"]
    period = ctx["period"]
    out = []
    for inv in _invoices(claim):
        try:
            issued = date.fromisoformat(str(inv.get("date")))
        except ValueError:
            out.append({"date": str(inv.get("date")), "period": period})
            continue
        if not (start <= issued <= end):
            out.append({"date": issued.isoformat(), "period": period})
    return out


def check_kind_mismatch(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R1.5 票种与费用类型不符：如拿餐饮票报办公费，按参数库 ``allowed_kinds`` 判定。"""
    allowed = (params.get("params", {}).get("allowed_kinds") or {}).get(claim.get("category"), [])
    if not allowed:
        return []
    return [
        {"category": claim.get("category"), "kind": inv.get("kind")}
        for inv in _invoices(claim)
        if inv.get("kind") not in allowed
    ]


def check_amount_sum_mismatch(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R2.1 金额勾稽：全部发票价税合计之和必须等于报销单填写的报销金额。"""
    total = dec(claim.get("total"))
    subtotal = sum((dec(inv.get("total")) for inv in _invoices(claim)), ZERO)
    if subtotal == total:
        return []
    return [{"sum": money(subtotal), "total": money(total), "diff": money(abs(total - subtotal))}]


def check_tax_math_mismatch(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R2.2 票面勾稽：每张发票的金额 + 税额必须等于价税合计。"""
    out = []
    for inv in _invoices(claim):
        amount, tax, total = dec(inv.get("amount")), dec(inv.get("tax")), dec(inv.get("total"))
        if amount + tax != total:
            out.append({"number": inv.get("number"), "amount": money(amount), "tax": money(tax), "total": money(total)})
    return out


CONTRACT_WORDS = ("合同", "订单", "协议", "contract")


def check_invoice_over_limit_without_contract(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R2.4 大额无合同：单张发票达到合同阈值、附件却没有合同/订单 → 警告（不阻断）。"""
    threshold = params.get("params", {}).get("invoice_amount_requires_contract")
    if threshold is None:
        return []
    limit = dec(threshold)
    attachments = " ".join(str(item) for item in claim.get("attachments") or []).lower()
    if any(word in attachments for word in CONTRACT_WORDS):
        return []
    out = []
    for inv in _invoices(claim):
        total = dec(inv.get("total"))
        if total >= limit:
            out.append({"number": inv.get("number"), "total": money(total), "threshold": money(limit)})
    return out


def check_lodging_over_cap(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R3.1 住宿超标：住宿类发票按"价税合计 ÷ 晚数"算每晚房价，对照城市档上限。"""
    caps = params.get("params", {}).get("lodging_cap_per_night") or {}
    out = []
    for inv in _invoices(claim):
        nights = int(inv.get("nights") or 0)
        if nights <= 0:
            continue
        names = " ".join(str(item.get("name", "")) for item in _items(inv))
        if "住宿" not in names and "房" not in names:
            continue
        tier = str(inv.get("city_tier") or "二线")
        cap = caps.get(tier)
        nightly = dec(inv.get("total")) / nights
        if cap is not None and nightly > dec(cap):
            out.append({"nightly": money(nightly), "tier": tier, "cap": cap})
    return out


def check_entertainment_over_cap(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R3.2 招待超标：招待费按"报销金额 ÷ 招待人数"算人均，超过标准给警告。"""
    if claim.get("category") != "招待":
        return []
    headcount = int((claim.get("meta") or {}).get("headcount") or 0)
    if headcount <= 0:
        return []
    cap = params.get("params", {}).get("entertainment_per_head")
    per_head = dec(claim.get("total")) / headcount
    if cap is not None and per_head > dec(cap):
        return [{"per_head": money(per_head), "cap": cap}]
    return []


def check_receipt_kind(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R5.3 凭证合规：收据 / 白条 / 内部单据不属于合规税前扣除凭证，需换开正式发票。"""
    bad = {"收据", "白条", "内部单据", "无票"}
    return [{"kind": inv.get("kind")} for inv in _invoices(claim) if inv.get("kind") in bad]


def check_self_approval(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R6.2 审批合规：报销人同时出现在审批链里，违反不相容岗位分离。"""
    name = (claim.get("employee") or {}).get("name")
    return [{"approver": item.get("name")} for item in claim.get("approvals") or [] if item.get("name") == name]


CATEGORY_WORDS = {
    "差旅": ("住宿", "酒店", "机票", "火车", "高铁", "打车", "出租", "交通", "行程", "航空"),
    "招待": ("餐饮", "餐费", "宴请", "酒水", "会议", "招待"),
    "办公": ("办公", "打印", "耗材", "文具", "电脑", "软件", "设备"),
    "交通": ("打车", "出租", "地铁", "公交", "加油", "停车", "交通"),
}


def check_purpose_mismatch(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R4.2 语义相关性（启发式）：按费用类型的关键词核对发票明细，对不上转人工确认。"""
    words = CATEGORY_WORDS.get(str(claim.get("category")), ())
    if not words:
        return []
    names = [str(item.get("name", "")) for inv in _invoices(claim) for item in _items(inv)]
    if any(word in name for name in names for word in words):
        return []
    return [{"category": claim.get("category"), "items": "、".join(names) or "（无明细）"}]


def check_consecutive_invoice(claim: dict, params: dict, ctx: dict) -> list[dict]:
    """R4.4 连号发票：发票号连续出现，疑似集中开票或拆分报销，提示人工。"""
    numbers = sorted(int(inv["number"]) for inv in _invoices(claim) if str(inv.get("number", "")).isdigit())
    runs: list[str] = []
    for prev, cur in zip(numbers, numbers[1:]):
        if cur - prev == 1:
            runs.extend([str(prev), str(cur)])
    if not runs:
        return []
    unique = sorted(set(runs), key=int)
    return [{"numbers": "、".join(unique)}]


CHECKS: dict[str, Callable[[dict, dict, dict], list[dict]]] = {
    "duplicate_invoice": check_duplicate_invoice,
    "buyer_mismatch": check_buyer_mismatch,
    "date_out_of_period": check_date_out_of_period,
    "kind_mismatch": check_kind_mismatch,
    "amount_sum_mismatch": check_amount_sum_mismatch,
    "tax_math_mismatch": check_tax_math_mismatch,
    "invoice_over_limit_without_contract": check_invoice_over_limit_without_contract,
    "lodging_over_cap": check_lodging_over_cap,
    "entertainment_over_cap": check_entertainment_over_cap,
    "receipt_kind": check_receipt_kind,
    "self_approval": check_self_approval,
    "purpose_mismatch": check_purpose_mismatch,
    "consecutive_invoice": check_consecutive_invoice,
}


def run_stage(stage: str, claim: dict, params: dict, ctx: dict) -> list[Finding]:
    """跑某一 stage 下的全部规则，按严重度排序返回命中的 Finding。

    每条规则先按 ``check`` 名找判定器、拿到证据字典，再用规则模板渲染文案；
    图里的 G3 / G4 / G5 三个节点都通过它执行各自的规则子集。
    """
    findings: list[Finding] = []
    for rule in load_rules():
        if rule.stage != stage:
            continue
        check = CHECKS.get(rule.check)
        if check is None:
            continue
        for evidence in check(claim, params, ctx):
            findings.append(
                Finding(
                    rule_id=rule.id,
                    name=rule.name,
                    category=rule.category,
                    severity=rule.severity,
                    stage=rule.stage,
                    message=safe_format(rule.message, evidence),
                    citation=rule.citation,
                    evidence=evidence,
                )
            )
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.rule_id))
    return findings


def period_bounds(period: str) -> tuple[date, date]:
    """把 ``YYYY-MM`` 的报销期间换算成该月首末日；格式不合法时退回当月（1–28 日）。"""
    match = re.fullmatch(r"(\d{4})-(\d{2})", period or "")
    if not match:
        today = date.today()
        return date(today.year, today.month, 1), date(today.year, today.month, 28)
    year, month = int(match.group(1)), int(match.group(2))
    start = date(year, month, 1)
    end = date(year + (month == 12), (month % 12) + 1, 1)
    return start, date.fromordinal(end.toordinal() - 1)
