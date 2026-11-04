"""样例报销单：覆盖「通过 / 退回 / 转人工」三条路径。

全部为虚构数据，仅用于演示与回归测试；金额、发票号、税号均不对应真实主体。
"""

from __future__ import annotations

from typing import Any

COMPANY = {"name": "示例科技（上海）有限公司", "tax_id": "91310000MA1FL0000X"}

# 历史已报销过的发票号（演示"重复报销"）
HISTORY = {"24410007"}


def _inv(
    number: str,
    day: str,
    seller: str,
    kind: str,
    amount: str,
    tax: str,
    total: str,
    items: list[tuple[str, str]],
    *,
    buyer: str = COMPANY["name"],
    buyer_tax_id: str = COMPANY["tax_id"],
    nights: int = 0,
    city_tier: str = "",
) -> dict[str, Any]:
    """构造一张样例发票：购买方默认是示例公司，住宿类再补晚数与城市档。"""
    return {
        "number": number,
        "date": day,
        "seller": seller,
        "buyer": buyer,
        "buyer_tax_id": buyer_tax_id,
        "kind": kind,
        "amount": amount,
        "tax": tax,
        "total": total,
        "items": [{"name": name, "amount": value} for name, value in items],
        "nights": nights,
        "city_tier": city_tier,
    }


SAMPLES: list[dict[str, Any]] = [
    {
        "sample_id": "C001",
        "title": "标准差旅报销",
        "expected": "通过",
        "note": "住宿 550 元/晚未超标，金额与票据一致",
        "claim": {
            "claim_id": "BX-2026-10-001",
            "employee": {"id": "E001", "name": "张伟", "dept": "技术部"},
            "period": "2026-10",
            "category": "差旅",
            "purpose": "上海客户现场支持 10/12-10/15",
            "total": "2900.00",
            "submit_date": "2026-10-20",
            "ocr_confidence": 0.95,
            "approvals": [{"role": "部门经理", "name": "李强", "approved_at": "2026-10-18"}],
            "attachments": ["行程单.pdf"],
            "invoices": [
                _inv("24410001", "2026-10-13", "上海某酒店", "专票", "1037.74", "62.26", "1100.00",
                     [("住宿费", "1037.74")], nights=2, city_tier="一线"),
                _inv("24410013", "2026-10-12", "某航空公司", "电子普票", "1698.11", "101.89", "1800.00",
                     [("机票", "1698.11")]),
            ],
        },
    },
    {
        "sample_id": "C002",
        "title": "住宿超标 + 金额不符",
        "expected": "退回",
        "note": "800 元/晚超一线标准；且票据合计 2400 与报销 2300 不符",
        "claim": {
            "claim_id": "BX-2026-10-002",
            "employee": {"id": "E002", "name": "李娜", "dept": "市场部"},
            "period": "2026-10",
            "category": "差旅",
            "purpose": "北京行业展会 10/09-10/12",
            "total": "2300.00",
            "submit_date": "2026-10-16",
            "ocr_confidence": 0.94,
            "approvals": [{"role": "部门经理", "name": "王强", "approved_at": "2026-10-15"}],
            "attachments": [],
            "invoices": [
                _inv("24410003", "2026-10-10", "北京某酒店", "专票", "2264.15", "135.85", "2400.00",
                     [("住宿费", "2264.15")], nights=3, city_tier="一线"),
            ],
        },
    },
    {
        "sample_id": "C003",
        "title": "招待费报成办公用品",
        "expected": "转人工",
        "note": "语义疑点：招待事由，票据是办公用品，转人工确认业务实质",
        "claim": {
            "claim_id": "BX-2026-10-003",
            "employee": {"id": "E006", "name": "刘洋", "dept": "销售部"},
            "period": "2026-10",
            "category": "招待",
            "purpose": "客户拜访接待",
            "total": "860.00",
            "submit_date": "2026-10-21",
            "ocr_confidence": 0.93,
            "meta": {"headcount": 3},
            "approvals": [{"role": "部门经理", "name": "李娜", "approved_at": "2026-10-20"}],
            "attachments": [],
            "invoices": [
                _inv("24410004", "2026-10-19", "某办公用品商行", "电子普票", "761.06", "98.94", "860.00",
                     [("办公用品", "761.06")]),
            ],
        },
    },
    {
        "sample_id": "C004",
        "title": "连号发票",
        "expected": "转人工",
        "note": "两张发票号码连续，疑似集中开票或拆分，转人工",
        "claim": {
            "claim_id": "BX-2026-10-004",
            "employee": {"id": "E004", "name": "赵敏", "dept": "财务部"},
            "period": "2026-10",
            "category": "办公",
            "purpose": "部门打印耗材采购",
            "total": "1000.00",
            "submit_date": "2026-10-22",
            "ocr_confidence": 0.96,
            "approvals": [{"role": "财务经理", "name": "陈静", "approved_at": "2026-10-21"}],
            "attachments": [],
            "invoices": [
                _inv("24410005", "2026-10-18", "某耗材商行", "电子普票", "442.48", "57.52", "500.00",
                     [("打印机耗材", "442.48")]),
                _inv("24410006", "2026-10-18", "某耗材商行", "电子普票", "442.48", "57.52", "500.00",
                     [("打印纸", "442.48")]),
            ],
        },
    },
    {
        "sample_id": "C005",
        "title": "重复报销 + 抬头错误",
        "expected": "退回",
        "note": "发票号已在历史记录中，且购买方抬头不是本公司",
        "claim": {
            "claim_id": "BX-2026-10-005",
            "employee": {"id": "E003", "name": "王强", "dept": "技术部"},
            "period": "2026-10",
            "category": "交通",
            "purpose": "客户现场往返打车",
            "total": "620.00",
            "submit_date": "2026-10-19",
            "ocr_confidence": 0.92,
            "approvals": [{"role": "部门经理", "name": "张伟", "approved_at": "2026-10-18"}],
            "attachments": [],
            "invoices": [
                _inv("24410007", "2026-10-15", "某出行平台", "电子普票", "396.23", "23.77", "420.00",
                     [("打车费", "396.23")]),
                _inv("24410021", "2026-10-16", "某出行平台", "电子普票", "188.68", "11.32", "200.00",
                     [("打车费", "188.68")], buyer="张三", buyer_tax_id=""),
            ],
        },
    },
    {
        "sample_id": "C006",
        "title": "用收据代替发票",
        "expected": "退回",
        "note": "收据不属于税前扣除凭证，需换开正式发票",
        "claim": {
            "claim_id": "BX-2026-10-006",
            "employee": {"id": "E005", "name": "陈静", "dept": "人事部"},
            "period": "2026-10",
            "category": "差旅",
            "purpose": "校招出差住宿",
            "total": "320.00",
            "submit_date": "2026-10-23",
            "ocr_confidence": 0.9,
            "approvals": [{"role": "人事经理", "name": "赵敏", "approved_at": "2026-10-22"}],
            "attachments": [],
            "invoices": [
                _inv("24410009", "2026-10-17", "某快捷酒店", "收据", "320.00", "0", "320.00",
                     [("住宿费", "320.00")], nights=1, city_tier="三线"),
            ],
        },
    },
    {
        "sample_id": "C007",
        "title": "大额采购未附合同",
        "expected": "通过",
        "note": "单张发票 6800 元达到合同阈值却没附合同，命中警告但不阻断",
        "claim": {
            "claim_id": "BX-2026-10-007",
            "employee": {"id": "E004", "name": "赵敏", "dept": "财务部"},
            "period": "2026-10",
            "category": "办公",
            "purpose": "部门采购两台显示器",
            "total": "6800.00",
            "submit_date": "2026-10-24",
            "ocr_confidence": 0.95,
            "approvals": [{"role": "部门经理", "name": "王强", "approved_at": "2026-10-23"}],
            "attachments": [],
            "invoices": [
                _inv("24410031", "2026-10-20", "某数码商城", "电子普票", "6017.70", "782.30", "6800.00",
                     [("办公设备（显示器）", "6017.70")]),
            ],
        },
    },
]

SAMPLES_BY_ID = {item["sample_id"]: item for item in SAMPLES}
