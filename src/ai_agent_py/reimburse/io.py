"""单据模板：xlsx 导出 / 导入，json 直读。

模板与解析保持同一套字段映射，保证「下载样板 → 填写 → 上传试跑」能闭环。
"""

from __future__ import annotations

import io
import json
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

SHEET_INFO = "单据信息"
SHEET_INVOICES = "发票明细"
SHEET_HELP = "填写说明"

HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=10)
LABEL_FONT = Font(bold=True)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
MONEY = "#,##0.00"

# 单据信息： (显示名, claim 字段路径)
INFO_FIELDS: list[tuple[str, str]] = [
    ("单据号", "claim_id"),
    ("报销人", "employee.name"),
    ("部门", "employee.dept"),
    ("报销期间", "period"),
    ("费用类型", "category"),
    ("报销事由", "purpose"),
    ("报销金额", "total"),
    ("提交日期", "submit_date"),
    ("抽取置信度", "ocr_confidence"),
    ("招待人数", "meta.headcount"),
    ("审批链", "approvals"),
    ("附件", "attachments"),
]

INVOICE_COLUMNS = [
    "发票号",
    "开票日期",
    "销售方",
    "购买方",
    "购买方税号",
    "票种",
    "金额",
    "税额",
    "价税合计",
    "明细",
    "住宿晚数",
    "城市档",
]


def _get(claim: dict[str, Any], path: str) -> Any:
    """按 ``employee.name`` 这类点路径取嵌套字段，取不到返回 None。"""
    current: Any = claim
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _display(value: Any) -> str:
    """把单据字段渲染成 Excel 单元格文本：日期截到日，审批链/附件按模板约定拼接。"""
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    if isinstance(value, list):
        if not value:
            return ""
        if isinstance(value[0], dict):
            first = value[0]
            if "role" in first or "name" in first:
                return ";".join(f"{item.get('role', '')}:{item.get('name', '')}" for item in value)
            if "name" in first:
                return "、".join(str(item.get("name", "")) for item in value)
            return json.dumps(value, ensure_ascii=False)
        return ",".join(str(item) for item in value)
    return str(value)


def _money(value: Any) -> str:
    """金额统一成两位小数字符串（去千分位）；非法输入原样返回，便于人工发现填错。"""
    if value is None or value == "":
        return ""
    try:
        return f"{Decimal(str(value).replace(',', '')):.2f}"
    except (InvalidOperation, ValueError):
        return str(value)


def _num(value: Any) -> float | str:
    """金额列写成数字，Excel 的千分位格式才生效。"""
    text = _money(value)
    if not text:
        return ""
    try:
        return float(text)
    except ValueError:
        return text


def _invoice_items(invoice: dict[str, Any]) -> str:
    """把发票明细压成「品名:金额、品名:金额」，与上传解析时的拆解格式保持对称。"""
    items = invoice.get("items") or []
    return "、".join(
        f"{item.get('name', '')}:{_money(item.get('amount'))}" if item.get("amount") else str(item.get("name", ""))
        for item in items
    )


def _autosize(ws, widths: dict[str, int]) -> None:
    """批量设置列宽：模板按固定宽度排版，填写与打印时不会挤在一起。"""
    for column, width in widths.items():
        ws.column_dimensions[column].width = width


def build_template(claim: dict[str, Any] | None = None) -> bytes:
    """生成可填写的报销单模板；传 claim 时预填成套打样板。"""
    data = claim or {}
    book = Workbook()

    info = book.active
    info.title = SHEET_INFO
    info["A1"] = "字段"
    info["B1"] = "值"
    for cell in ("A1", "B1"):
        info[cell].fill = HEADER_FILL
        info[cell].font = HEADER_FONT
        info[cell].alignment = Alignment(horizontal="center")
        info[cell].border = BORDER
    for row, (label, path) in enumerate(INFO_FIELDS, start=2):
        info.cell(row=row, column=1, value=label).font = LABEL_FONT
        info.cell(row=row, column=2, value=_display(_get(data, path)))
        for column in (1, 2):
            info.cell(row=row, column=column).border = BORDER
            info.cell(row=row, column=column).alignment = Alignment(vertical="center", wrap_text=True)
    _autosize(info, {"A": 16, "B": 58})
    info.freeze_panes = "A2"

    invoices = book.create_sheet(SHEET_INVOICES)
    for column, name in enumerate(INVOICE_COLUMNS, start=1):
        cell = invoices.cell(row=1, column=column, value=name)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = BORDER
    for row, invoice in enumerate(data.get("invoices") or [], start=2):
        values = [
            invoice.get("number", ""),
            _display(invoice.get("date")),
            invoice.get("seller", ""),
            invoice.get("buyer", ""),
            invoice.get("buyer_tax_id", ""),
            invoice.get("kind", ""),
            _num(invoice.get("amount")),
            _num(invoice.get("tax")),
            _num(invoice.get("total")),
            _invoice_items(invoice),
            invoice.get("nights") or "",
            invoice.get("city_tier") or "",
        ]
        for column, value in enumerate(values, start=1):
            cell = invoices.cell(row=row, column=column, value=value)
            cell.border = BORDER
            if column in (7, 8, 9):
                cell.number_format = MONEY
                cell.alignment = Alignment(horizontal="right")
    _autosize(
        invoices,
        {"A": 16, "B": 12, "C": 26, "D": 26, "E": 22, "F": 10, "G": 12, "H": 11, "I": 12, "J": 26, "K": 10, "L": 10},
    )
    invoices.freeze_panes = "A2"

    help_sheet = book.create_sheet(SHEET_HELP)
    lines = [
        "报销单模板使用说明",
        "",
        "1. 只需填写「单据信息」与「发票明细」两张表，表头文字不要修改。",
        "2. 单据信息：审批链格式为「角色:姓名;角色:姓名」，附件用逗号分隔。",
        "3. 发票明细：一张发票一行；「明细」格式为「项目名:金额」，多个项目用「、」分隔。",
        "4. 住宿类发票必须填「住宿晚数」和「城市档」（一线/二线/三线），否则无法校验住宿标准。",
        "5. 金额列直接填数字，不要带货币符号。",
        "6. 本模板为演示用途，参数为占位值；上传后由 Agent 按规则库审核。",
    ]
    for row, text in enumerate(lines, start=1):
        cell = help_sheet.cell(row=row, column=1, value=text)
        if row == 1:
            cell.font = Font(bold=True, size=13)
    help_sheet.column_dimensions["A"].width = 92

    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


# --------------------------------------------------------------------------- 解析


def _as_text(value: Any) -> str:
    """读入单元格文本：日期取 ISO 日期，整数浮点去掉 ``.0``，其余转字符串并去空白。"""
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _as_money(value: Any) -> str:
    """把单元格金额洗成两位小数字符串（去掉千分位与货币符号），失败时原样保留。"""
    text = _as_text(value)
    if not text:
        return ""
    try:
        return f"{Decimal(text.replace(',', '').replace('￥', '')):.2f}"
    except (InvalidOperation, ValueError):
        return text


def _parse_info(rows: list[tuple[Any, Any]]) -> dict[str, Any]:
    """把「单据信息」表还原成 claim 上半部分：报销人、期间、审批链、附件等。

    以"显示名 → 字段路径"映射为准，不依赖行的先后顺序；解析不了的值进 warnings。
    """
    labels = dict(INFO_FIELDS)
    claim: dict[str, Any] = {"employee": {}, "meta": {}, "approvals": [], "attachments": []}
    warnings: list[str] = []
    for label, value in rows:
        text = _as_text(label)
        if not text or text not in labels:
            continue
        raw = _as_text(value)
        path = labels[text]
        if path == "employee.name":
            claim["employee"]["name"] = raw
        elif path == "employee.dept":
            claim["employee"]["dept"] = raw
        elif path == "meta.headcount":
            if raw:
                claim["meta"]["headcount"] = int(float(raw)) if raw.replace(".", "").isdigit() else 0
        elif path == "approvals":
            for chunk in filter(None, (item.strip() for item in raw.replace("；", ";").split(";"))):
                role, _, name = chunk.partition(":")
                claim["approvals"].append({"role": role.strip(), "name": name.strip()})
        elif path == "attachments":
            claim["attachments"] = [item for item in raw.replace("，", ",").split(",") if item.strip()]
        elif path == "ocr_confidence":
            try:
                claim["ocr_confidence"] = float(raw) if raw else 0.95
            except ValueError:
                claim["ocr_confidence"] = 0.95
                warnings.append(f"置信度「{raw}」无法解析，按 0.95 处理")
        elif path == "total":
            claim["total"] = _as_money(value)
        else:
            claim[path] = raw
    claim.setdefault("ocr_confidence", 0.95)
    claim["_warnings"] = warnings
    return claim


def _parse_invoices(ws) -> list[dict[str, Any]]:
    """把「发票明细」表按表头还原成发票列表，明细列再拆成 items。"""
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    header = [_as_text(cell) for cell in rows[0]]
    index = {name: position for position, name in enumerate(header)}
    result: list[dict[str, Any]] = []
    for row in rows[1:]:
        if not any(cell not in (None, "") for cell in row):
            continue

        def pick(name: str, cells: tuple[Any, ...] = row) -> Any:
            """按表头名取值，缺列返回 None，列顺序调整后仍能对上。"""
            position = index.get(name)
            return cells[position] if position is not None and position < len(cells) else None

        items = []
        for chunk in filter(None, (part.strip() for part in _as_text(pick("明细")).replace("，", "、").split("、"))):
            name, _, amount = chunk.partition(":")
            items.append({"name": name.strip(), "amount": _as_money(amount) if amount else ""})
        nights = pick("住宿晚数")
        result.append(
            {
                "number": _as_text(pick("发票号")),
                "date": _as_text(pick("开票日期")),
                "seller": _as_text(pick("销售方")),
                "buyer": _as_text(pick("购买方")),
                "buyer_tax_id": _as_text(pick("购买方税号")),
                "kind": _as_text(pick("票种")),
                "amount": _as_money(pick("金额")),
                "tax": _as_money(pick("税额")),
                "total": _as_money(pick("价税合计")),
                "items": items,
                "nights": int(float(_as_text(nights))) if _as_text(nights).replace(".", "").isdigit() else 0,
                "city_tier": _as_text(pick("城市档")),
            }
        )
    return result


def parse_upload(filename: str, content: bytes) -> tuple[dict[str, Any], list[str]]:
    """把上传的单据（xlsx / json）解析成 claim，并返回需人工确认的 warnings。

    业务兜底：单据号缺失用文件名、金额缺失用发票合计、期间缺失按当月；发票缺票种
    按「普票」处理——这些都不静默跳过，一律进 warnings 让审核人看到。
    """
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if suffix == "json":
        try:
            claim = json.loads(content.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"JSON 解析失败：{exc}") from exc
        if not isinstance(claim, dict):
            raise ValueError("JSON 顶层必须是对象")
        warnings: list[str] = []
    elif suffix in ("xlsx", "xlsm"):
        try:
            book = load_workbook(io.BytesIO(content), data_only=True)
        except Exception as exc:  # openpyxl 抛出的异常类型很杂
            raise ValueError(f"Excel 解析失败：{exc}") from exc
        if SHEET_INFO not in book.sheetnames or SHEET_INVOICES not in book.sheetnames:
            raise ValueError(f"缺少工作表：「{SHEET_INFO}」或「{SHEET_INVOICES}」")
        info_rows = list(book[SHEET_INFO].iter_rows(values_only=True))
        claim = _parse_info([(row[0], row[1] if len(row) > 1 else None) for row in info_rows[1:]])
        warnings = claim.pop("_warnings", [])
        claim["invoices"] = _parse_invoices(book[SHEET_INVOICES])
    else:
        raise ValueError("仅支持 .xlsx / .json 文件")

    if not claim.get("claim_id"):
        claim["claim_id"] = filename.rsplit(".", 1)[0]
        warnings.append("未填写单据号，已用文件名代替")

    invoices = claim.get("invoices") or []
    if not invoices:
        raise ValueError("发票明细为空，至少需要一张发票")
    for position, invoice in enumerate(invoices, start=1):
        if not invoice.get("number"):
            raise ValueError(f"第 {position} 张发票缺少发票号")
        if not invoice.get("total"):
            raise ValueError(f"第 {position} 张发票缺少价税合计")
        if not invoice.get("kind"):
            warnings.append(f"第 {position} 张发票未填票种，按「普票」处理")
            invoice["kind"] = "普票"
    if not claim.get("total"):
        claim["total"] = f"{sum(Decimal(str(inv['total'])) for inv in invoices):.2f}"
        warnings.append("未填写报销金额，已按发票合计填入")
    if not (claim.get("employee") or {}).get("name"):
        warnings.append("未填写报销人")
    if not claim.get("period"):
        claim["period"] = date.today().strftime("%Y-%m")
        warnings.append(f"未填写报销期间，按当前月份 {claim['period']} 处理")

    claim.setdefault("category", "差旅")
    claim.setdefault("purpose", "")
    claim.setdefault("submit_date", "")
    claim.setdefault("approvals", [])
    claim.setdefault("attachments", [])
    claim.setdefault("employee", {})
    claim.setdefault("meta", {})
    return claim, warnings


def dump_claim_json(claim: dict[str, Any]) -> bytes:
    """把单据导出成 UTF-8 JSON 字节流：样例下载用，字段与 claim 完全一致。"""
    return json.dumps(claim, ensure_ascii=False, indent=2).encode("utf-8")
