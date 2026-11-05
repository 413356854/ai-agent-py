"""答案引用校验：存在性、句级支持度、数字归一化与 citation span。

模型给出的 ``[n]`` 可能是三种东西：合法来源编号、越界编号、被误写成方括号的
法条序号/年份。旧实现（``normalize_citations``）只处理第三种，而且会把越界编号
反向改写成“第 N 条”，等于把幻觉引用洗成合法引用（评审 D-18）。这里改成一条
确定性的后处理流水线：

    [n] ──▶ 存在性（1..len(documents)）──▶ 句级归因 ──▶ 数字归一化
        ──▶ 术语/法条号支持度 ──▶ 改写答案（只删硬失败）+ span + 统计

判定分三档：

* ``ok``：编号存在，数字一致，术语重合度足够；
* ``weak``：编号存在，但术语重合度低或法条号对不上——**保留引用**，只做提示；
* ``invalid`` / ``unsupported``：编号越界 / 数字与引用片段冲突——**删除引用**。

为什么不按术语重合度硬删？拿真实法条语料实测过：概括性正确引用（“扣除项目包括
成本、费用、税金和损失” vs “企业实际发生的…准予扣除”）的字符 bigram 重合度只有
0.07，而同主题但错误的引用能有 0.67。重合度分不开“概括”和“错误”，硬删会误伤大量
正确引用，所以它只作为 ``weak`` 信号。真正硬删的只有两个确定性信号：编号越界和
数字冲突。同主题错误需要 NLI / LLM-as-judge，等 R-05 有评测基线后再做。

数字归一化把 ``60%`` / ``百分之六十`` / ``0.6`` 视为同一个值，``5‰`` / ``千分之五``
/ ``0.5%`` 同理；比较用 :class:`decimal.Decimal` 精确相等，不用浮点。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, cast

from langchain_core.documents import Document

#: 一条引用的校验结论。
CitationStatus = Literal["ok", "weak", "invalid", "unsupported"]

#: 回答里的方括号数字（与提示词约定的 ``[1]`` 一致）。
CITATION_RE = re.compile(r"\[(\d{1,4})\]")

#: 句读：用于切出「这条引用负责的那句话」。
_SENTENCE_MARKS = "。！？；\n"

#: Markdown 列表前缀，避免“1. ”把重合度算歪。
_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d+[.、)]|[-*•])\s*")

#: 中文数字与单位（法条、比例够用；不支持“万”）。
_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_NUMERAL = r"[零〇一二三四五六七八九十百点两]+"
_NUMBER = rf"(?:{_CN_NUMERAL}|\d+(?:\.\d+)?)"

#: 比例/千分比数字：中文写法、阿拉伯写法、以及裸 ``0.x``。
#: 法条号、年份、普通金额（``0.6 万元``）不在这里，避免误伤。
_QUANTITY_RE = re.compile(
    rf"百分之{_NUMBER}|千分之{_NUMBER}|\d+(?:\.\d+)?\s*[%％‰]"
    r"|(?<![\d.])0\.\d+(?!\s*[万亿千百十元角分])"
)

#: 法条序号（只用于“弱支持”判断，不做改写）。
_ARTICLE_RE = re.compile(r"第[零〇一二三四五六七八九十百\d]+条")

_HUNDRED = Decimal(100)
_THOUSAND = Decimal(1000)

#: 术语重合度达到多少算“支持”；低于它只标弱，不删引用。
SUPPORT_OVERLAP = 0.45

#: 太短的句子（“依据上述规定”）不做重合度判断，避免误伤。
MIN_CLAIM_CHARS = 10


@dataclass(frozen=True, slots=True)
class Citation:
    """答案里的一条引用，及它在最终答案中的字符区间。"""

    index: int
    chunk_id: str
    doc_id: str
    quote: str
    start: int | None
    end: int | None
    status: CitationStatus
    reason: str = ""

    @property
    def removed(self) -> bool:
        """是否已从答案正文中删除（编号越界或数字冲突）。"""
        return self.status in {"invalid", "unsupported"}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Citation:
        """从 ``as_dict()`` 的字典还原（图状态 / 接口层用）。"""
        return cls(
            index=int(data.get("index") or 0),
            chunk_id=str(data.get("chunk_id") or ""),
            doc_id=str(data.get("doc_id") or ""),
            quote=str(data.get("quote") or ""),
            start=_optional_int(data.get("start")),
            end=_optional_int(data.get("end")),
            status=cast(CitationStatus, data.get("status") or "ok"),
            reason=str(data.get("reason") or ""),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "quote": self.quote,
            "start": self.start,
            "end": self.end,
            "status": self.status,
            "reason": self.reason,
            "removed": self.removed,
        }


@dataclass(frozen=True, slots=True)
class CitationStats:
    """一次答案的引用统计；``total == ok + weak + removed``。"""

    total: int = 0
    ok: int = 0
    weak: int = 0
    removed: int = 0
    invalid: int = 0
    unsupported: int = 0

    @classmethod
    def from_citations(cls, citations: Sequence[Citation]) -> CitationStats:
        invalid = sum(1 for citation in citations if citation.status == "invalid")
        unsupported = sum(1 for citation in citations if citation.status == "unsupported")
        return cls(
            total=len(citations),
            ok=sum(1 for citation in citations if citation.status == "ok"),
            weak=sum(1 for citation in citations if citation.status == "weak"),
            removed=invalid + unsupported,
            invalid=invalid,
            unsupported=unsupported,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "ok": self.ok,
            "weak": self.weak,
            "removed": self.removed,
            "invalid": self.invalid,
            "unsupported": self.unsupported,
        }


@dataclass(frozen=True, slots=True)
class CitationReport:
    """校验后的答案、逐条引用与汇总统计。"""

    answer: str
    citations: tuple[Citation, ...] = ()
    stats: CitationStats = field(default_factory=CitationStats)


def verify_citations(
    answer: str,
    documents: Sequence[Document],
) -> CitationReport:
    """校验答案里的 ``[n]``，删除不可信编号并返回答案内 span。

    只改写正文里**确定错误**的引用：编号越界、引用片段为空、数字与片段冲突。
    术语重合度低 / 法条号对不上只标记 ``weak``，引用保留。
    """
    text = answer or ""
    total = len(documents)
    citations: list[Citation] = []
    output: list[str] = []
    cursor = 0
    output_length = 0
    previous_end = 0

    for match in CITATION_RE.finditer(text):
        output.append(text[cursor : match.start()])
        output_length += match.start() - cursor

        index = int(match.group(1))
        document = documents[index - 1] if 1 <= index <= total else None
        if document is None:
            output_length -= _trim_space_before(output, text, match.end())
            citations.append(
                _removed_citation(index, match.group(0), "", "", "invalid", "out_of_range")
            )
        elif not document.page_content.strip():
            output_length -= _trim_space_before(output, text, match.end())
            citations.append(
                _removed_citation(
                    index,
                    match.group(0),
                    _chunk_id(document),
                    _doc_id(document),
                    "invalid",
                    "empty_chunk",
                )
            )
        else:
            claim = _claim_before(text, match.start(), previous_end)
            status, reason = _assess(claim, document.page_content)
            if status in {"invalid", "unsupported"}:
                output_length -= _trim_space_before(output, text, match.end())
                citations.append(
                    _removed_citation(
                        index,
                        match.group(0),
                        _chunk_id(document),
                        _doc_id(document),
                        status,
                        reason,
                    )
                )
            else:
                output.append(match.group(0))
                start = output_length
                output_length += len(match.group(0))
                citations.append(
                    Citation(
                        index=index,
                        chunk_id=_chunk_id(document),
                        doc_id=_doc_id(document),
                        quote=match.group(0),
                        start=start,
                        end=output_length,
                        status=status,
                        reason=reason,
                    )
                )

        cursor = match.end()
        previous_end = match.end()

    output.append(text[cursor:])
    return CitationReport(
        answer="".join(output),
        citations=tuple(citations),
        stats=CitationStats.from_citations(citations),
    )


def strip_citations(text: str) -> str:
    """删掉文本里的 ``[n]``（历史消息进提示词前用，见 D-01）。"""
    return CITATION_RE.sub("", text or "")


def ratio_numbers(text: str) -> set[Decimal]:
    """提取文本里的比例值（百分号/千分号/中文写法/裸 ``0.x``）。"""
    numbers: set[Decimal] = set()
    for match in _QUANTITY_RE.finditer(unicodedata.normalize("NFKC", text or "")):
        value = _parse_quantity(match.group(0))
        if value is not None:
            numbers.add(value)
    return numbers


# ---- 支持度判定 -----------------------------------------------------------
def _assess(claim: str, chunk: str) -> tuple[CitationStatus, str]:
    """判断 ``claim`` 是否被 ``chunk`` 支持；只返回 ok / weak / unsupported。"""
    if not claim:
        return "ok", ""

    if ratio_numbers(claim) - ratio_numbers(chunk):
        # 答案里的数字在引用片段里找不到：法条问答里这是最高置信度的错引。
        return "unsupported", "numeric_mismatch"

    labels = set(_ARTICLE_RE.findall(claim))
    if labels and not any(label in chunk for label in labels):
        # 切分点可能落在条中间，条号留在上一块：只弱提示，不硬删。
        return "weak", "article_mismatch"

    if len(_normalize_for_overlap(claim)) < MIN_CLAIM_CHARS:
        return "ok", ""

    if _overlap(claim, chunk) < SUPPORT_OVERLAP:
        return "weak", "low_overlap"

    return "ok", ""


def _claim_before(text: str, start: int, previous_end: int) -> str:
    """取当前引用负责的那段话。

    从「上一条引用结束」到「当前引用」之间截取，再退到最后一个句读之后。这样
    ``A 为 60%[1]，B 为 5‰[2]`` 的 ``[2]`` 只对 ``B 为 5‰`` 负责，不会把 ``[1]``
    的数字算到它头上。
    """
    segment = text[previous_end:start]
    boundary = max(segment.rfind(mark) for mark in _SENTENCE_MARKS)
    if boundary >= 0:
        segment = segment[boundary + 1 :]
    return _LIST_PREFIX_RE.sub("", segment).strip()


def _normalize_for_overlap(text: str) -> str:
    """去掉标点与比例数字后的小写文本，只用于字符 bigram 比较。"""
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = _QUANTITY_RE.sub("", normalized)
    return "".join(char for char in normalized if char.isalnum())


def _overlap(claim: str, chunk: str) -> float:
    """claim 的字符 bigram 有多大比例出现在 chunk 里。"""
    claim_grams = _bigrams(_normalize_for_overlap(claim))
    if not claim_grams:
        return 1.0
    chunk_grams = _bigrams(_normalize_for_overlap(chunk))
    return len(claim_grams & chunk_grams) / len(claim_grams)


def _bigrams(text: str) -> set[str]:
    return {text[index : index + 2] for index in range(len(text) - 1)}


# ---- 数字解析 -------------------------------------------------------------
def _parse_quantity(token: str) -> Decimal | None:
    token = unicodedata.normalize("NFKC", token).strip()
    if token.startswith("百分之"):
        value = _parse_number(token[len("百分之") :])
        return None if value is None else value / _HUNDRED
    if token.startswith("千分之"):
        value = _parse_number(token[len("千分之") :])
        return None if value is None else value / _THOUSAND
    if token.endswith("%") or token.endswith("‰"):
        value = _parse_number(token[:-1].strip())
        divisor = _HUNDRED if token.endswith("%") else _THOUSAND
        return None if value is None else value / divisor
    return _parse_number(token)


def _parse_number(raw: str) -> Decimal | None:
    raw = raw.strip()
    if not raw:
        return None
    if all(char.isdigit() or char == "." for char in raw):
        try:
            return Decimal(raw)
        except InvalidOperation:
            return None
    return _parse_chinese_number(raw)


def _parse_chinese_number(raw: str) -> Decimal | None:
    if "点" in raw:
        whole, _, fraction = raw.partition("点")
        base = _parse_chinese_integer(whole) if whole else Decimal(0)
        if base is None:
            return None
        digits = "".join(str(_CN_DIGITS[char]) for char in fraction if char in _CN_DIGITS)
        if not digits:
            return base
        return base + Decimal(f"0.{digits}")
    return _parse_chinese_integer(raw)


def _parse_chinese_integer(raw: str) -> Decimal | None:
    if not raw:
        return None
    total = 0
    current = 0
    seen = False
    for char in raw:
        if char in _CN_DIGITS:
            current = _CN_DIGITS[char]
            seen = True
        elif char in _CN_UNITS:
            total += (current or 1) * _CN_UNITS[char]
            current = 0
            seen = True
        else:
            return None
    if not seen:
        return None
    return Decimal(total + current)


# ---- 小工具 ---------------------------------------------------------------
def _trim_space_before(output: list[str], text: str, end: int) -> int:
    """删除引用后，若前文以空格结尾且后面紧跟标点/结尾，顺手去掉这个空格。

    返回从 ``output_length`` 中减去的字符数（0 或 1）。
    """
    if not output:
        return 0
    tail = output[-1]
    if not tail.endswith((" ", "\u3000")):
        return 0
    following = text[end : end + 1]
    if following and following not in "。！？；，、.,!?;:":
        return 0
    output[-1] = tail[:-1]
    return 1


def _removed_citation(
    index: int,
    quote: str,
    chunk_id: str,
    doc_id: str,
    status: CitationStatus,
    reason: str,
) -> Citation:
    return Citation(
        index=index,
        chunk_id=chunk_id,
        doc_id=doc_id,
        quote=quote,
        start=None,
        end=None,
        status=status,
        reason=reason,
    )


def _chunk_id(document: Document) -> str:
    return str(document.metadata.get("chunk_id") or document.id or "")


def _doc_id(document: Document) -> str:
    return str(document.metadata.get("doc_id") or "")


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
