"""答案引用校验：存在性、数字归一化、句级支持度与 span。"""

from __future__ import annotations

from decimal import Decimal

from langchain_core.documents import Document

from ai_agent_py.rag.citations import (
    ratio_numbers,
    strip_citations,
    verify_citations,
)


def _doc(content: str, *, chunk_id: str = "chunk-1", doc_id: str = "doc-a") -> Document:
    return Document(page_content=content, metadata={"chunk_id": chunk_id, "doc_id": doc_id})


def test_out_of_range_citation_is_removed() -> None:
    report = verify_citations("只有两条资料[3]。", [_doc("甲"), _doc("乙")])

    assert report.answer == "只有两条资料。"
    assert report.stats.total == 1
    assert report.stats.invalid == 1
    assert report.stats.removed == 1
    assert report.citations[0].removed is True
    assert report.citations[0].start is None


def test_zero_and_year_brackets_are_removed() -> None:
    report = verify_citations("2024 年[2024]和[0]。", [_doc("甲")])

    assert report.answer == "2024 年和。"
    assert report.stats.invalid == 2


def test_out_of_range_article_number_is_not_rewritten() -> None:
    """D-18：``[5]`` 即使命中资料里的「第五条」，也只能删除，不能法条化。"""
    documents = [_doc("第五条 发票的种类由国务院税务主管部门规定。"), _doc("第六条 其他规定。")]

    report = verify_citations("依据第五条[5]。", documents)

    assert report.answer == "依据第五条。"
    assert "[5]" not in report.answer
    assert "（第五条）" not in report.answer
    assert report.stats.invalid == 1


def test_valid_citation_span_points_at_the_marker() -> None:
    report = verify_citations("业务招待费按60%扣除[1]。", [_doc("业务招待费按发生额的60%扣除。")])

    citation = report.citations[0]
    assert report.answer == "业务招待费按60%扣除[1]。"
    assert citation.status == "ok"
    assert citation.chunk_id == "chunk-1"
    assert citation.start is not None
    assert citation.end is not None
    assert report.answer[citation.start : citation.end] == "[1]"


def test_spans_stay_correct_after_removal() -> None:
    report = verify_citations("A[3]B[1]C", [_doc("B")])

    assert report.answer == "AB[1]C"
    citation = report.citations[1]
    assert citation.start == 2
    assert citation.end == 5
    assert report.answer[2:5] == "[1]"


def test_chinese_percent_matches_arabic() -> None:
    report = verify_citations("按照60%扣除[1]。", [_doc("按照发生额的百分之六十扣除。")])

    assert report.answer == "按照60%扣除[1]。"
    assert report.citations[0].status == "ok"
    assert report.stats.removed == 0


def test_permille_matches_chinese_writing() -> None:
    report = verify_citations(
        "最高不得超过当年销售（营业）收入的5‰[1]。",
        [_doc("最高不得超过当年销售（营业）收入的千分之五。")],
    )

    assert report.citations[0].status == "ok"
    assert report.citations[0].removed is False


def test_bare_decimal_matches_percent() -> None:
    report = verify_citations("扣除比例为0.6[1]。", [_doc("扣除比例为60%。")])

    assert report.citations[0].status == "ok"
    assert report.citations[0].removed is False


def test_fullwidth_percent_is_normalized() -> None:
    report = verify_citations("扣除比例为60％[1]。", [_doc("扣除比例为60%。")])

    assert report.citations[0].status == "ok"


def test_numeric_mismatch_is_removed() -> None:
    report = verify_citations("税率为15%[1]。", [_doc("税率为25%。")])

    assert report.answer == "税率为15%。"
    assert report.citations[0].status == "unsupported"
    assert report.citations[0].reason == "numeric_mismatch"
    assert report.citations[0].chunk_id == "chunk-1"
    assert report.stats.unsupported == 1


def test_low_overlap_is_weak_but_kept() -> None:
    """概括性引用重合度很低，但内容是正确概括：只标弱，不删。"""
    documents = [_doc("企业实际发生的与取得收入有关的、合理的支出，准予在计算应纳税所得额时扣除。")]

    report = verify_citations("扣除项目包括成本、费用、税金和损失[1]。", documents)

    assert report.answer == "扣除项目包括成本、费用、税金和损失[1]。"
    assert report.citations[0].status == "weak"
    assert report.citations[0].reason == "low_overlap"
    assert report.citations[0].removed is False


def test_short_claim_is_not_judged() -> None:
    report = verify_citations("依据上述规定[1]。", [_doc("企业实际发生的支出，准予扣除。")])

    assert report.citations[0].status == "ok"


def test_multi_citation_sentence_attributes_each_claim() -> None:
    documents = [
        _doc("业务招待费按60%扣除。", chunk_id="chunk-1"),
        _doc("最高不得超过当年销售收入的5‰。", chunk_id="chunk-2"),
    ]

    report = verify_citations(
        "业务招待费按60%扣除[1]，最高不得超过当年销售收入的5‰[2]。",
        documents,
    )

    assert [citation.status for citation in report.citations] == ["ok", "ok"]
    assert [citation.chunk_id for citation in report.citations] == ["chunk-1", "chunk-2"]


def test_article_label_mismatch_is_weak() -> None:
    documents = [_doc("第四十四条 广告费和业务宣传费支出，不超过当年销售收入15%的部分，准予扣除。")]

    report = verify_citations("第四十三条规定的扣除标准[1]。", documents)

    assert report.citations[0].status == "weak"
    assert report.citations[0].reason == "article_mismatch"
    assert report.citations[0].removed is False


def test_empty_chunk_is_removed() -> None:
    report = verify_citations("内容[1]。", [_doc("")])

    assert report.answer == "内容。"
    assert report.citations[0].status == "invalid"
    assert report.citations[0].reason == "empty_chunk"


def test_no_citations_returns_text_unchanged() -> None:
    report = verify_citations("没有编号的回答。", [_doc("甲")])

    assert report.answer == "没有编号的回答。"
    assert report.citations == ()
    assert report.stats.total == 0


def test_strip_citations_keeps_article_text() -> None:
    assert strip_citations("依据[1]和[2]，见第四十三条。") == "依据和，见第四十三条。"


def test_ratio_numbers_normalizes_written_forms() -> None:
    assert ratio_numbers("60%、百分之六十、0.6") == {Decimal("0.6")}
    assert ratio_numbers("5‰、千分之五、0.5%") == {Decimal("0.005")}
    assert ratio_numbers("2.5% 和 百分之二点五") == {Decimal("0.025")}
    assert ratio_numbers("2024 年第四十三条，金额 100 万元") == set()
    # 0.x 是比例；带金额单位时不能当成“60% 那种比例”。
    assert ratio_numbers("0.6 万元") == set()
