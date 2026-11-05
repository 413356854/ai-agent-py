"""检索过滤条件：类型化、规范化、表达式转义（R-01 文档级过滤）。"""

from __future__ import annotations

import pytest

from ai_agent_py.rag.filters import (
    MAX_DOC_ID_LENGTH,
    MAX_FILTER_DOC_IDS,
    SearchFilter,
    build_filter_expr,
    normalize_doc_ids,
)


# ---- 规范化 ---------------------------------------------------------------
def test_doc_ids_are_trimmed_deduped_and_ordered() -> None:
    assert normalize_doc_ids([" doc-b ", "doc-a", "", "doc-b", "doc-a"]) == ("doc-b", "doc-a")


def test_none_and_empty_mean_no_filter() -> None:
    assert normalize_doc_ids(None) == ()
    assert normalize_doc_ids([]) == ()
    assert SearchFilter.from_doc_ids(None) is None
    assert SearchFilter.from_doc_ids([]) is None
    assert SearchFilter.from_doc_ids([""]) is None
    assert SearchFilter().is_empty is True


def test_from_doc_ids_builds_typed_filter() -> None:
    search_filter = SearchFilter.from_doc_ids(["doc-a", "doc-b"])

    assert search_filter == SearchFilter(doc_ids=("doc-a", "doc-b"))
    assert search_filter is not None
    assert search_filter.is_empty is False


def test_direct_construction_is_normalized_too() -> None:
    assert SearchFilter(doc_ids=(" doc-a ", "doc-a")).doc_ids == ("doc-a",)
    assert SearchFilter(doc_ids=("  ",)).is_empty is True


def test_bare_string_is_rejected() -> None:
    # 传字符串会被逐字符拆成单字 id，静默改变检索范围，必须直接报错。
    with pytest.raises(TypeError, match="不能直接传字符串"):
        normalize_doc_ids("doc-a")  # type: ignore[arg-type]


def test_too_many_doc_ids_is_rejected() -> None:
    ids = [f"doc-{index}" for index in range(MAX_FILTER_DOC_IDS + 1)]

    with pytest.raises(ValueError, match="一次最多按"):
        normalize_doc_ids(ids)


def test_overlong_doc_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="doc_id 超过"):
        normalize_doc_ids(["x" * (MAX_DOC_ID_LENGTH + 1)])


# ---- 表达式 ---------------------------------------------------------------
def test_no_filter_builds_empty_expression() -> None:
    assert build_filter_expr(None) == ""
    assert build_filter_expr(SearchFilter()) == ""


def test_expression_uses_in_with_double_quotes() -> None:
    assert build_filter_expr(SearchFilter(doc_ids=("doc-a",))) == 'doc_id in ["doc-a"]'
    assert (
        build_filter_expr(SearchFilter(doc_ids=("doc-a", "doc-b")))
        == 'doc_id in ["doc-a", "doc-b"]'
    )


def test_filter_literals_are_escaped() -> None:
    expr = build_filter_expr(SearchFilter(doc_ids=('x" or doc_id != ""',)))

    assert expr == 'doc_id in ["x\\" or doc_id != \\"\\""]'
    # 去掉转义后的引号，表达式里只剩最外层两个定界引号：注入值没有逃出字面量。
    assert expr.replace('\\"', "").count('"') == 2


def test_backslashes_are_escaped() -> None:
    expr = build_filter_expr(SearchFilter(doc_ids=("a\\b",)))

    assert expr == 'doc_id in ["a\\\\b"]'
