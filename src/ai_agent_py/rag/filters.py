"""检索过滤条件：只接受类型化字段，表达式统一由服务端生成。

Milvus 的 ``search`` / ``hybrid_search`` 原生支持 ``filter`` 表达式，但把调用方
传入的字符串直接拼进表达式，等于把过滤语义交给调用方：``doc_id`` 传成
``x" or doc_id != ""`` 就从「只看这一篇」变成「看全库」。所以这里只暴露
`SearchFilter`（类型化字段），表达式由 `build_filter_expr` 生成并转义字面量。

当前只开放文档级过滤（``doc_ids``）：``rag_chunk.doc_id`` 已有 INVERTED 索引，
接上即可用，不需要改集合结构。法规元数据（``doc_type`` / ``status`` /
``effective_date``）要先给集合增列并回灌，属于元数据补齐；schema 落地后再往
`SearchFilter` 加字段，并把表达式拼装收敛到本模块（见 docs/rag-agent-review.md R-01）。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

#: 一次过滤最多接受的文档数：避免把巨型表达式塞进检索请求。
MAX_FILTER_DOC_IDS = 50

#: 与 ``docs/milvus/create_rag_chunk.py`` 一致：``doc_id`` 是 VARCHAR(64)。
MAX_DOC_ID_LENGTH = 64


@dataclass(frozen=True, slots=True)
class SearchFilter:
    """检索范围；``doc_ids`` 为空表示不限制（全库检索）。

    只放**类型化**字段：这里没有「原始表达式」入口，调用方无法绕过
    `build_filter_expr` 的转义。构造时就会规范化 ``doc_ids``（去空白 / 去重），
    所以任何实例的 ``is_empty`` 与 `build_filter_expr` 的结果都一致。
    """

    doc_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # frozen dataclass 只能用 object.__setattr__ 写入；规范化后的实例不会再变。
        object.__setattr__(self, "doc_ids", normalize_doc_ids(self.doc_ids))

    @classmethod
    def from_doc_ids(cls, doc_ids: Iterable[str] | None) -> SearchFilter | None:
        """规范化（去空白 / 去重）后构造；没有有效 id 时返回 ``None``（不过滤）。"""
        if doc_ids is None:
            return None
        if isinstance(doc_ids, str):
            raise TypeError("doc_ids 必须是文档 id 列表，不能直接传字符串")
        search_filter = cls(doc_ids=tuple(doc_ids))
        return None if search_filter.is_empty else search_filter

    @property
    def is_empty(self) -> bool:
        """是否没有任何过滤条件。"""
        return not self.doc_ids


def normalize_doc_ids(doc_ids: Iterable[str] | None) -> tuple[str, ...]:
    """把外部传入的文档 id 收敛成去重后的元组（保持原顺序）。

    * ``None`` / 空列表 → 空元组（调用方据此判断「不限定文档」）；
    * 去首尾空白、丢空串、按出现顺序去重；
    * 单个 id 超长或文档数超上限直接报错，不静默改变检索范围。
    """
    if doc_ids is None:
        return ()
    if isinstance(doc_ids, str):
        # 传字符串会被当成字符序列，静默变成几十个单字 id，是最容易踩的坑。
        raise TypeError("doc_ids 必须是文档 id 列表，不能直接传字符串")
    unique: dict[str, None] = {}
    for raw in doc_ids:
        value = str(raw).strip()
        if not value:
            continue
        if len(value) > MAX_DOC_ID_LENGTH:
            raise ValueError(
                f"doc_id 超过 {MAX_DOC_ID_LENGTH} 个字符（集合里 doc_id 是 VARCHAR(64)）："
                f"{value[:32]}…"
            )
        unique.setdefault(value, None)
    if len(unique) > MAX_FILTER_DOC_IDS:
        raise ValueError(
            f"一次最多按 {MAX_FILTER_DOC_IDS} 篇文档过滤（收到 {len(unique)} 篇），请缩小范围"
        )
    return tuple(unique)


def build_filter_expr(search_filter: SearchFilter | None) -> str:
    """生成 Milvus 过滤表达式；没有过滤条件时返回空字符串（等于不过滤）。"""
    if search_filter is None or not search_filter.doc_ids:
        return ""
    values = ", ".join(f'"{escape_literal(doc_id)}"' for doc_id in search_filter.doc_ids)
    return f"doc_id in [{values}]"


def escape_literal(value: str) -> str:
    """转义 Milvus 过滤表达式里的字符串字面量（反斜杠 + 双引号）。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')
