"""Text splitting tuned for mixed Chinese / English documents.

按**文件后缀**选择切分器，前端「切分」按钮、RAG 入库与离线 ingestion 共用同一套规则：

===============  ==========================================  ==========================
后缀             切分器                                      适用场景
===============  ==========================================  ==========================
``.md``          `MarkdownHeaderTextSplitter`                标题层级即语义边界
``.html/.htm``   `HTMLHeaderTextSplitter`                    h1~h4 章节
``.json``        `RecursiveJsonSplitter`                     键值/嵌套结构
其他             `RecursiveCharacterTextSplitter`            中文标点优先的递归切分
===============  ==========================================  ==========================

结构化切分器只决定「在哪里切」，**不保证块大小**（`MarkdownHeaderTextSplitter`
根本没有 `chunk_size` 参数）。所以每一条切分结果最后都会再过一遍递归切分兜底，
确保单块不超过 `chunk_size`，避免超大章节把 embedding 请求撑爆。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.documents import Document
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)

from ai_agent_py.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Chinese punctuation first: it keeps clauses whole before falling back to
# whitespace splitting, which matters for CJK text without spaces.
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", "，", ",", " ", ""]

SplitStrategy = Literal["markdown", "html", "json", "recursive"]

STRATEGY_LABELS: dict[str, str] = {
    "markdown": "Markdown 标题切分",
    "html": "HTML 标题切分",
    "json": "JSON 结构切分",
    "recursive": "递归字符切分",
}

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
HTML_SUFFIXES = frozenset({".html", ".htm"})
JSON_SUFFIXES = frozenset({".json"})

# (标题标记, metadata 键)：层级越深，块越细。
MARKDOWN_HEADERS: list[tuple[str, str]] = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]
HTML_HEADERS: list[tuple[str, str]] = [
    ("h1", "h1"),
    ("h2", "h2"),
    ("h3", "h3"),
    ("h4", "h4"),
]


def normalize_suffix(suffix: str | Path | None) -> str:
    """``PDF`` / ``.pdf`` / ``Path('a.PDF')`` / ``"a.md"`` -> ``.pdf`` / ``.md``。"""
    if suffix is None:
        return ""
    if isinstance(suffix, Path):
        return suffix.suffix.lower()
    raw = str(suffix).strip().lower()
    if not raw:
        return ""
    if raw.startswith("."):
        return raw
    if "." in raw:
        return f".{raw.rsplit('.', 1)[1]}"
    return f".{raw}"


def splitter_for_suffix(suffix: str | Path | None) -> SplitStrategy:
    """按后缀选切分器；未知后缀回落到递归字符切分。"""
    normalized = normalize_suffix(suffix)
    if normalized in MARKDOWN_SUFFIXES:
        return "markdown"
    if normalized in HTML_SUFFIXES:
        return "html"
    if normalized in JSON_SUFFIXES:
        return "json"
    return "recursive"


def splitter_label(strategy: str) -> str:
    """前端展示用的中文名称。"""
    return STRATEGY_LABELS.get(strategy, strategy)


def split_text(
    text: str,
    *,
    suffix: str | Path | None = None,
    strategy: SplitStrategy | None = None,
    settings: Settings | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """把一段文本按后缀对应的策略切分，返回带 ``chunk_index`` 的文档块。"""
    if not text or not text.strip():
        return []

    cfg = settings or get_settings()
    size, overlap = _limits(cfg, chunk_size, chunk_overlap)
    chosen: SplitStrategy = strategy or splitter_for_suffix(suffix)
    base = dict(metadata or {})

    if chosen == "markdown":
        pieces = _markdown_pieces(text, base)
    elif chosen == "html":
        pieces = _html_pieces(text, base)
    elif chosen == "json":
        pieces = _json_pieces(text, size, base)
    else:
        pieces = [Document(page_content=text, metadata=dict(base))]

    bounded = _bounded(pieces, size, overlap, strategy=chosen)
    return _finalize(bounded, chosen)


def split_documents(
    documents: Sequence[Document],
    *,
    settings: Settings | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """切分已加载的文档（供离线 ingestion 使用），按每个文档的后缀分别处理。"""
    cfg = settings or get_settings()
    size, overlap = _limits(cfg, chunk_size, chunk_overlap)

    chunks: list[Document] = []
    for document in documents:
        suffix = _document_suffix(document)
        chunks.extend(
            split_text(
                document.page_content,
                suffix=suffix,
                settings=cfg,
                chunk_size=size,
                chunk_overlap=overlap,
                metadata=document.metadata,
            )
        )

    # 多个源文件拼在一起时，chunk_index 必须全局连续（chunk_id 依赖它）。
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["characters"] = len(chunk.page_content)
    return chunks


# ---- 各策略的「第一刀」 ---------------------------------------------------
def _markdown_pieces(text: str, metadata: dict[str, Any]) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,
    )
    pieces = splitter.split_text(text)
    for piece in pieces:
        piece.metadata = {**metadata, **piece.metadata}
    return pieces


def _html_pieces(text: str, metadata: dict[str, Any]) -> list[Document]:
    splitter = HTMLHeaderTextSplitter(headers_to_split_on=HTML_HEADERS)
    pieces = splitter.split_text(text)
    for piece in pieces:
        piece.metadata = {**metadata, **piece.metadata}
    return _merge_html_headers(pieces)


def _merge_html_headers(pieces: Sequence[Document]) -> list[Document]:
    """把「只有标题」的块并入下一段正文。

    ``HTMLHeaderTextSplitter`` 会为每个标题单独产出一个块（正文另起一块），
    标题块只有几个字，直接送进向量库几乎不可检索；这里把标题文本补到正文前，
    metadata 里的标题层级保持不变。
    """
    merged: list[Document] = []
    pending: Document | None = None
    for piece in pieces:
        content = piece.page_content.strip()
        if not content:
            continue
        headings = {
            str(piece.metadata[key]) for key in ("h1", "h2", "h3", "h4") if piece.metadata.get(key)
        }
        if content in headings and len(content) <= 200:
            pending = (
                piece
                if pending is None
                else Document(
                    page_content=f"{pending.page_content}\n\n{content}",
                    metadata={**pending.metadata, **piece.metadata},
                )
            )
            continue

        if pending is not None:
            piece = Document(
                page_content=f"{pending.page_content}\n\n{content}",
                metadata={**pending.metadata, **piece.metadata},
            )
            pending = None
        merged.append(piece)

    if pending is not None:
        merged.append(pending)
    return merged


def _json_pieces(text: str, size: int, metadata: dict[str, Any]) -> list[Document]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        # 后缀是 .json 但内容坏了：不阻断预览，退回纯文本切分并在 metadata 标记。
        logger.warning("JSON 解析失败（第 %d 行），退回递归切分: %s", exc.lineno, exc.msg)
        return [
            Document(
                page_content=text,
                metadata={**metadata, "json_error": f"第 {exc.lineno} 行: {exc.msg}"},
            )
        ]

    if not isinstance(payload, dict | list):
        # 标量 JSON（"abc" / 42）没有结构可切，直接作为一段。
        return [
            Document(
                page_content=json.dumps(payload, ensure_ascii=False),
                metadata={**metadata, "json": True},
            )
        ]

    splitter = RecursiveJsonSplitter(max_chunk_size=size)
    try:
        pieces = splitter.split_json(
            # 运行时 split_json 支持顶层数组（convert_lists=True），类型标注只写了 dict。
            json_data=cast("dict[str, Any]", payload),
            convert_lists=True,
        )
    except (TypeError, ValueError) as exc:  # 结构过于特殊时兜底
        logger.warning("JSON 结构化切分失败，退回递归切分: %s", exc)
        return [
            Document(
                page_content=json.dumps(payload, ensure_ascii=False),
                metadata={**metadata, "json": True},
            )
        ]

    return [
        Document(
            # 紧凑格式：RecursiveJsonSplitter 按紧凑 JSON 的字符数控制大小，
            # 用 indent=2 会额外膨胀 30%+，导致块被兜底切分、JSON 不再合法。
            page_content=json.dumps(piece, ensure_ascii=False, separators=(",", ":")),
            metadata={**metadata, "json": True},
        )
        for piece in pieces
        if piece
    ]


# ---- 大小兜底与编号 -------------------------------------------------------
def _bounded(
    pieces: Sequence[Document],
    size: int,
    overlap: int,
    *,
    strategy: SplitStrategy,
) -> list[Document]:
    """把超过 ``chunk_size`` 的结构化块再递归切开（metadata 原样保留）。

    JSON 例外：``RecursiveJsonSplitter`` 遇到单个超长字符串值时无法再切（再切就不再是
    合法 JSON），所以保留原块，允许它超过 ``chunk_size``。
    """
    if strategy == "json":
        return [
            Document(page_content=piece.page_content.strip(), metadata=dict(piece.metadata))
            for piece in pieces
            if piece.page_content.strip()
        ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        add_start_index=True,
    )
    out: list[Document] = []
    for piece in pieces:
        content = piece.page_content.strip()
        if not content:
            continue
        if len(content) <= size:
            out.append(Document(page_content=content, metadata=dict(piece.metadata)))
            continue
        for part in splitter.split_text(content):
            part = part.strip()
            if part:
                out.append(Document(page_content=part, metadata=dict(piece.metadata)))
    return out


def _finalize(pieces: Sequence[Document], strategy: SplitStrategy) -> list[Document]:
    label = splitter_label(strategy)
    chunks: list[Document] = []
    for piece in pieces:
        content = piece.page_content.strip()
        if not content:
            continue
        metadata = dict(piece.metadata)
        metadata["chunk_index"] = len(chunks)
        metadata["splitter"] = strategy
        metadata["splitter_label"] = label
        metadata["characters"] = len(content)
        chunks.append(Document(page_content=content, metadata=metadata))
    return chunks


def _limits(
    cfg: Settings,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> tuple[int, int]:
    size = cfg.chunk_size if chunk_size is None else chunk_size
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    # 只改 chunk_size、没传 overlap 时，把默认重叠压到合法范围（否则 100/120 这种
    # 组合会直接报错，前端只调一个输入框时会很困惑）。
    overlap = min(cfg.chunk_overlap, max(0, size - 1)) if chunk_overlap is None else chunk_overlap
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    return size, overlap


def _document_suffix(document: Document) -> str:
    """从 loader 写入的 metadata 推断后缀，兼容旧的 ``source`` 字段。"""
    file_type = document.metadata.get("file_type")
    if isinstance(file_type, str) and file_type:
        return file_type
    source = document.metadata.get("source")
    if isinstance(source, str) and source:
        return Path(source).suffix
    return ""
