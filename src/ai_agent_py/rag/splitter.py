"""Text splitting tuned for mixed Chinese / English documents."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_agent_py.config import Settings, get_settings

# Chinese punctuation first: it keeps clauses whole before falling back to
# whitespace splitting, which matters for CJK text without spaces.
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", "，", ",", " ", ""]


def split_documents(
    documents: Sequence[Document],
    *,
    settings: Settings | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into overlapping chunks and number them."""
    cfg = settings or get_settings()
    size = chunk_size or cfg.chunk_size
    overlap = cfg.chunk_overlap if chunk_overlap is None else chunk_overlap
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        add_start_index=True,
    )
    chunks: list[Document] = []
    for piece in splitter.split_documents(list(documents)):
        content = piece.page_content.strip()
        if not content:
            continue
        piece.page_content = content
        # `chunk_index` is what makes re-ingestion ids stable and de-duplicable.
        piece.metadata["chunk_index"] = len(chunks)
        chunks.append(piece)
    return chunks
