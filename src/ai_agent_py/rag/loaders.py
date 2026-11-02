"""Document loading: text/markdown, PDF and DOCX -> LangChain `Document`s."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document

from ai_agent_py.config import PROJECT_ROOT
from ai_agent_py.errors import DocumentLoadError

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".text", ".log", ".csv"})
PDF_SUFFIXES = frozenset({".pdf"})
DOCX_SUFFIXES = frozenset({".docx"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES

# utf-8 first, then a GB family fallback for legacy Chinese files, latin-1 last.
_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "latin-1")


def read_text(path: Path) -> str:
    """Read a text file, tolerating the encodings Chinese documents use."""
    raw = path.read_bytes()
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _source_label(path: Path) -> str:
    """Project-relative posix path when possible, absolute otherwise."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _base_metadata(path: Path) -> dict[str, object]:
    return {
        "source": _source_label(path),
        "file_name": path.name,
        "file_type": path.suffix.lower().lstrip("."),
    }


def load_file(path: Path) -> list[Document]:
    """Load a single file into one or more documents."""
    if not path.exists():
        raise DocumentLoadError(f"文件不存在: {path}")
    if not path.is_file():
        raise DocumentLoadError(f"不是文件: {path}")

    suffix = path.suffix.lower()
    metadata = _base_metadata(path)

    if suffix in TEXT_SUFFIXES:
        content = read_text(path)
        return _wrap(content, metadata)

    if suffix in PDF_SUFFIXES:
        return _load_pdf(path, metadata)

    if suffix in DOCX_SUFFIXES:
        from docx import Document as DocxDocument

        docx = DocxDocument(str(path))
        parts = [paragraph.text for paragraph in docx.paragraphs if paragraph.text.strip()]
        for table in docx.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return _wrap("\n".join(parts), metadata)

    raise DocumentLoadError(
        f"不支持的文件类型: {suffix or '(无扩展名)'} ({path.name})。"
        f"支持: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )


def _wrap(content: str, metadata: dict[str, object]) -> list[Document]:
    if not content.strip():
        logger.warning("跳过空文件: %s", metadata.get("source"))
        return []
    return [Document(page_content=content, metadata=dict(metadata))]


def _load_pdf(path: Path, metadata: dict[str, object]) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        page_metadata = dict(metadata)
        page_metadata["page"] = page_number
        documents.append(Document(page_content=text, metadata=page_metadata))

    if not documents:
        logger.warning("PDF 未提取到文字（可能是扫描件，需要 OCR）: %s", metadata.get("source"))
    return documents


def iter_files(paths: Sequence[str | Path]) -> list[Path]:
    """Expand files/directories into a sorted, de-duplicated file list."""
    files: dict[Path, None] = {}
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                    files[candidate.resolve()] = None
        elif path.is_file():
            files[path.resolve()] = None
        else:
            raise DocumentLoadError(f"路径不存在: {path}")
    return list(files)


def load_documents(paths: Sequence[str | Path]) -> list[Document]:
    """Load every supported file below `paths` (directories are walked)."""
    documents: list[Document] = []
    for path in iter_files(paths):
        documents.extend(load_file(path))
    return documents
