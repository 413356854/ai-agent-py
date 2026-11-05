"""Generic file tools for the agent: read / write / edit / glob / grep.

All paths are resolved inside the workspace (see :mod:`ai_agent_py.tools.workspace`),
reads are line numbered, and every result is a plain string the model can
consume. Expected failures raise :class:`~ai_agent_py.errors.ToolError`, which
the tool wrappers turn into a ``错误：...`` message, so a wrong path or a bad
regex never breaks the graph.
"""

from __future__ import annotations

import codecs
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from ai_agent_py.config import Settings
from ai_agent_py.errors import ToolError
from ai_agent_py.tools.workspace import (
    display_path,
    iter_files,
    relative_matcher,
    resolve_path,
    resolve_workspace,
)

READ_TOOL_NAME = "read"
WRITE_TOOL_NAME = "write"
EDIT_TOOL_NAME = "edit"
GLOB_TOOL_NAME = "glob"
GREP_TOOL_NAME = "grep"

READ_DESCRIPTION = (
    "读取工作目录内的文本文件，返回带行号的内容（如 `  12→...`），便于引用和定位。"
    "offset 为起始行号（从 1 开始），limit 为本次最多返回行数（默认 2000）。"
    "文件较长时末尾会提示下一段 offset；二进制文件与超过 20MB 的文件会被拒绝。"
)
WRITE_DESCRIPTION = (
    "把完整内容写入工作目录内的文件（UTF-8，换行原样保留）：不存在则创建，存在则整体覆盖，"
    "父目录自动创建。只改局部内容请用 edit，不要用 write 整文件重写。"
)
EDIT_DESCRIPTION = (
    "对工作目录内的文本文件做精确字符串替换。old_string 必须与原文完全一致（含缩进与空白）；"
    "默认要求唯一出现，出现多次会返回次数，可设 replace_all=true 全部替换。编辑前先用 read 查看原文。"
)
GLOB_DESCRIPTION = (
    "按 glob 模式查找工作目录内的文件（只返回文件，不返回目录）。"
    "模式不含 '/' 时按文件名在任意层级匹配（如 `*.py`）；显式递归用 `**`（如 `src/**/*.py`）。"
    "自动跳过 .git / .venv / node_modules / __pycache__ 等目录，结果按修改时间倒序。"
)
GREP_DESCRIPTION = (
    "用正则搜索工作目录内的文件内容，返回按文件分组的『行号: 内容』。"
    "include 用 glob 限定文件类型（如 `*.py`），ignore_case 忽略大小写。"
    "自动跳过二进制文件与 .git / .venv / node_modules 等目录，结果默认最多 250 行。"
)

MAX_LINE_CHARS = 2_000
MAX_READ_BYTES = 20 * 1024 * 1024
MAX_GREP_FILE_BYTES = 2 * 1024 * 1024
MAX_RESULTS = 1_000
DEFAULT_GLOB_RESULTS = 200
DEFAULT_GREP_RESULTS = 250
MAX_SCAN_MATCHES = 10_000
MAX_READ_LINES = 5_000


@dataclass(frozen=True)
class Decoded:
    """Text decoded from disk plus the metadata needed to write it back."""

    text: str
    encoding: str
    bom: bool = False
    lossy: bool = False


def _decode(data: bytes) -> Decoded:
    if data.startswith(codecs.BOM_UTF8):
        try:
            return Decoded(data.decode("utf-8-sig"), "utf-8", bom=True)
        except UnicodeDecodeError:
            pass
    for encoding in ("utf-8", "gb18030"):
        try:
            return Decoded(data.decode(encoding), encoding)
        except UnicodeDecodeError:
            continue
    return Decoded(data.decode("utf-8", errors="replace"), "utf-8", lossy=True)


def _encoding_label(decoded: Decoded) -> str:
    label = decoded.encoding.upper()
    if decoded.bom:
        label += "（带 BOM）"
    if decoded.lossy:
        label += "，含无法解码的字节"
    return label


def _encode(text: str, decoded: Decoded) -> tuple[bytes, str | None]:
    """Encode with the file's original encoding; return a note when that fails."""
    try:
        data = text.encode(decoded.encoding)
    except UnicodeEncodeError:
        return text.encode("utf-8"), "新内容无法用原编码表示，已改写为 UTF-8"
    if decoded.bom:
        data = codecs.BOM_UTF8 + data
    return data, None


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ToolError(f"读取失败：{exc.strerror or exc}") from exc


def _write_bytes(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError as exc:
        raise ToolError(f"写入失败：{exc.strerror or exc}") from exc


def _as_int(value: Any, default: int, *, name: str, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{name} 必须是整数：{value!r}") from exc
    return max(minimum, min(number, maximum))


def _ensure_text(data: bytes, name: str) -> None:
    if b"\x00" in data[:8192]:
        raise ToolError(f"{name} 看起来是二进制文件，无法按文本处理")


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def read_file(root: Path, path: str, offset: int = 1, limit: int = 2000) -> str:
    """Read a text file and render at most ``limit`` lines from ``offset``."""
    target = resolve_path(root, path, must_exist=True)
    name = display_path(root, target)
    if target.is_dir():
        raise ToolError(f"{name} 是目录；列出文件请用 glob，搜索内容请用 grep")
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise ToolError(f"读取失败：{exc.strerror or exc}") from exc
    if size > MAX_READ_BYTES:
        raise ToolError(f"{name} 太大（{size} 字节，上限 {MAX_READ_BYTES} 字节）")

    data = _read_bytes(target)
    _ensure_text(data, name)
    decoded = _decode(data)
    lines = decoded.text.splitlines()
    start_line = _as_int(offset, 1, name="offset", minimum=1, maximum=10**9)
    count = _as_int(limit, 2000, name="limit", minimum=1, maximum=MAX_READ_LINES)
    header = f"{name}（共 {len(lines)} 行，{_encoding_label(decoded)}）"

    if not lines:
        return f"{header}\n（空文件）"
    start = start_line - 1
    if start >= len(lines):
        return f"{header}\n（offset={start_line} 超出文件行数）"

    end = min(len(lines), start + count)
    width = max(3, len(str(end)))
    rendered: list[str] = []
    for number, line in enumerate(lines[start:end], start=start_line):
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + "…（本行过长已截断）"
        rendered.append(f"{number:>{width}}→{line}")

    body = "\n".join(rendered)
    if end < len(lines):
        body += f"\n…（还有 {len(lines) - end} 行，用 offset={end + 1} 继续读取）"
    return f"{header}\n{body}"


def write_file(root: Path, path: str, content: str) -> str:
    """Create or overwrite ``path`` with ``content`` (UTF-8, exact newlines)."""
    target = resolve_path(root, path)
    name = display_path(root, target)
    if target.exists() and target.is_dir():
        raise ToolError(f"{name} 是目录，不能写入")
    previous = target.stat().st_size if target.exists() else None
    data = content.encode("utf-8")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # newline="" keeps LF as LF on Windows (no CRLF translation).
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
    except OSError as exc:
        raise ToolError(f"写入 {name} 失败：{exc.strerror or exc}") from exc

    line_count = len(content.splitlines())
    if previous is None:
        return f"已创建 {name}（{line_count} 行，{len(data)} 字节，UTF-8）"
    message = f"已覆盖 {name}（{line_count} 行，{previous} → {len(data)} 字节）"
    if previous > 400 and len(data) < previous * 0.5:
        message += "；注意：文件比原来小了一半以上，确认没有误删内容"
    return message


def edit_file(
    root: Path,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace ``old_string`` with ``new_string`` in an existing text file."""
    if old_string == "":
        raise ToolError("old_string 不能为空")
    if old_string == new_string:
        raise ToolError("old_string 与 new_string 相同，编辑不会产生变化")

    target = resolve_path(root, path, must_exist=True)
    name = display_path(root, target)
    if target.is_dir():
        raise ToolError(f"{name} 是目录，不能编辑")
    data = _read_bytes(target)
    _ensure_text(data, name)
    decoded = _decode(data)

    raw_text = decoded.text
    crlf = raw_text.count("\r\n")
    lf = raw_text.count("\n")
    use_crlf = crlf > lf - crlf
    text = raw_text.replace("\r\n", "\n")
    old = old_string.replace("\r\n", "\n")
    new = new_string.replace("\r\n", "\n")

    occurrences = text.count(old)
    if occurrences == 0:
        raise ToolError(
            f"在 {name} 中未找到 old_string；请先用 read 查看原文，"
            "注意缩进、空行与空白字符必须完全一致"
        )
    if occurrences > 1 and not replace_all:
        raise ToolError(
            f"old_string 在 {name} 中出现 {occurrences} 次；"
            "请提供更长的唯一片段，或设置 replace_all=true"
        )

    first = text.find(old)
    line = text.count("\n", 0, first) + 1
    updated = text.replace(old, new)
    output = updated.replace("\n", "\r\n") if use_crlf else updated
    payload, note = _encode(output, decoded)
    try:
        target.write_bytes(payload)
    except OSError as exc:
        raise ToolError(f"写入 {name} 失败：{exc.strerror or exc}") from exc

    message = f"已编辑 {name}：替换 {occurrences} 处（第 {line} 行）"
    if note:
        message += f"；{note}"
    return message


def glob_files(
    root: Path,
    pattern: str,
    search_path: str = ".",
    max_results: int = DEFAULT_GLOB_RESULTS,
) -> str:
    """List files matching a glob pattern, newest first."""
    display_pattern = pattern.strip()
    matcher = relative_matcher(pattern)
    base = resolve_path(root, search_path or ".", must_exist=True)
    limit = _as_int(
        max_results,
        DEFAULT_GLOB_RESULTS,
        name="max_results",
        minimum=1,
        maximum=MAX_RESULTS,
    )
    scope = display_path(root, base)

    if base.is_file():
        if matcher.match(base.name):
            return f"匹配「{display_pattern}」的文件共 1 个：\n{display_path(root, base)}"
        return f"没有匹配「{display_pattern}」的文件（搜索目录：{scope}）"

    matches: list[Path] = []
    scan_truncated = False
    for candidate in iter_files(root, base):
        relative = candidate.relative_to(base).as_posix()
        if not matcher.match(relative):
            continue
        matches.append(candidate)
        if len(matches) >= MAX_SCAN_MATCHES:
            scan_truncated = True
            break

    if not matches:
        return f"没有匹配「{display_pattern}」的文件（搜索目录：{scope}）"

    matches.sort(key=lambda item: (-_mtime(item), display_path(root, item)))
    shown = matches[:limit]
    total = f"{len(matches)}+" if scan_truncated else str(len(matches))
    lines = [f"匹配「{display_pattern}」的文件共 {total} 个（按修改时间倒序）："]
    lines.extend(display_path(root, item) for item in shown)
    if len(matches) > len(shown):
        lines.append(f"…（仅显示前 {len(shown)} 个；缩小搜索目录或调大 max_results）")
    return "\n".join(lines)


def grep_files(
    root: Path,
    pattern: str,
    search_path: str = ".",
    include: str | None = None,
    ignore_case: bool = False,
    max_results: int = DEFAULT_GREP_RESULTS,
) -> str:
    """Search file contents with a regex, grouped by file."""
    if not pattern:
        raise ToolError("pattern 不能为空")
    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise ToolError(f"正则表达式无效：{exc}") from exc

    base = resolve_path(root, search_path or ".", must_exist=True)
    include_matcher = relative_matcher(include) if include else None
    limit = _as_int(
        max_results,
        DEFAULT_GREP_RESULTS,
        name="max_results",
        minimum=1,
        maximum=MAX_RESULTS,
    )
    scope = display_path(root, base)

    hits: list[tuple[str, list[str]]] = []
    total = 0
    stopped = False
    for candidate in iter_files(root, base):
        relative = candidate.name if base.is_file() else candidate.relative_to(base).as_posix()
        if include_matcher is not None and not include_matcher.match(relative):
            continue
        try:
            if candidate.stat().st_size > MAX_GREP_FILE_BYTES:
                continue
        except OSError:
            continue
        try:
            data = candidate.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8192]:
            continue

        rows: list[str] = []
        for number, line in enumerate(_decode(data).text.splitlines(), start=1):
            if not regex.search(line):
                continue
            text = line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + "…"
            rows.append(f"  {number}: {text}")
            total += 1
            if total >= limit:
                stopped = True
                break
        if rows:
            hits.append((display_path(root, candidate), rows))
        if stopped:
            break

    if total == 0:
        return f"没有匹配「{pattern}」的内容（搜索目录：{scope}）"

    suffix = "+" if stopped else ""
    header = f"在 {len(hits)} 个文件中找到 {total}{suffix} 行匹配「{pattern}」"
    body = "\n\n".join(f"{name}\n" + "\n".join(rows) for name, rows in hits)
    if stopped:
        body += f"\n…（已达 {limit} 行上限，缩小搜索范围或用 include 过滤）"
    return f"{header}\n\n{body}"


def _guard(action: Callable[[], str]) -> str:
    """Run a tool body and turn expected failures into a model friendly string."""
    try:
        return action()
    except ToolError as exc:
        return f"错误：{exc}"


def build_filesystem_tools(
    *,
    workspace: str | Path | None = None,
    settings: Settings | None = None,
) -> list[BaseTool]:
    """Build the ``read`` / ``write`` / ``edit`` / ``glob`` / ``grep`` tools.

    All tools share one workspace root: paths, glob scopes and grep targets are
    resolved inside it, and anything escaping it is refused.
    """
    root = resolve_workspace(workspace, settings)

    @tool(READ_TOOL_NAME, description=READ_DESCRIPTION)
    def read(path: str, offset: int = 1, limit: int = 2000) -> str:
        """读取工作目录内的文本文件，返回带行号的内容。"""
        return _guard(lambda: read_file(root, path, offset=offset, limit=limit))

    @tool(WRITE_TOOL_NAME, description=WRITE_DESCRIPTION)
    def write(path: str, content: str) -> str:
        """把完整内容写入工作目录内的文件（不存在则创建，存在则覆盖）。"""
        return _guard(lambda: write_file(root, path, content))

    @tool(EDIT_TOOL_NAME, description=EDIT_DESCRIPTION)
    def edit(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """在文件中把 old_string 精确替换为 new_string。"""
        return _guard(lambda: edit_file(root, path, old_string, new_string, replace_all))

    @tool(GLOB_TOOL_NAME, description=GLOB_DESCRIPTION)
    def glob(pattern: str, path: str = ".", max_results: int = DEFAULT_GLOB_RESULTS) -> str:
        """按 glob 模式查找工作目录内的文件。"""
        return _guard(lambda: glob_files(root, pattern, path, max_results))

    @tool(GREP_TOOL_NAME, description=GREP_DESCRIPTION)
    def grep(
        pattern: str,
        path: str = ".",
        include: str | None = None,
        ignore_case: bool = False,
        max_results: int = DEFAULT_GREP_RESULTS,
    ) -> str:
        """用正则搜索工作目录内的文件内容。"""
        return _guard(lambda: grep_files(root, pattern, path, include, ignore_case, max_results))

    return [read, write, edit, glob, grep]
