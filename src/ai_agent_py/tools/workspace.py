"""Workspace confinement and file traversal shared by the agent tools.

Every file tool resolves paths through :func:`resolve_path`, so a model that
hallucinates ``../../etc/passwd`` (or follows a symlink out of the project)
gets a clear error instead of touching the host filesystem. :func:`iter_files`
prunes the usual noise directories so ``glob`` / ``grep`` stay fast and quiet.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import ToolError

#: Directories that never contain project source worth showing the model.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        ".tox",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
    }
)

_DRIVE_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def resolve_workspace(
    workspace: str | Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Return the absolute directory the file tools may touch.

    ``workspace`` wins over ``settings``; when both are missing the
    ``AGENT_WORKSPACE`` setting is used (default: project root).
    """
    if workspace is not None:
        root = Path(workspace).expanduser()
    else:
        root = (settings or get_settings()).agent_workspace
    root = root.resolve()
    if not root.is_dir():
        raise ToolError(f"工作目录不存在或不是目录：{root}")
    return root


def resolve_path(root: Path, raw: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve ``raw`` against ``root``, rejecting anything outside the root.

    Symlinks are resolved before the check, so a link inside the workspace
    cannot be used to escape it. ``must_exist`` raises a model friendly error
    instead of a bare ``FileNotFoundError``.
    """
    text = str(raw).strip()
    if not text:
        raise ToolError("路径不能为空")
    path = Path(text).expanduser()
    if os.name == "nt" and text[:1] in {"/", "\\"} and not _DRIVE_ABSOLUTE.match(text):
        # Agents often send POSIX style "/src/app.py"; on Windows that would
        # otherwise mean "root of the current drive".
        path = root / text.lstrip("/\\")
    target = path if path.is_absolute() else root / path
    resolved = target.resolve()
    if not resolved.is_relative_to(root):
        raise ToolError(f"路径超出工作目录：{text}（工作目录：{root}）")
    if must_exist and not resolved.exists():
        raise ToolError(f"文件或目录不存在：{text}")
    return resolved


def display_path(root: Path, path: Path) -> str:
    """Path shown to the model: relative to the workspace, POSIX slashes."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.as_posix()
    return relative.as_posix() or "."


def iter_files(root: Path, base: Path) -> Iterator[Path]:
    """Yield files under ``base``, skipping :data:`IGNORED_DIRS`.

    Symlinked directories are not descended into and symlinked files pointing
    outside the workspace are skipped, which keeps ``glob``/``grep`` inside the
    sandbox and avoids symlink loops.
    """
    if base.is_file():
        yield base
        return
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in IGNORED_DIRS and not (current / name).is_symlink()
        )
        for name in sorted(filenames):
            candidate = current / name
            if candidate.is_symlink() and not _inside(root, candidate):
                continue
            yield candidate


def _inside(root: Path, path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root)
    except OSError:
        return False


@lru_cache(maxsize=256)
def compile_glob(pattern: str) -> re.Pattern[str]:
    """Translate a workspace glob into a regex.

    ``**`` spans directories (``src/**/*.py``), ``*`` and ``?`` never cross a
    ``/``. Matching is case insensitive on Windows, like the filesystem.
    """
    expression: list[str] = []
    index = 0
    length = len(pattern)
    while index < length:
        char = pattern[index]
        if char == "*":
            if pattern.startswith("**", index):
                index += 2
                if index < length and pattern[index] == "/":
                    expression.append("(?:.*/)?")
                    index += 1
                else:
                    expression.append(".*")
            else:
                expression.append("[^/]*")
                index += 1
        elif char == "?":
            expression.append("[^/]")
            index += 1
        elif char == "[":
            end = pattern.find("]", index + 1)
            if end == -1:
                expression.append(re.escape(char))
                index += 1
            else:
                body = pattern[index + 1 : end]
                if body.startswith("!"):
                    body = "^" + body[1:]
                expression.append(f"[{body}]")
                index = end + 1
        else:
            expression.append(re.escape(char))
            index += 1
    flags = re.IGNORECASE if os.name == "nt" else 0
    return re.compile("".join(expression) + r"\Z", flags)


def relative_matcher(pattern: str) -> re.Pattern[str]:
    """Compile ``pattern`` for paths relative to the search directory.

    A pattern without ``/`` (``*.py``) matches the file name at any depth; an
    explicit ``**`` is kept as written.
    """
    expression = pattern.strip().replace("\\", "/")
    if not expression:
        raise ToolError("glob 模式不能为空")
    if Path(expression).is_absolute() or expression.startswith("~"):
        raise ToolError(f"glob 模式必须是相对路径：{pattern}")
    if expression.startswith("./"):
        expression = expression[2:]
    if "/" not in expression:
        expression = f"**/{expression}"
    return compile_glob(expression)
