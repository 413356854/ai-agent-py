"""解析结果 zip 的解包与校验。

MinerU 的结果 zip 结构（非 HTML 文件）::

    full.md                 Markdown 正文
    content_list.json       内容块列表
    layout.json / middle.json
    images/*.jpg            文档中抽出的图片（Markdown 里以相对路径引用）

HTML 文件的结果是 ``full.md`` + ``main.html``，同样兼容。
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ai_agent_py.errors import MineruError

MAX_MEMBERS = 5000
MAX_TOTAL_BYTES = 1024 * 1024 * 1024  # 1 GB，防 zip bomb


@dataclass(slots=True)
class ExtractSummary:
    """解包结果摘要。"""

    markdown_name: str
    characters: int
    assets: int
    files: int
    bytes_written: int
    pages: int = 0


def _safe_member(name: str) -> PurePosixPath:
    """拒绝绝对路径与 `..`，避免 zip slip。"""
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise MineruError(f"解析结果压缩包包含非法路径: {name}")
    return path


def extract_result_zip(payload: bytes, target_dir: Path) -> ExtractSummary:
    """把结果 zip 解压到 `target_dir`，返回 Markdown 文件名与统计信息。

    解包在临时目录里完成，成功后才替换 `target_dir` 中上一次的结果，
    因此解析失败不会破坏已经存在的旧结果。
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    staging = target_dir / ".extract"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            if len(members) > MAX_MEMBERS:
                raise MineruError(f"解析结果文件过多（{len(members)}），已中止解包")
            total = sum(m.file_size for m in members)
            if total > MAX_TOTAL_BYTES:
                raise MineruError(f"解析结果过大（{total / 1024 / 1024:.0f} MB），已中止解包")
            root = staging.resolve()
            for member in members:
                relative = _safe_member(member.filename)
                destination = (staging / relative).resolve()
                if not destination.is_relative_to(root):
                    raise MineruError(f"解析结果压缩包包含非法路径: {member.filename}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise MineruError(f"下载的结果不是合法的 zip 文件: {exc}") from exc
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    markdown = _pick_markdown(staging)
    if markdown is None:
        shutil.rmtree(staging, ignore_errors=True)
        raise MineruError("解析结果中没有找到 Markdown 文件（full.md）")

    markdown_name = markdown.relative_to(staging).as_posix()
    text = markdown.read_text(encoding="utf-8", errors="replace")
    files = sum(1 for path in staging.rglob("*") if path.is_file())
    image_dir = staging / "images"
    assets = sum(1 for path in image_dir.rglob("*") if path.is_file()) if image_dir.is_dir() else 0
    pages = _count_pages(staging)

    for existing in target_dir.iterdir():
        if existing.name == ".extract" or existing.name.startswith("source."):
            continue
        if existing.is_dir():
            shutil.rmtree(existing, ignore_errors=True)
        else:
            existing.unlink(missing_ok=True)
    for item in staging.iterdir():
        shutil.move(str(item), str(target_dir / item.name))
    shutil.rmtree(staging, ignore_errors=True)

    return ExtractSummary(
        markdown_name=markdown_name,
        characters=len(text),
        assets=assets,
        files=files,
        bytes_written=total,
        pages=pages,
    )


def _count_pages(root: Path) -> int:
    """从 `content_list.json` 的 `page_idx` 推断页数（批量接口不返回页数时兜底）。"""
    for pattern in ("*_content_list.json", "content_list.json"):
        for path in sorted(root.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(payload, list):
                continue
            indexes = [
                item["page_idx"]
                for item in payload
                if isinstance(item, dict) and isinstance(item.get("page_idx"), int)
            ]
            if indexes:
                return max(indexes) + 1
    return 0


def _pick_markdown(root: Path) -> Path | None:
    """优先根目录的 full.md，其次任意 `.md`。"""
    preferred = root / "full.md"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        (path for path in root.rglob("*.md") if path.is_file()),
        key=lambda path: (len(path.parts), path.as_posix()),
    )
    return candidates[0] if candidates else None
