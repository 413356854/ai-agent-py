"""RAG 切片补删队列：删除文档时 Milvus 不可用的兜底（删除对称）。

删除解析任务时必须同时清掉 ``rag_chunk`` 里该文档的全部切片，否则问答还会引用
「已删除」的文档。但 Milvus 是外部服务，不能因为删不掉向量就拒绝删除任务，也不能
把待删的 ``doc_id`` 只留在内存里（进程重启就丢）——所以用**一个 doc_id 一个 JSON
标记文件**的落盘队列：

* 目录：``RAG_DELETE_PENDING_DIR``（默认 ``data/rag_delete_pending``）；
* 写入先写 ``*.json.tmp`` 再 `os.replace`，别的进程不会读到半个文件；
* 同一 ``doc_id`` 只保留一条标记，``attempts`` 累加尝试次数，``last_error`` 留原因；
* 补删成功后删除标记，失败则保留，由 `mineru.service.DocumentService` 后台重试。

为什么不用 MySQL 表：补删正好发生在 Milvus 故障时，MySQL 与 Milvus 是两个独立
故障域，队列不该因为数据库也不可用而丢；标记文件和 ``data/parsed`` 一样是本地
运行态资产。多实例部署时各实例会扫到同一批标记，重复删除是幂等的；将来多实例
不共享磁盘时再换成 Redis / MySQL 队列（A-10）。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 标记文件名里只允许这些字符；路径分隔符、控制字符等一律替换成 ``_``。
_UNSAFE_CHARS = re.compile(r"[^0-9A-Za-z_.-]+")

#: 文件名最长保留的 doc_id 字符数（doc_id 一般是 uuid hex 或 ``doc-<sha1前16位>``）。
_MAX_DOC_ID_CHARS = 96

#: ``last_error`` 只留一段，避免异常栈把标记文件撑大。
_MAX_ERROR_CHARS = 500


@dataclass(slots=True)
class PendingRagDelete:
    """一条「Milvus 切片还没删掉」的记录（磁盘上一个 JSON 标记文件）。"""

    doc_id: str
    file_name: str | None = None
    collection: str | None = None
    attempts: int = 0
    last_error: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "collection": self.collection,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PendingRagDelete:
        """标记文件内容 → 记录；缺 ``doc_id`` 时抛 `ValueError`（调用方忽略该文件）。"""
        doc_id = str(raw.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError("标记缺少 doc_id")
        return cls(
            doc_id=doc_id,
            file_name=_optional_text(raw.get("file_name")),
            collection=_optional_text(raw.get("collection")),
            attempts=_as_attempts(raw.get("attempts")),
            last_error=str(raw.get("last_error") or "")[:_MAX_ERROR_CHARS],
            created_at=_as_time(raw.get("created_at")),
            updated_at=_as_time(raw.get("updated_at")),
        )


class RagDeleteQueue:
    """补删标记的落盘队列；``directory`` 可注入，单元测试用 ``tmp_path``。"""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def __len__(self) -> int:
        return len(self.items())

    def record(
        self,
        doc_id: str,
        *,
        file_name: str | None = None,
        collection: str | None = None,
        error: str = "",
    ) -> PendingRagDelete:
        """记一次删除失败：同一 ``doc_id`` 只留一条，``attempts`` 累加。"""
        clean = (doc_id or "").strip()
        if not clean:
            raise ValueError("doc_id 不能为空")

        previous = self._load(self._path(clean))
        now = datetime.now(UTC)
        item = PendingRagDelete(
            doc_id=clean,
            file_name=file_name or (previous.file_name if previous else None),
            collection=collection or (previous.collection if previous else None),
            attempts=(previous.attempts + 1) if previous else 1,
            last_error=(error or "").strip()[:_MAX_ERROR_CHARS],
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        self._save(item)
        return item

    def items(self) -> list[PendingRagDelete]:
        """全部待补删记录，按创建时间排序（损坏 / 非法的标记跳过并告警）。"""
        if not self.directory.is_dir():
            return []
        found: list[PendingRagDelete] = []
        for path in sorted(self.directory.glob("*.json")):
            item = self._load(path)
            if item is not None:
                found.append(item)
        return sorted(found, key=lambda item: (item.created_at, item.doc_id))

    def discard(self, doc_id: str) -> None:
        """补删成功后移除标记（文件已不存在时静默返回）。"""
        path = self._path_if_valid(doc_id)
        if path is None:
            return
        with contextlib.suppress(FileNotFoundError):
            path.unlink()

    # ---- 内部 -----------------------------------------------------------
    def _path(self, doc_id: str) -> Path:
        path = self._path_if_valid(doc_id)
        if path is None:
            raise ValueError("doc_id 不能为空")
        return path

    def _path_if_valid(self, doc_id: str) -> Path | None:
        stem = _file_stem((doc_id or "").strip())
        return None if stem is None else self.directory / f"{stem}.json"

    def _save(self, item: PendingRagDelete) -> None:
        """先写临时文件再原子替换：读的一方永远看不到半截 JSON。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(item.doc_id)
        tmp = path.with_name(f"{path.name}.tmp")
        try:
            tmp.write_text(
                json.dumps(item.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        finally:
            # 写失败 / 被取消时别把临时文件留在目录里（glob 只认 *.json，但脏）。
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()

    def _load(self, path: Path) -> PendingRagDelete | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("补删标记无法读取，已忽略：%s（%s）", path, exc)
            return None
        if not isinstance(raw, dict):
            logger.warning("补删标记内容不是对象，已忽略：%s", path)
            return None
        try:
            return PendingRagDelete.from_dict(raw)
        except (TypeError, ValueError) as exc:
            logger.warning("补删标记内容不合法，已忽略：%s（%s）", path, exc)
            return None


# ---- 内部工具 ---------------------------------------------------------------
def _file_stem(doc_id: str) -> str | None:
    """doc_id → 安全的文件名主干；非法字符被替换，变化时附加短 hash 防撞名。"""
    if not doc_id:
        return None
    safe = _UNSAFE_CHARS.sub("_", doc_id).strip("._")
    if not safe:
        return None
    if safe == doc_id and len(safe) <= _MAX_DOC_ID_CHARS:
        return safe
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:_MAX_DOC_ID_CHARS]}-{digest}"


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _as_attempts(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _as_time(value: Any) -> datetime:
    """ISO 时间字符串 → 带 UTC 时区的时间；解析不了就用当前时间。"""
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            return datetime.now(UTC)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return datetime.now(UTC)
