"""切分结果的进程内缓存（HashMap），供「查阅后确认入库」按 ``split_id`` 取回。

前端只回传 ``split_id``，不再把几千个切片塞进请求体；服务端按 id 从缓存里取出
**完整**切片列表（预览可能截断，缓存不截断）直接写 Milvus。

缓存特性：

* 只存在本进程内存，服务重启 / 换实例即失效。多实例部署要么做粘性会话，要么
  改用无状态的 ``chunks`` 入参；
* 三重上限：TTL（默认 30 分钟）、条数（默认 16）、总字符（默认 8M，约 16MB
  中文），超出后淘汰最久未使用的条目；
* ``get`` 会刷新 LRU 位置，但**不续期** TTL，避免反复点击把内存一直占着；
* 删除文档任务时用 :meth:`ChunkCache.drop_job` 主动清理。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from uuid import uuid4

from langchain_core.documents import Document

#: 默认缓存有效期：30 分钟，够用户慢慢查阅。
DEFAULT_TTL_SECONDS = 30 * 60
#: 默认最多缓存多少次切分结果。
DEFAULT_MAX_ENTRIES = 16
#: 默认缓存总字符上限（中文按 1 字符 ≈ 2~3 字节，8M 字符约 16~24MB）。
DEFAULT_MAX_CHARS = 8_000_000


@dataclass(slots=True)
class CachedSplit:
    """一次切分的完整结果与回传给前端的标识 / 统计。"""

    split_id: str
    job_id: str
    chunks: list[Document]
    chunk_size: int
    chunk_overlap: int
    strategy: str
    characters: int
    created_at: float
    expires_at: float

    @property
    def total(self) -> int:
        return len(self.chunks)


class ChunkCache:
    """``split_id -> CachedSplit`` 的有界缓存（线程安全的 LRU + TTL）。"""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max(1, max_entries)
        self.max_chars = max(1, max_chars)
        self._items: OrderedDict[str, CachedSplit] = OrderedDict()
        self._characters = 0
        self._lock = threading.Lock()

    def put(
        self,
        *,
        job_id: str,
        chunks: list[Document],
        chunk_size: int,
        chunk_overlap: int,
        strategy: str,
    ) -> CachedSplit:
        """缓存一次切分结果，返回带新 ``split_id`` 的条目。"""
        now = time.monotonic()
        characters = sum(len(chunk.page_content) for chunk in chunks)
        entry = CachedSplit(
            split_id=uuid4().hex,
            job_id=job_id,
            chunks=list(chunks),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
            characters=characters,
            created_at=now,
            expires_at=now + max(0.0, self.ttl_seconds),
        )
        with self._lock:
            self._expire(now)
            self._items[entry.split_id] = entry
            self._characters += characters
            self._trim()
        return entry

    def get(self, split_id: str) -> CachedSplit | None:
        """按 id 取回切片；不存在或已过期返回 ``None``。"""
        now = time.monotonic()
        with self._lock:
            entry = self._items.get(split_id)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._remove(split_id)
                return None
            self._items.move_to_end(split_id)
            return entry

    def drop_job(self, job_id: str) -> int:
        """删除某个任务的全部缓存条目（删除任务时调用），返回清理条数。"""
        with self._lock:
            keys = [key for key, entry in self._items.items() if entry.job_id == job_id]
            for key in keys:
                self._remove(key)
            return len(keys)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._characters = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    # ---- 内部（调用方需已持有 _lock）-----------------------------------
    def _remove(self, split_id: str) -> None:
        entry = self._items.pop(split_id, None)
        if entry is not None:
            self._characters -= entry.characters

    def _expire(self, now: float) -> None:
        for key in [key for key, entry in self._items.items() if entry.expires_at <= now]:
            self._remove(key)

    def _trim(self) -> None:
        # 至少保留一条：单次切分本身就超过 max_chars 时，也要能入库。
        while len(self._items) > 1 and (
            len(self._items) > self.max_entries or self._characters > self.max_chars
        ):
            self._remove(next(iter(self._items)))
