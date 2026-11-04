"""切分结果缓存：读写、TTL、LRU、字符预算与按任务清理。"""

from __future__ import annotations

from langchain_core.documents import Document

from ai_agent_py.rag.chunk_cache import ChunkCache


def _chunks(*texts: str) -> list[Document]:
    return [
        Document(page_content=text, metadata={"chunk_index": index})
        for index, text in enumerate(texts)
    ]


def _put(cache: ChunkCache, job_id: str, *texts: str) -> str:
    entry = cache.put(
        job_id=job_id,
        chunks=_chunks(*texts),
        chunk_size=100,
        chunk_overlap=10,
        strategy="markdown",
    )
    return entry.split_id


def test_put_and_get_roundtrip() -> None:
    cache = ChunkCache()
    entry = cache.put(
        job_id="job-1",
        chunks=_chunks("甲", "乙乙"),
        chunk_size=100,
        chunk_overlap=10,
        strategy="markdown",
    )
    assert len(entry.split_id) == 32
    assert entry.total == 2
    assert entry.characters == 3
    assert entry.strategy == "markdown"
    assert cache.get(entry.split_id) is entry
    assert cache.get("missing") is None
    assert len(cache) == 1


def test_entry_expires_after_ttl() -> None:
    cache = ChunkCache(ttl_seconds=0)
    split_id = _put(cache, "job-1", "甲")
    assert cache.get(split_id) is None
    assert len(cache) == 0


def test_lru_evicts_least_recently_used_entry() -> None:
    cache = ChunkCache(max_entries=2)
    first = _put(cache, "job-1", "甲")
    second = _put(cache, "job-1", "乙")
    # 读一次 first，让它成为最近使用；second 变成最久未用
    assert cache.get(first) is not None
    third = _put(cache, "job-1", "丙")

    assert cache.get(second) is None
    assert cache.get(first) is not None
    assert cache.get(third) is not None


def test_character_budget_drops_old_entries() -> None:
    cache = ChunkCache(max_chars=10)
    first = _put(cache, "job-1", "x" * 8)
    second = _put(cache, "job-1", "y" * 8)
    assert cache.get(first) is None
    assert cache.get(second) is not None


def test_single_oversized_entry_is_kept() -> None:
    """单次切分本身超过字符预算时也保留，否则大文档无法入库。"""
    cache = ChunkCache(max_chars=5)
    split_id = _put(cache, "job-1", "x" * 20)
    assert cache.get(split_id) is not None


def test_drop_job_only_removes_that_job() -> None:
    cache = ChunkCache()
    first = _put(cache, "job-a", "a1")
    second = _put(cache, "job-a", "a2")
    other = _put(cache, "job-b", "b1")

    assert cache.drop_job("job-a") == 2
    assert cache.get(first) is None
    assert cache.get(second) is None
    assert cache.get(other) is not None
    assert cache.drop_job("job-a") == 0

    cache.clear()
    assert len(cache) == 0
