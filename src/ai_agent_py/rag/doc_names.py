"""按 ``doc_id`` 回查 MySQL 文件名，把检索片段的来源补成可读文件名。

Milvus ``rag_chunk`` 只存 ``doc_id``：文件名、后缀、页数这些文档级属性留在 MySQL
``tb_document_parse_job``，这样改文件名不用回写 N 条切片。代价是刚检索回来的片段
只有 ``doc_id``，直接拼进提示词后模型看到的是 ``doc-1a2b…``，既不知道出处是哪个
文件，也没法在回答里正确指代它。

这个模块补的就是这一步——**检索之后、拼提示词之前**批量回查一次文件名：

    search() ──▶ apply_document_names() ──▶ format_documents() / 引用展示

* 只查本次召回的 ``doc_id``（``WHERE id IN (...)``），不扫全表；
* 结果在进程内缓存：``doc_id`` 由文件路径哈希得到，文件名不会变；
* MySQL 未配置 / 宕机 / 表不存在只记 warning，``source`` 退回 ``doc_id``，
  问答本身不受影响；失败后短时间内不再重试，避免每轮问答都撞一次连接超时。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Protocol

from langchain_core.documents import Document

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.db.document_jobs import MySQLJobStore
from ai_agent_py.errors import AiAgentError

logger = logging.getLogger(__name__)

#: ``Document.metadata`` 里可读文件名的键；``source`` 与 RAG 链的引用字段保持一致。
SOURCE_KEY = "source"
DOC_NAME_KEY = "doc_name"

#: 进程内最多缓存多少个 ``doc_id -> 文件名``（超了清空重查，内存有界）。
MAX_CACHE_ENTRIES = 4096

#: 回查失败后的冷却秒数：MySQL 临时不可用时不让每轮问答都等一次连接超时。
RETRY_SECONDS = 60.0


class DocumentNameStore(Protocol):
    """文件名读取接口：生产用 `MySQLJobStore`，单元测试注入内存实现。"""

    def load_file_names(self, doc_ids: Sequence[str]) -> Mapping[str, str]:
        """按 ``doc_id`` 批量返回文件名；查不到的 id 直接不出现在结果里。"""
        ...


class DocumentNameResolver(Protocol):
    """``doc_id`` → 文件名；查不到时可以不返回该 key。"""

    def resolve(self, doc_ids: Sequence[str]) -> Mapping[str, str]: ...


class MySQLDocumentNameResolver:
    """从 ``tb_document_parse_job`` 批量读文件名；失败降级为空映射。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: DocumentNameStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._store = store
        self._cache: dict[str, str] = {}
        self._retry_at = 0.0
        self._lock = threading.Lock()

    def resolve(self, doc_ids: Sequence[str]) -> dict[str, str]:
        """返回本次片段里能查到的文件名；重复 id 只查一次。"""
        wanted = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids if doc_id))
        if not wanted:
            return {}
        with self._lock:
            missing = [doc_id for doc_id in wanted if doc_id not in self._cache]
            if missing:
                self._load(missing)
            return {doc_id: self._cache[doc_id] for doc_id in wanted if doc_id in self._cache}

    def _load(self, doc_ids: Sequence[str]) -> None:
        """查库并写入缓存；调用方已经持有 ``self._lock``。"""
        now = time.monotonic()
        if now < self._retry_at:
            return
        store = self._store
        if store is None:
            # 没配 MySQL 时静默跳过：离线 demo / 单测不需要每次检索都报一次错。
            if not self.settings.mysql_url:
                return
            try:
                store = MySQLJobStore.from_settings(self.settings)
            except AiAgentError as exc:
                self._retry_at = now + RETRY_SECONDS
                logger.warning("连接 MySQL 失败，引用来源退回 doc_id：%s", exc)
                return
            self._store = store

        try:
            names = store.load_file_names(doc_ids)
        except AiAgentError as exc:
            self._retry_at = now + RETRY_SECONDS
            logger.warning("按 doc_id 回查文件名失败，引用来源退回 doc_id：%s", exc)
            return

        if len(self._cache) + len(names) > MAX_CACHE_ENTRIES:
            self._cache.clear()
        self._cache.update(names)


def apply_document_names(
    documents: Sequence[Document],
    resolver: DocumentNameResolver | None,
) -> list[Document]:
    """把文件名写回片段的 ``metadata["source"]`` / ``metadata["doc_name"]``。

    返回同一批片段（``Document.metadata`` 原地更新），方便节点直接放进状态。
    解析失败时保留调用方已有的 ``source``（通常是 ``doc_id``），不抛出异常。
    """
    result = list(documents)
    if not result or resolver is None:
        return result

    doc_ids = list(dict.fromkeys(str(document.metadata.get("doc_id") or "") for document in result))
    try:
        names = resolver.resolve(doc_ids)
    except Exception as exc:  # 注入的解析器不应把问答拖垮
        logger.warning("解析文档文件名失败，引用来源退回 doc_id：%s", exc)
        return result

    for document in result:
        doc_id = str(document.metadata.get("doc_id") or "")
        name = names.get(doc_id)
        if name:
            document.metadata[SOURCE_KEY] = name
            document.metadata[DOC_NAME_KEY] = name
    return result
