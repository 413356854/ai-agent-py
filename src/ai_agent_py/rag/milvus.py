"""把切分后的文档块写入 Milvus ``rag_chunk``（稠密向量 + BM25 稀疏向量）。

集合结构由 ``docs/milvus/create_rag_chunk.py`` 创建，应用只负责写：

* 一行 = 一个检索块，字段 ``chunk_id`` / ``doc_id`` / ``chunk_index`` / ``page`` /
  ``content`` / ``embedding`` / ``sparse``；
* ``sparse`` 由 Milvus 的 BM25 内置函数根据 ``content`` 生成，**写入时不要赋值**；
* ``chunk_id = "chunk-" + sha1(doc_id|chunk_index|content)[:32]``，单条记录幂等；
* 重新入库时**先删除该 ``doc_id`` 的全部旧切片，再写入本次切片**：既支持用不同
  chunk_size 反复入库，也不会因为块序号 / 正文变化留下孤儿块或重复块。

写入使用 `pymilvus.MilvusClient`（而不是某个 LangChain VectorStore），因为
``rag_chunk`` 依赖 BM25 函数、nullable 字段等 LangChain 封装没有暴露的能力。
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import ConfigurationError, MilvusError, MilvusSchemaError
from ai_agent_py.llm import get_embeddings
from ai_agent_py.rag.filters import escape_literal

logger = logging.getLogger(__name__)

#: ``rag_chunk`` 里应用必须写入的字段（``sparse`` 由服务端生成）。
REQUIRED_FIELDS = ("chunk_id", "doc_id", "chunk_index", "page", "content", "embedding")

#: VARCHAR.max_length 的单位是 UTF-8 字节，建表脚本里 content 用满 65535。
MAX_CONTENT_BYTES = 65_535


@dataclass(slots=True)
class RagIngestReport:
    """一次 RAG 入库的结果摘要。"""

    chunks: int
    characters: int
    database: str
    collection: str
    store: str = "Milvus"
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "characters": self.characters,
            "database": self.database,
            "collection": self.collection,
            "store": self.store,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


def chunk_id(doc_id: str, index: int, content: str) -> str:
    """与建表脚本约定的稳定主键（重灌幂等）。"""
    digest = hashlib.sha1(f"{doc_id}|{index}|{content}".encode()).hexdigest()
    return f"chunk-{digest[:32]}"


def create_milvus_client(settings: Settings | None = None) -> Any:
    """按 `.env` 创建 Milvus 客户端并切到目标库。

    写入、记录查询、混合检索共用这一份连接逻辑，避免三处各写一遍。
    """
    cfg = settings or get_settings()
    try:
        import pymilvus
    except ImportError as exc:  # pragma: no cover - 依赖已声明
        raise ConfigurationError("缺少 pymilvus：请执行 `uv sync` 安装依赖后再使用 Milvus") from exc

    kwargs: dict[str, Any] = {"uri": cfg.milvus_uri}
    token = cfg.resolved_milvus_token
    if token:
        kwargs["token"] = token
    try:
        client = pymilvus.MilvusClient(**kwargs)
        if cfg.milvus_database and cfg.milvus_database != "default":
            client.use_database(cfg.milvus_database)
    except Exception as exc:
        raise MilvusError(
            f"连接 Milvus 失败（{cfg.milvus_uri}, database={cfg.milvus_database}）：{exc}"
        ) from exc
    return client


class MilvusRagWriter:
    """``rag_chunk`` 的写入器。

    `client` / `embeddings` 可注入，单元测试不需要真实 Milvus 或 API Key。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._embeddings = embeddings

    # ---- 连接与校验 -----------------------------------------------------
    @property
    def database(self) -> str:
        return self.settings.milvus_database

    @property
    def collection(self) -> str:
        return self.settings.milvus_collection

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = get_embeddings(self.settings)
        return self._embeddings

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = create_milvus_client(self.settings)
        return self._client

    def _check_collection(self) -> None:
        """集合不存在或维度不符时，给出可以直接照做的中文提示。"""
        try:
            exists = self.client.has_collection(self.collection)
        except Exception as exc:
            raise MilvusError(f"检查 Milvus 集合失败：{exc}") from exc
        if not exists:
            raise MilvusSchemaError(
                f"Milvus 集合 {self.database}.{self.collection} 不存在。"
                "请先执行 `uv run python docs/milvus/create_rag_chunk.py --dim "
                f"{self.settings.milvus_dim}`"
            )

        try:
            description = self.client.describe_collection(self.collection)
        except Exception as exc:
            raise MilvusError(f"读取 Milvus 集合结构失败：{exc}") from exc

        fields = {field["name"]: field for field in description.get("fields", [])}
        missing = [name for name in REQUIRED_FIELDS if name not in fields]
        if missing:
            raise MilvusSchemaError(
                f"Milvus 集合 {self.collection} 缺少字段 {', '.join(missing)}；"
                "请按 docs/milvus/create_rag_chunk.py 重建集合"
            )

        params = fields["embedding"].get("params") or {}
        dim = int(params.get("dim") or 0)
        if dim != self.settings.milvus_dim:
            raise MilvusSchemaError(
                f"Milvus 集合 {self.collection} 的向量维度是 {dim}，"
                f"配置的 MILVUS_DIM 是 {self.settings.milvus_dim}。"
                "换 embedding 模型必须新建集合并回填（不能原地改维度）。"
            )

    # ---- 写入 -----------------------------------------------------------
    def index_document(self, doc_id: str, chunks: Sequence[Document]) -> RagIngestReport:
        """覆盖写入一篇文档的所有块。

        重复入库（换切分参数 / 正文有改动）时按 ``doc_id`` 删除全部旧切片，
        再 ``upsert`` 本次的新切片：
        * 删除发生在 embedding 计算成功之后，向量模型失败时不会先丢掉旧数据；
        * 旧块不会残留，也不会产生重复块；
        * 最后只 ``flush`` 一次，避免两次同步 flush 把请求拖到 20s 以上。
        """
        started = time.perf_counter()
        self._check_collection()
        try:
            self.client.load_collection(self.collection)
        except Exception as exc:
            raise MilvusError(f"加载 Milvus 集合 {self.collection} 失败：{exc}") from exc

        if not chunks:
            self._delete_document(doc_id)
            self._flush()
            return RagIngestReport(
                chunks=0,
                characters=0,
                database=self.database,
                collection=self.collection,
                elapsed_seconds=time.perf_counter() - started,
            )

        contents = [chunk.page_content for chunk in chunks]
        vectors = self._embed(contents)
        if len(vectors) != len(chunks):
            raise MilvusError(f"embedding 数量 ({len(vectors)}) 与块数 ({len(chunks)}) 不一致")

        rows = [
            self._row(doc_id, index, chunk, vector)
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]

        # 先删除旧切片，再写新切片：重复入库不会留下上一版的块。
        self._delete_document(doc_id)

        batch_size = max(1, self.settings.milvus_batch_size)
        try:
            for start in range(0, len(rows), batch_size):
                self.client.upsert(self.collection, rows[start : start + batch_size])
            self._flush()
        except Exception as exc:
            raise MilvusError(
                f"写入 Milvus 集合 {self.database}.{self.collection} 失败：{exc}"
            ) from exc

        characters = sum(len(content) for content in contents)
        logger.info(
            "写入 Milvus %s.%s：doc_id=%s, %d chunks",
            self.database,
            self.collection,
            doc_id,
            len(rows),
        )
        return RagIngestReport(
            chunks=len(rows),
            characters=characters,
            database=self.database,
            collection=self.collection,
            elapsed_seconds=time.perf_counter() - started,
        )

    def _delete_document(self, doc_id: str) -> None:
        """删除该文档在集合里的全部切片（重复入库前先清空旧版本）。"""
        try:
            self.client.delete(self.collection, filter=f'doc_id == "{escape_literal(doc_id)}"')
        except Exception as exc:
            raise MilvusError(f"删除旧切片失败（doc_id={doc_id}）：{exc}") from exc

    def _flush(self) -> None:
        try:
            self.client.flush(self.collection)
        except Exception as exc:
            raise MilvusError(f"刷新 Milvus 集合失败：{exc}") from exc

    def _embed(self, contents: Sequence[str]) -> list[list[float]]:
        batch_size = max(1, self.settings.embedding_batch_size)
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(contents), batch_size):
                window = list(contents[start : start + batch_size])
                vectors.extend(self.embeddings.embed_documents(window))
        except ConfigurationError:
            raise
        except Exception as exc:
            raise MilvusError(f"生成 embedding 失败：{exc}") from exc
        return vectors

    def _row(
        self,
        doc_id: str,
        index: int,
        chunk: Document,
        vector: Sequence[float],
    ) -> dict[str, Any]:
        content = chunk.page_content
        if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
            raise MilvusError(
                f"第 {index} 块超过 Milvus content 字段上限（65535 字节），"
                "请调小 chunk_size 后重新切分"
            )
        return {
            "chunk_id": chunk_id(doc_id, index, content),
            "doc_id": doc_id,
            "chunk_index": index,
            "page": _page_of(chunk),
            "content": content,
            "embedding": list(vector),
        }


def _page_of(chunk: Document) -> int | None:
    page = chunk.metadata.get("page")
    if isinstance(page, bool):
        return None
    if isinstance(page, int):
        return page if page > 0 else None
    if isinstance(page, str) and page.isdigit():
        return int(page) or None
    return None
