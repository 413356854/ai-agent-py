"""Typed application settings, loaded from `.env` / environment variables.

Every knob of the pipeline lives here so the rest of the code never reads
`os.environ` directly. Precedence: real env vars > `.env` file > defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from ai_agent_py.db.dsn import MySQLTarget

PROJECT_ROOT = Path(os.getenv("AI_AGENT_PY_ROOT", Path(__file__).resolve().parents[2])).resolve()

SearchType = Literal["similarity", "mmr", "similarity_score_threshold"]
VectorStoreKind = Literal["chroma", "memory"]
MineruModelVersion = Literal["pipeline", "vlm", "MinerU-HTML"]


def _mask(secret: str | None) -> str:
    if not secret:
        return "missing"
    return f"set (***{secret[-4:]})" if len(secret) > 4 else "set"


class Settings(BaseSettings):
    """Runtime configuration for the whole application."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- application ---------------------------------------------------
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # ---- chat model: any OpenAI compatible endpoint --------------------
    # OpenAI / DeepSeek / Qwen / Moonshot / Ollama / vLLM all work here.
    llm_model: str = "gpt-4o-mini"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int | None = None
    llm_timeout: float = 60.0
    agent_max_turns: int = 8

    # ---- embeddings ----------------------------------------------------
    # 阿里云百炼（DashScope）OpenAI 兼容模式：text-embedding-v4，默认 1024 维，
    # 可通过 EMBEDDING_DIMENSIONS 指定 1536/1024/768...，必须与 MILVUS_DIM 一致。
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: SecretStr | None = None
    dashscope_api_key: SecretStr | None = None
    embedding_base_url: str | None = None
    embedding_dimensions: int | None = None
    embedding_batch_size: int = 64
    # `None` -> auto: only OpenAI hosts get tiktoken based chunking.
    embedding_check_ctx_length: bool | None = None

    # ---- vector store --------------------------------------------------
    vector_store: VectorStoreKind = "chroma"
    collection_name: str = "rag_docs"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"

    # ---- Milvus（通用业务 RAG 表，建表脚本 docs/milvus/create_rag_chunk.py）----
    # 与上面的 `vector_store` 并行：chroma 走 LangChain VectorStore，Milvus 存检索块。
    # 这里放连接、维度与检索参数，写入/混合检索在应用侧按这张表实现。
    milvus_uri: str = "http://localhost:19530"
    milvus_token: SecretStr | None = None
    milvus_database: str = "db_tax_rag"
    milvus_collection: str = "rag_chunk"
    # 必须与 EMBEDDING_MODEL 的输出维度一致；换模型 = 建新集合 + 重建索引。
    milvus_dim: int = 1536
    milvus_metric: Literal["COSINE", "IP", "L2"] = "COSINE"
    # 单租户部署写 default 即可；多租户时 tenant 是分区键。
    milvus_tenant: str = "default"
    milvus_kb: str = "default"
    milvus_num_partitions: int = 16
    # 中文文档用 chinese（jieba 分词）；纯英文语料用 english；混合语料可试 standard。
    milvus_analyzer: Literal["chinese", "english", "standard"] = "chinese"
    milvus_batch_size: int = 500
    milvus_candidates: int = 30
    milvus_ef: int = 64
    milvus_rrf_k: int = 60

    # ---- rerank（RRF 融合之后的精排）------------------------------------
    # 阿里云百炼 qwen3.7-text-rerank：交叉编码器，直接给 (query, document) 打 0~1 分。
    # 链路：两路召回各 MILVUS_CANDIDATES 条 → RRF 融合取 RERANK_CANDIDATES 条
    #       → 精排取 RETRIEVER_K 条。没配 Key 时自动跳过（等价于只做 RRF）。
    rerank_enabled: bool = True
    rerank_model: str = "qwen3.7-text-rerank"
    rerank_api_key: SecretStr | None = None
    # 完整端点；留空按 EMBEDDING_BASE_URL 推导（百炼 MaaS 域名），否则用公共网关。
    rerank_endpoint: str | None = None
    # 进入精排的候选条数：太小精排没有腾挪空间，太大拖慢响应且更费 token。
    rerank_candidates: int = 30
    # 0~1；只在单次请求内可比（百炼文档明确说明），留空表示不按分数丢弃。
    rerank_min_score: float | None = None
    # 任务说明，仅 qwen3 系列 rerank 支持；留空用模型默认的问答检索任务。
    rerank_instruct: str | None = None
    rerank_timeout: float = 10.0

    # ---- ingestion -----------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ---- retrieval -----------------------------------------------------
    retriever_k: int = 4
    retriever_search_type: SearchType = "similarity"
    retriever_fetch_k: int = 20
    retriever_score_threshold: float = 0.2

    # ---- MinerU 精准解析 API -------------------------------------------
    # 文档地址：https://mineru.net/apiManage/docs
    mineru_api_key: SecretStr | None = None
    mineru_base_url: str = "https://mineru.net/api/v4"
    mineru_model_version: MineruModelVersion = "vlm"
    mineru_language: str = "ch"
    mineru_enable_formula: bool = True
    mineru_enable_table: bool = True
    mineru_is_ocr: bool = False
    # 单个任务最长等待时间（秒）与轮询间隔（秒）
    mineru_timeout: float = 600.0
    mineru_poll_interval: float = 3.0
    # 本地同时进行的解析任务上限
    mineru_max_concurrent: int = 2
    mineru_max_file_mb: int = 200
    mineru_output_dir: Path = PROJECT_ROOT / "data" / "parsed"

    # ---- MySQL（本地已有库；只做连接与健康检查）------------------------
    # 既接受 Java 项目的 JDBC 形式（jdbc:mysql://...），也接受 SQLAlchemy 的
    # mysql+pymysql://... 形式；连接参数由 `db.dsn` 统一翻译。
    mysql_url: str | None = None
    mysql_username: str | None = None
    mysql_password: SecretStr | None = None
    mysql_pool_size: int = 5
    mysql_max_overflow: int = 10
    mysql_pool_recycle: int = 1800
    mysql_connect_timeout: int = 5

    # ---- derived helpers -----------------------------------------------
    @property
    def mysql_target(self) -> MySQLTarget | None:
        """解析后的 MySQL 连接目标；未配置时返回 `None`。"""
        if not self.mysql_url:
            return None
        from ai_agent_py.db.dsn import parse_mysql_url

        return parse_mysql_url(
            self.mysql_url,
            username=self.mysql_username,
            password=self.mysql_password.get_secret_value() if self.mysql_password else None,
            connect_timeout=self.mysql_connect_timeout,
        )

    @property
    def mysql_dsn(self) -> str | None:
        """SQLAlchemy 连接串（含密码，只应交给引擎，不要写日志）。"""
        target = self.mysql_target
        return target.dsn if target else None

    @property
    def mysql_safe_dsn(self) -> str | None:
        """可展示的连接串（密码显示为 `***`）。"""
        target = self.mysql_target
        return target.safe_dsn if target else None

    @property
    def resolved_llm_api_key(self) -> str | None:
        """对话模型 Key：``LLM_API_KEY`` → ``DASHSCOPE_API_KEY`` → ``OPENAI_API_KEY``。

        阿里云百炼同时提供对话与向量接口，很多环境只导出了 ``DASHSCOPE_API_KEY``；
        把它作为 ``LLM_API_KEY`` 的第二个来源（环境变量或 ``.env`` 均可），
        ``serve`` / ``ai-agent-py chat`` 就不必重复导出同一个 Key。
        ``OPENAI_API_KEY`` 仍作为最后的兜底，保持原有行为。
        """
        if self.llm_api_key is not None:
            return self.llm_api_key.get_secret_value()
        if self.dashscope_api_key is not None:
            return self.dashscope_api_key.get_secret_value()
        return os.getenv("OPENAI_API_KEY") or None

    @property
    def resolved_embedding_api_key(self) -> str | None:
        """Embedding key: dedicated key, else DashScope, else OpenAI, else chat key."""
        if self.embedding_api_key is not None:
            return self.embedding_api_key.get_secret_value()
        if self.dashscope_api_key is not None:
            return self.dashscope_api_key.get_secret_value()
        return os.getenv("OPENAI_API_KEY") or self.resolved_llm_api_key

    @property
    def is_dashscope_embedding(self) -> bool:
        """``EMBEDDING_BASE_URL`` 是否指向百炼（含业务空间 MaaS 域名）。"""
        host = urlsplit(self.embedding_base_url or "").hostname or ""
        return host.endswith("aliyuncs.com") and ("dashscope" in host or ".maas." in host)

    @property
    def resolved_rerank_api_key(self) -> str | None:
        """Rerank key：专用 Key → 百炼 Key → 同业务空间的 embedding Key。

        精排走的是百炼 ``/api/v1/services/rerank/...``，所以：
        * 只有 Key 属于百炼时才能复用；``EMBEDDING_API_KEY`` 仅在
          ``EMBEDDING_BASE_URL`` 也指向百炼域名时才算数，避免把 OpenAI /
          硅基流动的 Key 发到阿里云去（那只会换来一个 401 和一个无用的告警）；
        * 永远不回退 ``LLM_API_KEY``：对话模型可能接的是 DeepSeek。
        """
        if self.rerank_api_key is not None:
            return self.rerank_api_key.get_secret_value()
        if self.dashscope_api_key is not None:
            return self.dashscope_api_key.get_secret_value()
        if self.embedding_api_key is not None and self.is_dashscope_embedding:
            return self.embedding_api_key.get_secret_value()
        return None

    @property
    def resolved_mineru_api_key(self) -> str | None:
        """MinerU token, falling back to the original `MINERU_TOKEN` name."""
        if self.mineru_api_key is not None:
            return self.mineru_api_key.get_secret_value()
        return os.getenv("MINERU_TOKEN") or None

    @property
    def resolved_milvus_token(self) -> str | None:
        """Milvus / Zilliz Cloud 的 API Key 或 `user:password` 形式 token。"""
        return self.milvus_token.get_secret_value() if self.milvus_token else None

    def describe(self) -> dict[str, Any]:
        """Human readable snapshot with secrets masked (used by `info`)."""
        target = self.mysql_target
        mysql = (
            f"{target.username}@{target.host}:{target.port}/{target.database}"
            if target
            else "未配置"
        )
        return {
            "env": self.app_env,
            "llm": f"{self.llm_model} @ {self.llm_base_url or 'api.openai.com'}",
            "llm_api_key": _mask(self.resolved_llm_api_key),
            "embeddings": f"{self.embedding_model} @ {self.embedding_base_url or 'api.openai.com'}",
            "embedding_api_key": _mask(self.resolved_embedding_api_key),
            "embedding_dimensions": self.embedding_dimensions or "auto",
            "vector_store": f"{self.vector_store} / {self.collection_name}",
            "chroma_dir": str(self.chroma_dir),
            "milvus": (
                f"{self.milvus_uri} / {self.milvus_database}.{self.milvus_collection} "
                f"(dim={self.milvus_dim}, {self.milvus_metric}, {self.milvus_analyzer})"
            ),
            "milvus_token": "set" if self.milvus_token else "missing",
            "data_dir": str(self.data_dir),
            "chunking": f"size={self.chunk_size}, overlap={self.chunk_overlap}",
            "retrieval": (
                f"{self.retriever_search_type}, k={self.retriever_k}, "
                f"score_threshold={self.retriever_score_threshold}"
            ),
            "rerank": (
                f"{self.rerank_model}, candidates={self.rerank_candidates}, "
                f"{'on' if self.rerank_enabled and self.resolved_rerank_api_key else 'off'}"
            ),
            "rerank_api_key": _mask(self.resolved_rerank_api_key),
            "mineru": f"{self.mineru_model_version} @ {self.mineru_base_url}",
            "mineru_api_key": _mask(self.resolved_mineru_api_key),
            "mineru_output_dir": str(self.mineru_output_dir),
            "mysql": mysql,
            # 数据库口令不做“后四位”提示：短口令会被直接猜出来。
            "mysql_password": "set" if self.mysql_password else "missing",
        }

    # ---- normalisation -------------------------------------------------
    @field_validator(
        "llm_base_url",
        "embedding_base_url",
        "llm_api_key",
        "embedding_api_key",
        "dashscope_api_key",
        "mineru_api_key",
        "mysql_url",
        "mysql_username",
        "mysql_password",
        "milvus_token",
        "rerank_api_key",
        "rerank_endpoint",
        "rerank_instruct",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("chroma_dir", "data_dir", "mineru_output_dir", mode="after")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value)

    @field_validator("mysql_pool_size", "mysql_connect_timeout", mode="after")
    @classmethod
    def _positive_mysql_number(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("MySQL 连接池大小与连接超时必须是正整数")
        return value

    @field_validator("mysql_max_overflow", "mysql_pool_recycle", mode="after")
    @classmethod
    def _non_negative_mysql_number(cls, value: int) -> int:
        if value < -1:
            raise ValueError("MySQL 溢出连接数与回收秒数不能小于 -1")
        return value

    @field_validator("mineru_base_url", mode="after")
    @classmethod
    def _strip_base_url(cls, value: str | None) -> str:
        return (value or "https://mineru.net/api/v4").rstrip("/")

    @field_validator("embedding_base_url", mode="after")
    @classmethod
    def _strip_embedding_base_url(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value else None

    @field_validator("embedding_dimensions", mode="after")
    @classmethod
    def _positive_dimensions(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS 必须是正整数")
        return value

    @field_validator(
        "milvus_dim",
        "milvus_num_partitions",
        "milvus_batch_size",
        "milvus_candidates",
        "milvus_ef",
        "milvus_rrf_k",
        "rerank_candidates",
        mode="after",
    )
    @classmethod
    def _positive_milvus_number(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Milvus 维度、分区数、批大小、召回数、ef 与精排候选数必须是正整数")
        return value

    @field_validator("rerank_timeout", mode="after")
    @classmethod
    def _positive_rerank_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("RERANK_TIMEOUT 必须是正数")
        return value

    @field_validator("rerank_min_score", mode="after")
    @classmethod
    def _rerank_score_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("RERANK_MIN_SCORE 必须在 0~1 之间（百炼精排分数是 0~1）")
        return value

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @model_validator(mode="after")
    def _validate_mysql(self) -> Settings:
        # 配置了就必须能解析：宁可启动时失败，也不要等健康检查才发现 URL 写错。
        if self.mysql_url:
            _ = self.mysql_target
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process wide (cached) settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cache, e.g. after changing env vars in tests."""
    get_settings.cache_clear()
