"""Typed application settings, loaded from `.env` / environment variables.

Every knob of the pipeline lives here so the rest of the code never reads
`os.environ` directly. Precedence: real env vars > `.env` file > defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(os.getenv("AI_AGENT_PY_ROOT", Path(__file__).resolve().parents[2])).resolve()

SearchType = Literal["similarity", "mmr", "similarity_score_threshold"]
VectorStoreKind = Literal["chroma", "memory"]


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
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None
    embedding_batch_size: int = 64
    # `None` -> auto: only OpenAI hosts get tiktoken based chunking.
    embedding_check_ctx_length: bool | None = None

    # ---- vector store --------------------------------------------------
    vector_store: VectorStoreKind = "chroma"
    collection_name: str = "rag_docs"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"

    # ---- ingestion -----------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ---- retrieval -----------------------------------------------------
    retriever_k: int = 4
    retriever_search_type: SearchType = "similarity"
    retriever_fetch_k: int = 20
    retriever_score_threshold: float = 0.2

    # ---- derived helpers -----------------------------------------------
    @property
    def resolved_llm_api_key(self) -> str | None:
        """`LLM_API_KEY`, falling back to the conventional `OPENAI_API_KEY`."""
        if self.llm_api_key is not None:
            return self.llm_api_key.get_secret_value()
        return os.getenv("OPENAI_API_KEY") or None

    @property
    def resolved_embedding_api_key(self) -> str | None:
        """Embedding key: dedicated key, else OpenAI's, else the chat key."""
        if self.embedding_api_key is not None:
            return self.embedding_api_key.get_secret_value()
        return os.getenv("OPENAI_API_KEY") or self.resolved_llm_api_key

    def describe(self) -> dict[str, Any]:
        """Human readable snapshot with secrets masked (used by `info`)."""
        return {
            "env": self.app_env,
            "llm": f"{self.llm_model} @ {self.llm_base_url or 'api.openai.com'}",
            "llm_api_key": _mask(self.resolved_llm_api_key),
            "embeddings": f"{self.embedding_model} @ {self.embedding_base_url or 'api.openai.com'}",
            "embedding_api_key": _mask(self.resolved_embedding_api_key),
            "vector_store": f"{self.vector_store} / {self.collection_name}",
            "chroma_dir": str(self.chroma_dir),
            "data_dir": str(self.data_dir),
            "chunking": f"size={self.chunk_size}, overlap={self.chunk_overlap}",
            "retrieval": (
                f"{self.retriever_search_type}, k={self.retriever_k}, "
                f"score_threshold={self.retriever_score_threshold}"
            ),
        }

    # ---- normalisation -------------------------------------------------
    @field_validator(
        "llm_base_url", "embedding_base_url", "llm_api_key", "embedding_api_key", mode="before"
    )
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("chroma_dir", "data_dir", mode="after")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value)

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process wide (cached) settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cache, e.g. after changing env vars in tests."""
    get_settings.cache_clear()
