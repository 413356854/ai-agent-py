"""Factories for the chat model and the embedding model.

Both are built through `langchain-openai` against an *OpenAI compatible* API,
which covers OpenAI, DeepSeek, Qwen/DashScope, Moonshot, SiliconFlow, Ollama,
vLLM and friends - only `*_BASE_URL` has to change.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import ConfigurationError


def get_chat_model(
    settings: Settings | None = None,
    *,
    streaming: bool = False,
    temperature: float | None = None,
) -> BaseChatModel:
    """Build the chat model used by the RAG chain and the agent."""
    cfg = settings or get_settings()
    api_key = cfg.resolved_llm_api_key
    if not api_key:
        raise ConfigurationError(
            "Missing chat model API key. Copy `.env.example` to `.env` and set LLM_API_KEY "
            "(or export DASHSCOPE_API_KEY / OPENAI_API_KEY). "
            "Run `ai-agent-py demo` for a key-less offline check."
        )

    kwargs: dict[str, Any] = {
        "model": cfg.llm_model,
        "api_key": api_key,
        "temperature": cfg.llm_temperature if temperature is None else temperature,
        "timeout": cfg.llm_timeout,
        "max_retries": 2,
    }
    if cfg.llm_base_url:
        kwargs["base_url"] = cfg.llm_base_url
    if cfg.llm_max_tokens:
        kwargs["max_tokens"] = cfg.llm_max_tokens
    if streaming:
        kwargs["streaming"] = True
    return ChatOpenAI(**kwargs)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Build the embedding model used for indexing and querying."""
    cfg = settings or get_settings()
    api_key = cfg.resolved_embedding_api_key
    if not api_key:
        raise ConfigurationError(
            "Missing embedding API key. Copy `.env.example` to `.env` and set "
            "EMBEDDING_API_KEY or DASHSCOPE_API_KEY (or OPENAI_API_KEY)."
        )

    kwargs: dict[str, Any] = {
        "model": cfg.embedding_model,
        "api_key": api_key,
        # DashScope 兼容接口单次最多接收 10 条文本；与入库批大小保持一致。
        "chunk_size": max(1, cfg.embedding_batch_size),
    }
    if cfg.embedding_base_url:
        kwargs["base_url"] = cfg.embedding_base_url
    if cfg.embedding_dimensions is not None:
        # text-embedding-v4 默认 1024 维；Milvus 集合按 1536 维建时必须显式指定。
        kwargs["dimensions"] = cfg.embedding_dimensions
    # Non-OpenAI endpoints usually reject tiktoken based input chunking.
    check_ctx = cfg.embedding_check_ctx_length
    if check_ctx is None:
        check_ctx = not (cfg.embedding_base_url and "openai.com" not in cfg.embedding_base_url)
    kwargs["check_embedding_ctx_length"] = check_ctx
    return OpenAIEmbeddings(**kwargs)
