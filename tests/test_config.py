"""Settings behaviour: defaults, env overrides, path handling, masking."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_agent_py.config import PROJECT_ROOT, Settings, get_settings, reset_settings_cache


def test_defaults_are_sane() -> None:
    cfg = Settings(_env_file=None)
    assert cfg.vector_store == "chroma"
    assert cfg.chunk_overlap < cfg.chunk_size
    assert cfg.resolved_llm_api_key is None
    assert cfg.describe()["llm_api_key"] == "missing"


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("RETRIEVER_K", "7")
    monkeypatch.setenv("LLM_API_KEY", "sk-test-1234567890")
    cfg = Settings(_env_file=None)
    assert cfg.chunk_size == 256
    assert cfg.retriever_k == 7
    assert cfg.resolved_llm_api_key == "sk-test-1234567890"


def test_secret_is_masked_in_describe() -> None:
    cfg = Settings(_env_file=None, llm_api_key="sk-test-1234567890")  # type: ignore[arg-type]
    described = cfg.describe()["llm_api_key"]
    assert described.startswith("set")
    assert "1234567890" not in described
    assert described.endswith("7890)")


def test_blank_values_become_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "   ")
    monkeypatch.setenv("LLM_BASE_URL", "  ")
    cfg = Settings(_env_file=None)
    assert cfg.llm_api_key is None
    assert cfg.llm_base_url is None


def test_dashscope_key_is_used_for_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-test")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
    cfg = Settings(_env_file=None)

    assert cfg.resolved_embedding_api_key == "sk-dashscope-test"
    assert cfg.embedding_dimensions == 1536
    assert "1536" in str(cfg.describe()["embedding_dimensions"])


def test_embedding_key_takes_priority_over_dashscope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-test")
    ephemeral = Settings(_env_file=None, embedding_api_key="sk-explicit")
    assert ephemeral.resolved_embedding_api_key == "sk-explicit"


def test_dashscope_key_is_used_for_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-test")
    cfg = Settings(_env_file=None)

    assert cfg.resolved_llm_api_key == "sk-dashscope-test"
    assert cfg.describe()["llm_api_key"].startswith("set")


def test_llm_key_takes_priority_over_dashscope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-test")
    cfg = Settings(_env_file=None, llm_api_key="sk-explicit")
    assert cfg.resolved_llm_api_key == "sk-explicit"


def test_openai_key_is_last_resort_for_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    cfg = Settings(_env_file=None)
    assert cfg.resolved_llm_api_key == "sk-openai-test"


def test_relative_paths_resolve_against_project_root() -> None:
    cfg = Settings(_env_file=None, chroma_dir="build/chroma")  # type: ignore[arg-type]
    assert cfg.chroma_dir.is_absolute()
    assert cfg.chroma_dir.parent == PROJECT_ROOT / "build"


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_get_settings_is_cached_and_resettable() -> None:
    reset_settings_cache()
    assert get_settings() is get_settings()
    reset_settings_cache()
    assert get_settings() is not None
