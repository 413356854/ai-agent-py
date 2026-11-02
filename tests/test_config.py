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
