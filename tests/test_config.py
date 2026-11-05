"""Settings behaviour: defaults, env overrides, path handling, masking."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_agent_py.config import PROJECT_ROOT, Settings, get_settings, reset_settings_cache


def test_defaults_are_sane() -> None:
    cfg = Settings(_env_file=None)
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
    cfg = Settings(_env_file=None, data_dir="build/raw")  # type: ignore[arg-type]
    assert cfg.data_dir.is_absolute()
    assert cfg.data_dir.parent == PROJECT_ROOT / "build"


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_get_settings_is_cached_and_resettable() -> None:
    reset_settings_cache()
    assert get_settings() is get_settings()
    reset_settings_cache()
    assert get_settings() is not None


def test_agent_tool_settings() -> None:
    cfg = Settings(_env_file=None)
    assert cfg.agent_workspace == PROJECT_ROOT
    assert cfg.agent_shell_timeout > 0
    assert cfg.agent_shell_max_output > 0
    assert cfg.web_search_model == "qwen-plus"
    assert cfg.resolved_web_search_api_key is None
    assert cfg.web_extractor_timeout > 0
    assert cfg.web_extractor_max_chars > 0

    with pytest.raises(ValidationError):
        Settings(_env_file=None, agent_shell_timeout=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, agent_shell_max_output=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, web_search_timeout=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, web_extractor_timeout=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, web_extractor_max_chars=0)


def test_web_search_key_prefers_dedicated_then_dashscope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope")
    assert Settings(_env_file=None).resolved_web_search_api_key == "sk-dashscope"

    cfg = Settings(_env_file=None, web_search_api_key="sk-dedicated")  # type: ignore[arg-type]
    assert cfg.resolved_web_search_api_key == "sk-dedicated"


def test_web_extractor_follows_llm_endpoint_by_default() -> None:
    cfg = Settings(
        _env_file=None,
        llm_base_url="https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        llm_model="qwen-plus",
        llm_api_key="sk-llm",  # type: ignore[arg-type]
    )
    assert cfg.resolved_web_extractor_base_url == cfg.llm_base_url
    assert cfg.resolved_web_extractor_model == "qwen-plus"
    assert cfg.resolved_web_extractor_api_key == "sk-llm"

    override = Settings(
        _env_file=None,
        web_extractor_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
        web_extractor_model="qwen3.5-plus",
        web_extractor_api_key="sk-extractor",  # type: ignore[arg-type]
    )
    assert override.resolved_web_extractor_base_url == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert override.resolved_web_extractor_model == "qwen3.5-plus"
    assert override.resolved_web_extractor_api_key == "sk-extractor"


def test_web_extractor_falls_back_to_dashscope_for_non_bailian_llm() -> None:
    cfg = Settings(
        _env_file=None,
        llm_base_url="https://api.deepseek.com/v1",
        llm_api_key="sk-deepseek",  # type: ignore[arg-type]
    )
    assert cfg.resolved_web_extractor_base_url == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    # 不能把 DeepSeek 的 Key 发到阿里云去；没配百炼 Key 就返回配置提示。
    assert cfg.resolved_web_extractor_api_key is None
