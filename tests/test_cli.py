"""CLI smoke tests (typer runner, offline models)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import ai_agent_py.cli as cli

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert "ai-agent-py" in result.output


def test_demo_runs_the_offline_pipeline(monkeypatch, settings) -> None:
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sample.md").write_text(
        "# 说明\n\n本项目使用 uv 管理依赖，用 LangGraph 编排智能体。",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    result = runner.invoke(cli.app, ["demo", "--question", "用什么管理依赖？"])

    assert result.exit_code == 0, result.output
    assert "索引完成" in result.output
    assert "占位回答" in result.output
