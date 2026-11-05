"""Shell tool: output capture, exit codes, timeouts and workspace confinement."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai_agent_py.config import Settings
from ai_agent_py.errors import ToolError
from ai_agent_py.tools.shell import build_shell_tool, run_shell

WINDOWS = os.name == "nt"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    return tmp_path


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def test_shell_prints_output(workspace: Path, settings: Settings) -> None:
    tool = build_shell_tool(workspace=workspace, settings=settings)
    command = "Write-Output 'hello 世界'" if WINDOWS else "echo 'hello 世界'"
    output = str(tool.invoke({"command": command}))
    assert "hello 世界" in output
    assert "[exit code: 0]" in output


def test_shell_reports_exit_code_and_stderr(workspace: Path) -> None:
    command = "[Console]::Error.WriteLine('boom'); exit 3" if WINDOWS else "echo boom >&2; exit 3"
    output = run_shell(workspace, command)
    assert "boom" in output
    assert "stderr:" in output
    assert "[exit code: 3]" in output


def test_shell_times_out(workspace: Path) -> None:
    command = "Start-Sleep -Seconds 5" if WINDOWS else "sleep 5"
    output = run_shell(workspace, command, timeout=0.5)
    assert "[超时]" in output


def test_shell_cwd_is_confined(workspace: Path) -> None:
    command = "(Get-Location).Path" if WINDOWS else "pwd"
    output = run_shell(workspace, command, cwd="src")
    assert "src" in output

    with pytest.raises(ToolError, match="超出工作目录"):
        run_shell(workspace, command, cwd="..")


def test_shell_rejects_empty_command(workspace: Path) -> None:
    with pytest.raises(ToolError, match="不能为空"):
        run_shell(workspace, "   ")
