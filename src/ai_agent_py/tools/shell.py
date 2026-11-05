"""Shell tool: run commands inside the workspace and return a compact transcript.

The tool is the escape hatch for builds, tests and git; it is deliberately
not the way to read or edit files (``read`` / ``edit`` are safer and cheaper).
Commands inherit the parent environment, run with a timeout, and their stdout
/ stderr are truncated head-and-tail so a huge log cannot flood the context.
"""

from __future__ import annotations

import locale
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.errors import ToolError
from ai_agent_py.tools.workspace import resolve_path, resolve_workspace

SHELL_TOOL_NAME = "shell"
DEFAULT_TIMEOUT = 120.0
MAX_TIMEOUT = 3600.0
DEFAULT_MAX_OUTPUT = 20_000


class ShellArgs(BaseModel):
    """``shell`` 的入参；``timeout`` 上限与工具内部一致（``MAX_TIMEOUT``）。"""

    command: str = Field(min_length=1, description="要执行的 shell 命令")
    timeout: float | None = Field(
        default=None,
        gt=0,
        le=MAX_TIMEOUT,
        description=f"超时秒数（默认 AGENT_SHELL_TIMEOUT，上限 {MAX_TIMEOUT:g}）",
    )
    cwd: str = Field(default=".", description="工作目录内的子目录")


SHELL_DESCRIPTION = (
    "在工作目录内执行 shell 命令（Windows 优先 pwsh，其次 powershell / cmd；"
    "macOS/Linux 用 bash），返回 stdout、stderr 与退出码。适合构建、测试、git 等命令；"
    "读写文件请优先用 read / write / edit / glob / grep。"
    "timeout 为超时秒数（默认 120，超时会终止并返回已有输出），cwd 为工作目录内的子目录。"
)


def run_shell(
    root: Path,
    command: str,
    *,
    timeout: float | None = None,
    cwd: str = ".",
    default_timeout: float = DEFAULT_TIMEOUT,
    max_output: int = DEFAULT_MAX_OUTPUT,
) -> str:
    """Run ``command`` with ``cwd`` inside ``root`` and render the result."""
    text = command.strip()
    if not text:
        raise ToolError("命令不能为空")
    seconds = _timeout_value(timeout, default_timeout)
    workdir = resolve_path(root, cwd or ".", must_exist=True)
    if not workdir.is_dir():
        raise ToolError(f"cwd 不是目录：{cwd}")

    argv = _shell_argv(text)
    env = os.environ.copy()
    for key, value in {
        "NO_COLOR": "1",
        "TERM": "dumb",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
    }.items():
        env.setdefault(key, value)

    options: dict[str, Any] = {}
    if os.name == "nt":
        # Hide the console window when the agent runs inside the web server.
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            argv,
            cwd=str(workdir),
            capture_output=True,
            timeout=seconds,
            env=env,
            check=False,
            **options,
        )
    except subprocess.TimeoutExpired as exc:
        return _render(
            text,
            _decode_output(exc.stdout),
            _decode_output(exc.stderr),
            None,
            seconds,
            max_output,
        )
    except OSError as exc:
        raise ToolError(f"无法执行命令：{exc}") from exc

    return _render(
        text,
        _decode_output(completed.stdout),
        _decode_output(completed.stderr),
        completed.returncode,
        seconds,
        max_output,
    )


def _timeout_value(value: float | None, default: float) -> float:
    seconds = default if value is None else value
    try:
        seconds = float(seconds)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"timeout 必须是数字：{value!r}") from exc
    if seconds <= 0:
        raise ToolError("timeout 必须大于 0")
    return min(seconds, MAX_TIMEOUT)


def _shell_argv(command: str) -> list[str]:
    if os.name == "nt":
        for executable in ("pwsh", "powershell"):
            found = shutil.which(executable)
            if found:
                return [
                    found,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    command,
                ]
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", command]
    return [shutil.which("bash") or "/bin/sh", "-c", command]


def _decode_output(data: bytes | str | None) -> str:
    """Decode captured output: UTF-8 first, then the console/ANSI code page."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data.replace("\r\n", "\n")
    if not data:
        return ""
    encodings = ["utf-8", locale.getpreferredencoding(False), "gb18030"]
    for encoding in dict.fromkeys(encodings):
        try:
            return data.decode(encoding).replace("\r\n", "\n")
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n…（省略 {omitted} 个字符）…\n{text[-tail:]}"


def _render(
    command: str,
    stdout: str,
    stderr: str,
    code: int | None,
    timeout: float,
    max_output: int,
) -> str:
    stdout = _truncate(stdout, max_output)
    stderr = _truncate(stderr, max_output)
    parts = [f"$ {command}"]
    if stdout.strip():
        parts.extend(["", stdout.rstrip()])
    if stderr.strip():
        parts.extend(["", "stderr:", stderr.rstrip()])
    if not stdout.strip() and not stderr.strip():
        parts.append("(无输出)")
    if code is None:
        parts.append(f"[超时] 超过 {timeout:g} 秒，进程已终止")
    else:
        parts.append(f"[exit code: {code}]")
    return "\n".join(parts)


def build_shell_tool(
    *,
    workspace: str | Path | None = None,
    settings: Settings | None = None,
) -> BaseTool:
    """Build the ``shell`` tool pinned to the workspace root."""
    cfg = settings or get_settings()
    root = resolve_workspace(workspace, cfg)
    default_timeout = cfg.agent_shell_timeout
    max_output = cfg.agent_shell_max_output

    @tool(SHELL_TOOL_NAME, description=SHELL_DESCRIPTION, args_schema=ShellArgs)
    def shell(command: str, timeout: float | None = None, cwd: str = ".") -> str:
        """在工作目录内执行 shell 命令，返回输出与退出码。"""
        try:
            return run_shell(
                root,
                command,
                timeout=timeout,
                cwd=cwd,
                default_timeout=default_timeout,
                max_output=max_output,
            )
        except ToolError as exc:
            return f"错误：{exc}"

    return shell
