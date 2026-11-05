"""File tools: numbered reads, exact edits, glob pruning, content search."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ai_agent_py.config import Settings
from ai_agent_py.errors import ToolError
from ai_agent_py.graph.agent import build_agent_graph
from ai_agent_py.tools import build_workspace_tools
from ai_agent_py.tools.filesystem import (
    build_filesystem_tools,
    edit_file,
    glob_files,
    grep_files,
    read_file,
    write_file,
)
from fakes import ScriptedChatModel


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src" / "deep").mkdir(parents=True)
    (tmp_path / "src" / "app.py").write_text("import os\n\nprint('hello')\n", encoding="utf-8")
    (tmp_path / "src" / "deep" / "util.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo\nhello world\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("第一行\n第二行\n第三行\n", encoding="utf-8")
    return tmp_path


def test_read_returns_numbered_page(workspace: Path) -> None:
    output = read_file(workspace, "notes.txt", offset=2, limit=1)
    assert "共 3 行" in output
    assert "  2→第二行" in output
    assert "第一行" not in output
    assert "offset=3" in output


def test_read_handles_empty_and_gb18030(workspace: Path) -> None:
    (workspace / "empty.txt").write_text("", encoding="utf-8")
    assert "空文件" in read_file(workspace, "empty.txt")

    (workspace / "gbk.txt").write_bytes("报销流程\n".encode("gbk"))
    assert "报销流程" in read_file(workspace, "gbk.txt")


def test_read_truncates_overlong_lines(workspace: Path) -> None:
    (workspace / "long.txt").write_text("x" * 3000 + "\n", encoding="utf-8")
    output = read_file(workspace, "long.txt")
    assert "本行过长已截断" in output
    assert len(output) < 2500


def test_read_rejects_binary_missing_and_escapes(workspace: Path) -> None:
    (workspace / "blob.bin").write_bytes(b"abc\x00def")
    with pytest.raises(ToolError, match="二进制"):
        read_file(workspace, "blob.bin")

    with pytest.raises(ToolError, match="不存在"):
        read_file(workspace, "nope.txt")

    outside = workspace.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ToolError, match="超出工作目录"):
        read_file(workspace, str(outside))
    with pytest.raises(ToolError, match="超出工作目录"):
        read_file(workspace, "../outside.txt")


def test_write_creates_parent_dirs_and_keeps_lf(workspace: Path) -> None:
    message = write_file(workspace, "new/nested/file.txt", "a\nb\n")
    target = workspace / "new" / "nested" / "file.txt"
    assert target.read_bytes() == b"a\nb\n"
    assert "已创建" in message
    assert "2 行" in message


def test_write_warns_when_file_shrinks(workspace: Path) -> None:
    write_file(workspace, "big.txt", "x" * 1000)
    message = write_file(workspace, "big.txt", "y")
    assert "已覆盖" in message
    assert "一半以上" in message


def test_edit_replaces_and_reports_line(workspace: Path) -> None:
    write_file(workspace, "code.py", "def add(a, b):\n    return a + b\n")
    message = edit_file(workspace, "code.py", "return a + b", "return a - b")
    assert "替换 1 处" in message
    assert "第 2 行" in message
    assert (workspace / "code.py").read_text(encoding="utf-8") == (
        "def add(a, b):\n    return a - b\n"
    )


def test_edit_requires_unique_match_then_replace_all(workspace: Path) -> None:
    write_file(workspace, "dup.txt", "x = 1\nx = 1\n")
    with pytest.raises(ToolError, match="2 次"):
        edit_file(workspace, "dup.txt", "x = 1", "x = 2")

    message = edit_file(workspace, "dup.txt", "x = 1", "x = 2", replace_all=True)
    assert "替换 2 处" in message
    assert (workspace / "dup.txt").read_text(encoding="utf-8") == "x = 2\nx = 2\n"


def test_edit_missing_text_hints_read(workspace: Path) -> None:
    with pytest.raises(ToolError, match="未找到"):
        edit_file(workspace, "notes.txt", "不存在", "x")


def test_edit_preserves_crlf(workspace: Path) -> None:
    target = workspace / "crlf.txt"
    target.write_bytes(b"line1\r\nline2\r\n")
    edit_file(workspace, "crlf.txt", "line2", "line2 changed")
    assert target.read_bytes() == b"line1\r\nline2 changed\r\n"


def test_glob_matches_any_depth_and_prunes_noise(workspace: Path) -> None:
    for directory in (".venv/lib", "node_modules/pkg", "__pycache__"):
        (workspace / directory).mkdir(parents=True, exist_ok=True)
    (workspace / ".venv" / "lib" / "ghost.py").write_text("", encoding="utf-8")
    (workspace / "node_modules" / "pkg" / "ghost.py").write_text("", encoding="utf-8")

    output = glob_files(workspace, "*.py")
    assert "src/app.py" in output
    assert "src/deep/util.py" in output
    assert "ghost.py" not in output


def test_glob_respects_scope_and_limit(workspace: Path) -> None:
    output = glob_files(workspace, "*.py", "src")
    assert "src/app.py" in output
    assert "src/deep/util.py" in output
    assert "README.md" not in output

    assert "没有匹配" in glob_files(workspace, "*.rs")

    for index in range(4):
        (workspace / f"many{index}.txt").write_text("x", encoding="utf-8")
    limited = glob_files(workspace, "many*.txt", max_results=2)
    assert "仅显示前 2 个" in limited


def test_grep_groups_matches_by_file(workspace: Path) -> None:
    output = grep_files(workspace, "hello")
    assert "src/app.py" in output
    assert "README.md" in output
    assert "3: print('hello')" in output


def test_grep_include_and_ignore_case(workspace: Path) -> None:
    output = grep_files(workspace, "HELLO", include="*.md", ignore_case=True)
    assert "README.md" in output
    assert "src/app.py" not in output


def test_grep_rejects_invalid_regex(workspace: Path) -> None:
    with pytest.raises(ToolError, match="正则"):
        grep_files(workspace, "([")


def test_grep_respects_limit(workspace: Path) -> None:
    (workspace / "many.txt").write_text(
        "\n".join(f"hit {index}" for index in range(10)), encoding="utf-8"
    )
    output = grep_files(workspace, "hit", max_results=3)
    assert "3+ 行" in output
    assert "上限" in output


def test_filesystem_tools_are_langchain_tools(workspace: Path) -> None:
    tools = build_filesystem_tools(workspace=workspace)
    assert [item.name for item in tools] == ["read", "write", "edit", "glob", "grep"]

    read_tool = tools[0]
    assert "第一行" in str(read_tool.invoke({"path": "notes.txt"}))
    assert str(read_tool.invoke({"path": "missing.txt"})).startswith("错误：")


def test_workspace_toolset_can_skip_shell(workspace: Path) -> None:
    settings = Settings(_env_file=None)
    without_shell = build_workspace_tools(
        workspace=workspace, settings=settings, include_shell=False
    )
    assert [item.name for item in without_shell] == ["read", "write", "edit", "glob", "grep"]

    full = build_workspace_tools(workspace=workspace, settings=settings)
    assert [item.name for item in full][-1] == "shell"


def test_workspace_tools_run_inside_the_agent_graph(workspace: Path) -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read",
                        "args": {"path": "notes.txt"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="notes.txt 有三行。"),
        ]
    )
    graph = build_agent_graph(
        llm=model,
        tools=build_workspace_tools(workspace=workspace, include_shell=False),
        settings=Settings(_env_file=None),
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content="看看 notes.txt")]},
        {"recursion_limit": 6},
    )
    tool_messages = [message for message in result["messages"] if message.type == "tool"]
    assert tool_messages
    assert "第一行" in str(tool_messages[0].content)
    assert result["messages"][-1].content == "notes.txt 有三行。"
