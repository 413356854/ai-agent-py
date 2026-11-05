"""Tools exposed to the LangGraph agent.

`build_knowledge_tool` wraps the Milvus knowledge base; `build_workspace_tools` adds
the generic file and shell tools (``read`` / ``write`` / ``edit`` / ``glob`` /
``grep`` / ``shell``); `build_web_search_tool` and `build_web_extractor_tool`
add Bailian web search and page extraction. Pass them to
``build_agent_graph(tools=[...])`` - the graph itself needs no changes.

File tools are pinned to one workspace root (``AGENT_WORKSPACE``, default the
project root): every path is resolved inside it, so the model cannot read or
write outside the project by accident.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

from ai_agent_py.config import Settings
from ai_agent_py.rag.chain import format_documents
from ai_agent_py.tools.filesystem import build_filesystem_tools
from ai_agent_py.tools.shell import build_shell_tool
from ai_agent_py.tools.web_extractor import build_web_extractor_tool
from ai_agent_py.tools.web_search import build_web_search_tool
from ai_agent_py.tools.workspace import resolve_workspace

if TYPE_CHECKING:
    from ai_agent_py.rag.milvus_search import DocumentSearcher

__all__ = [
    "KNOWLEDGE_TOOL_DESCRIPTION",
    "KNOWLEDGE_TOOL_NAME",
    "build_filesystem_tools",
    "build_knowledge_tool",
    "build_shell_tool",
    "build_web_extractor_tool",
    "build_web_search_tool",
    "build_workspace_tools",
]

KNOWLEDGE_TOOL_NAME = "search_knowledge_base"
KNOWLEDGE_TOOL_DESCRIPTION = (
    "在内部知识库中做向量检索，返回与查询最相关的文档片段（带 [编号] 和来源）。"
    "任何关于公司制度、产品资料、文档细节的事实性问题，都应先调用本工具。"
)


def build_knowledge_tool(
    searcher: DocumentSearcher,
    *,
    name: str = KNOWLEDGE_TOOL_NAME,
    description: str = KNOWLEDGE_TOOL_DESCRIPTION,
) -> BaseTool:
    """Wrap the Milvus knowledge base as a LangChain tool the LLM can call."""

    @tool(name, description=description)
    def search_knowledge_base(query: str) -> str:
        """在内部知识库中检索与 query 相关的文档片段。"""
        documents = list(searcher.search(query))
        if not documents:
            return "未检索到相关资料。"
        return format_documents(documents)

    return search_knowledge_base


def build_workspace_tools(
    *,
    workspace: str | Path | None = None,
    settings: Settings | None = None,
    include_shell: bool = True,
) -> list[BaseTool]:
    """Build the generic agent toolset, all confined to ``workspace``.

    ``include_shell=False`` returns the read/write-only set, for graphs that may
    inspect files but must not execute commands.
    """
    root = resolve_workspace(workspace, settings)
    tools = build_filesystem_tools(workspace=root, settings=settings)
    if include_shell:
        tools.append(build_shell_tool(workspace=root, settings=settings))
    return tools
