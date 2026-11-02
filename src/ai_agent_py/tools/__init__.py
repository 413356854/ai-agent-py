"""Tools exposed to the LangGraph agent.

Add your own tools here (SQL query, HTTP call, calculator, ...) and pass them
to `build_agent_graph(tools=[...])` - the graph itself needs no changes.
"""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

from ai_agent_py.rag.chain import format_documents

RETRIEVER_TOOL_NAME = "search_knowledge_base"
RETRIEVER_TOOL_DESCRIPTION = (
    "在内部知识库中做向量检索，返回与查询最相关的文档片段（带 [编号] 和来源）。"
    "任何关于公司制度、产品资料、文档细节的事实性问题，都应先调用本工具。"
)


def build_retriever_tool(
    retriever: BaseRetriever,
    *,
    name: str = RETRIEVER_TOOL_NAME,
    description: str = RETRIEVER_TOOL_DESCRIPTION,
) -> BaseTool:
    """Wrap a retriever as a LangChain tool the LLM can call."""

    @tool(name, description=description)
    def search_knowledge_base(query: str) -> str:
        """在内部知识库中检索与 query 相关的文档片段。"""
        documents = list(retriever.invoke(query))
        if not documents:
            return "未检索到相关资料。"
        return format_documents(documents)

    return search_knowledge_base
