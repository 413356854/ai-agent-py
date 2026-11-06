"""通用 Agent：把项目里的**全部工具**挂到同一个 LangChain / LangGraph 智能体上。

工具分组（都来自 :mod:`ai_agent_py.tools`）：

======== =====================================================
分组     工具
======== =====================================================
知识库   ``search_knowledge_base``
文件      ``read`` / ``write`` / ``edit`` / ``glob`` / ``grep``
命令      ``shell``
联网      ``web_search`` / ``web_extract``
======== =====================================================

图本身复用 :func:`ai_agent_py.graph.agent.build_agent_graph`（``agent ⇄ tools`` 的
ReAct 循环），这里只负责「工具清单 + 提示词」。

会话记忆不放进 checkpointer：每轮由 API 层从 MySQL（``db.agent_chat``）读出历史消息，
和本轮提问一起交给图。这样服务重启、换实例后会话都还在，也方便界面回放。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.general.context import clip_tool_result, compress_tool_messages
from ai_agent_py.graph.budget import LoopBudget
from ai_agent_py.graph.nodes import (
    build_finalize_node,
    build_route_after_agent,
    build_tool_node,
)
from ai_agent_py.graph.state import AgentState
from ai_agent_py.llm import get_chat_model
from ai_agent_py.rag.milvus_search import DocumentSearcher, get_milvus_searcher
from ai_agent_py.tools import (
    KNOWLEDGE_TOOL_NAME,
    build_knowledge_tool,
    build_web_extractor_tool,
    build_web_search_tool,
    build_workspace_tools,
)
from ai_agent_py.tools.filesystem import (
    EDIT_TOOL_NAME,
    GLOB_TOOL_NAME,
    GREP_TOOL_NAME,
    READ_TOOL_NAME,
    WRITE_TOOL_NAME,
)
from ai_agent_py.tools.shell import SHELL_TOOL_NAME
from ai_agent_py.tools.web_extractor import WEB_EXTRACT_TOOL_NAME
from ai_agent_py.tools.web_search import WEB_SEARCH_TOOL_NAME

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class GeneralAgentState(AgentState):
    """通用 Agent 的图状态：消息 + 已压缩历史的摘要。"""

    summary: NotRequired[str]


GENERAL_AGENT_SYSTEM_PROMPT = """你是「通用 Agent」——一个可以真正动手干活的智能体，运行在用户的 ai-agent-py 项目里。

【可用工具】
- `search_knowledge_base(query)`：在公司知识库（Milvus `rag_chunk`）里检索文档片段，
  返回带 [编号] 与来源的片段。公司制度、产品资料等事实性问题先用它。
- `read` / `write` / `edit` / `glob` / `grep`：读写工作目录内的文件。
  `read` 带行号分页读取；`glob` 按模式找文件；`grep` 用正则搜内容；
  `write` 整体覆盖；`edit` 精确替换（改局部内容优先用它）。
- `shell(command, timeout?, cwd?)`：在工作目录内执行命令（Windows 上是 PowerShell），
  返回 stdout / stderr 与退出码。
- `web_search(query)`：联网搜索实时公开信息，返回摘要与来源链接。
- `web_extract(url)`：抓取指定网页的正文。

【工作方式】
1. 先判断需要什么工具：查资料用 `search_knowledge_base` / `web_search`，看代码或文件用
   `read` / `glob` / `grep`，改文件用 `edit` / `write`，需要运行命令、跑测试用 `shell`。
2. 多步任务就分多轮调用工具：上一轮的结果决定下一步，不要一次把所有工具都调一遍，
   也不要为了「看起来用了工具」而重复调用同一个工具。
3. 路径一律用相对工作目录的路径。改文件前先 `read` 看清原文；局部修改用 `edit`，
   不要用 `write` 整文件覆盖，更不要用 shell 的 echo 重定向代替 `write`。
4. 工具返回「错误：...」时先读错误信息，修正参数或换一种做法；不要把错误原样丢给用户。
5. 引用知识库片段时标注 [1] [2]（编号只能取工具返回内容里的 [n]）；联网信息给出具体
   来源链接；不要编造工具没有返回的内容、文件或命令输出。
6. 做完再回答：先给结论 / 结果，再列关键步骤；涉及文件给出路径，涉及命令给出退出码。
7. 使用简体中文回答。会话历史保存在服务端 MySQL 中，可以结合上文继续用户的任务。
8. 用户可能上传图片。图片会以 LangChain 标准 `image` 内容块随用户消息发送，历史回放时
   也会还原；当前模型支持视觉时直接根据图片内容回答，看不清或模型不支持时不要凭空描述，
   可以把图片链接展示给用户。
"""

#: 工具名 → 分组（界面按组展示工具清单）。
TOOL_GROUPS: dict[str, str] = {
    KNOWLEDGE_TOOL_NAME: "knowledge",
    READ_TOOL_NAME: "filesystem",
    WRITE_TOOL_NAME: "filesystem",
    EDIT_TOOL_NAME: "filesystem",
    GLOB_TOOL_NAME: "filesystem",
    GREP_TOOL_NAME: "filesystem",
    SHELL_TOOL_NAME: "shell",
    WEB_SEARCH_TOOL_NAME: "web",
    WEB_EXTRACT_TOOL_NAME: "web",
}

GROUP_LABELS: dict[str, str] = {
    "knowledge": "知识库",
    "filesystem": "文件",
    "shell": "命令",
    "web": "联网",
}

#: 工具清单里每个工具的描述长度（完整的工具说明几十行，界面上只显示摘要）。
DESCRIPTION_CHARS = 160


@dataclass(frozen=True, slots=True)
class ToolMeta:
    """界面展示用的工具条目（分组 + 摘要）。"""

    name: str
    group: str
    group_label: str
    description: str


def describe_tools(tools: list[BaseTool]) -> list[ToolMeta]:
    """把 LangChain 工具转成「名称 / 分组 / 摘要」，顺序与传入顺序一致。"""
    metas: list[ToolMeta] = []
    for tool in tools:
        group = TOOL_GROUPS.get(tool.name, "other")
        metas.append(
            ToolMeta(
                name=tool.name,
                group=group,
                group_label=GROUP_LABELS.get(group, "其他"),
                description=_summarise(tool.description or ""),
            )
        )
    return metas


def build_general_tools(
    settings: Settings | None = None,
    *,
    searcher: DocumentSearcher | None = None,
    include_shell: bool = True,
) -> list[BaseTool]:
    """装配全部工具。

    ``searcher`` 可注入（测试用假检索器），默认用 Milvus 混合检索；
    联网搜索 / 网页抓取即使没配 Key 也会返回工具：调用时给出配置提示，
    图仍能正常构建与运行。
    """
    cfg = settings or get_settings()
    knowledge = build_knowledge_tool(searcher or get_milvus_searcher(settings=cfg))
    tools: list[BaseTool] = [knowledge]
    tools.extend(build_workspace_tools(settings=cfg, include_shell=include_shell))
    tools.append(build_web_search_tool(settings=cfg))
    tools.append(build_web_extractor_tool(settings=cfg))
    return tools


def build_general_agent_graph(
    *,
    llm: BaseChatModel | None = None,
    tools: list[BaseTool] | None = None,
    settings: Settings | None = None,
    system_prompt: str = GENERAL_AGENT_SYSTEM_PROMPT,
) -> CompiledStateGraph:
    """编译通用 Agent 图（``agent`` 节点 + ``tools`` 节点 + 条件边）。

    与通用 ReAct 图相比多了三处：

    * ``tools`` 节点的结果先做「头 + 标记 + 尾」修剪，完整结果放 ``artifact``；
    * ``agent`` 节点每次调模型前，把本轮较早的工具输出压缩成占位（保留工具配对），
      并把状态里的 ``summary``（更早历史的摘要）拼进系统提示词；
    * 循环预算用尽时走 ``finalize`` 节点（A-02）：基于已有资料作答并说明信息可能
      不完整，不会抛 ``GraphRecursionError`` 变成 502。
    """
    cfg = settings or get_settings()
    model = llm or get_chat_model(cfg, streaming=True)
    tool_list = list(tools) if tools is not None else build_general_tools(cfg)
    budget = LoopBudget.from_settings(cfg)

    agent_node = _build_general_agent_node(
        model,
        tool_list,
        system_prompt=system_prompt,
        compressor=lambda messages: _compress_prompt(cfg, messages),
    )
    tool_node = build_tool_node(
        tool_list,
        result_formatter=lambda text: clip_tool_result(text, cfg),
        settings=cfg,
        budget=budget,
    )
    finalize_node = build_finalize_node(
        model,
        system_prompt=system_prompt,
        budget=budget,
        compressor=lambda messages: _compress_prompt(cfg, messages),
        system_extra=lambda state: summary_block(str(state.get("summary") or "")),
    )

    builder = StateGraph(GeneralAgentState)
    builder.add_node("agent", agent_node)  # type: ignore[call-overload]
    builder.add_node("tools", tool_node)  # type: ignore[call-overload]
    builder.add_node("finalize", finalize_node)  # type: ignore[call-overload]
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        build_route_after_agent(budget),
        {"tools": "tools", "finalize": "finalize", "end": END},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)
    return builder.compile()


def _build_general_agent_node(
    llm: BaseChatModel,
    tools: list[BaseTool],
    *,
    system_prompt: str,
    compressor: Callable[[list[AnyMessage]], list[AnyMessage]] | None = None,
) -> Callable[[GeneralAgentState], Any]:
    """构造通用 Agent 的模型节点：系统提示 + 摘要 + 工具预算压缩。"""
    model = llm.bind_tools(tools) if tools else llm

    def agent_node(state: GeneralAgentState) -> dict[str, Any]:
        messages: list[AnyMessage] = list(state["messages"])
        update: dict[str, Any] = {}
        if messages and isinstance(messages[-1], HumanMessage):
            # 每轮第一次调用模型时记下开始时间，供时间预算使用（每轮覆盖）。
            update["started_at"] = time.monotonic()
        system = _system_with_summary(system_prompt, str(state.get("summary") or ""))
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system))
        elif system != system_prompt:
            messages[0] = SystemMessage(content=system)
        prompt = compressor(messages) if compressor is not None else messages
        response = model.invoke(prompt)
        update["messages"] = [response]
        return update

    return agent_node


def _system_with_summary(system_prompt: str, summary: str) -> str:
    """系统提示 + 更早历史的摘要（通用 Agent 与 finalize 节点共用）。"""
    block = summary_block(summary)
    return f"{system_prompt}\n\n{block}" if block else system_prompt


def summary_block(summary: str) -> str:
    """「更早对话摘要」段落；没有摘要时返回空串。"""
    text = summary.strip()
    if not text:
        return ""
    return (
        f"【更早对话摘要（已压缩）】\n{text}\n\n"
        "把这段摘要当作已确立的背景继续工作，不要重复确认或复述它。"
    )


def _compress_prompt(
    settings: Settings,
    messages: list[AnyMessage],
) -> list[AnyMessage]:
    """单轮工具输出累计预算：只改模型看到的 prompt，不动物化消息与数据库。"""
    result = compress_tool_messages(
        messages,
        max_total_chars=settings.agent_tool_turn_max_chars,
        keep_recent_results=3,
    )
    if result.trimmed:
        logger.info(
            "通用 Agent 单轮工具输出压缩：%d 条，%d → %d 字符",
            result.trimmed,
            result.before_chars,
            result.after_chars,
        )
    return result.messages


def _summarise(description: str, limit: int = DESCRIPTION_CHARS) -> str:
    flattened = " ".join(description.split())
    return flattened if len(flattened) <= limit else f"{flattened[:limit]}…"
