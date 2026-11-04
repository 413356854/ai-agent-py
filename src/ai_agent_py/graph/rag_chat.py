"""RAG 问答的 LangGraph 编排：condense → retrieve → answer。

与 `graph.agent` 里的 ReAct 智能体不同，这里是一条**固定的检索增强链**，
把「多轮改写、检索、生成」拆成三个可单独测试的节点：

    START ──▶ condense ──▶ retrieve ──▶ answer ──▶ END
              改写追问      混合检索      带引用作答

* ``condense``：只有存在历史对话时才调 LLM，把「那它的扣除标准呢？」改写成
  「业务招待费的扣除标准是什么？」这种可独立检索的问题；首轮直接使用原问题，
  不额外消耗一次模型调用。
* ``retrieve``：委托 `MilvusChunkSearcher` 在 ``rag_chunk`` 上做混合检索，
  融合结果再交给可选的精排模型（``qwen3.7-text-rerank``）重新打分。
* ``answer``：把召回片段按 ``[1] [2]`` 编号拼进提示词；没有召回时**不调用 LLM**，
  直接返回兜底文案（省钱、避免幻觉）。

多轮上下文由 LangGraph 的 checkpointer 保存，API 层为每个 ``thread_id`` 建一条会话。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ai_agent_py.config import Settings, get_settings
from ai_agent_py.graph.agent import message_text
from ai_agent_py.llm import get_chat_model
from ai_agent_py.rag.chain import format_documents, normalize_citations
from ai_agent_py.rag.milvus_search import DocumentSearcher, get_milvus_searcher

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

#: 一次问答最多跑到多少步（3 个节点 + 余量，防止图配置出错时无限循环）。
RAG_CHAT_RECURSION_LIMIT = 12

#: 带进提示词的历史消息条数（一问一答算两条）。
MAX_HISTORY_MESSAGES = 6

NO_CONTEXT_ANSWER = (
    "知识库中没有检索到与这个问题相关的内容。\n"
    "可以换一种说法再问，或先在「文档解析」里上传文档、切分后点「入库 RAG」。"
)

#: 只在多轮追问时使用：把指代消解成可独立检索的问题。
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是检索问题改写助手。结合对话历史，把用户的最新问题改写成一个独立、"
            "完整、适合知识库检索的问题。要求：\n"
            "1. 只输出改写后的问题本身，不要解释、不要回答问题、不要加序号；\n"
            "2. 补全“它 / 这个 / 上面说的”等指代，保留关键词与限定条件；法条序号"
            "（如「第四十三条」）、文号、主体名称、税率等原样保留；\n"
            "3. 最新问题本身已经完整时，原样输出；\n"
            "4. 使用简体中文，不超过 100 字。",
        ),
        ("human", "【对话历史】\n{history}\n\n【最新问题】\n{question}"),
    ]
)

SYSTEM_PROMPT = """你是「企业知识库问答助手」，只能依据【参考资料】回答用户问题。

【引用编号】
1. 引用编号只能取【参考资料】里每条开头的方括号序号；资料写明「共 N 条」时，合法编号只有 [1]~[N]。
2. 禁止把法条序号、文号、年份写进方括号：第四十三条要写成「第四十三条」或「实施条例第四十三条」，
   绝不能写成 [43]；财税〔2018〕XX 号、2024 年同理。
   资料正文里的「第X条」只是法条号，和被引用的资料是第几条无关——即使这句话讲的是第四十三条，
   也要标注它所在资料的编号（例如它是第 2 条资料，就写 [2]）。
3. 每句事实都要在句末标注来源，例如 [1]；同一事实来自多条资料时写成 [1][3]。
4. 输出前逐句自查：每个 [n] 都能在参考资料里找到第 n 条，且该条确实支持这句话；
   对不上就删掉这句话，或改写为「资料中未提及」。

【事实与数字】
5. 只使用参考资料中的信息作答，不要依赖先验知识，不要编造事实、文件名、文号、日期。
6. 资料不足以回答时，直接说明「资料中未找到相关内容」，并指出还缺少什么信息。
7. 涉及税率、金额、期限、比例等数字时原样引用，不要改写、换算或估算。

【表达】
8. 结合对话历史理解追问，但事实依据只能来自本次检索到的资料。
9. 使用简体中文：先给结论，再分点说明，必要时用列表；不要大段复制资料原文。
10. 不要在回答末尾另列「参考资料」清单，界面已经展示引用来源。

【示例】
参考资料（共 3 条，编号 1~3）：
[1] 来源：实施条例（第 6 页）
第四十二条 职工教育经费支出，不超过工资薪金总额2.5%的部分，准予扣除。
[2] 来源：实施条例（第 7 页）
第四十三条 企业发生的与生产经营活动有关的业务招待费支出，按照发生额的60%扣除，但最高不得超过当年销售（营业）收入的5‰。
[3] 来源：实施条例（第 8 页）
第四十四条 广告费和业务宣传费支出，不超过当年销售（营业）收入15%的部分，准予扣除。
问：业务招待费的扣除标准是什么？
答：业务招待费按发生额的 60% 扣除，但最高不得超过当年销售（营业）收入的 5‰[2]。
错误：……5‰[43]。
说明：这句话出自第 2 条资料，所以写 [2]；43 是法条序号，与资料编号无关，资料只有 3 条，不存在 [43]。"""


class RagChatState(TypedDict, total=False):
    """图状态：消息走 add_messages 追加，其余键每轮覆盖。"""

    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    documents: list[Document]
    answer: str


@dataclass(slots=True)
class RagChatTurn:
    """一轮问答的结果（同步接口用）。"""

    answer: str
    query: str = ""
    documents: list[Document] = field(default_factory=list)


# ---- 节点 -----------------------------------------------------------------
def build_condense_node(
    llm: BaseChatModel,
    *,
    prompt: ChatPromptTemplate = CONDENSE_PROMPT,
) -> Callable[[RagChatState], Any]:
    """把多轮追问改写成可独立检索的问题（首轮不调用模型）。"""

    def condense(state: RagChatState) -> dict[str, Any]:
        """节点：把多轮追问改写成可独立检索的问题（首轮直接用原问题，不调模型）。"""
        messages = list(state.get("messages") or [])
        if not messages:
            return {"query": ""}
        question = _one_line(message_text(messages[-1]), 2000)
        history = _history(messages[:-1])
        if not history or not question:
            return {"query": question}

        response = llm.invoke(
            prompt.format_messages(
                history=_render_history(history),
                question=question,
            )
        )
        rewritten = _one_line(message_text(response), 500)
        return {"query": rewritten or question}

    return condense


def build_retrieve_node(
    searcher: DocumentSearcher,
) -> Callable[[RagChatState], Any]:
    """在 ``rag_chunk`` 上检索与 ``query`` 相关的片段。"""

    def retrieve(state: RagChatState) -> dict[str, Any]:
        """节点：在 ``rag_chunk`` 上检索与 query 相关的片段。"""
        query = str(state.get("query") or "").strip()
        if not query:
            messages = list(state.get("messages") or [])
            query = message_text(messages[-1]).strip() if messages else ""
        documents = searcher.search(query) if query else []
        return {"query": query, "documents": list(documents)}

    return retrieve


def build_answer_node(
    llm: BaseChatModel,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    max_history: int = MAX_HISTORY_MESSAGES,
) -> Callable[[RagChatState], Any]:
    """把编号后的召回片段交给模型生成答案；没有召回时不调用模型。"""

    def answer(state: RagChatState) -> dict[str, Any]:
        """节点：把编号后的召回片段交给模型作答；没有召回时直接返回兜底文案。"""
        messages = list(state.get("messages") or [])
        question = message_text(messages[-1]).strip() if messages else ""
        documents = [
            document for document in (state.get("documents") or []) if document.page_content.strip()
        ]
        if not documents:
            return {
                "answer": NO_CONTEXT_ANSWER,
                "messages": [AIMessage(content=NO_CONTEXT_ANSWER)],
            }

        history = _history(messages[:-1])[-max_history:]
        prompt_messages: list[AnyMessage] = [SystemMessage(content=system_prompt)]
        prompt_messages.extend(history)
        total = len(documents)
        prompt_messages.append(
            HumanMessage(
                content=(
                    f"【参考资料】（共 {total} 条，编号 1~{total}）\n"
                    f"{format_documents(documents)}\n\n"
                    f"【问题】\n{question}"
                )
            )
        )
        response = llm.invoke(prompt_messages)
        text = message_text(response).strip() or NO_CONTEXT_ANSWER
        # 兜底：模型偶尔把法条序号当来源编号（[43]），用本轮片段反查真实来源。
        text, fixed = normalize_citations(text, documents)
        if fixed:
            logger.warning(
                "回答里出现越界引用 %s（本轮资料 %d 条），已按正文校正",
                fixed,
                len(documents),
            )
        return {"answer": text, "messages": [AIMessage(content=text)]}

    return answer


# ---- 组装 ----------------------------------------------------------------
def _add_node(builder: StateGraph, name: str, node: Callable[[RagChatState], Any]) -> None:
    """注册节点（mypy 匹配不上 LangGraph 的节点协议重载，统一在这里 ignore）。"""
    builder.add_node(name, node)  # type: ignore[call-overload]


def build_rag_chat_graph(
    *,
    llm: BaseChatModel | None = None,
    searcher: DocumentSearcher | None = None,
    checkpointer: Any | None = None,
    settings: Settings | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    condense_prompt: ChatPromptTemplate = CONDENSE_PROMPT,
) -> CompiledStateGraph:
    """编译 RAG 问答图；``llm`` / ``searcher`` 可注入，测试不需要真实模型或 Milvus。"""
    cfg = settings or get_settings()
    model = llm or get_chat_model(cfg, streaming=True)
    chunk_searcher = searcher or get_milvus_searcher(settings=cfg)

    builder = StateGraph(RagChatState)
    _add_node(builder, "condense", build_condense_node(model, prompt=condense_prompt))
    _add_node(builder, "retrieve", build_retrieve_node(chunk_searcher))
    _add_node(
        builder,
        "answer",
        build_answer_node(model, system_prompt=system_prompt),
    )
    builder.add_edge(START, "condense")
    builder.add_edge("condense", "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer)


def rag_chat_config(
    *,
    thread_id: str,
    recursion_limit: int = RAG_CHAT_RECURSION_LIMIT,
) -> RunnableConfig:
    """一次问答的 Runnable 配置：会话 id + 步数上限。"""
    return cast(
        "RunnableConfig",
        {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit,
        },
    )


def ask_rag_chat(
    graph: CompiledStateGraph,
    message: str,
    *,
    thread_id: str | None = None,
) -> RagChatTurn:
    """同步跑一轮问答；``thread_id`` 相同则复用 checkpointer 里的历史。"""
    config: RunnableConfig = (
        rag_chat_config(thread_id=thread_id)
        if thread_id
        else {"recursion_limit": RAG_CHAT_RECURSION_LIMIT}
    )
    result = graph.invoke({"messages": [HumanMessage(content=message)]}, config)
    return RagChatTurn(
        answer=str(result.get("answer") or ""),
        query=str(result.get("query") or ""),
        documents=list(result.get("documents") or []),
    )


# ---- 历史与文本处理 -------------------------------------------------------
def _history(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """只保留有正文的人机消息（工具消息、空消息不进提示词）。"""
    kept: list[AnyMessage] = []
    for message in messages:
        if message.type not in {"human", "ai"}:
            continue
        if message_text(message).strip():
            kept.append(message)
    return kept


def _render_history(messages: Sequence[AnyMessage]) -> str:
    """把历史消息渲染成「用户：…／助手：…」的紧凑文本，供改写提示词使用。"""
    lines: list[str] = []
    for message in messages[-MAX_HISTORY_MESSAGES:]:
        text = _one_line(message_text(message), 300)
        if not text:
            continue
        role = "用户" if message.type == "human" else "助手"
        lines.append(f"{role}：{text}")
    return "\n".join(lines) or "（无）"


def _one_line(text: str, limit: int = 500) -> str:
    """压成单行并截断：防止把整段历史 / 超长问题塞进改写提示词。"""
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else f"{flattened[:limit]}…"
