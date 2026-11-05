"""通用 Agent 的历史摘要（L3）：把更早的 user/assistant 历史压缩成结构化检查点。

参考 deepseek-harness 的 8 节摘要格式，改为中文并适配我们的 MySQL 会话记忆：
原始消息不删除，摘要单独存 ``tb_agent_chat_summary``，只在组装提示词时替换旧历史。

摘要生成是一次独立的模型调用（不经过 LangGraph），因此不会把摘要 token 混进 SSE；
调用失败时由 API 降级为纯窗口截断，不影响本轮回答。
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ai_agent_py.config import Settings
from ai_agent_py.db.agent_chat import ChatMessage
from ai_agent_py.errors import AiAgentError
from ai_agent_py.general.context import build_summarization_transcript, estimate_text_tokens
from ai_agent_py.graph.agent import message_text


class SummaryError(AiAgentError):
    """摘要生成失败（模型报错、空输出等）；调用方应降级到窗口截断。"""


@dataclass(frozen=True, slots=True)
class SummaryOutcome:
    """一次摘要调用的结果。"""

    summary: str
    tokens: int
    model: str


SUMMARY_SYSTEM_PROMPT = """你是 AI 编码助手的「对话压缩引擎」。你的唯一任务是把给定的较早对话压缩成一份结构化检查点，让另一个模型在不丢失关键上下文的情况下继续工作。

要求：
1. 严格输出下面 8 个 Markdown 章节，顺序不变；每节用简洁的条目，不要写散文；没有内容写「（无）」。
2. 保留精确的文件路径、命令、错误字符串、标识符、数值、函数签名与关键代码片段。
3. 忠实记录用户的明确要求和纠正，尤其是「不要这样做」这类反馈。
4. 不要提及「压缩」「摘要」「检查点」这件事本身，也不要请求工具或输出任何额外解释。
5. 如果输入里已有旧摘要，它是更早的检查点：保留仍然成立的事实，丢弃过时信息，把新内容合并成一份摘要，不要原样照抄。
6. 使用简体中文，除非原文的路径、命令、错误信息本身是英文。

固定章节：
## 主要请求与意图
## 关键技术概念
## 文件与代码
## 错误与修复
## 待办任务
## 当前工作
## 下一步
## 关键上下文"""


def summarize_history(
    llm: BaseChatModel,
    records: list[ChatMessage],
    *,
    settings: Settings,
    previous_summary: str = "",
) -> SummaryOutcome:
    """把一批历史记录（按时间正序）压缩成一段结构化摘要。"""
    transcript = build_summarization_transcript(
        records,
        max_chars=settings.agent_summary_input_max_chars,
    )
    if not transcript.strip():
        raise SummaryError("没有可压缩的历史消息")

    sections: list[str] = []
    if previous_summary.strip():
        sections.append(f"【已有摘要（需要合并，不要照抄）】\n{previous_summary.strip()}")
    sections.append(f"【需要压缩的对话】\n{transcript}")
    sections.append(
        "请按系统要求输出合并后的检查点，只输出检查点正文；"
        f"总长度尽量控制在 {settings.agent_summary_max_chars} 字以内。"
    )
    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(sections)),
    ]

    try:
        response = llm.invoke(messages, max_tokens=settings.agent_summary_max_tokens)
    except Exception as exc:  # 上游模型错误
        raise SummaryError(f"摘要模型调用失败：{exc}") from exc

    summary = message_text(response).strip()
    if not summary:
        raise SummaryError("摘要模型返回了空内容")
    if len(summary) > settings.agent_summary_max_chars * 2:
        # 模型偶尔会超长；超得太多说明这次压缩不可靠，交给 API 降级。
        raise SummaryError(
            f"摘要过长（{len(summary)} 字，上限约 {settings.agent_summary_max_chars} 字）"
        )

    model_name = str(getattr(llm, "model_name", "") or getattr(llm, "model", "") or "chat-model")
    return SummaryOutcome(
        summary=summary,
        tokens=estimate_text_tokens(summary, chars_per_token=settings.agent_token_chars),
        model=model_name,
    )


__all__ = ["SummaryError", "SummaryOutcome", "summarize_history"]
