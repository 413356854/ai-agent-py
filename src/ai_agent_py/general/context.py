"""通用 Agent 的上下文预算：token 估算、历史选择与工具输出修剪。

对应 LangChain 的 ``trim_messages`` / ``SummarizationMiddleware`` 语义，但适配
「MySQL 外部会话记忆 + 手写 LangGraph」：每次请求前由 API 层组装提示词。

三层预算：

* **单条工具结果**（L1）：``clip_tool_result`` 保留头 + 标记 + 尾；
* **单轮工具累计**（L1）：``compress_tool_messages`` 把较早的工具输出替换成占位，
  但保留 ``ToolMessage`` 对象与 ``tool_call_id``（provider 要求 tool_call 配对）；
* **跨轮历史**（L2）：``select_history`` 从最新往前按 token 预算保留连续尾部，
  单条超长消息做中间截断。

token 估算用 ``langchain_core`` 的 ``count_tokens_approximately``；默认按中文
``chars_per_token=2.0``（官方默认 4.0 会低估中文约一倍）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool

from ai_agent_py.config import Settings
from ai_agent_py.db.agent_chat import ChatMessage

#: 单条工具结果被修剪时插入的标记（模型可见）。
TOOL_PRUNE_MARKER = "\n\n[... 工具结果中间已省略 {omitted} 字，完整结果见界面/数据库 ...]\n\n"
#: 单条历史消息被截断时插入的标记（模型可见）。
MESSAGE_PRUNE_MARKER = "\n\n[... 消息中间已省略 {omitted} 字 ...]\n\n"
#: 单轮累计工具输出超预算时，旧结果被替换成的占位文本。
TOOL_RESULT_PLACEHOLDER = "[较早的工具输出已省略：完整结果保留在右侧工具卡片与 MySQL 会话记录中]"

#: 历史预算的下限，避免窗口配置异常时把可用上下文算成 0。
MIN_HISTORY_BUDGET_TOKENS = 512


@dataclass(frozen=True, slots=True)
class HistorySelection:
    """按 token 预算选出的历史消息。"""

    messages: list[AnyMessage]
    estimated_tokens: int
    used: int
    dropped: int
    budget_tokens: int


@dataclass(frozen=True, slots=True)
class ToolCompression:
    """单轮工具输出压缩结果（只改模型可见的 prompt，不改数据库）。"""

    messages: list[AnyMessage]
    trimmed: int
    before_chars: int
    after_chars: int


def estimate_text_tokens(text: str, *, chars_per_token: float) -> int:
    """估算一段文本的 token 数（含一条消息的结构开销）。"""
    if not text:
        return 0
    return count_tokens_approximately(
        [HumanMessage(content=text)],
        chars_per_token=max(1.0, chars_per_token),
    )


def estimate_messages_tokens(
    messages: Sequence[AnyMessage],
    *,
    chars_per_token: float,
) -> int:
    """估算消息列表的 token 数（tool schema 不含在内）。"""
    if not messages:
        return 0
    return count_tokens_approximately(
        list(messages),
        chars_per_token=max(1.0, chars_per_token),
    )


def estimate_tools_tokens(
    tools: Sequence[BaseTool],
    *,
    chars_per_token: float,
) -> int:
    """估算工具 schema 随每次请求带来的固定开销。"""
    if not tools:
        return 0
    try:
        return count_tokens_approximately(
            [],
            tools=list(tools),
            chars_per_token=max(1.0, chars_per_token),
        )
    except Exception:  # pragma: no cover - 估算失败时退回 JSON 长度
        import json

        payload = json.dumps(
            [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": getattr(tool, "args", {}),
                }
                for tool in tools
            ],
            ensure_ascii=False,
            default=str,
        )
        return max(1, int(len(payload) / max(1.0, chars_per_token)) + 4 * len(tools))


def build_history_budget(
    settings: Settings,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str,
    summary: str = "",
) -> int:
    """计算可以分配给历史的 token 预算。

    ``AGENT_CONTEXT_MAX_TOKENS`` 为 0 时按模型窗口自动推导；否则与自动值取较小者。
    """
    system_tokens = estimate_text_tokens(system_prompt, chars_per_token=settings.agent_token_chars)
    summary_tokens = estimate_text_tokens(summary, chars_per_token=settings.agent_token_chars)
    tools_tokens = estimate_tools_tokens(tools, chars_per_token=settings.agent_token_chars)
    auto = (
        settings.agent_model_context_tokens
        - settings.agent_output_reserve_tokens
        - settings.agent_context_safety_tokens
        - system_tokens
        - summary_tokens
        - tools_tokens
    )
    budget = settings.agent_context_max_tokens or auto
    if settings.agent_context_max_tokens:
        budget = min(budget, settings.agent_context_max_tokens)
    return max(MIN_HISTORY_BUDGET_TOKENS, budget)


def clip_text(
    text: str,
    *,
    max_chars: int,
    head_chars: int,
    tail_chars: int,
    marker: str = MESSAGE_PRUNE_MARKER,
) -> str:
    """保留头 + 中间标记 + 尾；超长文本按字符数收紧。

    ``marker`` 里的 ``{omitted}`` 会被替换成被省略的字符数。不会切断 UTF-16
    surrogate pair（按 Python 字符切分）。
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    overhead = len(marker.format(omitted="0"))
    available = max(8, max_chars - overhead)
    head = min(max(1, head_chars), available)
    tail = min(max(0, tail_chars), max(0, available - head))
    omitted = max(0, len(text) - head - tail)
    result = text[:head] + marker.format(omitted=omitted) + (text[-tail:] if tail else "")
    # marker 实际长度随 omitted 变化，必要时继续收紧尾部/头部，保证不超过预算。
    while len(result) > max_chars and tail > 0:
        tail = max(0, tail - max(1, len(result) - max_chars))
        omitted = max(0, len(text) - head - tail)
        result = text[:head] + marker.format(omitted=omitted) + (text[-tail:] if tail else "")
    while len(result) > max_chars and head > 1:
        head = max(1, head - max(1, len(result) - max_chars))
        omitted = max(0, len(text) - head - tail)
        result = text[:head] + marker.format(omitted=omitted) + (text[-tail:] if tail else "")
    return result


def clip_tool_result(text: str, settings: Settings) -> str:
    """单次工具结果回给模型前的修剪（头 + 标记 + 尾）。"""
    return clip_text(
        text,
        max_chars=settings.agent_tool_result_max_chars,
        head_chars=settings.agent_tool_result_head_chars,
        tail_chars=settings.agent_tool_result_tail_chars,
        marker=TOOL_PRUNE_MARKER,
    )


def compress_tool_messages(
    messages: Sequence[AnyMessage],
    *,
    max_total_chars: int,
    keep_recent_results: int = 3,
    placeholder: str = TOOL_RESULT_PLACEHOLDER,
) -> ToolCompression:
    """把本轮较早的工具输出替换成占位，控制模型可见的工具输出总量。

    - 只改 ``ToolMessage.content``，保留对象、``tool_call_id`` 与 ``artifact``；
    - 最近 ``keep_recent_results`` 条工具结果保持完整；
    - 已落库/展示的完整结果在 ``artifact`` 里，不受影响。
    """
    result: list[AnyMessage] = list(messages)
    tool_indexes = [
        index for index, message in enumerate(result) if isinstance(message, ToolMessage)
    ]
    if not tool_indexes:
        return ToolCompression(messages=result, trimmed=0, before_chars=0, after_chars=0)

    before_chars = sum(len(_message_text(result[index])) for index in tool_indexes)
    keep = max(0, keep_recent_results)
    if keep >= len(tool_indexes):
        # 工具结果数量不多时，至少让最老的一条可以被压缩，否则累计预算永远不生效。
        keep = max(1, len(tool_indexes) - 1) if len(tool_indexes) > 1 else len(tool_indexes)
    recent = set(tool_indexes[-keep:]) if keep else set()
    total = sum(len(_message_text(result[index])) for index in tool_indexes if index in recent)
    trimmed = 0

    for index in reversed(tool_indexes):
        if index in recent:
            continue
        content = _message_text(result[index])
        if total + len(content) <= max_total_chars:
            total += len(content)
            continue
        result[index] = result[index].model_copy(update={"content": placeholder})
        total += len(placeholder)
        trimmed += 1

    after_chars = sum(len(_message_text(result[index])) for index in tool_indexes)
    return ToolCompression(
        messages=result,
        trimmed=trimmed,
        before_chars=before_chars,
        after_chars=after_chars,
    )


def _message_text(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


#: 历史用户消息里由 `_user_text` 追加的 Markdown 图片链接。
IMAGE_MARKDOWN = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")


def _user_content_blocks(
    content: str,
    *,
    max_chars: int,
    head_chars: int,
    tail_chars: int,
) -> str | list[Any]:
    """历史用户消息 → 文本 / LangChain 标准图片内容块。

    MySQL 里存的是「文字 + Markdown 图片链接」，回放时把链接还原成
    ``{"type": "image", "url": ...}``，这样视觉模型在后续轮次仍能看到图片；
    没有图片的旧消息保持纯字符串，行为不变。
    """
    urls = IMAGE_MARKDOWN.findall(content)
    if not urls:
        return clip_text(
            content,
            max_chars=max_chars,
            head_chars=head_chars,
            tail_chars=tail_chars,
        )
    text = IMAGE_MARKDOWN.sub("", content).strip() or "（用户上传了图片）"
    clipped = clip_text(
        text,
        max_chars=max_chars,
        head_chars=head_chars,
        tail_chars=tail_chars,
    )
    blocks: list[Any] = [{"type": "text", "text": clipped}]
    blocks.extend({"type": "image", "url": url} for url in urls)
    return blocks


def record_to_message(
    record: ChatMessage,
    *,
    max_chars: int,
    head_chars: int,
    tail_chars: int,
) -> AnyMessage | None:
    """MySQL 历史消息 → LangChain 消息；空内容与工具行不进上下文。"""
    if record.role not in {"user", "assistant"}:
        return None
    if not record.content.strip():
        return None
    if record.role == "user":
        return HumanMessage(
            content=_user_content_blocks(
                record.content,
                max_chars=max_chars,
                head_chars=head_chars,
                tail_chars=tail_chars,
            )
        )
    content = clip_text(
        record.content,
        max_chars=max_chars,
        head_chars=head_chars,
        tail_chars=tail_chars,
    )
    return AIMessage(content=content)


def estimate_records_tokens(
    records: Sequence[ChatMessage],
    *,
    settings: Settings,
) -> int:
    """估算一组历史记录的 token（单条超长消息先按同样的规则截断）。"""
    messages: list[AnyMessage] = []
    for record in records:
        message = record_to_message(
            record,
            max_chars=settings.agent_message_max_chars,
            head_chars=settings.agent_message_max_chars // 2,
            tail_chars=settings.agent_message_max_chars // 4,
        )
        if message is not None:
            messages.append(message)
    return estimate_messages_tokens(messages, chars_per_token=settings.agent_token_chars)


def select_history(
    records: Sequence[ChatMessage],
    *,
    settings: Settings,
    budget_tokens: int,
) -> HistorySelection:
    """从最新往前按 token 预算保留连续历史尾部。

    - 最近 ``AGENT_HISTORY_KEEP_MESSAGES`` 条优先保留；
    - 预算不够时丢掉最老的整条消息（不会把一条消息拆成两条）；
    - 单条消息超过 ``AGENT_MESSAGE_MAX_CHARS`` 时做中间截断；
    - 结果从一条 user 消息开始，保证对话序列合法。
    """
    converted: list[tuple[ChatMessage, AnyMessage]] = []
    for record in records:
        message = record_to_message(
            record,
            max_chars=settings.agent_message_max_chars,
            head_chars=settings.agent_message_max_chars // 2,
            tail_chars=settings.agent_message_max_chars // 4,
        )
        if message is not None:
            converted.append((record, message))

    kept: list[tuple[ChatMessage, AnyMessage, int]] = []
    used_tokens = 0
    keep_messages = max(1, settings.agent_history_keep_messages)
    for record, message in reversed(converted):
        tokens = estimate_messages_tokens([message], chars_per_token=settings.agent_token_chars)
        force = len(kept) < keep_messages
        if force or used_tokens + tokens <= budget_tokens:
            kept.append((record, message, tokens))
            used_tokens += tokens
            continue
        break

    # 强制保留的最近消息也可能超预算：从最老的开始丢，至少留下最新一条。
    while kept and used_tokens > budget_tokens and len(kept) > 1:
        _, _, tokens = kept.pop()
        used_tokens -= tokens
    # 预算非常紧时，至少保留最新的一条（优先最新 user）。
    if not kept and converted:
        record, message = converted[-1]
        kept.append(
            (
                record,
                message,
                estimate_messages_tokens([message], chars_per_token=settings.agent_token_chars),
            )
        )
        used_tokens = kept[-1][2]

    kept.reverse()

    # 历史必须以 user 消息开头（部分 provider 对 assistant 开头不友好）。
    while kept and isinstance(kept[0][1], AIMessage):
        _, _, tokens = kept.pop(0)
        used_tokens -= tokens

    used = len(kept)
    dropped = max(0, len(converted) - used)
    return HistorySelection(
        messages=[message for _, message, _ in kept],
        estimated_tokens=used_tokens,
        used=used,
        dropped=dropped,
        budget_tokens=budget_tokens,
    )


def build_summarization_transcript(
    records: Sequence[ChatMessage],
    *,
    max_chars: int,
    head_chars: int | None = None,
    tail_chars: int | None = None,
) -> str:
    """把待压缩历史渲染成「用户：… / 助手：…」的紧凑文本。"""
    head = max_chars // 2 if head_chars is None else head_chars
    tail = max_chars // 4 if tail_chars is None else tail_chars
    lines: list[str] = []
    for record in records:
        if record.role not in {"user", "assistant"}:
            continue
        content = " ".join(record.content.split())
        if not content:
            continue
        role = "用户" if record.role == "user" else "助手"
        lines.append(f"{role}：{content}")
    transcript = "\n\n".join(lines)
    if len(transcript) <= max_chars:
        return transcript
    return clip_text(
        transcript,
        max_chars=max_chars,
        head_chars=head,
        tail_chars=tail,
        marker="\n\n[... 较早对话中间已省略 {omitted} 字 ...]\n\n",
    )


__all__ = [
    "MESSAGE_PRUNE_MARKER",
    "MIN_HISTORY_BUDGET_TOKENS",
    "TOOL_PRUNE_MARKER",
    "TOOL_RESULT_PLACEHOLDER",
    "HistorySelection",
    "ToolCompression",
    "build_history_budget",
    "build_summarization_transcript",
    "clip_text",
    "clip_tool_result",
    "compress_tool_messages",
    "estimate_messages_tokens",
    "estimate_records_tokens",
    "estimate_text_tokens",
    "estimate_tools_tokens",
    "record_to_message",
    "select_history",
]
