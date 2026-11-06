"""Message helpers shared by the graph nodes and their callers.

`message_text` lives in its own module so both :mod:`ai_agent_py.graph.nodes`
and :mod:`ai_agent_py.graph.agent` can use it without a circular import.
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage

__all__ = ["message_text"]


def message_text(message: AnyMessage) -> str:
    """Flatten message content (string or content-block list) to plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
    return "".join(parts)
