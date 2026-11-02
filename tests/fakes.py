"""Test doubles: a scripted chat model that can pretend to call tools."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def tool_call(name: str, query: str, *, call_id: str = "call-1") -> AIMessage:
    """An AI message that requests a tool call (not a final answer)."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": query}, "id": call_id, "type": "tool_call"}],
    )


class ScriptedChatModel(BaseChatModel):
    """Returns `responses` in order, repeating the last one when exhausted."""

    responses: list[AIMessage]
    calls: int = 0
    received: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        self.received.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses[index])])
