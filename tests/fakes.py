"""Test doubles: a scripted chat model that can pretend to call tools."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from ai_agent_py.mineru.models import ParseJob


class InMemoryJobStore:
    """`JobStore` 的内存实现：单元测试不连 MySQL 也能验证调度与续跑。

    读写都做深拷贝，模拟「进程重启后从库里读回一份独立快照」的语义。
    """

    def __init__(self) -> None:
        self._jobs: dict[str, ParseJob] = {}

    def load_all(self) -> list[ParseJob]:
        return [job.model_copy(deep=True) for job in self._jobs.values()]

    def get(self, job_id: str) -> ParseJob | None:
        job = self._jobs.get(job_id)
        return None if job is None else job.model_copy(deep=True)

    def save(self, job: ParseJob) -> None:
        self._jobs[job.id] = job.model_copy(deep=True)

    def delete(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def __len__(self) -> int:
        return len(self._jobs)


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


class StreamingScriptedChatModel(ScriptedChatModel):
    """`ScriptedChatModel` 的流式版本：把回复按 ``chunk_size`` 逐段吐出。

    用来验证「RAG 问答」的 SSE 逐 token 推送——真实模型（ChatOpenAI streaming=True）
    走的是同一条 ``_stream`` 路径。
    """

    chunk_size: int = 4

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        content = result.generations[0].message.content
        text = content if isinstance(content, str) else ""
        size = max(1, self.chunk_size)
        pieces = [text[start : start + size] for start in range(0, len(text), size)] or [""]
        for piece in pieces:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager is not None:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk
