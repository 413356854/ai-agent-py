"""Offline building blocks: a stub chat model and a lexical embedding model.

They keep `ai-agent-py demo` and the test suite runnable without an API key or
network access, while exercising exactly the same code paths as the real models.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class StaticChatModel(BaseChatModel):
    """A chat model that always replies with the same text (offline demo)."""

    response: str = "(offline demo)"

    @property
    def _llm_type(self) -> str:
        return "static"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])


class LexicalHashEmbeddings(Embeddings):
    """Hashing bag-of-words embeddings: deterministic, offline, dependency free.

    Obviously not semantic, but queries that share surface forms (words or
    character bigrams) with a chunk do score higher, which is enough to
    smoke-test ingestion, retrieval and the whole chain without an API key.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _tokens(self, text: str) -> list[str]:
        lowered = text.lower()
        words = re.findall(r"[a-z0-9_]+", lowered)
        chars = [ch for ch in lowered if not ch.isspace()]
        bigrams = [first + second for first, second in itertools.pairwise(chars)]
        return words + bigrams

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
