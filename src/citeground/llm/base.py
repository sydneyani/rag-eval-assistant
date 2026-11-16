"""Provider protocols.

Chat and embedding providers are plain objects. The offline implementations
honor the same methods as the OpenAI client, so the ingestion pipeline, the
API, and the evaluation runner do not branch on provider names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


class Embedder(Protocol):
    provider_name: str
    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class ChatModel(Protocol):
    provider_name: str
    model_name: str

    def complete(
        self, messages: list[dict[str, str]], *, temperature: float = 0.0
    ) -> Completion: ...
