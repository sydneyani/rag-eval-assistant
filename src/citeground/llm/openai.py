"""OpenAI chat and embedding providers.

Implemented on httpx rather than the official SDK so the dependency surface
stays small and tests can inject a mock transport. The API key is sent as a
header and is never logged.
"""

from __future__ import annotations

import httpx
import numpy as np

from citeground.errors import ProviderError
from citeground.llm.base import Completion


class OpenAIEmbedder:
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dimensions: int | None = None,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is required for the OpenAI embedder")
        self.model_name = model
        self._dimensions = dimensions
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self.last_prompt_tokens = 0
        self._known_dim: int | None = dimensions

    @property
    def dim(self) -> int:
        if self._known_dim is None:
            raise ProviderError("OpenAI embedding dimension is not known until the first call")
        return self._known_dim

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            width = self._known_dim or self._dimensions or 0
            return np.zeros((0, width), dtype=np.float32)
        url = f"{self._base_url}/embeddings"
        payload: dict[str, object] = {"model": self.model_name, "input": texts}
        if self._dimensions:
            payload["dimensions"] = self._dimensions
        try:
            response = self._client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI embeddings request failed: {exc}") from exc
        body = response.json()
        rows = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
        if len(rows) != len(texts):
            raise ProviderError("OpenAI embeddings response did not contain one vector per input")
        matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        self._known_dim = int(matrix.shape[1])
        usage = body.get("usage") or {}
        self.last_prompt_tokens += int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # The embeddings endpoint accepts batches; keep them modest.
        batches: list[np.ndarray] = []
        step = 64
        for start in range(0, len(texts), step):
            batches.append(self._embed(texts[start : start + step]))
        if not batches:
            return self._embed([])
        return np.vstack(batches)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


class OpenAIChatModel:
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is required for the OpenAI chat provider")
        self.model_name = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> Completion:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": messages,
        }
        try:
            response = self._client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI chat request failed: {exc}") from exc
        body = response.json()
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI chat response did not include a message") from exc
        usage = body.get("usage") or {}
        return Completion(
            text=text.strip(),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            model=str(body.get("model") or self.model_name),
        )
