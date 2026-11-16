"""Signed feature-hashing embedder.

This is the offline dense encoder. Each token and bigram is hashed into a
fixed vector with a random sign (the hashing trick), weighted by log term
frequency, then L2-normalized. Cosine similarity is a dot product.

It does not understand synonyms. It exists so dense retrieval, hybrid fusion,
and the evaluation harness run in CI without an API key or a model download.
Swap it for the OpenAI embedder by setting a key or `CITEGROUND_EMBEDDING_PROVIDER`.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from citeground.text import tokenize


class HashingEmbedder:
    provider_name = "hashing"
    model_name = "signed-hash-tf"

    def __init__(self, dim: int = 256) -> None:
        if dim < 16:
            raise ValueError("hashing dimension must be at least 16")
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
        grams = tokens + bigrams
        counts = Counter(grams)
        vector = np.zeros(self.dim, dtype=np.float32)
        for token, frequency in counts.items():
            digest = hashlib.blake2s(token.encode("utf-8"), digest_size=8).digest()
            hashed = int.from_bytes(digest, "little")
            index = hashed % self.dim
            sign = 1.0 if (hashed >> 8) & 1 else -1.0
            vector[index] += np.float32(sign * np.log1p(frequency))
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= np.float32(norm)
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)
