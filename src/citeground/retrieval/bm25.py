"""Okapi BM25 over in-memory tokenized chunks.

IDF uses the always-positive form `log(1 + (N - df + 0.5) / (df + 0.5))`
so a term that appears in more than half the chunks does not contribute a
negative score. Defaults are the usual `k1 = 1.5` and `b = 0.75`.
"""

from __future__ import annotations

import math
from collections import Counter


class BM25Index:
    def __init__(
        self,
        documents: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.count = len(documents)
        self.lengths = [len(document) for document in documents]
        self.average_length = (sum(self.lengths) / self.count) if self.count else 0.0
        self.term_freqs: list[Counter[str]] = []
        document_freq: Counter[str] = Counter()
        for document in documents:
            frequencies = Counter(document)
            self.term_freqs.append(frequencies)
            document_freq.update(frequencies.keys())
        self.idf = {
            term: math.log(1.0 + (self.count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    def score(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.count
        if self.count == 0 or self.average_length == 0:
            return scores
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self.term_freqs):
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                length_norm = 1.0 - self.b + self.b * (self.lengths[index] / self.average_length)
                denominator = frequency + self.k1 * length_norm
                scores[index] += idf * (frequency * (self.k1 + 1.0)) / denominator
        return scores

    def topk(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        scored = self.score(query_tokens)
        order = sorted(range(len(scored)), key=lambda index: scored[index], reverse=True)
        hits: list[tuple[int, float]] = []
        for index in order:
            if scored[index] <= 0:
                break
            hits.append((index, scored[index]))
            if len(hits) >= k:
                break
        return hits
