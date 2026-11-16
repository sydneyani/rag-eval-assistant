"""Reciprocal rank fusion.

RRF combines ranked lists without requiring their scores to live on the same
scale. BM25 is unbounded and cosine is in [-1, 1], so a weighted sum needs a
normalization story that changes with the query. Rank fusion does not.

    score(d) = sum over lists of 1 / (k + rank(d))

`k = 60` is the value from Cormack, Clarke, and Buettcher (SIGIR 2009).
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = 60,
) -> list[tuple[int, float]]:
    if k < 0:
        raise ValueError("RRF k must be non-negative")
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
