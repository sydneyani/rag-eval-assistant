"""Feature reranker that runs with no extra model.

A hosted cross-encoder is the right production reranker once you are paying
for a second model call. This one is a transparent linear combination of
features already in hand:

* query-term coverage of the chunk
* BM25 score normalized by the max score in the candidate pool
* dense cosine mapped from [-1, 1] into [0, 1]

Weights favor coverage, then BM25, then the hashing embedder. The hashing
embedder is lexical, so giving it equal weight would double-count term
overlap without adding a semantic signal. When the OpenAI embedder is on,
these weights are the first thing to retune against the evaluation set.
"""

from __future__ import annotations

from citeground.schemas import ScoredChunk
from citeground.text import tokenize


def rerank(query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
    if not hits:
        return []
    query_terms = set(tokenize(query))
    max_bm25 = max((hit.bm25_score or 0.0) for hit in hits) or 1.0
    reranked: list[ScoredChunk] = []
    for hit in hits:
        chunk_terms = set(tokenize(hit.chunk.text))
        coverage = (len(query_terms & chunk_terms) / len(query_terms)) if query_terms else 0.0
        bm25_norm = (hit.bm25_score or 0.0) / max_bm25 if hit.bm25_score is not None else None
        dense_norm = None
        if hit.dense_score is not None:
            dense_norm = max(0.0, min(1.0, (hit.dense_score + 1.0) / 2.0))
        score, weights = _combine(coverage, bm25_norm, dense_norm)
        updated = hit.model_copy(
            update={
                "score": score,
                "rerank_score": score,
                "features": {
                    "coverage": coverage,
                    "bm25_norm": 0.0 if bm25_norm is None else bm25_norm,
                    "dense_norm": 0.0 if dense_norm is None else dense_norm,
                    **weights,
                },
            }
        )
        reranked.append(updated)
    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked


def _combine(
    coverage: float,
    bm25_norm: float | None,
    dense_norm: float | None,
) -> tuple[float, dict[str, float]]:
    if bm25_norm is None and dense_norm is None:
        return coverage, {"w_coverage": 1.0, "w_bm25": 0.0, "w_dense": 0.0}
    if dense_norm is None:
        return (
            0.55 * coverage + 0.45 * (bm25_norm or 0.0),
            {"w_coverage": 0.55, "w_bm25": 0.45, "w_dense": 0.0},
        )
    if bm25_norm is None:
        return (
            0.55 * coverage + 0.45 * dense_norm,
            {"w_coverage": 0.55, "w_bm25": 0.0, "w_dense": 0.45},
        )
    return (
        0.45 * coverage + 0.35 * bm25_norm + 0.20 * dense_norm,
        {"w_coverage": 0.45, "w_bm25": 0.35, "w_dense": 0.20},
    )
