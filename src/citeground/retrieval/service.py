"""Hybrid retriever.

Candidate generation:
  BM25 top `candidate_k` and dense top `candidate_k`, fused with RRF when
  the mode is hybrid. Single-retriever modes skip fusion and rank on their
  own score.

Reranking, when enabled, rescores the fused pool with coverage, BM25, and
dense features. Component scores are computed for every fused candidate,
including candidates that only one retriever returned, so a chunk is not
penalized for missing the other retriever's top-k cutoff.
"""

from __future__ import annotations

import numpy as np

from citeground.errors import ConfigError
from citeground.index.store import CorpusIndex, dense_search
from citeground.llm.base import Embedder
from citeground.retrieval.bm25 import BM25Index
from citeground.retrieval.fusion import reciprocal_rank_fusion
from citeground.retrieval.rerank import rerank
from citeground.schemas import ScoredChunk
from citeground.text import tokenize

_MODES = {"hybrid", "bm25", "dense"}


class HybridRetriever:
    def __init__(
        self,
        index: CorpusIndex,
        embedder: Embedder,
        *,
        mode: str = "hybrid",
        rerank_results: bool = True,
        rrf_k: int = 60,
        candidate_k: int = 20,
        vector_backend: str = "numpy",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Expected one of {sorted(_MODES)}.")
        self.index = index
        self.embedder = embedder
        self.mode = mode
        self.rerank_results = rerank_results
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.vector_backend = vector_backend
        self.bm25 = BM25Index([tokenize(chunk.text) for chunk in index.chunks])

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        mode: str | None = None,
        rerank_results: bool | None = None,
    ) -> list[ScoredChunk]:
        selected_mode = mode or self.mode
        if selected_mode not in _MODES:
            raise ValueError(f"Unknown retrieval mode '{selected_mode}'.")
        do_rerank = self.rerank_results if rerank_results is None else rerank_results
        if top_k <= 0 or not query.strip() or not self.index.chunks:
            return []

        query_tokens = tokenize(query)
        pool = min(max(self.candidate_k, top_k), len(self.index.chunks))
        want_bm25 = selected_mode in {"bm25", "hybrid"}
        want_dense = selected_mode in {"dense", "hybrid"}

        bm25_hits = self.bm25.topk(query_tokens, pool) if want_bm25 else []
        query_vector: np.ndarray | None = None
        dense_hits: list[tuple[int, float]] = []
        if want_dense:
            query_vector = self.embedder.embed_query(query)
            if self.index.vectors.size and query_vector.shape[0] != self.index.vectors.shape[1]:
                raise ConfigError(
                    f"Query embedding dim {query_vector.shape[0]} does not match the index "
                    f"dim {self.index.vectors.shape[1]}. Re-ingest with the same embedder."
                )
            dense_hits = dense_search(
                self.index.vectors,
                query_vector,
                pool,
                backend=self.vector_backend,
            )

        if selected_mode == "bm25":
            fused = bm25_hits
        elif selected_mode == "dense":
            fused = dense_hits
        else:
            fused = reciprocal_rank_fusion(
                [[row for row, _score in bm25_hits], [row for row, _score in dense_hits]],
                k=self.rrf_k,
            )

        bm25_scores = self.bm25.score(query_tokens) if want_bm25 else None
        dense_scores = None
        if query_vector is not None and self.index.vectors.size:
            dense_scores = self.index.vectors @ query_vector.astype(np.float32)

        hits: list[ScoredChunk] = []
        for row, score in fused[:pool]:
            bm25_score = float(bm25_scores[row]) if bm25_scores is not None else None
            dense_score = float(dense_scores[row]) if dense_scores is not None else None
            hits.append(
                ScoredChunk(
                    chunk=self.index.chunks[row],
                    score=float(score),
                    bm25_score=bm25_score,
                    dense_score=dense_score,
                )
            )
        if do_rerank:
            hits = rerank(query, hits)
        return hits[:top_k]
