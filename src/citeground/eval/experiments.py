"""The comparison the README reports.

Four stacks, one corpus, one labeled set. The point of the table is the
difference between them, not a single headline number. Chunk text and the
embedder change the index, so each spec is ingested on its own.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExperimentSpec(BaseModel):
    name: str
    description: str
    retrieval_mode: str
    chunker: str
    chunk_size: int = 900
    chunk_overlap: int = 150
    rerank: bool = False
    answer_k: int = 5
    retrieve_k: int = 10


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="bm25-recursive-900",
        description="Lexical baseline. Section-aware recursive chunks, no dense channel.",
        retrieval_mode="bm25",
        chunker="recursive",
        rerank=False,
    ),
    ExperimentSpec(
        name="dense-hash-recursive-900",
        description="Hashed TF embeddings only. This is the offline stand-in for a dense embedder.",
        retrieval_mode="dense",
        chunker="recursive",
        rerank=False,
    ),
    ExperimentSpec(
        name="hybrid-rerank-recursive-900",
        description="BM25 plus hashed TF, fused with RRF, then the feature reranker. This is the default product stack.",
        retrieval_mode="hybrid",
        chunker="recursive",
        rerank=True,
    ),
    ExperimentSpec(
        name="hybrid-rerank-fixed-500",
        description="The same hybrid stack with smaller fixed character windows.",
        retrieval_mode="hybrid",
        chunker="fixed",
        chunk_size=500,
        chunk_overlap=80,
        rerank=True,
    ),
)


def specs_by_name(names: list[str] | None = None) -> list[ExperimentSpec]:
    if not names:
        return list(EXPERIMENTS)
    lookup = {spec.name: spec for spec in EXPERIMENTS}
    missing = [name for name in names if name not in lookup]
    if missing:
        known = ", ".join(lookup)
        raise ValueError(f"Unknown experiment {missing}. Known experiments: {known}")
    return [lookup[name] for name in names]


class Aggregate(BaseModel):
    n: int
    n_answerable: int
    n_abstain: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    faithfulness: float
    answer_relevance: float
    citation_support: float
    abstention_accuracy: float
    mean_latency_ms: float = 0.0
    extra: dict[str, float] = Field(default_factory=dict)
