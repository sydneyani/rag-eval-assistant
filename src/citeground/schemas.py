"""Shared data types for documents, retrieval hits, and answers."""

from pydantic import BaseModel, Field


class Section(BaseModel):
    """One headed section of a source document.

    Evaluation labels sections, not chunk offsets. Chunkers may split a
    section, but every chunk keeps the section id, so recall stays comparable
    across chunking experiments.
    """

    doc_id: str
    section_id: str
    title: str
    text: str
    heading_path: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    doc_id: str
    title: str
    source_path: str
    metadata: dict[str, str] = Field(default_factory=dict)
    sections: list[Section]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    section_id: str
    doc_title: str
    section_title: str
    text: str
    position: int
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A retrieved chunk plus the scores that put it there.

    `score` is the value the caller should sort on: the fusion score before
    reranking, and the reranker score after. Component scores are kept so
    traces can explain a ranking.
    """

    chunk: Chunk
    score: float
    bm25_score: float | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
    features: dict[str, float] = Field(default_factory=dict)


class Usage(BaseModel):
    """Token counts and a cost figure.

    `cost_basis` says how `cost_usd` was produced. Offline providers report
    zero cost. OpenAI runs multiply provider-reported tokens by the list
    prices in settings, which are estimates, not an invoice.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float = 0.0
    cost_basis: str = "offline-zero"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.embedding_tokens


class StageTiming(BaseModel):
    name: str
    duration_ms: float


class Citation(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    section_id: str
    doc_title: str
    section_title: str
    score: float
    quote: str


class AnswerResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[ScoredChunk]
    provider: str
    model: str
    usage: Usage
    trace_id: str
    latency_ms: float
    abstained: bool = False
    stages: list[StageTiming] = Field(default_factory=list)
