from citeground.retrieval.rerank import rerank
from citeground.schemas import Chunk, ScoredChunk


def _hit(text: str, bm25: float, dense: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id="d#000",
            doc_id="d",
            section_id="d/s",
            doc_title="Doc",
            section_title="Section",
            text=text,
            position=0,
        ),
        score=0.0,
        bm25_score=bm25,
        dense_score=dense,
    )


def test_rerank_prefers_coverage_over_a_higher_unrelated_bm25() -> None:
    query = "Lattice Silicon remaining purchase commitment"
    relevant = _hit(
        "The remaining minimum purchase commitment to Lattice Silicon was $900 million.",
        bm25=1.0,
        dense=0.2,
    )
    distractor = _hit(
        "Revenue grew because customers ordered more racks in the second half.",
        bm25=5.0,
        dense=0.9,
    )
    ranked = rerank(query, [distractor, relevant])
    assert ranked[0].chunk.text.startswith("The remaining")
    assert ranked[0].rerank_score == ranked[0].score
    assert "coverage" in ranked[0].features
