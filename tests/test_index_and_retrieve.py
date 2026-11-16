import numpy as np

from citeground.config import Settings
from citeground.index.store import CorpusIndex, dense_search
from citeground.ingest.pipeline import ingest_corpus
from citeground.llm.factory import build_embedder
from citeground.retrieval.service import HybridRetriever


def test_dense_search_returns_the_best_dot_product() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    hits = dense_search(vectors, np.asarray([1.0, 0.0], dtype=np.float32), k=2)
    assert hits[0][0] == 0
    assert hits[1][0] == 2


def test_index_roundtrip(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        embedding_provider="hashing",
        embedding_dim=64,
        chunker="recursive",
        chunk_size=700,
        chunk_overlap=80,
    )
    result = ingest_corpus(
        settings,
        corpus_dir=_corpus(),
        index_dir=tmp_path / "index",
    )
    assert result.documents == 3
    assert result.chunks > 10
    loaded = CorpusIndex.load(tmp_path / "index")
    assert len(loaded.chunks) == result.chunks
    assert loaded.vectors.shape[0] == result.chunks
    assert loaded.manifest["embedding_provider"] == "hashing"


def test_hybrid_retrieval_finds_labeled_sections(tmp_path) -> None:
    corpus = _corpus()
    settings = Settings(
        _env_file=None,
        embedding_provider="hashing",
        embedding_dim=256,
        chunker="recursive",
        chunk_size=900,
        chunk_overlap=150,
        vector_store="numpy",
    )
    ingest_corpus(settings, corpus_dir=corpus, index_dir=tmp_path / "index")
    index = CorpusIndex.load(tmp_path / "index")
    retriever = HybridRetriever(
        index,
        build_embedder(settings),
        mode="hybrid",
        rerank_results=True,
        vector_backend="numpy",
    )
    expectations = {
        "When does the Quarry Ridge power purchase agreement expire?": "helios-10k-fy2024/risk-power",
        "When does Helios expect HIR-4 customer sampling?": "helios-q4-fy2024-letter/hir4",
        "What revenue did the fiscal 2023 Form 10-K report?": "helios-10k-fy2023/results-2023",
        "What was the Alvarez v. Helios Systems case about?": "helios-10k-fy2024/risk-litigation",
    }
    for question, section_id in expectations.items():
        hits = retriever.retrieve(question, top_k=5)
        found = [hit.chunk.section_id for hit in hits]
        assert section_id in found, (question, found)


def test_bm25_and_dense_modes_both_return_hits(tmp_path) -> None:
    settings = Settings(_env_file=None, embedding_provider="hashing", embedding_dim=128)
    ingest_corpus(settings, corpus_dir=_corpus(), index_dir=tmp_path / "index")
    retriever = HybridRetriever(
        CorpusIndex.load(tmp_path / "index"),
        build_embedder(settings),
        mode="hybrid",
    )
    question = "What was fiscal 2024 revenue compared with fiscal 2023 revenue?"
    for mode in ("bm25", "dense", "hybrid"):
        hits = retriever.retrieve(question, top_k=3, mode=mode)
        assert hits
        assert hits[0].chunk.section_id == "helios-10k-fy2024/results"


def _corpus():
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "data" / "corpus"
