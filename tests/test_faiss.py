import numpy as np
import pytest

from citeground.index.store import dense_search_faiss, dense_search_numpy

pytest.importorskip("faiss")


def test_faiss_flat_ip_matches_numpy_on_normalized_vectors() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(12, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = rng.normal(size=16).astype(np.float32)
    numpy_hits = dense_search_numpy(vectors, query, k=4)
    faiss_hits = dense_search_faiss(vectors, query, k=4)
    assert [row for row, _score in faiss_hits] == [row for row, _score in numpy_hits]
    for (_, numpy_score), (_, faiss_score) in zip(numpy_hits, faiss_hits, strict=True):
        assert faiss_score == pytest.approx(numpy_score, abs=1e-5)
