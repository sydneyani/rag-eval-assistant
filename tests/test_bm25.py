import math

import pytest

from citeground.retrieval.bm25 import BM25Index


def test_bm25_ranks_repeated_terms_above_a_single_mention() -> None:
    index = BM25Index(
        [
            ["cat", "sat", "mat"],
            ["dog", "sat", "log"],
            ["cat", "cat", "cat"],
        ]
    )
    ranking = [row for row, _score in index.topk(["cat"], k=3)]
    assert ranking[0] == 2
    assert ranking[1] == 0
    assert 1 not in ranking

    idf = math.log(1.0 + (3 - 2 + 0.5) / (2 + 0.5))
    scores = index.score(["cat"])
    # Document length equals the average, so the length-normalization term is 1.
    assert scores[0] == pytest.approx(idf)
    assert scores[2] == pytest.approx(idf * (3 * 2.5) / 4.5)


def test_bm25_ignores_unknown_terms() -> None:
    index = BM25Index([["revenue", "billion"], ["power", "campus"]])
    assert index.topk(["not-in-corpus"], k=2) == []
