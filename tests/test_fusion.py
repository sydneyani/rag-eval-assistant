from citeground.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_matches_the_sigir_formula() -> None:
    fused = reciprocal_rank_fusion([[0, 1, 2], [2, 0, 3]], k=60)
    scores = dict(fused)
    assert scores[0] == 1 / 61 + 1 / 62
    assert scores[2] == 1 / 63 + 1 / 61
    assert scores[1] == 1 / 62
    assert scores[3] == 1 / 63
    assert [doc_id for doc_id, _score in fused] == [0, 2, 1, 3]


def test_rrf_keeps_a_hit_that_only_one_list_returned() -> None:
    fused = dict(reciprocal_rank_fusion([[5], []], k=60))
    assert fused == {5: 1 / 61}
