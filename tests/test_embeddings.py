import numpy as np

from citeground.llm.hashing import HashingEmbedder


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=64)
    first = embedder.embed_query("fiscal 2024 revenue was $4.82 billion")
    second = embedder.embed_query("fiscal 2024 revenue was $4.82 billion")
    assert np.allclose(first, second)
    assert first.shape == (64,)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_hashing_embedder_ranks_a_near_copy_above_an_unrelated_sentence() -> None:
    embedder = HashingEmbedder(dim=128)
    query = embedder.embed_query("fiscal 2024 revenue was $4.82 billion")
    matrix = embedder.embed_documents(
        [
            "The quarry ridge power purchase agreement expires on June 30, 2026.",
            "Fiscal 2024 revenue was $4.82 billion compared with the prior year.",
        ]
    )
    scores = matrix @ query
    assert scores[1] > scores[0]


def test_empty_input_has_the_right_shape() -> None:
    embedder = HashingEmbedder(dim=32)
    assert embedder.embed_documents([]).shape == (0, 32)
    assert np.allclose(embedder.embed_query(""), np.zeros(32))
