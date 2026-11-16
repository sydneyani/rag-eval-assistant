import httpx
import numpy as np

from citeground.errors import ProviderError
from citeground.llm.openai import OpenAIChatModel, OpenAIEmbedder


def test_embedder_normalizes_and_records_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/embeddings")
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 9, "total_tokens": 9},
            },
        )

    embedder = OpenAIEmbedder(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    matrix = embedder.embed_documents(["one", "two"])
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix[0], [1.0, 0.0])
    assert np.allclose(matrix[1], [0.0, 1.0])
    assert embedder.last_prompt_tokens == 9


def test_chat_model_reads_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": "Revenue was $4.82 billion. [1]"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 7},
            },
        )

    model = OpenAIChatModel(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    completion = model.complete([{"role": "user", "content": "question"}])
    assert completion.text.startswith("Revenue")
    assert completion.prompt_tokens == 20
    assert completion.completion_tokens == 7


def test_missing_key_is_a_provider_error() -> None:
    try:
        OpenAIChatModel(api_key="")
    except ProviderError:
        return
    raise AssertionError("expected ProviderError")
